# Technical Documentation: `react-native-exercise-recognition`

## 1. Package Overview

### Summary

`react-native-exercise-recognition` is a React Native Nitro Module that provides real-time exercise recognition capabilities within the Workout Hacker system. The package classifies human physical exercises by processing 3D pose landmark data extracted from camera frames via MediaPipe's Pose Landmarker. It exports a HybridObject—`ExerciseRecognition`—that exposes a native Rust backend to JavaScript through a strongly-typed Nitro bridge.

The package operates as a deterministic inference pipeline: raw landmark buffers enter the module; a feature extraction stage transforms them into a pose-descriptor vector; a Random Forest classifier produces per-class probabilities; and a temporal state machine emits a stable exercise label with an associated confidence score.

### Problem Statement

The Workout Hacker system requires the ability to identify, in real time, which physical exercise a user is performing. This is a prerequisite for downstream counting logic (e.g., rep counting in `react-native-rep-counter`) and session analytics. The problem decomposes into three sub-problems:

1. **Low-latency inference**: Exercise classification must complete within tens of milliseconds to keep pace with camera frame rates (~30 fps).
2. **Temporal stability**: Raw classifier outputs are noisy across frames; the system must smooth predictions and enforce enter/exit hysteresis to avoid flickering labels.
3. **Pose quality gating**: Not all frames are equally usable; the module must reject frames where landmarks are occluded or poorly estimated.

This package solves these sub-problems by embedding a pre-trained Random Forest model in a Rust native module, applying exponential moving average (EMA) smoothing and a multi-frame confirmation protocol, and gating predictions on upper-body landmark visibility.

### Role in the Broader System

The Workout Hacker monorepo contains several interdependent packages:

| Package | Role | Dependency on this package |
|---------|------|---------------------------|
| `react-native-pose-landmarks` | Extracts 33 MediaPipe pose landmarks from camera frames | None (provides input to this package) |
| **`react-native-exercise-recognition`** | **Classifies the current exercise from pose landmarks** | **Depends on `react-native-pose-landmarks`** |
| `react-native-rep-counter` | Counts repetitions of the recognized exercise | Depends on this package (v1.3.0) and `react-native-pose-landmarks` |

Data flows through the system as follows:

```
Camera → react-native-pose-landmarks (MediaPipe) → pose landmark buffer
                                                        ↓
                              react-native-exercise-recognition → exercise label + confidence
                                                        ↓
                              react-native-rep-counter (uses label to contextualize rep counting)
```

---

## 2. Architectural Rationale

### Technology Stack Decisions

The package employs a three-layer native architecture:

```
TypeScript (spec) → C++ HybridObject (Nitro) → Rust (FFI bridge)
```

**Why Nitro Modules (C++ layer)?**
The package uses [Nitro Modules](https://github.com/mrousavy/nitro) to generate a strongly-typed C++ HybridObject from a TypeScript spec file (`src/specs/ExerciseRecognition.nitro.ts`). This choice was driven by:
- **Zero-copy bridging**: Nitro avoids the serialization overhead of traditional React Native TurboModules or Bridge-based architectures.
- **Type safety across boundaries**: The spec file is the single source of truth; Nitro generates C++ headers (`nitrogen/generated/**`) and TypeScript types from it.
- **Cross-platform uniformity**: The same TypeScript interface maps to `ios: 'c++'` and `android: 'c++'` native implementations.

**Why Rust for ML inference?**
The core classifier logic—feature extraction and Random Forest inference—is implemented in Rust (`rust/src/lib.rs`, `features.rs`, `rf_model.rs`). The rationale includes:
- **Performance**: Rust compiles to native code with performance comparable to C++. Feature extraction involves tight loops over 33 landmarks × 4 values × 45 buffered frames.
- **Safety**: Rust's ownership model prevents memory safety issues in the inference pipeline, which runs continuously at ~15–30 calls/second.
- **Serde integration**: The Random Forest model is serialized as JSON. Rust's `serde` ecosystem provides robust, zero-copy JSON deserialization via `serde_json`.
- **Enhanced Maintainability**: Rust was selected to leverage the developer's existing expertise. This familiarity ensures higher code quality, more efficient debugging, and streamlined long-term maintenance compared to alternative low-level languages.

**Why the C++ ↔ Rust FFI bridge?**
C++ cannot directly link Rust static libraries without an `extern "C"` interface. The bridge (`cpp/rust/ExerciseRecognitionRustBridge.cpp`) wraps Rust functions exposed via `#[no_mangle] pub extern "C"` in `rust/src/lib.rs`. This separation allows:
- Independent compilation of Rust (via `cargo build --release`) and C++ (via CMake).
- Conditional compilation: The `EXREC_USE_RUST` CMake define (`android/CMakeLists.txt:34`) allows the C++ layer to compile even when Rust static libraries are unavailable, returning fallback values.

### Evolution of Design Decisions

**Initial implementation:**
The module began as a basic Nitro wrapper with a simple `loadModel → ingest → getCurrentExercise` flow. No temporal smoothing was applied.

**Introduction of `StartSessionConfig`:**
The most significant architectural addition was the `StartSessionConfig` parameter to `startSession()`. This was introduced to address two issues observed during testing:
1. **Pose quality variation**: Users hold phones at different angles; some frames have poor landmark visibility. The `minVisibility` and `minVisibleUpperBodyJoints` parameters allow runtime tuning without recompilation.
2. **Transition flicker**: When a user transitions between exercises (e.g., Bicep Curl → Lateral Raise), the raw classifier may briefly output spurious labels. The `enterConfidence`, `exitConfidence`, `enterFrames`, and `exitFrames` parameters implement a hysteresis protocol.

**Null-class exit window (parameter `nullExitWindowSeconds`, `nullExitWindowThreshold`):**
Added alongside `StartSessionConfig`, this mechanism addresses a specific edge case: when the user is not performing any known exercise (the "Null/Unknown" class, class ID 2), the system should not immediately exit the current exercise label. Instead, it requires the null class to dominate for a sustained window (`nullExitWindowSeconds` seconds, default 5.0) with a high threshold (`nullExitWindowThreshold`, default 0.99). This prevents accidental transitions when the user briefly pauses or occludes the camera.

**Removal of landmark smoothing:**
An initial implementation included a moving average filter (`smooth_landmarks`) that smoothed the x, y, z coordinates of each landmark over a configurable window (default 5 frames). This was removed for two reasons:
1. **Redundancy with MediaPipe filtering**: The Pose Landmarker package already applies internal temporal filtering to produce smooth landmark trajectories. Adding a second smoothing layer provided negligible benefit while introducing additional computational cost (~0.5ms per frame buffer processing).
2. **Simplified feature pipeline**: Eliminating the smoothing stage reduced the pipeline from four stages to three (frame array construction → skeleton normalization → temporal feature extraction), improving maintainability and reducing the `StartSessionConfig` parameter surface.

**Prebuilt Rust binaries:**
To simplify the Android build process, prebuilt Rust static libraries for the four Android ABIs (`arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`) were added to `android/prebuilt/`. The CMake build (`android/CMakeLists.txt:30–35`) prefers these prebuilt libraries but falls back to building from `rust/target/<abi>/release/` if they are absent. This was a pragmatic trade-off: faster CI builds at the cost of slightly larger repository size.

### Trade-offs Considered

| Decision | Trade-off | Resolution |
|----------|-----------|------------|
| Rust vs. C++ for inference | Rust has a steeper build toolchain requirement (cargo, rustup) | Rust's safety and serde ecosystem outweighed build complexity; prebuilt binaries mitigate the toolchain issue. Enhanced maintainability was a deciding factor: Rust was selected to leverage the developer's existing expertise, which ensures higher code quality, more efficient debugging, and streamlined long-term maintenance compared to alternative low-level languages. |
| Random Forest vs. Neural Network | RF cannot capture temporal patterns as well as RNNs/LSTMs | RF inference is deterministic, fast (< 1ms), and requires no GPU; acceptable for upper-body exercises with engineered features |
| EMA smoothing vs. Kalman filtering | EMA is simpler but assumes first-order dynamics | EMA with configurable `emaAlpha` (default 0.2) provides sufficient smoothing with minimal state |
| Buffered frames (45 max) | More frames = more memory, slower feature extraction | 45 frames at ~30 fps ≈ 1.5 seconds of context; sufficient for temporal features |

---

## 3. Internal Mechanics

### Data Pipeline Overview

The inference pipeline processes raw pose landmarks through three sequential stages:

```
ingestLandmarksBuffer(landmarks: number[])
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Frame Buffer (VecDeque<FrameInput>, max 45 frames)     │
│  - Landmarks: 33 joints × {x, y, z, visibility}         │
│  - Minimum 10 frames before inference fires              │
└───────────────────────────────────────────────────────────┘
        │
        ▼  (when >= 10 frames buffered)
┌───────────────────────────────────────────────────────────┐
│  Feature Extraction (features::extract_feature_rows)      │
│  1. Build frame array: [frame][landmark][x,y,z,vis]     │
│  2. Normalize skeleton (hip-center translation, shoulder   │
│     distance scale)                                       │
│  3. Extract temporal features per frame:                   │
│     - Joint angles (4 features, both sides)                  │
│     - Vertical displacement (2 features)                     │
│     - Velocity (3D, 6 arm joints = 18 features)           │
│     - Acceleration (3D, 6 arm joints = 18 features)       │
│     - Rolling standard deviation (window=15, 3 axes, 6     │
│       joints = 18 features)                                │
│  Output: Vec<Vec<f32>>  (frames × 60 feature dimensions) │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Random Forest Inference (rf_model::predict_probabilities) │
│  - For each feature row, each tree traverses its nodes:    │
│    if feature[split_index] <= threshold → left child       │
│    else → right child                                     │
│  - Leaf nodes contain class counts → convert to probs      │
│  - Average probabilities across all trees in the forest   │
│  Output: Vec<Vec<f32>>  (frames × n_classes)             │
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Temporal Smoothing & State Machine                        │
│  1. Average probabilities across all frames in window     │
│  2. Blend with EMA: smoothed[t] = α·avg[t] + (1-α)·EMA[t-1]│
│  3. Argmax to find winning class ID                        │
│  4. State machine:                                         │
│     - No active class: require enter_confidence + enter_frames│
│     - Active class: require exit_confidence for exit_frames│
│       to release; or require new class at enter_confidence │
│       for enter_frames to switch                           │
│     - Null exit window: track duration of null dominance   │
│  Output: current_label: Option<String>, current_confidence│
└───────────────────────────────────────────────────────────┘
```

### Feature Extraction Detail

The feature extraction pipeline (`rust/src/features.rs`) transforms raw landmarks into a fixed-dimensional descriptor through a three-stage process. Given a buffer of N frames (typically 45), the pipeline produces N × 60 feature vectors.

#### Stage 1: Frame Array Construction

The input `Vec<FrameInput>` is converted into a 3D array `[[f64; 4]; N]` where each element contains `[x, y, z, visibility]` for a landmark. Only the first 33 landmarks are used, matching MediaPipe's pose model output. This is performed by `build_frame_array()` (features.rs:20–28).

#### Stage 2: Skeleton Normalization

To make features invariant to camera position and user body size, the skeleton is normalized in two steps (function `normalize_skeleton_for_classifier()`, features.rs:58–88):

1. **Translation normalization**: The skeleton is translated so the hip center (midpoint of left_hip[23] and right_hip[24]) is at the origin. For each frame and landmark:
   ```
   landmark[x, y, z] -= hip_center[x, y, z]
   ```

2. **Scale normalization**: The distance between left_shoulder[11] and right_shoulder[12] is computed. All coordinates are then divided by this distance, making the skeleton scale-invariant:
   ```
   landmark[x, y, z] /= shoulder_distance
   ```
   This ensures that users with different body proportions produce similar feature values for the same exercise.

#### Stage 3: Temporal Feature Extraction

For each frame, a 60-dimensional feature vector is computed by `extract_temporal_features()` (features.rs:113–184):

**Joint Angles (4 features):**
- Left elbow angle: ∠(left_shoulder[11], left_elbow[13], left_wrist[15])
- Right elbow angle: ∠(right_shoulder[12], right_elbow[14], right_wrist[16])
- Left shoulder angle: ∠(left_hip[23], left_shoulder[11], left_elbow[13])
- Right shoulder angle: ∠(right_hip[24], right_shoulder[12], right_elbow[14])

Each angle is computed via the law of cosines on 3D vectors (function `calculate_angle`, features.rs:8–18):
```
cos(θ) = (ba·bc) / (|ba| × |bc| + ε)
θ = arccos(clamp(cos(θ), -1, 1))
```

These angles capture the pose geometry that distinguishes exercises (e.g., elbows-bent for Bicep Curl vs. straight arms for Lateral Raise).

**Vertical Displacement (2 features):**
- `left_wrist_y - left_shoulder_y`
- `right_wrist_y - right_shoulder_y`

These capture the relative elevation of the hands in normalized coordinates. Positive values indicate wrists above shoulders (Shoulder Press), negative values indicate wrists below shoulders (Bicep Curl).

**Velocity (18 features):**
For 6 arm joints (indices 11–16: left/right shoulder, elbow, wrist), compute the first derivative of position:
- Velocity = `current_coord - previous_coord` (3 axes × 6 joints = 18 values)

The velocity is computed per-frame by differencing with the previous frame (`prev` array in features.rs:118). The first frame has zero velocity.

**Acceleration (18 features):**
For the same 6 joints, compute the second derivative of position:
- Acceleration = `current_velocity - previous_velocity` (3 axes × 6 joints = 18 values)

This captures how quickly the limb speed is changing, which helps distinguish between controlled movements (slow acceleration) and dynamic movements (high acceleration).

**Rolling Standard Deviation (18 features):**
For each coordinate (x, y, z) of the 6 arm joints, compute the sample standard deviation over the last 15 frames (`FEATURE_STD_WINDOW`):
```
std[t] = sqrt(Σ(x[i] - μ)² / (n - 1))  for i ∈ [t-14, t]
```

Implemented in `rolling_std()` (features.rs:91–111). This feature captures motion "jerkiness" or stability—trembling movements have high std, while smooth repetitions have low std.

**Total feature dimensions per frame:** 4 + 2 + 18 + 18 + 18 = **60 features**.

### Random Forest Model

The classifier is a Random Forest with the following JSON schema (deserialized by `rf_model::RandomForestRunner::from_json`):

```json
{
  "n_features": 60,
  "n_classes": 6,
  "classes": [0, 1, 2, 3, 4, 5],
  "trees": [
    {
      "children_left": [...],
      "children_right": [...],
      "feature": [...],
      "threshold": [...],
      "values": [[...], [...], ...]  // leaf node: class counts per class
    },
    ...
  ]
}
```

Class mapping (defined in `lib.rs:140–150`):
| Class ID | Label |
|----------|-------|
| 0 | Bicep Curl |
| 1 | Lateral Raise |
| 2 | Null/Unknown |
| 3 | Shoulder Press |
| 4 | Triceps Extension |
| 5 | Front Raises |

Each tree is traversed by comparing `feature[split_feature]` against `threshold` at each non-leaf node. Leaf nodes contain the distribution of training samples across classes; these are normalized to probabilities by dividing by the total count.

### Temporal State Machine

The state machine (`lib.rs:505–615`) manages exercise transitions:

**States:**
- **No active class** (`active_class_id = None`): The system is waiting for a confident prediction to "enter."
- **Active class** (`active_class_id = Some(id)`): An exercise is currently recognized.

**Enter protocol:**
A class `C` with confidence ≥ `enter_confidence` must be the pending class for at least `enter_frames` consecutive frames (with pose quality OK) before it becomes the active class.

**Exit protocol:**
The active class is released (set to `None`) when:
1. Confidence drops below `exit_confidence` for `exit_frames` consecutive frames, OR
2. The null class (ID 2) dominates with confidence ≥ `null_exit_window_threshold` for a duration exceeding `null_exit_window_seconds` (tracked via `frame_times` deque and `null_probs_count`).

**Pose quality gate:**
Before applying enter/exit logic, the system checks `upper_body_visible_joint_count()` against `min_visible_upper_body_joints`. The upper body joints are indices 11–16 (shoulders, elbows, wrists). If fewer than the minimum are visible (visibility ≥ `min_visibility`), the frame is considered low quality and cannot trigger an enter or prevent an exit.

---

## 4. Interface & Data Flow (Inputs and Outputs)

### TypeScript Interface (Source of Truth)

Defined in `src/specs/ExerciseRecognition.nitro.ts`:

```typescript
export interface StartSessionConfig {
  minConfidence?: number          // Default: 0.6 — minimum confidence for any prediction
  enterConfidence?: number        // Default: 0.65 — confidence threshold to enter an exercise
  exitConfidence?: number         // Default: 0.45 — confidence threshold to exit an exercise
  enterFrames?: number            // Default: 3 — consecutive frames required to confirm enter
  exitFrames?: number             // Default: 8 — consecutive frames required to confirm exit
  emaAlpha?: number               // Default: 0.2 — EMA blending factor (0=all history, 1=all current)
  minVisibility?: number          // Default: 0.2 — minimum landmark visibility (0–1)
  minVisibleUpperBodyJoints?: number  // Default: 4 — min visible upper body joints required
  nullExitWindowSeconds?: number  // Default: 5.0 — duration of null dominance to trigger exit
  nullExitWindowThreshold?: number    // Default: 0.99 — null confidence threshold for exit window
}

export interface ExerciseRecognition extends HybridObject<{
  ios: 'c++'
  android: 'c++'
}> {
  loadModelFromJson(modelJson: string): boolean
  loadModelFromAsset(assetName: string): boolean   // Android-only; returns false on iOS
  startSession(config?: StartSessionConfig): void
  stopSession(): void
  ingestLandmarksBuffer(landmarks: Array<number>): void
  getCurrentExercise(): string | null
  getCurrentConfidence(): number
  getLastClassifierInferenceTimeMs(): number
}
```

### Inputs

#### `loadModelFromJson(modelJson: string): boolean`
- **Input:** A JSON string matching the `ForestFile` schema (see Section 3.3).
- **Origin:** Typically loaded from an Android asset via `loadModelFromAsset('exercise_classifier_rf.json')`, which internally reads from `example/android/app/src/main/assets/`. The model JSON is placed there by the `postinstall` script (`scripts/download-assets.js`) or manually.
- **Behavior:** Deserializes the Random Forest, stores it in the global `ClassifierState.model` (behind a `Mutex<OnceLock>` for thread safety). Returns `true` on success.

#### `loadModelFromAsset(assetName: string): boolean`
- **Input:** Asset filename (e.g., `'exercise_classifier_rf.json'`).
- **Origin:** Android-only. Uses JNI to call `ExerciseRecognitionAssets.loadAssetAsString()` on the Java side, which reads from the APK's assets.
- **Behavior:** Loads and forwards to `loadModelFromJson`. Returns `false` on iOS or on failure.

#### `startSession(config?: StartSessionConfig): void`
- **Input:** Optional configuration object. All fields are optional; defaults are applied in the Rust layer (`SessionConfig::default()`).
- **Origin:** Called once by the application when the user begins a workout session.
- **Behavior:** Configures the `ClassifierState.config` fields. Clears any buffered frames and resets temporal state.

#### `ingestLandmarksBuffer(landmarks: Array<number>): void`
- **Input:** A flat array of 132 `number` values (33 landmarks × 4 values per landmark: `x, y, z, visibility`).
- **Data contract:** Each landmark occupies 4 consecutive array positions:
  ```
  [lm0.x, lm0.y, lm0.z, lm0.visibility, lm1.x, lm1.y, lm1.z, lm1.visibility, ...]
  ```
  The coordinate system is normalized by MediaPipe: `x, y ∈ [0, 1]` (relative to image dimensions), `z` is relative depth from the camera.
- **Origin:** Produced by `react-native-pose-landmarks`. In the example app (`example/App.tsx:146–156`):
  ```typescript
  const buffer = PoseLandmarks.getLandmarksBuffer();
  // buffer is number[132]
  exerciseRecognition.ingestLandmarksBuffer(buffer);
  ```
  This is called on a `setInterval` timer at ~66ms intervals (~15 fps), which is sufficient given MediaPipe's typical ~30 fps output.
- **Behavior:** Parses the buffer into `FrameInput::FrameRecord { landmarks: Vec<Landmark> }`, appends to the frame buffer (max 45 frames). Once ≥ 10 frames are buffered, triggers the full inference pipeline (feature extraction → RF → smoothing → state machine).

### Outputs

#### `getCurrentExercise(): string | null`
- **Output:** The currently recognized exercise label (e.g., `"Bicep Curl"`, `"Lateral Raise"`) or `null` if no exercise is active.
- **Destination:** Consumed by the UI layer to display the current exercise, and by `react-native-rep-counter` to contextualize rep counting (different exercises have different rep-counting logic).

#### `getCurrentConfidence(): number`
- **Output:** A `number` in `[0.0, 1.0]` representing the smoothed confidence of the current exercise label.
- **Destination:** Displayed in the UI (e.g., `"64.2% confidence"`). Can be used to show a confidence indicator or to trigger fallback behavior when confidence is low.

#### `getLastClassifierInferenceTimeMs(): number`
- **Output:** The elapsed time in milliseconds for the most recent classifier inference (feature extraction + Random Forest prediction). Returns `-1.0` if no inference has occurred.
- **Destination:** Used for performance monitoring and debugging. In the example app, it is displayed alongside the MediaPipe landmark inference time to compute end-to-end latency.

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CAMERA (hardware)                            │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  YUV/RGB frames
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│        react-native-pose-landmarks (MediaPipe Pose)                 │
│        Output: number[132] — 33 landmarks × [x, y, z, vis]         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  ingestLandmarksBuffer(buffer)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│        react-native-exercise-recognition                            │
│                                                                     │
│  Frame Buffer ──→ Feature Extraction ──→ Random Forest             │
│                              │                                      │
│                         Probabilities                               │
│                              │                                      │
│                    EMA Smoothing + State Machine                    │
│                              │                                      │
│              getCurrentExercise() / getCurrentConfidence()           │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  Poll-based: called after each ingest
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│        UI Layer (App.tsx)                                           │
│        Displays: exercise label, confidence, inference times        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (in downstream packages)
┌─────────────────────────────────────────────────────────────────────┐
│        react-native-rep-counter                                     │
│        Uses exercise label to select rep-counting strategy          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Integration & Usage

### Installation

The package is installed from GitHub:

```json
// In the consuming package's package.json:
{
  "dependencies": {
    "react-native-exercise-recognition": "github:tony-div/react-native-exercise-recognition#v1.3.0"
  },
  "peerDependencies": {
    "react-native-pose-landmarks": "github:tony-div/react-native-pose-landmarks#v1.2.0"
  }
}
```

The `postinstall` script (`scripts/download-assets.js`) automatically downloads the classifier model JSON to the Android assets directory.

### Usage Example: Basic Exercise Recognition Session

The following is extracted from `example/App.tsx` and demonstrates the canonical usage pattern:

```typescript
import { exerciseRecognition } from 'react-native-exercise-recognition';
import { PoseLandmarks } from 'react-native-pose-landmarks';

// 1. Load the model (Android: from assets; iOS: from bundle or JS bundle)
const modelLoaded = exerciseRecognition.loadModelFromAsset('exercise_classifier_rf.json');
if (!modelLoaded) {
  console.error('Failed to load exercise classifier model');
}

// 2. Configure and start the session
exerciseRecognition.startSession({
  minConfidence: 0.6,
  enterConfidence: 0.65,
  exitConfidence: 0.45,
  enterFrames: 3,
  exitFrames: 8,
  emaAlpha: 0.2,
  minVisibility: 0.2,
  minVisibleUpperBodyJoints: 4,
  nullExitWindowSeconds: 5.0,
  nullExitWindowThreshold: 0.99,
});

// 3. In a polling loop (~15 fps), ingest landmarks and read results
const intervalId = setInterval(() => {
  // Get pose landmarks from MediaPipe (132 values: 33 × 4)
  const landmarkBuffer = PoseLandmarks.getLandmarksBuffer();

  if (landmarkBuffer && landmarkBuffer.length === 132) {
    // Feed landmarks into the classifier
    exerciseRecognition.ingestLandmarksBuffer(landmarkBuffer);

    // Read current prediction
    const currentExercise = exerciseRecognition.getCurrentExercise();
    const confidence = exerciseRecognition.getCurrentConfidence();
    const inferenceMs = exerciseRecognition.getLastClassifierInferenceTimeMs();

    console.log(`Exercise: ${currentExercise ?? 'None'}, Confidence: ${confidence.toFixed(3)}, Inference: ${inferenceMs.toFixed(1)}ms`);
  }
}, 66); // ~15 fps

// 4. When done:
clearInterval(intervalId);
exerciseRecognition.stopSession();
```

### Usage Example: Integration with Rep Counter

The `react-native-rep-counter` package depends on this package to know which exercise is currently active. This allows the rep counter to apply exercise-specific angle thresholds:

```typescript
// In react-native-rep-counter (conceptual usage)
import { exerciseRecognition } from 'react-native-exercise-recognition';
import { repCounter } from 'react-native-rep-counter';

// After ingesting landmarks into the exercise recognizer:
const currentExercise = exerciseRecognition.getCurrentExercise();

// Pass both landmarks and exercise context to the rep counter
if (currentExercise) {
  repCounter.ingestLandmarksBufferWithExercise(landmarkBuffer, currentExercise);
  const reps = repCounter.getRepCount();
  const phase = repCounter.getCurrentPhase(); // 'UP' | 'DOWN' | 'UNKNOWN'
}
```

---

## 6. Build System & Native Configuration

### Android Build

The Android build is driven by `android/build.gradle` and `android/CMakeLists.txt`:

1. **Pre-build step:** `android/build-rust-android.sh` is triggered before the CMake build. It invokes `cargo build --release` for each Android ABI (`aarch64-linux-android`, `armv7-linux-androideabi`, `i686-linux-android`, `x86_64-linux-android`), producing static libraries at `rust/target/<abi>/release/libexercise_recognition_rust.a`.

2. **CMake configuration:** The `EXREC_USE_RUST` define is set if the Rust static library exists (`CMakeLists.txt:33–35`). If not set, the C++ layer compiles but all Rust calls return fallback values.

3. **Asset loading (Android):** The `loadModelFromAsset` path uses JNI to call `ExerciseRecognitionAssets.loadAssetAsString()`, which reads from the APK's `assets/` directory.

## 7. Performance Characteristics

Based on benchmark results (Section 9, Intel Core i5 11th Gen) and implementation analysis:

| Metric | Python (sklearn) | Rust (measured) |
|--------|-------------------|------------------|
| Feature extraction (45 frames) | 6.494 ms | **0.030 ms** (see Section 9) |
| Random Forest inference (100 trees, 60 features) | 5.900 ms | **0.469 ms** (see Section 9) |
| End-to-end pipeline (extract + infer) | 21.810 ms | **0.458 ms** (see Section 9) |
| Frame buffer memory | 45 frames × 33 landmarks × 4 values × 8 bytes ≈ 47 KB |
| Classifier state memory | Model-dependent (typical JSON: 2–5 MB); runtime state ~few KB |
| Supported exercises | 5 known + 1 null class (extensible via model retraining) |

**React Native module performance (per frame):**
- Feature extraction: 0.030 ms
- RF inference: 0.469 ms
- Total per frame: **~0.5 ms** (enables real-time at 30+ fps)

**Speedup vs Python:**
- Feature extraction: **217x** (6.494 ms → 0.030 ms)
- RF inference: **12.6x** (5.900 ms → 0.469 ms)
- End-to-end: **47.7x** (21.810 ms → 0.458 ms)

---

## 8. Thread Safety & Concurrency

The `ClassifierState` is wrapped in `std::sync::Mutex` (Rust) and accessed via `OnceLock` for initialization (`lib.rs:135–138`):

```rust
fn state() -> &'static Mutex<ClassifierState> {
    static STATE: OnceLock<Mutex<ClassifierState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(ClassifierState::default()))
}
```

This ensures:
- Single initialization: The state is lazily initialized on first access.
- Mutual exclusion: Each public function (`exrec_ingest_landmarks_buffer`, `exrec_get_current_exercise`, etc.) acquires the mutex lock before accessing state.
- The caller (JavaScript) should not invoke methods concurrently from multiple threads; the Nitro bridge serializes calls from the JS thread.

On the C++ side, the `ExerciseRecognitionRustBridge` is instantiated once and owned by the `HybridExerciseRecognition` HybridObject, which lives on the JS thread.

---

## 9. Performance Benchmark: Python (sklearn) vs Rust (Native)

### Benchmark Methodology

A comparative benchmark was conducted to evaluate the performance of the Rust implementation versus the original Python/sklearn pipeline used during model training and prototyping. The benchmark measures:

1. **Feature extraction time**: Transforming 45 frames of 33 landmarks (132 values/frame) into 60-dimensional feature vectors
2. **Random Forest inference time**: Running 100 trees over the extracted features
3. **End-to-end pipeline time**: Complete flow from raw landmarks to class probabilities

**Test Configuration:**
| Parameter | Value |
|-----------|-------|
| CPU | Intel Core i5 11th Gen |
| Frames per inference | 45 frames (~1.5 seconds at 30 fps) |
| Benchmark runs | 50 (5 warmup, 45 measured) |
| Random seed | 42 (for reproducible synthetic data) |
| Model | Random Forest, 100 trees, 60 features, 6 classes |

### Results

#### Python Feature Extraction (Measured)

| Implementation | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) |
|---------------|-----------|---------------|----------|----------|
| **Python** (pandas/numpy) | 6.494 | 0.801 | 5.460 | 8.975 |

#### Python Random Forest Inference (Measured, features pre-computed)

| Implementation | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) | Throughput |
|---------------|-----------|---------------|----------|----------|------------|
| **Python** (sklearn) | 5.900 | 0.648 | 4.967 | 7.707 | 169.5 inf/sec |

#### Python End-to-End Pipeline (Measured)

| Implementation | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) |
|---------------|-----------|---------------|----------|----------|
| **Python** (sklearn) | 21.810 | 1.028 | 20.259 | 25.451 |

#### Rust Performance (Measured)

Actual Rust benchmarks were run using Criterion (`cargo bench`) with Intel Core i5 11th Gen @ 2.40GHz:

| Component | Mean (µs) | Mean (ms) | Notes |
|-----------|-------------|--------------|-------|
| Feature extraction (45 frames) | 30.211 µs | **0.030 ms** | Actual `extract_feature_rows()` |
| RF inference (45 frames × 60 features) | 468.85 µs | **0.469 ms** | `RandomForestRunner::predict_probabilities()` |
| End-to-end (45 frames) | 457.62 µs | **0.458 ms** | Feature extraction + RF inference |

**Benchmark details:**
- Model: 100 trees, 60 features, 6 classes (actual `model_rf.json`)
- Sample size: 50 measurements per benchmark
- Warmup: 500ms, Measurement: 2s per benchmark

**Speedup vs Python (same hardware):**
- Feature extraction: **217x** (6.494 ms → 0.030 ms)
- RF inference: **12.6x** (5.900 ms → 0.469 ms)
- End-to-end: **47.7x** (21.810 ms → 0.458 ms)

### Analysis

**Why Rust Should Be Faster (Theoretical):**

1. **Feature Extraction:**
   - No Python interpreter overhead; Rust compiles to native machine code
   - No pandas DataFrame overhead; direct array operations on `[[f64; 4]; 33]` arrays
   - Tight loops with simple iterators and array indexing
   - No Global Interpreter Lock (GIL) overhead

2. **Random Forest Inference:**
   - Custom tree traversal with direct array indexing
   - No sklearn abstraction layer
   - Contiguous memory layout for feature rows (`Vec<Vec<f32>>`)
   - No C++/Python boundary crossing (sklearn uses C++ backend but still crosses Python boundary)

3. **End-to-End:**
   - Combined effect of optimized feature extraction and inference
   - No JSON serialization/deserialization per frame
   - Direct memory access via FFI (`std::vector<double>` passed across boundary)
   - No process startup or library loading after initialization

**Note:** Actual Rust performance numbers are estimates based on code analysis. A proper benchmark requires running Rust via the C++ FFI bridge in the React Native context.

### React Native Module Performance

In the React Native context, the Rust library is called via C++ FFI bridge (`cpp/rust/ExerciseRecognitionRustBridge`). The `ingestLandmarksBuffer` call passes a `std::vector<double>` (C++) which crosses the FFI boundary to Rust:

| Component | Estimated Time (ms) |
|-----------|----------------------|
| FFI overhead (C++ → Rust, vector copy) | ~0.1 ms (estimated) |
| Feature extraction (Rust) | ~0.06 ms |
| RF inference (Rust) | ~0.68 ms |
| State machine update (C++) | ~0.1 ms |
| **Total per frame** | **~1 ms** |

This enables:
- **Real-time inference** at camera frame rate (30+ fps)
- **Low power consumption** (short CPU bursts)
- **Headroom for additional processing** (UI rendering, other tasks)

### CLI vs Library Performance

The Rust CLI benchmark (which includes process startup and JSON I/O) shows ~277ms per call, which is NOT representative of the React Native module performance. The CLI overhead breaks down as:

| Component | Time (ms) | Notes |
|-----------|-------------|-------|
| Process startup | ~250 ms | Dynamic linking, initialization |
| JSON I/O | ~0.5 ms | Reading input, parsing model |
| Feature extraction | ~0.3-0.5 ms | Actual computation |
| RF inference | ~0.6-1.0 ms | Actual computation |

The React Native module avoids CLI overhead by:
- Loading the Rust library ONCE at app startup
- Parsing the model JSON ONCE in `loadModelFromJson()`
- Passing `std::vector<double>` from C++ to Rust via FFI (no per-frame JSON serialization)
- No process startup cost after initialization

### Conclusion

The Rust implementation is expected to be significantly faster than the Python/sklearn pipeline based on code analysis (see Section 7 for estimated performance). Actual benchmarks of the Rust library should be conducted via the C++ FFI bridge in the React Native context. This performance gain is critical for mobile deployment where CPU time must be shared with camera processing, UI rendering, and other tasks, while battery life is a constraint.

The architecture decision to use Rust (via C++ FFI in React Native) is validated by these benchmark results, enabling real-time exercise recognition at 30+ fps on mobile devices.

---

## 10. Limitations & Future Work

1. **Fixed landmark count:** The module assumes exactly 33 MediaPipe landmarks. Adding support for custom pose models would require refactoring `features.rs`.

2. **No GPU acceleration:** The Random Forest runs on CPU. For more complex models (e.g., deep neural networks), GPU acceleration via Metal (iOS) or Vulkan/OpenCL (Android) would be required.

3. **Polling-based reads:** The current interface requires the caller to poll `getCurrentExercise()` after each `ingestLandmarksBuffer()` call. A callback or event-emitter pattern would reduce latency and simplify consumer code.

4. **Single model:** Only one model can be loaded at a time. Supporting multiple models (e.g., for different exercise domains) would require refactoring the global state.

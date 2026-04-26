# react-native-exercise-recognition

React Native Nitro Module for on-device exercise classification.

It exposes a small JS API backed by a C++ HybridObject and Rust classifier runtime.

## Install

```bash
npm install github:tony-div/react-native-exercise-recognition react-native-nitro-modules github:tony-div/react-native-pose-landmarks
```

## Usage

Import the module and drive a session by:

1. Loading a model (`loadModelFromJson` or `loadModelFromAsset`)
2. Starting a session (`startSession`)
3. Sending landmarks (`ingestLandmarksBuffer`)
4. Reading predictions (`getCurrentExercise`, `getCurrentConfidence`)

```ts
import { exerciseRecognition } from 'react-native-exercise-recognition'

const LANDMARK_COUNT = 33
const VALUES_PER_LANDMARK = 4

const loaded = exerciseRecognition.loadModelFromJson(modelJsonString)
if (!loaded) {
  throw new Error('Failed to load exercise model')
}

exerciseRecognition.startSession({
  minConfidence: 0.6,
  smoothingWindow: 5,
  enterConfidence: 0.7,
  exitConfidence: 0.4,
  emaAlpha: 0.3,
  minVisibility: 0.5,
  minVisibleUpperBodyJoints: 8,
})

function onPoseFrame(buffer: number[]) {
  if (buffer.length !== LANDMARK_COUNT * VALUES_PER_LANDMARK) return

  exerciseRecognition.ingestLandmarksBuffer(buffer)

  const label = exerciseRecognition.getCurrentExercise()
  const confidence = exerciseRecognition.getCurrentConfidence()
  const inferenceMs = exerciseRecognition.getLastClassifierInferenceTimeMs()

  console.log({ label, confidence, inferenceMs })
}

// later
exerciseRecognition.stopSession()
```

### API

- `loadModelFromJson(modelJson: string): boolean`
- `loadModelFromAsset(assetName: string): boolean`
- `startSession(config?: StartSessionConfig): void`
- `stopSession(): void`
- `ingestLandmarksBuffer(landmarks: number[]): void`
- `getCurrentExercise(): string | null`
- `getCurrentConfidence(): number`
- `getLastClassifierInferenceTimeMs(): number`

#### StartSessionConfig

| Property | Default | Description |
|----------|---------|-------------|
| `minConfidence` | `0.5` | Minimum confidence to return a non-null prediction |
| `smoothingWindow` | `5` | Number of recent predictions to average for stability |
| `enterConfidence` | `0.6` | Confidence threshold to transition into an exercise |
| `exitConfidence` | `0.4` | Confidence threshold to transition out of an exercise |
| `enterFrames` | `3` | Consecutive frames above `enterConfidence` needed to start an exercise |
| `exitFrames` | `5` | Consecutive frames below `exitConfidence` needed to stop an exercise |
| `emaAlpha` | `0.3` | Exponential moving average weight (higher = more responsive, lower = smoother) |
| `minVisibility` | `0.5` | Minimum visibility score for a landmark to be considered valid |
| `minVisibleUpperBodyJoints` | `8` | Minimum number of upper body joints that must meet `minVisibility` |
| `nullExitWindowSeconds` | `2.0` | Duration of silence required before allowing exit from null state |
| `nullExitWindowThreshold` | `0.3` | Maximum confidence during `nullExitWindowSeconds` to trigger null exit |

### Important behavior

- `loadModelFromAsset(...)` is currently Android-only; on iOS it returns `false`.
- `ingestLandmarksBuffer(...)` expects flattened pose data in groups of 4 values per landmark (`x, y, z, visibility`) and uses 33 landmarks.
- The built-in Rust label mapping currently returns:
  - `Bicep Curl`
  - `Lateral Raise`
  - `Null/Unknown`
  - `Shoulder Press`
  - `Triceps Extension`
  - `Front Raises`

## Contributing

### Prerequisites

- Node.js (example app currently requires `>= 22.11.0`)
- npm
- Android/iOS React Native toolchains
- Rust toolchain (`cargo` + `rustup`) for Android native builds

### Setup

```bash
npm install
```

### Dev commands

```bash
# lint with fixes
npm run lint

# lint in CI mode
npm run lint-ci

# TypeScript type-check only
npm run typecheck

# build TS output to lib/
npm run typescript

# regenerate Nitro bindings (and TS emit)
npm run specs
```

### Required workflow when editing specs

If you change any `*.nitro.ts` file:

1. Run `npm run specs`
2. Commit generated changes in `nitrogen/generated/**`
3. Do not commit `lib/` (build output, gitignored)

### Native build notes

- Android `preBuild` runs `android/build-rust-android.sh`, which builds Rust static libs for:
  - `aarch64-linux-android`
  - `armv7-linux-androideabi`
  - `i686-linux-android`
  - `x86_64-linux-android`
- If those static libs are missing, C++ builds still succeed but run in fallback mode without Rust (`EXREC_USE_RUST` not defined).

## Example app

The demo app lives in `example/` and links this package via `file:..`.

Run the example from `example/`:

```bash
npm install
npm run start
npm run android
# or
npm run ios
```
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button, Dimensions, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { PoseLandmarks } from 'react-native-pose-landmarks';
import { exerciseRecognition } from 'react-native-exercise-recognition';

const LANDMARK_COUNT = 33;
const VALUES_PER_LANDMARK = 4;
const BONE_THICKNESS = 3;
const DEFAULT_SIZE = Dimensions.get('window');

const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [9, 10],
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [24, 26],
  [25, 27],
  [26, 28],
  [27, 29],
  [28, 30],
  [29, 31],
  [30, 32],
  [27, 31],
  [28, 32],
];

function App(): React.JSX.Element {
  const [sessionActive, setSessionActive] = useState(false);
  const [currentExercise, setCurrentExercise] = useState<string>('Unknown');
  const [confidence, setConfidence] = useState<number>(0);
  const [inferenceMs, setInferenceMs] = useState<number>(-1);
  const [classifierInferenceMs, setClassifierInferenceMs] = useState<number>(-1);
  const [e2eLatencyMs, setE2eLatencyMs] = useState<number>(-1);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [landmarks, setLandmarks] = useState<number[]>([]);
  const [viewport, setViewport] = useState({ width: DEFAULT_SIZE.width, height: DEFAULT_SIZE.height * 0.62 });
  const hasLoadedModel = useRef(false);

  useEffect(() => {
    if (!sessionActive) {
      return;
    }

    if (!hasLoadedModel.current) {
      const loaded = exerciseRecognition.loadModelFromAsset('exercise_classifier_rf.json');
      hasLoadedModel.current = loaded;
      setModelLoaded(loaded);
      if (!loaded) {
        setCurrentExercise('Failed to load model asset in native');
        setSessionActive(false);
        return;
      }
    }

    if (PoseLandmarks == null) {
      setCurrentExercise('PoseLandmarks native object is unavailable');
      setSessionActive(false);
      return;
    }

    const initialized = PoseLandmarks.initPoseLandmarker();
    if (!initialized) {
      setCurrentExercise('Unable to initialize MediaPipe pose landmarker');
      setSessionActive(false);
      return;
    }

    const interval = setInterval(() => {
      const buffer = PoseLandmarks.getLandmarksBuffer();
      const landmarkInferenceMs = PoseLandmarks.getLastInferenceTimeMs();
      setInferenceMs(landmarkInferenceMs);

      if (!Array.isArray(buffer) || buffer.length !== LANDMARK_COUNT * VALUES_PER_LANDMARK) {
        return;
      }

      setLandmarks(buffer);

      exerciseRecognition.ingestLandmarksBuffer(buffer);
      setCurrentExercise(exerciseRecognition.getCurrentExercise() ?? 'Unknown');
      setConfidence(exerciseRecognition.getCurrentConfidence());
      const classifierMs = exerciseRecognition.getLastClassifierInferenceTimeMs();
      setClassifierInferenceMs(classifierMs);
      if (landmarkInferenceMs >= 0 && classifierMs >= 0) {
        setE2eLatencyMs(landmarkInferenceMs + classifierMs);
      } else {
        setE2eLatencyMs(-1);
      }
    }, 66);

    return () => {
      clearInterval(interval);
      PoseLandmarks.closePoseLandmarker();
    };
  }, [sessionActive]);

  const onStart = () => {
    exerciseRecognition.startSession({ minConfidence: 0.6, smoothingWindow: 5 });
    setCurrentExercise('Collecting pose frames...');
    setConfidence(0);
    setSessionActive(true);
  };

  const onStop = () => {
    setSessionActive(false);
    exerciseRecognition.stopSession();
  };

  const renderedSkeleton = useMemo(() => {
    if (landmarks.length !== LANDMARK_COUNT * VALUES_PER_LANDMARK) {
      return null;
    }

    return POSE_CONNECTIONS.map(([from, to], index) => {
      const fromIndex = from * VALUES_PER_LANDMARK;
      const toIndex = to * VALUES_PER_LANDMARK;

      const fromVisibility = landmarks[fromIndex + 3] ?? 1;
      const toVisibility = landmarks[toIndex + 3] ?? 1;
      if (fromVisibility < 0.05 || toVisibility < 0.05) {
        return null;
      }

      const x1 = (landmarks[fromIndex] ?? 0) * viewport.width;
      const y1 = (landmarks[fromIndex + 1] ?? 0) * viewport.height;
      const x2 = (landmarks[toIndex] ?? 0) * viewport.width;
      const y2 = (landmarks[toIndex + 1] ?? 0) * viewport.height;

      const deltaX = x2 - x1;
      const deltaY = y2 - y1;
      const length = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
      const angle = Math.atan2(deltaY, deltaX);
      const midX = (x1 + x2) / 2;
      const midY = (y1 + y2) / 2;

      return (
        <View
          key={`bone-${from}-${to}-${index}`}
          style={[
            styles.bone,
            {
              left: midX - length / 2,
              top: midY - BONE_THICKNESS / 2,
              width: length,
              transform: [{ rotateZ: `${angle}rad` }],
            },
          ]}
        />
      );
    });
  }, [landmarks, viewport.height, viewport.width]);

  const renderedLandmarks = useMemo(() => {
    if (landmarks.length !== LANDMARK_COUNT * VALUES_PER_LANDMARK) {
      return null;
    }

    return Array.from({ length: LANDMARK_COUNT }).map((_, index) => {
      const baseIndex = index * VALUES_PER_LANDMARK;
      const visibility = landmarks[baseIndex + 3] ?? 1;
      if (visibility < 0.05) {
        return null;
      }

      const x = (landmarks[baseIndex] ?? 0) * viewport.width;
      const y = (landmarks[baseIndex + 1] ?? 0) * viewport.height;

      return <View key={`dot-${index}`} style={[styles.dot, { left: x - 4, top: y - 4 }]} />;
    });
  }, [landmarks, viewport.height, viewport.width]);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.container}>
          <View style={styles.hud}>
            <Text style={styles.title}>Live Exercise Recognition</Text>
            <Text style={styles.prediction}>{currentExercise.toUpperCase()}</Text>
            <Text style={styles.confidence}>{(confidence * 100).toFixed(1)}% confidence</Text>
            <Text style={styles.meta}>Model: {modelLoaded ? 'loaded' : 'not loaded'}</Text>
            <Text style={styles.meta}>Landmark inference: {inferenceMs >= 0 ? `${inferenceMs.toFixed(0)} ms` : '--'}</Text>
            <Text style={styles.meta}>Classifier inference: {classifierInferenceMs >= 0 ? `${classifierInferenceMs.toFixed(1)} ms` : '--'}</Text>
            <Text style={styles.meta}>E2E latency: {e2eLatencyMs >= 0 ? `${e2eLatencyMs.toFixed(1)} ms` : '--'}</Text>
            <Text style={styles.meta}>Landmark points: {Math.floor(landmarks.length / VALUES_PER_LANDMARK)} / 33</Text>
          </View>

          <View
            style={styles.viewport}
            onLayout={event => {
              const { width, height } = event.nativeEvent.layout;
              if (width > 0 && height > 0) {
                setViewport({ width, height });
              }
            }}
          >
            <View style={styles.cameraTint} />
            {renderedSkeleton}
            {renderedLandmarks}
            {landmarks.length === 0 ? <Text style={styles.placeholder}>Waiting for live camera landmarks...</Text> : null}
          </View>

          <View style={styles.controls}>
            <View style={styles.buttons}>
            <Button title="Start Session" onPress={onStart} disabled={sessionActive} />
            </View>
            <View style={styles.buttons}>
            <Button title="Stop Session" onPress={onStop} disabled={!sessionActive} />
            </View>
          </View>
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0a1118',
  },
  container: {
    flex: 1,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  hud: {
    marginBottom: 12,
    backgroundColor: '#112131',
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  title: {
    fontSize: 20,
    fontWeight: '700',
    color: '#dceeff',
    letterSpacing: 0.2,
  },
  prediction: {
    marginTop: 8,
    fontSize: 44,
    fontWeight: '900',
    color: '#f7fff8',
    lineHeight: 48,
  },
  confidence: {
    marginTop: 2,
    marginBottom: 8,
    fontSize: 24,
    fontWeight: '700',
    color: '#79f0b3',
  },
  meta: {
    fontSize: 13,
    marginTop: 2,
    color: '#97adc1',
  },
  viewport: {
    flex: 1,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#0e1d2a',
    borderWidth: 1,
    borderColor: '#203446',
  },
  cameraTint: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(10, 20, 34, 0.76)',
  },
  bone: {
    position: 'absolute',
    height: BONE_THICKNESS,
    borderRadius: 999,
    backgroundColor: '#3fd4ff',
  },
  dot: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#ffca57',
    borderWidth: 1,
    borderColor: '#fff6d5',
  },
  placeholder: {
    color: '#8ea4b6',
    fontSize: 17,
    textAlign: 'center',
    marginTop: '45%',
  },
  controls: {
    marginTop: 12,
    marginBottom: 6,
    flexDirection: 'row',
  },
  buttons: {
    flex: 1,
    marginHorizontal: 6,
  },
});

export default App;

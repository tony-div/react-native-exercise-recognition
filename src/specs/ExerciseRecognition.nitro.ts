import { type HybridObject, NitroModules } from 'react-native-nitro-modules'

export interface StartSessionConfig {
  minConfidence?: number
  smoothingWindow?: number
}

export interface ExerciseRecognition extends HybridObject<{ ios: 'c++', android: 'c++' }> {
  loadModelFromJson(modelJson: string): boolean
  loadModelFromAsset(assetName: string): boolean
  startSession(config?: StartSessionConfig): void
  stopSession(): void
  ingestLandmarksBuffer(landmarks: Array<number>): void
  getCurrentExercise(): string | null
  getCurrentConfidence(): number
  getLastClassifierInferenceTimeMs(): number
}

export const exerciseRecognition = NitroModules.createHybridObject<ExerciseRecognition>(
  'ExerciseRecognition'
)

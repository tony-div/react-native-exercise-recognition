import { type HybridObject } from 'react-native-nitro-modules';
export interface StartSessionConfig {
    minConfidence?: number;
    smoothingWindow?: number;
    enterConfidence?: number;
    exitConfidence?: number;
    enterFrames?: number;
    exitFrames?: number;
    emaAlpha?: number;
    minVisibility?: number;
    minVisibleUpperBodyJoints?: number;
    nullExitWindowSeconds?: number;
    nullExitWindowThreshold?: number;
}
export interface ExerciseRecognition extends HybridObject<{
    ios: 'c++';
    android: 'c++';
}> {
    loadModelFromJson(modelJson: string): boolean;
    loadModelFromAsset(assetName: string): boolean;
    startSession(config?: StartSessionConfig): void;
    stopSession(): void;
    ingestLandmarksBuffer(landmarks: Array<number>): void;
    getCurrentExercise(): string | null;
    getCurrentConfidence(): number;
    getLastClassifierInferenceTimeMs(): number;
}
export declare const exerciseRecognition: ExerciseRecognition;

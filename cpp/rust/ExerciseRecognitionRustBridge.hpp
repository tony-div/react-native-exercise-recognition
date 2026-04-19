#pragma once

#include <string>
#include <vector>

namespace margelo::nitro::exerciserecognition {

class ExerciseRecognitionRustBridge {
 public:
  bool loadModelFromJson(const std::string& modelJson);
  void startSession(double minConfidence, int smoothingWindow);
  void stopSession();
  void ingestLandmarksBuffer(const std::vector<double>& landmarks);
  double getCurrentConfidence();
  double getLastClassifierInferenceTimeMs();
  std::string getCurrentExercise();
};

} // namespace margelo::nitro::exerciserecognition

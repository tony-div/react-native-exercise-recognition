#pragma once

#include "../nitrogen/generated/shared/c++/HybridExerciseRecognitionSpec.hpp"
#include "rust/ExerciseRecognitionRustBridge.hpp"

#include <optional>
#include <string>
#include <vector>
#include <variant>

namespace margelo::nitro::exerciserecognition {

class HybridExerciseRecognition : public HybridExerciseRecognitionSpec {
 public:
  HybridExerciseRecognition() : HybridObject(TAG) {}

 public:
  bool loadModelFromJson(const std::string& modelJson) override;
  bool loadModelFromAsset(const std::string& assetName) override;
  void startSession(const std::optional<StartSessionConfig>& config) override;
  void stopSession() override;
  void ingestLandmarksBuffer(const std::vector<double>& landmarks) override;
  std::variant<nitro::NullType, std::string> getCurrentExercise() override;
  double getCurrentConfidence() override;
  double getLastClassifierInferenceTimeMs() override;

 private:
  ExerciseRecognitionRustBridge rust_;
  std::variant<nitro::NullType, std::string> currentExercise_ = nitro::NullType();
  double currentConfidence_ = 0.0;
  double lastClassifierInferenceMs_ = -1.0;
  double minConfidence_ = 0.6;
  int smoothingWindow_ = 5;
  double enterConfidence_ = 0.65;
  double exitConfidence_ = 0.45;
  int enterFrames_ = 3;
  int exitFrames_ = 8;
  double emaAlpha_ = 0.2;
  double minVisibility_ = 0.2;
  int minVisibleUpperBodyJoints_ = 4;
  double nullExitWindowSeconds_ = 5.0;
  double nullExitWindowThreshold_ = 0.99;
};

} // namespace margelo::nitro::exerciserecognition

#include "ExerciseRecognitionRustBridge.hpp"

#include <cstddef>
#include <cstdio>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#if defined(EXREC_USE_RUST)
extern "C" {
int exrec_load_model_from_json(const char* model_json);
void exrec_start_session(
  double min_confidence,
  int smoothing_window,
  double enter_confidence,
  double exit_confidence,
  int enter_frames,
  int exit_frames,
  double ema_alpha,
  double min_visibility,
  int min_visible_upper_body_joints,
  double null_exit_window_seconds,
  double null_exit_window_threshold);
void exrec_stop_session();
void exrec_ingest_landmarks_buffer(const double* values, size_t len);
double exrec_get_current_confidence();
double exrec_get_last_classifier_inference_time_ms();
char* exrec_get_current_exercise();
void exrec_string_free(char* s);
}
#endif

namespace margelo::nitro::exerciserecognition {

namespace {

constexpr const char* kLogTag = "NitroExerciseRec";

void logDebug(const char* message) {
#if defined(__ANDROID__)
  __android_log_print(ANDROID_LOG_DEBUG, kLogTag, "%s", message);
#else
  std::fprintf(stderr, "[%s] %s\n", kLogTag, message);
#endif
}

void logDebugFmt(const char* format, double value) {
#if defined(__ANDROID__)
  __android_log_print(ANDROID_LOG_DEBUG, kLogTag, format, value);
#else
  std::fprintf(stderr, "[%s] ", kLogTag);
  std::fprintf(stderr, format, value);
  std::fprintf(stderr, "\n");
#endif
}

void logDebugFmt(const char* format, int value) {
#if defined(__ANDROID__)
  __android_log_print(ANDROID_LOG_DEBUG, kLogTag, format, value);
#else
  std::fprintf(stderr, "[%s] ", kLogTag);
  std::fprintf(stderr, format, value);
  std::fprintf(stderr, "\n");
#endif
}

void logDebugFmt(const char* format, const std::string& value) {
#if defined(__ANDROID__)
  __android_log_print(ANDROID_LOG_DEBUG, kLogTag, format, value.c_str());
#else
  std::fprintf(stderr, "[%s] ", kLogTag);
  std::fprintf(stderr, format, value.c_str());
  std::fprintf(stderr, "\n");
#endif
}

} // namespace

bool ExerciseRecognitionRustBridge::loadModelFromJson(const std::string& modelJson) {
#if defined(EXREC_USE_RUST)
  logDebugFmt("rust.loadModelFromJson(): input bytes=%d", static_cast<int>(modelJson.size()));
  const bool loaded = exrec_load_model_from_json(modelJson.c_str()) == 1;
  logDebugFmt("rust.loadModelFromJson(): result=%d", loaded ? 1 : 0);
  return loaded;
#else
  (void)modelJson;
  logDebug("rust.loadModelFromJson(): EXREC_USE_RUST disabled, returning false");
  return false;
#endif
}

void ExerciseRecognitionRustBridge::startSession(
  double minConfidence,
  int smoothingWindow,
  double enterConfidence,
  double exitConfidence,
  int enterFrames,
  int exitFrames,
  double emaAlpha,
  double minVisibility,
  int minVisibleUpperBodyJoints,
  double nullExitWindowSeconds,
  double nullExitWindowThreshold) {
#if defined(EXREC_USE_RUST)
  logDebugFmt("rust.startSession(): minConfidence=%.4f", minConfidence);
  logDebugFmt("rust.startSession(): smoothingWindow=%d", smoothingWindow);
  logDebugFmt("rust.startSession(): enterConfidence=%.4f", enterConfidence);
  logDebugFmt("rust.startSession(): exitConfidence=%.4f", exitConfidence);
  logDebugFmt("rust.startSession(): enterFrames=%d", enterFrames);
  logDebugFmt("rust.startSession(): exitFrames=%d", exitFrames);
  logDebugFmt("rust.startSession(): emaAlpha=%.4f", emaAlpha);
  logDebugFmt("rust.startSession(): minVisibility=%.4f", minVisibility);
  logDebugFmt("rust.startSession(): minVisibleUpperBodyJoints=%d", minVisibleUpperBodyJoints);
  logDebugFmt("rust.startSession(): nullExitWindowSeconds=%.2f", nullExitWindowSeconds);
  logDebugFmt("rust.startSession(): nullExitWindowThreshold=%.2f", nullExitWindowThreshold);
  exrec_start_session(
    minConfidence,
    smoothingWindow,
    enterConfidence,
    exitConfidence,
    enterFrames,
    exitFrames,
    emaAlpha,
    minVisibility,
    minVisibleUpperBodyJoints,
    nullExitWindowSeconds,
    nullExitWindowThreshold);
  logDebug("rust.startSession(): rust session started");
#else
  (void)minConfidence;
  (void)smoothingWindow;
  (void)enterConfidence;
  (void)exitConfidence;
  (void)enterFrames;
  (void)exitFrames;
  (void)emaAlpha;
  (void)minVisibility;
  (void)minVisibleUpperBodyJoints;
  (void)nullExitWindowSeconds;
  (void)nullExitWindowThreshold;
  logDebug("rust.startSession(): EXREC_USE_RUST disabled");
#endif
}

void ExerciseRecognitionRustBridge::stopSession() {
#if defined(EXREC_USE_RUST)
  logDebug("rust.stopSession(): begin");
  exrec_stop_session();
  logDebug("rust.stopSession(): complete");
#else
  logDebug("rust.stopSession(): EXREC_USE_RUST disabled");
#endif
}

void ExerciseRecognitionRustBridge::ingestLandmarksBuffer(const std::vector<double>& landmarks) {
#if defined(EXREC_USE_RUST)
  if (landmarks.empty()) {
    logDebug("rust.ingestLandmarksBuffer(): empty input, skipping");
    return;
  }
  logDebugFmt("rust.ingestLandmarksBuffer(): landmark values=%d", static_cast<int>(landmarks.size()));
  exrec_ingest_landmarks_buffer(landmarks.data(), landmarks.size());
  logDebug("rust.ingestLandmarksBuffer(): ingest call completed");
#else
  (void)landmarks;
  logDebug("rust.ingestLandmarksBuffer(): EXREC_USE_RUST disabled");
#endif
}

double ExerciseRecognitionRustBridge::getCurrentConfidence() {
#if defined(EXREC_USE_RUST)
  const double confidence = exrec_get_current_confidence();
  logDebugFmt("rust.getCurrentConfidence(): %.6f", confidence);
  return confidence;
#else
  logDebug("rust.getCurrentConfidence(): EXREC_USE_RUST disabled, returning 0");
  return 0.0;
#endif
}

double ExerciseRecognitionRustBridge::getLastClassifierInferenceTimeMs() {
#if defined(EXREC_USE_RUST)
  const double inferenceMs = exrec_get_last_classifier_inference_time_ms();
  logDebugFmt("rust.getLastClassifierInferenceTimeMs(): %.6f", inferenceMs);
  return inferenceMs;
#else
  logDebug("rust.getLastClassifierInferenceTimeMs(): EXREC_USE_RUST disabled, returning -1");
  return -1.0;
#endif
}

std::string ExerciseRecognitionRustBridge::getCurrentExercise() {
#if defined(EXREC_USE_RUST)
  char* raw = exrec_get_current_exercise();
  if (raw == nullptr) {
    logDebug("rust.getCurrentExercise(): returned null");
    return "";
  }

  std::string out(raw);
  exrec_string_free(raw);
  if (out.empty()) {
    logDebug("rust.getCurrentExercise(): returned empty string");
  } else {
    logDebugFmt("rust.getCurrentExercise(): '%s'", out);
  }
  return out;
#else
  logDebug("rust.getCurrentExercise(): EXREC_USE_RUST disabled, returning empty");
  return "";
#endif
}

} // namespace margelo::nitro::exerciserecognition

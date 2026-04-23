#include "HybridExerciseRecognition.hpp"

#include <cstdio>

#if defined(__ANDROID__)
#include <android/log.h>
#include <fbjni/fbjni.h>
#endif

namespace {

constexpr const char* kLogTag = "NitroExerciseRec";

void logDebug(const char* message) {
#if defined(__ANDROID__)
  __android_log_print(ANDROID_LOG_DEBUG, kLogTag, "%s", message);
#else
  std::fprintf(stderr, "[%s] %s\n", kLogTag, message);
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

#if defined(__ANDROID__)
std::string loadAndroidAssetText(const std::string& assetName) {
  logDebugFmt("loadAndroidAssetText(): requested asset='%s'", assetName);
  auto env = facebook::jni::Environment::current();
  if (env == nullptr) {
    logDebug("loadAndroidAssetText(): JNI environment is null");
    return "";
  }

  jclass cls = env->FindClass("com/margelo/nitro/exerciserecognition/ExerciseRecognitionAssets");
  if (cls == nullptr) {
    logDebug("loadAndroidAssetText(): ExerciseRecognitionAssets class not found");
    return "";
  }

  jmethodID method = env->GetStaticMethodID(
    cls,
    "loadAssetAsString",
    "(Ljava/lang/String;)Ljava/lang/String;");
  if (method == nullptr) {
    logDebug("loadAndroidAssetText(): loadAssetAsString method not found");
    env->DeleteLocalRef(cls);
    return "";
  }

  jstring jAssetName = env->NewStringUTF(assetName.c_str());
  jobject jResult = env->CallStaticObjectMethod(cls, method, jAssetName);
  env->DeleteLocalRef(jAssetName);
  env->DeleteLocalRef(cls);

  if (jResult == nullptr) {
    logDebug("loadAndroidAssetText(): Java returned null asset content");
    return "";
  }

  auto* resultStr = static_cast<jstring>(jResult);
  const char* utfChars = env->GetStringUTFChars(resultStr, nullptr);
  if (utfChars == nullptr) {
    logDebug("loadAndroidAssetText(): failed to access UTF chars");
    env->DeleteLocalRef(resultStr);
    return "";
  }

  std::string output(utfChars);
  env->ReleaseStringUTFChars(resultStr, utfChars);
  env->DeleteLocalRef(resultStr);
  logDebugFmt("loadAndroidAssetText(): loaded bytes=%d", static_cast<int>(output.size()));
  return output;
}
#endif

} // namespace

namespace margelo::nitro::exerciserecognition {

bool HybridExerciseRecognition::loadModelFromJson(const std::string& modelJson) {
  logDebugFmt("loadModelFromJson(): input bytes=%d", static_cast<int>(modelJson.size()));
  const bool loaded = rust_.loadModelFromJson(modelJson);
  logDebugFmt("loadModelFromJson(): result=%d", loaded ? 1 : 0);
  return loaded;
}

bool HybridExerciseRecognition::loadModelFromAsset(const std::string& assetName) {
  logDebugFmt("loadModelFromAsset(): asset='%s'", assetName);
#if defined(__ANDROID__)
  const auto modelJson = loadAndroidAssetText(assetName);
  if (modelJson.empty()) {
    logDebug("loadModelFromAsset(): asset load returned empty content");
    return false;
  }
  const bool loaded = rust_.loadModelFromJson(modelJson);
  logDebugFmt("loadModelFromAsset(): rust load result=%d", loaded ? 1 : 0);
  return loaded;
#else
  (void)assetName;
  logDebug("loadModelFromAsset(): unsupported on this platform");
  return false;
#endif
}

void HybridExerciseRecognition::startSession(const std::optional<StartSessionConfig>& config) {
  logDebug("startSession(): begin");
  if (config.has_value()) {
    logDebug("startSession(): config provided");
    if (config->minConfidence.has_value()) {
      minConfidence_ = config->minConfidence.value();
      logDebugFmt("startSession(): updated minConfidence=%.4f", minConfidence_);
      if (!config->enterConfidence.has_value()) {
        enterConfidence_ = minConfidence_;
        logDebugFmt("startSession(): enterConfidence follows minConfidence=%.4f", enterConfidence_);
      }
    }
    if (config->smoothingWindow.has_value()) {
      smoothingWindow_ = static_cast<int>(config->smoothingWindow.value());
      logDebugFmt("startSession(): updated smoothingWindow=%d", smoothingWindow_);
    }
    if (config->enterConfidence.has_value()) {
      enterConfidence_ = config->enterConfidence.value();
      logDebugFmt("startSession(): updated enterConfidence=%.4f", enterConfidence_);
    }
    if (config->exitConfidence.has_value()) {
      exitConfidence_ = config->exitConfidence.value();
      logDebugFmt("startSession(): updated exitConfidence=%.4f", exitConfidence_);
    }
    if (config->enterFrames.has_value()) {
      enterFrames_ = static_cast<int>(config->enterFrames.value());
      logDebugFmt("startSession(): updated enterFrames=%d", enterFrames_);
    }
    if (config->exitFrames.has_value()) {
      exitFrames_ = static_cast<int>(config->exitFrames.value());
      logDebugFmt("startSession(): updated exitFrames=%d", exitFrames_);
    }
    if (config->emaAlpha.has_value()) {
      emaAlpha_ = config->emaAlpha.value();
      logDebugFmt("startSession(): updated emaAlpha=%.4f", emaAlpha_);
    }
    if (config->minVisibility.has_value()) {
      minVisibility_ = config->minVisibility.value();
      logDebugFmt("startSession(): updated minVisibility=%.4f", minVisibility_);
    }
    if (config->minVisibleUpperBodyJoints.has_value()) {
      minVisibleUpperBodyJoints_ = static_cast<int>(config->minVisibleUpperBodyJoints.value());
      logDebugFmt("startSession(): updated minVisibleUpperBodyJoints=%d", minVisibleUpperBodyJoints_);
    }
  }

  rust_.startSession(
    minConfidence_,
    smoothingWindow_,
    enterConfidence_,
    exitConfidence_,
    enterFrames_,
    exitFrames_,
    emaAlpha_,
    minVisibility_,
    minVisibleUpperBodyJoints_);
  logDebug("startSession(): rust session started");
  currentExercise_ = nitro::NullType();
  currentConfidence_ = 0.0;
  lastClassifierInferenceMs_ = -1.0;
  logDebug("startSession(): state reset complete");
}

void HybridExerciseRecognition::stopSession() {
  logDebug("stopSession(): begin");
  rust_.stopSession();
  currentExercise_ = nitro::NullType();
  currentConfidence_ = 0.0;
  lastClassifierInferenceMs_ = -1.0;
  logDebug("stopSession(): rust stopped and state reset");
}

void HybridExerciseRecognition::ingestLandmarksBuffer(const std::vector<double>& landmarks) {
  logDebugFmt("ingestLandmarksBuffer(): landmark values=%d", static_cast<int>(landmarks.size()));
  rust_.ingestLandmarksBuffer(landmarks);
  currentConfidence_ = rust_.getCurrentConfidence();
  lastClassifierInferenceMs_ = rust_.getLastClassifierInferenceTimeMs();
  const auto label = rust_.getCurrentExercise();
  if (label.empty()) {
    currentExercise_ = nitro::NullType();
    logDebug("ingestLandmarksBuffer(): current exercise is null");
  } else {
    currentExercise_ = label;
    logDebugFmt("ingestLandmarksBuffer(): current exercise='%s'", label);
  }
  logDebugFmt("ingestLandmarksBuffer(): confidence=%.6f", currentConfidence_);
  logDebugFmt("ingestLandmarksBuffer(): inferenceMs=%.6f", lastClassifierInferenceMs_);
}

std::variant<nitro::NullType, std::string> HybridExerciseRecognition::getCurrentExercise() {
  return currentExercise_;
}

double HybridExerciseRecognition::getCurrentConfidence() {
  return currentConfidence_;
}

double HybridExerciseRecognition::getLastClassifierInferenceTimeMs() {
  return lastClassifierInferenceMs_;
}

} // namespace margelo::nitro::exerciserecognition

project(NitroExerciseRecognition)
cmake_minimum_required(VERSION 3.9.0)

set (PACKAGE_NAME NitroExerciseRecognition)
set (CMAKE_VERBOSE_MAKEFILE ON)
set (CMAKE_CXX_STANDARD 20)

# Define C++ library and add all sources
add_library(${PACKAGE_NAME} SHARED
        src/main/cpp/cpp-adapter.cpp
        ../cpp/HybridExerciseRecognition.cpp
        ../cpp/rust/ExerciseRecognitionRustBridge.cpp
)

if(ANDROID_ABI STREQUAL "arm64-v8a")
  set(RUST_TARGET "aarch64-linux-android")
elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
  set(RUST_TARGET "armv7-linux-androideabi")
elseif(ANDROID_ABI STREQUAL "x86")
  set(RUST_TARGET "i686-linux-android")
elseif(ANDROID_ABI STREQUAL "x86_64")
  set(RUST_TARGET "x86_64-linux-android")
else()
  set(RUST_TARGET "")
endif()

set(RUST_LIB_PATH "")
if(NOT RUST_TARGET STREQUAL "")
  set(RUST_LIB_PATH "${CMAKE_SOURCE_DIR}/../rust/target/${RUST_TARGET}/release/libexercise_recognition_rust.a")
  if(EXISTS "${RUST_LIB_PATH}")
    target_compile_definitions(${PACKAGE_NAME} PRIVATE EXREC_USE_RUST=1)
  endif()
endif()

# Add Nitrogen specs :)
include(${CMAKE_SOURCE_DIR}/../nitrogen/generated/android/NitroExerciseRecognition+autolinking.cmake)

# Set up local includes
include_directories(
        "src/main/cpp"
        "../cpp"
)

find_library(LOG_LIB log)

# Link all libraries together
target_link_libraries(
        ${PACKAGE_NAME}
        ${LOG_LIB}
        android                                   # <-- Android core
)

if(NOT RUST_LIB_PATH STREQUAL "" AND EXISTS "${RUST_LIB_PATH}")
  target_link_libraries(${PACKAGE_NAME} ${RUST_LIB_PATH})
endif()

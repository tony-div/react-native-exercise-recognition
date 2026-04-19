# AGENTS.md

## Scope

- Single-package React Native Nitro Module library (not a monorepo).
- JS API source of truth: `src/specs/ExerciseRecognition.nitro.ts`; public exports: `src/index.ts`.
- Runtime path is TS spec -> generated Nitro bindings in `nitrogen/generated/**` -> C++ HybridObject in `cpp/HybridExerciseRecognition.*` -> Rust FFI bridge in `cpp/rust/*` + `rust/src/lib.rs`.

## Commands

- Install: `npm install`
- Lint/fix: `npm run lint`
- Lint (CI style): `npm run lint-ci`
- Type check only: `npm run typecheck`
- Build TS output (`lib/`): `npm run typescript`
- Regenerate Nitro bindings (also runs TS emit): `npm run specs`

## Required workflows

- If any `*.nitro.ts` changes, run `npm run specs` before finishing.
- Commit `nitrogen/generated/**` changes produced by specs/codegen.
- Do not commit `lib/` (gitignored build output).
- There is no root `test` script; use lint/typecheck/specs plus targeted native/example runs for verification.

## Native gotchas

- Android build always triggers `android/build-rust-android.sh` from `preBuild`; this requires `cargo` + `rustup` installed.
- That script builds Rust staticlibs for 4 Android targets; missing Rust toolchain/targets breaks Android builds.
- CMake only defines `EXREC_USE_RUST` when `rust/target/<abi>/release/libexercise_recognition_rust.a` exists; without it, C++ bridge compiles but returns fallback values (easy to miss).
- `loadModelFromAsset(...)` is Android-only today (`HybridExerciseRecognition.cpp` returns `false` on iOS).
- Keep module naming aligned across `nitro.json`, `android/CMakeLists.txt`, `android/build.gradle`, and `NitroExerciseRecognition.podspec` (`NitroExerciseRecognition`).

## Example app

- Work from `example/` for app runs (`npm run start`, `npm run android`, `npm run ios`).
- Example links this package via `file:..` and pins `react-native-pose-landmarks` to `github:tony-div/react-native-pose-landmarks#v1.1.0`.
- `example/metro.config.js` still watches both repo root and parent folder; this is optional when using the pinned GitHub dependency, but useful for local sibling development.

## Trust config over docs

- `README.md` is mostly template text; prefer `package.json`, `nitro.json`, and native build files as source of truth.

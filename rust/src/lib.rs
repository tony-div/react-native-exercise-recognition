mod features;
mod rf_model;

use rf_model::RandomForestRunner;
use serde::Deserialize;
use std::collections::VecDeque;
use std::ffi::{c_char, c_double, c_int, CStr, CString};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const LOG_TAG: &[u8] = b"NitroExerciseRec\0";

#[cfg(target_os = "android")]
extern "C" {
    fn __android_log_write(prio: c_int, tag: *const c_char, text: *const c_char) -> c_int;
}

fn log_debug(message: &str) {
    #[cfg(target_os = "android")]
    {
        const ANDROID_LOG_DEBUG: c_int = 3;
        let safe_message = message.replace('\0', "\\0");
        if let Ok(c_message) = CString::new(safe_message) {
            unsafe {
                let _ = __android_log_write(
                    ANDROID_LOG_DEBUG,
                    LOG_TAG.as_ptr() as *const c_char,
                    c_message.as_ptr(),
                );
            }
        }
    }

    #[cfg(not(target_os = "android"))]
    {
        eprintln!("[NitroExerciseRec] {message}");
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum FrameInput {
    FrameRecord { landmarks: Vec<Landmark> },
    LandmarkList(Vec<Landmark>),
}

impl FrameInput {
    pub fn landmarks(&self) -> &[Landmark] {
        match self {
            FrameInput::FrameRecord { landmarks } => landmarks,
            FrameInput::LandmarkList(landmarks) => landmarks,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct Landmark {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub visibility: Option<f64>,
}

#[derive(Clone)]
struct SessionConfig {
    min_confidence: f32,
    smoothing_window: usize,
    enter_confidence: f32,
    exit_confidence: f32,
    enter_frames: usize,
    exit_frames: usize,
    ema_alpha: f32,
    min_visibility: f32,
    min_visible_upper_body_joints: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            smoothing_window: 5,
            enter_confidence: 0.65,
            exit_confidence: 0.45,
            enter_frames: 3,
            exit_frames: 8,
            ema_alpha: 0.2,
            min_visibility: 0.2,
            min_visible_upper_body_joints: 4,
        }
    }
}

struct ClassifierState {
    model: Option<RandomForestRunner>,
    config: SessionConfig,
    frames: VecDeque<FrameInput>,
    current_label: Option<String>,
    current_confidence: f32,
    last_inference_ms: f64,
    ema_probs: Vec<f32>,
    active_class_id: Option<i32>,
    pending_class_id: Option<i32>,
    pending_class_count: usize,
    low_confidence_count: usize,
}

impl Default for ClassifierState {
    fn default() -> Self {
        Self {
            model: None,
            config: SessionConfig::default(),
            frames: VecDeque::new(),
            current_label: None,
            current_confidence: 0.0,
            last_inference_ms: -1.0,
            ema_probs: Vec::new(),
            active_class_id: None,
            pending_class_id: None,
            pending_class_count: 0,
            low_confidence_count: 0,
        }
    }
}

fn state() -> &'static Mutex<ClassifierState> {
    static STATE: OnceLock<Mutex<ClassifierState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(ClassifierState::default()))
}

fn label_for_class_id(class_id: i32) -> &'static str {
    match class_id {
        0 => "Bicep Curl",
        1 => "Lateral Raise",
        2 => "Null/Unknown",
        3 => "Shoulder Press",
        4 => "Triceps Extension",
        5 => "Front Raises",
        _ => "Unknown",
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn average_probabilities(frame_probs: &[Vec<f32>]) -> Vec<f32> {
    if frame_probs.is_empty() {
        return Vec::new();
    }
    let mut avg = vec![0.0_f32; frame_probs[0].len()];
    for probs in frame_probs {
        for (i, &p) in probs.iter().enumerate().take(avg.len()) {
            avg[i] += p;
        }
    }

    let n = frame_probs.len() as f32;
    if n > 0.0 {
        for v in &mut avg {
            *v /= n;
        }
    }
    avg
}

fn upper_body_visible_joint_count(landmarks: &[Landmark], min_visibility: f32) -> usize {
    const UPPER_BODY_JOINTS: [usize; 6] = [11, 12, 13, 14, 15, 16];
    UPPER_BODY_JOINTS
        .iter()
        .filter_map(|&idx| landmarks.get(idx))
        .filter(|lm| lm.visibility.unwrap_or(0.0) as f32 >= min_visibility)
        .count()
}

fn probability_for_class_id(class_ids: &[i32], probs: &[f32], class_id: i32) -> f32 {
    class_ids
        .iter()
        .position(|&c| c == class_id)
        .and_then(|idx| probs.get(idx).copied())
        .unwrap_or(0.0)
}

fn blend_with_ema(prev: &[f32], current: &[f32], alpha: f32) -> Vec<f32> {
    if prev.len() != current.len() {
        return current.to_vec();
    }
    let mut blended = Vec::with_capacity(current.len());
    for (old, now) in prev.iter().zip(current.iter()) {
        blended.push((alpha * *now) + ((1.0 - alpha) * *old));
    }
    blended
}

fn reset_temporal_state(guard: &mut ClassifierState) {
    guard.ema_probs.clear();
    guard.active_class_id = None;
    guard.pending_class_id = None;
    guard.pending_class_count = 0;
    guard.low_confidence_count = 0;
}

fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        log_debug("cstr_to_string(): input pointer is null");
        return None;
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    let output = cstr.to_string_lossy().into_owned();
    log_debug(&format!("cstr_to_string(): decoded bytes={}", output.len()));
    Some(output)
}

#[no_mangle]
pub extern "C" fn exrec_load_model_from_json(model_json: *const c_char) -> c_int {
    log_debug("exrec_load_model_from_json(): begin");
    let Some(json) = cstr_to_string(model_json) else {
        log_debug("exrec_load_model_from_json(): failed to decode input json");
        return 0;
    };
    log_debug(&format!("exrec_load_model_from_json(): json bytes={}", json.len()));

    let Some(model) = RandomForestRunner::from_json(&json) else {
        log_debug("exrec_load_model_from_json(): RandomForestRunner::from_json failed");
        return 0;
    };
    log_debug("exrec_load_model_from_json(): model parsed successfully");

    if let Ok(mut guard) = state().lock() {
        guard.model = Some(model);
        guard.current_label = None;
        guard.current_confidence = 0.0;
        guard.last_inference_ms = -1.0;
        guard.frames.clear();
        reset_temporal_state(&mut guard);
        log_debug("exrec_load_model_from_json(): state updated, returning success");
        return 1;
    }

    log_debug("exrec_load_model_from_json(): failed to acquire state lock");
    0
}

#[no_mangle]
pub extern "C" fn exrec_start_session(
    min_confidence: c_double,
    smoothing_window: c_int,
    enter_confidence: c_double,
    exit_confidence: c_double,
    enter_frames: c_int,
    exit_frames: c_int,
    ema_alpha: c_double,
    min_visibility: c_double,
    min_visible_upper_body_joints: c_int,
) {
    log_debug(&format!(
        "exrec_start_session(): begin min_confidence={:.4} smoothing_window={} enter_confidence={:.4} exit_confidence={:.4} enter_frames={} exit_frames={} ema_alpha={:.4} min_visibility={:.4} min_visible_upper_body_joints={}",
        min_confidence,
        smoothing_window,
        enter_confidence,
        exit_confidence,
        enter_frames,
        exit_frames,
        ema_alpha,
        min_visibility,
        min_visible_upper_body_joints
    ));
    if let Ok(mut guard) = state().lock() {
        guard.config.min_confidence = if min_confidence > 0.0 {
            min_confidence as f32
        } else {
            0.6
        };
        guard.config.smoothing_window = if smoothing_window > 0 {
            smoothing_window as usize
        } else {
            5
        };
        guard.config.enter_confidence = if enter_confidence > 0.0 {
            enter_confidence as f32
        } else {
            guard.config.min_confidence
        };
        guard.config.exit_confidence = if exit_confidence >= 0.0 {
            exit_confidence as f32
        } else {
            0.45
        };
        if guard.config.exit_confidence > guard.config.enter_confidence {
            guard.config.exit_confidence = guard.config.enter_confidence;
        }
        guard.config.enter_frames = if enter_frames > 0 {
            enter_frames as usize
        } else {
            3
        };
        guard.config.exit_frames = if exit_frames > 0 {
            exit_frames as usize
        } else {
            8
        };
        guard.config.ema_alpha = if (0.0..=1.0).contains(&ema_alpha) {
            ema_alpha as f32
        } else {
            0.2
        };
        guard.config.min_visibility = if (0.0..=1.0).contains(&min_visibility) {
            min_visibility as f32
        } else {
            0.2
        };
        guard.config.min_visible_upper_body_joints = if min_visible_upper_body_joints > 0 {
            (min_visible_upper_body_joints as usize).min(6)
        } else {
            4
        };
        guard.frames.clear();
        guard.current_label = None;
        guard.current_confidence = 0.0;
        guard.last_inference_ms = -1.0;
        reset_temporal_state(&mut guard);
        log_debug(&format!(
            "exrec_start_session(): config applied min_confidence={:.4} smoothing_window={} enter_confidence={:.4} exit_confidence={:.4} enter_frames={} exit_frames={} ema_alpha={:.4} min_visibility={:.4} min_visible_upper_body_joints={}",
            guard.config.min_confidence,
            guard.config.smoothing_window,
            guard.config.enter_confidence,
            guard.config.exit_confidence,
            guard.config.enter_frames,
            guard.config.exit_frames,
            guard.config.ema_alpha,
            guard.config.min_visibility,
            guard.config.min_visible_upper_body_joints
        ));
        return;
    }

    log_debug("exrec_start_session(): failed to acquire state lock");
}

#[no_mangle]
pub extern "C" fn exrec_stop_session() {
    log_debug("exrec_stop_session(): begin");
    if let Ok(mut guard) = state().lock() {
        guard.frames.clear();
        guard.current_label = None;
        guard.current_confidence = 0.0;
        guard.last_inference_ms = -1.0;
        reset_temporal_state(&mut guard);
        log_debug("exrec_stop_session(): state cleared");
        return;
    }

    log_debug("exrec_stop_session(): failed to acquire state lock");
}

#[no_mangle]
pub extern "C" fn exrec_ingest_landmarks_buffer(values: *const c_double, len: usize) {
    log_debug(&format!("exrec_ingest_landmarks_buffer(): begin len={}", len));
    if values.is_null() || len < 33 * 4 {
        log_debug("exrec_ingest_landmarks_buffer(): invalid input pointer or insufficient length");
        return;
    }

    let slice = unsafe { std::slice::from_raw_parts(values, len) };
    let mut landmarks = Vec::with_capacity(33);
    for i in 0..33 {
        let base = i * 4;
        landmarks.push(Landmark {
            x: slice[base],
            y: slice[base + 1],
            z: slice[base + 2],
            visibility: Some(slice[base + 3]),
        });
    }

    if let Ok(mut guard) = state().lock() {
        if guard.model.is_none() {
            log_debug("exrec_ingest_landmarks_buffer(): model not loaded");
            return;
        }

        guard.frames.push_back(FrameInput::FrameRecord { landmarks });
        if guard.frames.len() > 45 {
            let _ = guard.frames.pop_front();
        }
        log_debug(&format!(
            "exrec_ingest_landmarks_buffer(): buffered frames={}",
            guard.frames.len()
        ));

        if guard.frames.len() < 10 {
            log_debug("exrec_ingest_landmarks_buffer(): waiting for minimum frame count (10)");
            return;
        }

        let infer_start = Instant::now();

        let frames_snapshot: Vec<FrameInput> = guard.frames.iter().cloned().collect();

        let feature_rows =
            features::extract_feature_rows(&frames_snapshot, guard.config.smoothing_window);
        log_debug(&format!(
            "exrec_ingest_landmarks_buffer(): extracted feature rows={}",
            feature_rows.len()
        ));
        let frame_probs = {
            let model = guard.model.as_ref().expect("model exists");
            model.predict_probabilities(&feature_rows)
        };

        let Some(frame_probs) = frame_probs else {
            log_debug("exrec_ingest_landmarks_buffer(): model.predict_probabilities returned None");
            return;
        };
        if frame_probs.is_empty() {
            log_debug("exrec_ingest_landmarks_buffer(): frame probabilities are empty");
            return;
        }
        log_debug(&format!(
            "exrec_ingest_landmarks_buffer(): frame probability rows={}",
            frame_probs.len()
        ));

        let avg_probs = average_probabilities(&frame_probs);
        if avg_probs.is_empty() {
            log_debug("exrec_ingest_landmarks_buffer(): averaged probabilities are empty");
            return;
        }

        let smoothed_probs = if guard.ema_probs.is_empty() {
            avg_probs.clone()
        } else {
            blend_with_ema(&guard.ema_probs, &avg_probs, guard.config.ema_alpha)
        };
        guard.ema_probs = smoothed_probs.clone();

        let winner_idx = argmax(&smoothed_probs);
        let class_ids = {
            let model = guard.model.as_ref().expect("model exists");
            model.class_ids().to_vec()
        };
        let winner_class_id = {
            class_ids
                .get(winner_idx)
                .copied()
                .unwrap_or(winner_idx as i32)
        };
        let winner_confidence = smoothed_probs.get(winner_idx).copied().unwrap_or(0.0);
        guard.current_confidence = winner_confidence;
        log_debug(&format!(
            "exrec_ingest_landmarks_buffer(): winner_idx={} class_id={} confidence={:.6}",
            winner_idx, winner_class_id, winner_confidence
        ));

        let visible_joint_count = upper_body_visible_joint_count(
            frames_snapshot
                .last()
                .map(|f| f.landmarks())
                .unwrap_or(&[]),
            guard.config.min_visibility,
        );
        let quality_ok = visible_joint_count >= guard.config.min_visible_upper_body_joints;
        log_debug(&format!(
            "exrec_ingest_landmarks_buffer(): visible_upper_body_joints={} quality_ok={} min_required={}",
            visible_joint_count,
            if quality_ok { 1 } else { 0 },
            guard.config.min_visible_upper_body_joints
        ));

        let null_class_id = 2;
        if let Some(active_class_id) = guard.active_class_id {
            let active_confidence = probability_for_class_id(&class_ids, &smoothed_probs, active_class_id);
            guard.current_confidence = active_confidence;

            let can_switch = quality_ok
                && winner_class_id != null_class_id
                && winner_class_id != active_class_id
                && winner_confidence >= guard.config.enter_confidence;

            if can_switch {
                if guard.pending_class_id == Some(winner_class_id) {
                    guard.pending_class_count += 1;
                } else {
                    guard.pending_class_id = Some(winner_class_id);
                    guard.pending_class_count = 1;
                }

                if guard.pending_class_count >= guard.config.enter_frames {
                    log_debug(&format!(
                        "exrec_ingest_landmarks_buffer(): switching active class {} -> {}",
                        active_class_id, winner_class_id
                    ));
                    guard.active_class_id = Some(winner_class_id);
                    guard.pending_class_id = None;
                    guard.pending_class_count = 0;
                    guard.low_confidence_count = 0;
                }
            } else if winner_class_id == active_class_id {
                guard.pending_class_id = None;
                guard.pending_class_count = 0;
            }

            let is_low_confidence = !quality_ok || active_confidence < guard.config.exit_confidence;
            if is_low_confidence {
                guard.low_confidence_count += 1;
            } else {
                guard.low_confidence_count = 0;
            }

            if guard.low_confidence_count >= guard.config.exit_frames {
                log_debug("exrec_ingest_landmarks_buffer(): release active class to null");
                guard.active_class_id = None;
                guard.pending_class_id = None;
                guard.pending_class_count = 0;
                guard.low_confidence_count = 0;
            }
        } else {
            let can_enter = quality_ok
                && winner_class_id != null_class_id
                && winner_confidence >= guard.config.enter_confidence;

            if can_enter {
                if guard.pending_class_id == Some(winner_class_id) {
                    guard.pending_class_count += 1;
                } else {
                    guard.pending_class_id = Some(winner_class_id);
                    guard.pending_class_count = 1;
                }

                if guard.pending_class_count >= guard.config.enter_frames {
                    log_debug(&format!(
                        "exrec_ingest_landmarks_buffer(): enter active class {}",
                        winner_class_id
                    ));
                    guard.active_class_id = Some(winner_class_id);
                    guard.pending_class_id = None;
                    guard.pending_class_count = 0;
                    guard.low_confidence_count = 0;
                }
            } else {
                guard.pending_class_id = None;
                guard.pending_class_count = 0;
            }
        }

        if let Some(active_class_id) = guard.active_class_id {
            let label = label_for_class_id(active_class_id).to_string();
            log_debug(&format!(
                "exrec_ingest_landmarks_buffer(): active label='{}'",
                label
            ));
            guard.current_label = Some(label);
        } else {
            log_debug("exrec_ingest_landmarks_buffer(): no active class, returning null label");
            guard.current_label = None;
        }
        guard.last_inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
        log_debug(&format!(
            "exrec_ingest_landmarks_buffer(): inference_ms={:.6}",
            guard.last_inference_ms
        ));
        return;
    }

    log_debug("exrec_ingest_landmarks_buffer(): failed to acquire state lock");
}

#[no_mangle]
pub extern "C" fn exrec_get_current_confidence() -> c_double {
    if let Ok(guard) = state().lock() {
        let confidence = guard.current_confidence as c_double;
        log_debug(&format!("exrec_get_current_confidence(): {:.6}", confidence));
        return confidence;
    }
    log_debug("exrec_get_current_confidence(): failed to acquire state lock");
    0.0
}

#[no_mangle]
pub extern "C" fn exrec_get_last_classifier_inference_time_ms() -> c_double {
    if let Ok(guard) = state().lock() {
        log_debug(&format!(
            "exrec_get_last_classifier_inference_time_ms(): {:.6}",
            guard.last_inference_ms
        ));
        return guard.last_inference_ms;
    }
    log_debug("exrec_get_last_classifier_inference_time_ms(): failed to acquire state lock");
    -1.0
}

#[no_mangle]
pub extern "C" fn exrec_get_current_exercise() -> *mut c_char {
    if let Ok(guard) = state().lock() {
        let text = guard.current_label.clone().unwrap_or_default();
        if text.is_empty() {
            log_debug("exrec_get_current_exercise(): no label available, returning null");
            return std::ptr::null_mut();
        }
        if let Ok(c) = CString::new(text) {
            log_debug("exrec_get_current_exercise(): returning current label");
            return c.into_raw();
        }
        log_debug("exrec_get_current_exercise(): failed to encode CString");
    }
    log_debug("exrec_get_current_exercise(): failed to acquire state lock");
    std::ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn exrec_string_free(s: *mut c_char) {
    if s.is_null() {
        log_debug("exrec_string_free(): pointer is null, skipping");
        return;
    }
    unsafe {
        let _ = CString::from_raw(s);
    }
    log_debug("exrec_string_free(): buffer released");
}

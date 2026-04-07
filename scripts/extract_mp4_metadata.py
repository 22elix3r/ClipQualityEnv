#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError as exc:
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError as exc:
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import whisper
except ImportError as exc:
    whisper = None
    _WHISPER_IMPORT_ERROR = exc
else:
    _WHISPER_IMPORT_ERROR = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clip_quality_env.models import ClipMetadata
from clip_quality_env.real_clips import derive_clip_difficulty
from clip_quality_env.rubric import RubricState


SUPPORTED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
_FACE_FALLBACK_WARNED = False
DEFAULT_ENV_TAG = "unknown_env"

ENV_KEYWORDS: dict[str, tuple[str, ...]] = {
    "podcast_studio": ("podcast", "studio", "talkinghead", "talking_head"),
    "office": ("office", "meeting", "work", "corp", "workspace"),
    "home_office": ("home", "bedroom", "livingroom", "living_room"),
    "webcam_room": ("webcam", "zoom", "teams", "meet"),
    "outdoor_interview": ("outdoor", "outside", "park", "field"),
    "crowded_event": ("event", "crowd", "conference", "stage"),
    "car_vlog": ("car", "vehicle", "dashboard"),
    "street_walk": ("street", "walk", "market", "sidewalk"),
}


@dataclass
class FaceStats:
    area_ratio: float
    confidence: float
    yaw_deg: float
    pitch_deg: float
    mouth_open_ratio: float
    blink_rate_hz: float
    occlusion_present: bool


def _run_ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid resolution from ffprobe for {path}")

    rate = str(stream.get("r_frame_rate", "0/1"))
    num, den = rate.split("/")
    fps = float(num) / max(float(den), 1.0)
    duration_s = float(payload.get("format", {}).get("duration", 0.0))
    if duration_s <= 0.0:
        raise RuntimeError(f"Invalid duration from ffprobe for {path}")

    return {
        "duration_s": round(duration_s, 3),
        "fps": max(1, int(round(fps))),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
    }


def _sample_frames(path: Path, sample_fps: float) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 24.0
    stride = max(1, int(round(source_fps / max(sample_fps, 0.1))))

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames sampled from {path}")
    return frames, source_fps


def _landmark_distance(a: Any, b: Any, width: int, height: int) -> float:
    ax, ay = float(a.x) * width, float(a.y) * height
    bx, by = float(b.x) * width, float(b.y) * height
    return math.hypot(ax - bx, ay - by)


def _extract_face_stats_fallback(frames: list[np.ndarray], sample_fps: float) -> FaceStats:
    """
    Fallback when MediaPipe face mesh API is unavailable in current build.
    Uses image heuristics that keep extraction running instead of dropping all clips.
    """
    global _FACE_FALLBACK_WARNED
    if not _FACE_FALLBACK_WARNED:
        print("[WARN] MediaPipe Face Mesh unavailable; using heuristic face fallback metrics.")
        _FACE_FALLBACK_WARNED = True

    h, w = frames[0].shape[:2]
    area_ratio = 0.28

    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    conf_proxy = 1.0 - float(np.std(gray0) / 128.0)
    confidence = float(np.clip(conf_proxy, 0.55, 0.90))

    # Head pose proxies from horizontal/vertical gradient imbalance.
    grad_x = np.mean(np.abs(np.diff(gray0.astype(np.float32), axis=1)))
    grad_y = np.mean(np.abs(np.diff(gray0.astype(np.float32), axis=0)))
    yaw_deg = float(np.clip((grad_x / max(grad_y, 1e-6)) * 8.0, 0.0, 35.0))
    pitch_deg = float(np.clip((grad_y / max(grad_x, 1e-6)) * 6.0, 0.0, 25.0))

    # Mouth-open proxy from lower-center variance.
    y0, y1 = int(0.55 * h), int(0.82 * h)
    x0, x1 = int(0.33 * w), int(0.67 * w)
    roi = gray0[y0:y1, x0:x1]
    mouth_open_ratio = float(np.clip(np.std(roi) / 96.0, 0.18, 0.62))

    # Blink proxy unavailable -> conservative nominal rate.
    blink_rate_hz = 0.25

    # Occlusion proxy via darkness saturation ratio in center region.
    center = gray0[int(0.2 * h) : int(0.85 * h), int(0.2 * w) : int(0.8 * w)]
    dark_ratio = float(np.mean(center < 18))
    occlusion_present = dark_ratio > 0.35

    return FaceStats(
        area_ratio=area_ratio,
        confidence=confidence,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        mouth_open_ratio=mouth_open_ratio,
        blink_rate_hz=blink_rate_hz,
        occlusion_present=occlusion_present,
    )


def _extract_face_stats(frames: list[np.ndarray], sample_fps: float) -> tuple[FaceStats, list[float]]:
    """Returns (FaceStats, per_frame_yaws) so callers can compute eye_contact_ratio."""
    if mp is None or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
        # Fallback: approximate per-frame yaw from horizontal gradient imbalance.
        per_frame_yaws: list[float] = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grad_x = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=1)))
            grad_y = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=0)))
            per_frame_yaws.append(float(np.clip((grad_x / max(grad_y, 1e-6)) * 8.0, 0.0, 35.0)))
        return _extract_face_stats_fallback(frames, sample_fps), per_frame_yaws

    mp_mesh = mp.solutions.face_mesh
    face_mesh = mp_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    area_ratios: list[float] = []
    confidences: list[float] = []
    yaws: list[float] = []
    pitches: list[float] = []
    mouth_open: list[float] = []
    blink_closures: list[float] = []
    per_frame_yaws: list[float] = []
    missing_face = 0

    # Mouth/eye landmarks from MediaPipe Face Mesh topology.
    mouth_top, mouth_bottom = 13, 14
    mouth_left, mouth_right = 78, 308
    l_eye_top, l_eye_bottom = 159, 145
    l_eye_left, l_eye_right = 33, 133
    r_eye_top, r_eye_bottom = 386, 374
    r_eye_left, r_eye_right = 362, 263
    nose_tip, left_cheek, right_cheek = 1, 234, 454
    forehead, chin = 10, 152

    for frame in frames:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            missing_face += 1
            continue

        lm = result.multi_face_landmarks[0].landmark

        xs = np.array([p.x for p in lm], dtype=np.float32)
        ys = np.array([p.y for p in lm], dtype=np.float32)
        x0 = float(np.clip(xs.min(), 0.0, 1.0))
        x1 = float(np.clip(xs.max(), 0.0, 1.0))
        y0 = float(np.clip(ys.min(), 0.0, 1.0))
        y1 = float(np.clip(ys.max(), 0.0, 1.0))
        area_ratio = max(0.0, (x1 - x0) * (y1 - y0))
        area_ratios.append(area_ratio)

        # MediaPipe FaceMesh does not expose direct confidence per face; use coverage proxy.
        conf = 1.0 - float(np.mean((xs <= 0.0) | (xs >= 1.0) | (ys <= 0.0) | (ys >= 1.0)))
        confidences.append(float(np.clip(conf, 0.0, 1.0)))

        nose = lm[nose_tip]
        lch = lm[left_cheek]
        rch = lm[right_cheek]
        fh = lm[forehead]
        ch = lm[chin]
        yaw = abs((nose.x - (lch.x + rch.x) / 2.0) * 120.0)
        pitch = abs((nose.y - (fh.y + ch.y) / 2.0) * 120.0)
        yaws.append(float(yaw))
        per_frame_yaws.append(float(yaw))
        pitches.append(float(pitch))

        mouth_h = _landmark_distance(lm[mouth_top], lm[mouth_bottom], w, h)
        mouth_w = max(_landmark_distance(lm[mouth_left], lm[mouth_right], w, h), 1e-6)
        mouth_open.append(float(np.clip(mouth_h / mouth_w, 0.0, 1.0)))

        l_h = _landmark_distance(lm[l_eye_top], lm[l_eye_bottom], w, h)
        l_w = max(_landmark_distance(lm[l_eye_left], lm[l_eye_right], w, h), 1e-6)
        r_h = _landmark_distance(lm[r_eye_top], lm[r_eye_bottom], w, h)
        r_w = max(_landmark_distance(lm[r_eye_left], lm[r_eye_right], w, h), 1e-6)
        eye_open_ratio = float(np.clip(0.5 * (l_h / l_w + r_h / r_w), 0.0, 1.0))
        blink_closures.append(eye_open_ratio)

    face_mesh.close()

    if not area_ratios:
        raise RuntimeError("No detectable face landmarks in sampled frames")

    # Blink estimate: count closures under threshold in sampled stream, then normalize by duration.
    blink_events = 0
    prev_closed = False
    for r in blink_closures:
        closed = r < 0.18
        if closed and not prev_closed:
            blink_events += 1
        prev_closed = closed
    sampled_seconds = len(frames) / max(sample_fps, 1e-6)
    blink_rate_hz = blink_events / max(sampled_seconds, 1e-6)

    occlusion_ratio = missing_face / len(frames)
    occlusion_present = occlusion_ratio > 0.30

    return FaceStats(
        area_ratio=float(np.clip(np.mean(area_ratios), 0.0, 1.0)),
        confidence=float(np.clip(np.mean(confidences), 0.0, 1.0)),
        yaw_deg=float(np.clip(np.mean(yaws), 0.0, 180.0)),
        pitch_deg=float(np.clip(np.mean(pitches), 0.0, 180.0)),
        mouth_open_ratio=float(np.clip(np.mean(mouth_open), 0.0, 1.0)),
        blink_rate_hz=max(0.0, float(blink_rate_hz)),
        occlusion_present=occlusion_present,
    ), per_frame_yaws


def _motion_score(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    scores = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev, gray)
        scores.append(float(np.mean(diff) / 255.0))
        prev = gray
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _sharpness_score(frames: list[np.ndarray]) -> float:
    """Laplacian variance over the central face ROI (approx top-60% of frame).
    Higher = sharper. Normalised to [0, 1] via soft-cap at 2000 variance units.
    """
    vals = []
    for frame in frames:
        h, w = frame.shape[:2]
        # Face ROI: horizontal centre third, upper 60% of frame.
        roi = frame[0 : int(0.60 * h), int(0.25 * w) : int(0.75 * w)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        vals.append(lap_var)
    mean_var = float(np.mean(vals)) if vals else 0.0
    return float(np.clip(mean_var / 2000.0, 0.0, 1.0))


def _temporal_flicker(frames: list[np.ndarray]) -> float:
    """Std-dev of per-frame mean brightness, normalised to [0, 1].
    High values indicate unstable / flickering lighting.
    """
    if not frames:
        return 0.0
    means = [float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in frames]
    flicker = float(np.std(means))
    return float(np.clip(flicker / 64.0, 0.0, 1.0))


def _bg_entropy(frames: list[np.ndarray]) -> float:
    """Shannon entropy of a background patch (outer 20% ring of frame).
    Higher = more complex / busy background.
    """
    entropies = []
    for frame in frames:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Build background mask: only keep the outer 20% border strip.
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask[int(0.20 * h) : int(0.80 * h), int(0.20 * w) : int(0.80 * w)] = 0
        pixels = gray[mask == 255].astype(np.float32)
        if pixels.size == 0:
            continue
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
        entropies.append(entropy)
    if not entropies:
        return 0.0
    # Max theoretical entropy for 256 bins is log2(256)=8; normalise.
    return float(np.clip(np.mean(entropies) / 8.0, 0.0, 1.0))


def _eye_contact_ratio(frames: list[np.ndarray], face_yaws: list[float]) -> float:
    """Fraction of sampled frames where head yaw < 5 degrees (frontal gaze).
    Computed from per-frame yaw list produced alongside face stats.
    Falls back to a single global yaw threshold when per-frame data is unavailable.
    """
    if not face_yaws:
        return 0.0
    frontal = sum(1 for y in face_yaws if y < 5.0)
    return float(np.clip(frontal / len(face_yaws), 0.0, 1.0))


def _bg_complexity(frames: list[np.ndarray]) -> tuple[str, float]:
    # Hybrid complexity proxy: edge density + intensity variation + local texture energy.
    edge_densities = []
    texture_energies = []
    intensity_vars = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.mean(edges > 0))
        edge_densities.append(edge_density)

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture_energy = float(np.mean(np.abs(lap)) / 255.0)
        texture_energies.append(texture_energy)

        intensity_vars.append(float(np.std(gray.astype(np.float32)) / 128.0))

    edge_score = float(np.clip(np.mean(edge_densities), 0.0, 1.0))
    texture_score = float(np.clip(np.mean(texture_energies), 0.0, 1.0))
    variation_score = float(np.clip(np.mean(intensity_vars), 0.0, 1.0))
    score = float(np.clip(0.45 * edge_score + 0.30 * texture_score + 0.25 * variation_score, 0.0, 1.0))

    if score < 0.07:
        tag = "solid_dark"
    elif score < 0.10:
        tag = "simple_room"
    elif score < 0.15:
        tag = "busy_room"
    else:
        tag = "crowded_event"
    return tag, score


def _lighting_uniformity(frames: list[np.ndarray]) -> float:
    vals = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        std = float(np.std(gray))
        vals.append(float(np.clip(1.0 - 2.0 * std, 0.0, 1.0)))
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def _extract_audio_track(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed for {video_path}: {result.stderr.strip()}")


def _audio_snr_db(wav_path: Path) -> float:
    # Approximate SNR using RMS percentile split.
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"amovie={wav_path},astats=metadata=1:reset=1",
        "-show_entries",
        "frame_tags=lavfi.astats.Overall.RMS_level",
        "-of",
        "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return 0.0
    payload = json.loads(result.stdout or "{}")
    frames = payload.get("frames", [])
    levels = []
    for frame in frames:
        tags = frame.get("tags", {})
        val = tags.get("lavfi.astats.Overall.RMS_level")
        if val is None:
            continue
        try:
            levels.append(float(val))
        except ValueError:
            continue
    if len(levels) < 10:
        return 0.0
    arr = np.array(levels, dtype=np.float32)
    speech = np.percentile(arr, 85)
    noise = np.percentile(arr, 20)
    return float(np.clip(speech - noise, 0.0, 80.0))


def _transcribe(wav_path: Path, model: Any) -> tuple[int, float]:
    result = model.transcribe(str(wav_path), language="en", fp16=False, verbose=False)
    text = str(result.get("text", "")).strip()
    segments = result.get("segments", []) or []
    # Whisper does not provide true confidence; use avg_logprob as bounded proxy.
    confs = []
    for seg in segments:
        avg_logprob = seg.get("avg_logprob")
        if avg_logprob is None:
            continue
        confs.append(float(1.0 / (1.0 + math.exp(-avg_logprob))))
    confidence = float(np.mean(confs)) if confs else 0.0
    return len(text.split()), float(np.clip(confidence, 0.0, 1.0))


def _load_environment_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            k = str(key).strip().lower().replace("\\", "/")
            v = str(value).strip()
            if not k or not v:
                continue
            mapping[k] = v
        return mapping
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            key = item.get("clip_id") or item.get("path") or item.get("name")
            value = item.get("environment_tag")
            if key is None or value is None:
                continue
            k = str(key).strip().lower().replace("\\", "/")
            v = str(value).strip()
            if not k or not v:
                continue
            mapping[k] = v
        return mapping
    raise ValueError("environment map must be JSON object or list of {clip_id/path,name,environment_tag}")


def _environment_tag_from_features(
    bg_complexity_score: float,
    motion_score: float,
    lighting_uniformity: float,
    audio_snr_db: float,
) -> str:
    activity = 0.60 * bg_complexity_score + 0.30 * motion_score + 0.10 * (1.0 - lighting_uniformity)
    if motion_score >= 0.075 or (activity >= 0.14 and bg_complexity_score >= 0.13):
        return "street_walk"
    if bg_complexity_score >= 0.18 or activity >= 0.18:
        return "crowded_event"
    if audio_snr_db < 14.5 and motion_score >= 0.035:
        return "car_vlog"
    if lighting_uniformity >= 0.76 and bg_complexity_score <= 0.09 and motion_score <= 0.03 and audio_snr_db >= 16.0:
        return "podcast_studio"
    if bg_complexity_score <= 0.12 and lighting_uniformity >= 0.67:
        return "office"
    if lighting_uniformity < 0.62:
        return "webcam_room"
    return "home_office"


def _environment_tag(
    path_hint: Path,
    clip_id: str,
    bg_complexity_score: float,
    motion_score: float,
    lighting_uniformity: float,
    audio_snr_db: float,
    environment_map: dict[str, str] | None = None,
) -> str:
    mapping = environment_map or {}
    norm_path = str(path_hint).strip().lower().replace("\\", "/")
    base_name = Path(norm_path).name
    base_stem = Path(norm_path).stem
    clip_key = clip_id.strip().lower()

    for key in (norm_path, base_name, base_stem, clip_key):
        if key in mapping:
            return mapping[key]

    # Use only user-intended path hint (relative clip path), not absolute system path.
    for tag, keys in ENV_KEYWORDS.items():
        if any(k in norm_path for k in keys):
            return tag

    return _environment_tag_from_features(
        bg_complexity_score=bg_complexity_score,
        motion_score=motion_score,
        lighting_uniformity=lighting_uniformity,
        audio_snr_db=audio_snr_db,
    )


def _framing_from_path_or_pose(path_hint: Path, yaw_deg: float) -> str:
    name = str(path_hint).lower()
    if "closeup" in name or "close_up" in name:
        return "closeup"
    if "offgaze" in name or "off_gaze" in name:
        return "offgaze"
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    if yaw_deg <= 6.0:
        return "front"
    if yaw_deg <= 11.0:
        return "offgaze"
    return "left" if ("left" in name) else "right"


def _extract_clip_metadata(
    video_path: Path,
    path_hint: Path,
    sample_fps: float,
    asr_model: Any,
    rubric: RubricState,
    with_difficulty: bool,
    environment_tag_override: str | None = None,
    environment_map: dict[str, str] | None = None,
    framing_override: str | None = None,
) -> dict[str, Any]:
    probe = _run_ffprobe(video_path)
    frames, _source_fps = _sample_frames(video_path, sample_fps=sample_fps)
    face, per_frame_yaws = _extract_face_stats(frames, sample_fps=sample_fps)
    motion = _motion_score(frames)
    bg_tag, bg_score = _bg_complexity(frames)
    light = _lighting_uniformity(frames)
    sharpness = _sharpness_score(frames)
    flicker = _temporal_flicker(frames)
    entropy = _bg_entropy(frames)
    eye_contact = _eye_contact_ratio(frames, per_frame_yaws)

    wav_path = video_path.with_suffix(".tmp16k.wav")
    _extract_audio_track(video_path, wav_path)
    try:
        snr = _audio_snr_db(wav_path)
        word_count, transcript_conf = _transcribe(wav_path, asr_model)
    finally:
        if wav_path.exists():
            wav_path.unlink()

    duration_s = probe["duration_s"]
    speech_rate_wpm = round((word_count / max(duration_s, 1e-6)) * 60.0, 2) if word_count > 0 else 0.0

    row: dict[str, Any] = {
        "clip_id": video_path.stem,
        "duration_s": duration_s,
        "fps": probe["fps"],
        "resolution": probe["resolution"],
        "face_area_ratio": round(face.area_ratio, 4),
        "face_confidence": round(face.confidence, 4),
        "head_pose_yaw_deg": round(face.yaw_deg, 3),
        "head_pose_pitch_deg": round(face.pitch_deg, 3),
        "motion_score": round(motion, 4),
        "bg_complexity": bg_tag,
        "bg_complexity_score": round(bg_score, 4),
        "mouth_open_ratio": round(face.mouth_open_ratio, 4),
        "blink_rate_hz": round(face.blink_rate_hz, 4),
        "audio_snr_db": round(float(snr), 3),
        "transcript_word_count": int(word_count),
        "transcript_confidence": round(float(transcript_conf), 4),
        "lighting_uniformity": round(light, 4),
        "occlusion_present": bool(face.occlusion_present),
        # ── Option-A enriched features ──────────────────────────────────────
        "sharpness_score": round(sharpness, 4),
        "temporal_flicker": round(flicker, 4),
        "bg_entropy": round(entropy, 4),
        "eye_contact_ratio": round(eye_contact, 4),
        "speech_rate_wpm": speech_rate_wpm,
        # ───────────────────────────────────────────────────────────────────
        "environment_tag": environment_tag_override
        or _environment_tag(
            path_hint=path_hint,
            clip_id=video_path.stem,
            bg_complexity_score=bg_score,
            motion_score=motion,
            lighting_uniformity=light,
            audio_snr_db=float(snr),
            environment_map=environment_map,
        ),
        "framing": framing_override or _framing_from_path_or_pose(path_hint=path_hint, yaw_deg=face.yaw_deg),
    }

    clip = ClipMetadata(**row)
    payload = clip.model_dump()
    if with_difficulty:
        payload["difficulty"] = derive_clip_difficulty(payload, rubric)
    return payload


def _iter_videos(input_dir: Path) -> list[Path]:
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ClipMetadata manifest from MP4 files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing video clips.")
    parser.add_argument("--output", type=str, required=True, help="Output manifest path (.jsonl).")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frame sampling FPS for visual features.")
    parser.add_argument("--max-clips", type=int, default=0, help="Optional cap; 0 means all clips.")
    parser.add_argument(
        "--difficulty-mode",
        type=str,
        choices=["none", "derive"],
        default="derive",
        help="Attach derived difficulty labels or not.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small",
        help="Whisper model size (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--environment-tag",
        type=str,
        default=None,
        help="Optional fixed environment_tag override for all clips in this run.",
    )
    parser.add_argument(
        "--framing",
        type=str,
        default=None,
        help="Optional fixed framing override for all clips in this run.",
    )
    parser.add_argument(
        "--environment-map",
        type=str,
        default=None,
        help="Optional JSON map/list to assign environment_tag by clip_id/path/name.",
    )
    args = parser.parse_args()

    dep_errors = [err for err in (_CV2_IMPORT_ERROR, _NUMPY_IMPORT_ERROR, _WHISPER_IMPORT_ERROR) if err is not None]
    if dep_errors:
        raise RuntimeError(
            "Missing extractor dependencies. Install: "
            "opencv-python-headless mediapipe openai-whisper numpy"
        ) from dep_errors[0]

    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    rubric = RubricState()
    asr_model = whisper.load_model(args.whisper_model)
    environment_map = _load_environment_map(args.environment_map)

    videos = _iter_videos(input_dir)
    if args.max_clips > 0:
        videos = videos[: args.max_clips]
    if not videos:
        raise RuntimeError(f"No supported video files found in {input_dir}")

    failures: list[tuple[str, str]] = []
    with open(output_path, "w", encoding="utf-8") as out:
        for video_path in videos:
            try:
                row = _extract_clip_metadata(
                    video_path=video_path,
                    path_hint=video_path.relative_to(input_dir),
                    sample_fps=args.sample_fps,
                    asr_model=asr_model,
                    rubric=rubric,
                    with_difficulty=(args.difficulty_mode == "derive"),
                    environment_tag_override=args.environment_tag,
                    environment_map=environment_map,
                    framing_override=args.framing,
                )
            except Exception as exc:
                failures.append((str(video_path), str(exc)))
                continue
            out.write(json.dumps(row, ensure_ascii=True) + "\n")

    with open(output_path, "r", encoding="utf-8") as f:
        written = sum(1 for _ in f)
    print(json.dumps({"output": str(output_path), "written_rows": written, "failed_clips": len(failures)}, indent=2))
    if failures:
        for clip, err in failures:
            print(f"[FAIL] {clip}: {err}")


if __name__ == "__main__":
    main()

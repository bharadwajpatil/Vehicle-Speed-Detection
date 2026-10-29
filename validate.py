"""
Validation and Accuracy Benchmark for Vehicle Speed Detection
Author: Bharadwaj Patil

Evaluates the speed estimation pipeline against known ground-truth velocities.
Supported detector modes:
- "colour" (default): Colour-mask detector isolating geometric/homography speed math.
- "yolo": Full YOLOv8 + ByteTrack pipeline from speed.py.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from speed import (
    PerspectiveTransformer,
    SpeedEstimator,
    VehicleSpeedDetector,
    generate_synthetic_demo_video,
)

THRESHOLD_PCT = 8.0


def run_benchmark(
    video_path: str = "demo_clip.mp4",
    config_path: str = "config/calibration_config.yaml",
    detector_type: str = "colour",
) -> Dict:
    """
    Runs speed validation benchmark on the test video.
    In the synthetic ground-truth test clip:
    - Car A (Compliant): Ground Truth = 42.0 km/h (Lane 1, left)
    - Car B (Speeding):  Ground Truth = 72.0 km/h (Lane 2, right)
    """
    print("=" * 65)
    print("   VEHICLE SPEED DETECTION - ACCURACY BENCHMARK & VALIDATION")
    print(f"   Detector Mode: {detector_type.upper()}")
    print("=" * 65)

    # 1. Ensure test video exists
    if not Path(video_path).exists():
        print(f"[INFO] Benchmark clip not found. Generating {video_path}...")
        generate_synthetic_demo_video(video_path, num_frames=180, fps=30)

    # 2. Load configuration parameters from YAML
    cfg_p = Path(config_path)
    if not cfg_p.exists() and (Path("config") / config_path).exists():
        cfg_p = Path("config") / config_path

    cfg_data = {}
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

    speed_cfg = cfg_data.get("speed_estimation", {})
    speed_limit = float(speed_cfg.get("speed_limit_kmh", 50.0))
    window_seconds = float(speed_cfg.get("window_seconds", 0.75))
    min_displacement = float(speed_cfg.get("min_displacement_meters", 0.8))
    ema_factor = float(speed_cfg.get("ema_smoothing_factor", 0.7))
    max_speed = float(speed_cfg.get("max_plausible_kmh", 220.0))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    transformer = PerspectiveTransformer.from_config(str(cfg_p), width, height)
    estimator = SpeedEstimator(
        speed_limit_kmh=speed_limit,
        max_plausible_kmh=max_speed,
        window_seconds=window_seconds,
        min_displacement_meters=min_displacement,
        ema_factor=ema_factor,
    )

    # Known ground truths for synthetic benchmark clip
    ground_truths = {
        "Car A (Compliant)": 42.0,
        "Car B (Speeding)": 72.0,
    }

    measured_speeds: Dict[str, List[float]] = {
        "Car A (Compliant)": [],
        "Car B (Speeding)": [],
    }

    frame_idx = 0

    if detector_type == "colour":
        # Target BGR colours from synthetic generator
        targets = [
            ("Car A (Compliant)", [190, 60, 40], 1),
            ("Car B (Speeding)", [40, 80, 220], 2),
        ]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            video_time = frame_idx / fps

            for label, target_bgr, tid in targets:
                bgr = np.array(target_bgr, dtype=np.uint8)
                mask = cv2.inRange(frame, bgr - 15, bgr + 15)
                pts = cv2.findNonZero(mask)
                if pts is not None and len(pts) > 20:
                    x, y, w, h = cv2.boundingRect(pts)
                    cx = x + w / 2.0
                    cy = y + h  # Ground contact point
                    mx, my = transformer.transform_point(cx, cy)
                    spd = estimator.update(tid, video_time, mx, my, int(cx), int(cy))
                    if spd is not None and frame_idx > 25:  # Allow warmup window
                        measured_speeds[label].append(spd)

            frame_idx += 1

    elif detector_type == "yolo":
        model_cfg = cfg_data.get("model", {})
        weights = model_cfg.get("weights", "yolov8n.pt")
        conf = float(model_cfg.get("confidence", 0.35))
        classes = model_cfg.get("classes", [2, 3, 5, 7])

        detector = VehicleSpeedDetector(
            weights=weights,
            speed_limit=speed_limit,
            config_path=str(cfg_p),
            conf=conf,
            classes=classes,
            window_seconds=window_seconds,
            max_speed=max_speed,
        )

        midpoint_x = width / 2.0
        total_detections = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            _, detections = detector.process_frame(frame, frame_idx, fps)
            total_detections += len(detections)

            for det in detections:
                spd = det.get("speed")
                if spd is not None and frame_idx > 25:
                    gx, _ = det["ground_pt"]
                    # Assign to lane: left lane = Car A, right lane = Car B
                    label = "Car A (Compliant)" if gx < midpoint_x else "Car B (Speeding)"
                    measured_speeds[label].append(spd)

            frame_idx += 1

        if total_detections == 0:
            print("[WARNING] YOLOv8 + ByteTrack detected 0 vehicles in this synthetic test clip.")
            print("[EXPLANATION] YOLOv8 COCO pre-trained weights expect natural photographic vehicle features")
            print("              (tires, headlights, windshield reflections). Flat colored rectangles in this")
            print("              synthetic clip are not recognized as COCO classes [car, bus, truck].")
            print("[TIP] Use --detector colour for the geometric speed math validation on synthetic clips.")
    else:
        raise ValueError(f"Unknown detector type: '{detector_type}'. Use 'colour' or 'yolo'.")

    cap.release()

    # Calculate metrics
    results = {}
    print(f"\nProcessed {frame_idx} frames for benchmark evaluation.")
    print("-" * 65)
    print(f"{'Vehicle':<22} | {'True Speed':<11} | {'Measured':<11} | {'Error (%)':<10}")
    print("-" * 65)

    total_pct_err = 0.0
    count = 0

    for label, gt in ground_truths.items():
        speeds = measured_speeds[label]
        if speeds:
            mean_measured = float(np.mean(speeds[-30:])) if len(speeds) >= 30 else float(np.mean(speeds))
            abs_err = abs(mean_measured - gt)
            pct_err = (abs_err / gt) * 100.0
            total_pct_err += pct_err
            count += 1

            results[label] = {
                "ground_truth_kmh": gt,
                "measured_kmh": round(mean_measured, 2),
                "absolute_error_kmh": round(abs_err, 2),
                "percentage_error": round(pct_err, 2),
            }

            print(f"{label:<22} | {gt:6.1f} km/h | {mean_measured:6.1f} km/h | {pct_err:5.2f}%")
        else:
            results[label] = {
                "ground_truth_kmh": gt,
                "measured_kmh": None,
                "absolute_error_kmh": None,
                "percentage_error": None,
            }
            print(f"{label:<22} | {gt:6.1f} km/h | {'N/A':<11} | N/A")

    if count > 0:
        mape = total_pct_err / count
        passed = (count == len(ground_truths)) and (mape < THRESHOLD_PCT)
    else:
        mape = float("nan")
        passed = False

    status_str = "PASSED" if passed else "FAILED"
    test_type_str = (
        "synthetic clip, colour-mask detector (geometry/speed-math check only)"
        if detector_type == "colour"
        else "synthetic clip, YOLOv8 + ByteTrack detector"
    )

    results["summary"] = {
        "test_type": test_type_str,
        "detector": detector_type,
        "MAPE": round(mape, 2) if not np.isnan(mape) else None,
        "threshold_pct": THRESHOLD_PCT,
        "status": status_str,
        "passed": bool(passed),
        "frames_evaluated": frame_idx,
    }

    print("-" * 65)
    if not np.isnan(mape):
        print(f"Mean Absolute Percentage Error (MAPE): ±{mape:.2f}% (Threshold: < ±{THRESHOLD_PCT:.1f}%)")
    else:
        print(f"Mean Absolute Percentage Error (MAPE): N/A (Threshold: < ±{THRESHOLD_PCT:.1f}%)")
    print(f"Validation Status: {status_str}")
    print("=" * 65)

    # Save results to benchmark_results.json
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[INFO] Benchmark metrics saved to benchmark_results.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate speed detection pipeline against known ground truths.")
    parser.add_argument("--video", default="demo_clip.mp4", help="Path to video clip with known vehicle speeds.")
    parser.add_argument("--config", default="config/calibration_config.yaml", help="Path to calibration YAML.")
    parser.add_argument(
        "--detector",
        choices=["colour", "yolo"],
        default="colour",
        help="Detector mode: 'colour' (default colour-mask geometric check) or 'yolo' (YOLOv8 + ByteTrack).",
    )
    args = parser.parse_args()

    results = run_benchmark(args.video, args.config, detector_type=args.detector)
    passed = results.get("summary", {}).get("passed", False)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

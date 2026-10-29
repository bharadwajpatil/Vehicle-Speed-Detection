"""
Vehicle Speed Detection using YOLOv8 & ByteTrack with Perspective Calibration
Author: Bharadwaj Patil

Features:
- Ultralytics YOLOv8 for vehicle detection
- ByteTrack for stable, persistent multi-object tracking
- Perspective transformation (Homography) mapping road pixels to real-world meters
- Automatic resolution scaling & ROI boundary verification
- Video-time based calculation (frame_index / fps) for deterministic, frame-rate independent accuracy
- Configurable temporal sliding window with exponential smoothing and outlier filtering
- CLI with video file, webcam, headless mode, and video export support
"""

import argparse
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


DEFAULT_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class PerspectiveTransformer:
    """
    Computes and applies perspective transformation (Homography) to convert
    image pixel coordinates into real-world metric coordinates (meters) on a top-down ground plane.
    """

    def __init__(self, source_points: np.ndarray, target_width: float, target_length: float):
        self.source_points = np.float32(source_points)
        self.target_width = float(target_width)
        self.target_length = float(target_length)

        # Destination rectangle in meters: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        self.destination_points = np.float32([
            [0.0, 0.0],
            [self.target_width, 0.0],
            [self.target_width, self.target_length],
            [0.0, self.target_length],
        ])

        # Compute Homography matrix
        self.matrix = cv2.getPerspectiveTransform(self.source_points, self.destination_points)

    @classmethod
    def from_config(cls, config_path: str, frame_width: int, frame_height: int) -> "PerspectiveTransformer":
        """Load calibration from YAML config with auto-scaling to current frame resolution."""
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            calib = data.get("calibration", {})
            src = calib.get("source_points")
            dims = calib.get("target_dimensions", {})
            ref_res = calib.get("reference_resolution", [1280, 720])
            width = dims.get("width", 8.0)
            length = dims.get("length", 40.0)

            if src and len(src) == 4:
                src_arr = np.array(src, dtype=np.float32)
                # Scale points if input resolution differs from reference resolution
                if ref_res and len(ref_res) == 2:
                    ref_w, ref_h = ref_res
                    if ref_w > 0 and ref_h > 0 and (ref_w != frame_width or ref_h != frame_height):
                        scale_x = frame_width / ref_w
                        scale_y = frame_height / ref_h
                        src_arr[:, 0] *= scale_x
                        src_arr[:, 1] *= scale_y

                return cls(src_arr, width, length)

        # Fallback default trapezoid adapted to current frame resolution
        src = np.float32([
            [frame_width * 0.35, frame_height * 0.45],  # Top-Left
            [frame_width * 0.65, frame_height * 0.45],  # Top-Right
            [frame_width * 0.90, frame_height * 0.95],  # Bottom-Right
            [frame_width * 0.10, frame_height * 0.95],  # Bottom-Left
        ])
        return cls(src, target_width=8.0, target_length=40.0)

    def is_inside_roi(self, x: float, y: float, margin_px: float = 30.0) -> bool:
        """Checks if a point is within or near the calibrated road polygon."""
        dist = cv2.pointPolygonTest(self.source_points.astype(np.int32), (float(x), float(y)), True)
        return dist >= -margin_px

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Maps an image coordinate (x, y) to metric ground plane (meters_x, meters_y)."""
        pts = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, self.matrix)
        mx, my = warped[0][0]
        return float(mx), float(my)

    def draw_roi(self, frame: np.ndarray, alpha: float = 0.22) -> np.ndarray:
        """Visualizes the calibrated road detection zone overlay on the frame."""
        overlay = frame.copy()
        pts = self.source_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (255, 140, 0))  # Semi-transparent amber/cyan
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw boundary and corner markers
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)
        labels = ["TL", "TR", "BR", "BL"]
        for idx, (px, py) in enumerate(self.source_points.astype(int)):
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                labels[idx],
                (px + 8, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return frame


class SpeedEstimator:
    """
    Estimates vehicle speeds using timestamps calculated from video frame index / FPS
    and top-down metric displacements over a rolling temporal window.
    """

    def __init__(
        self,
        speed_limit_kmh: float = 50.0,
        max_plausible_kmh: float = 220.0,
        window_seconds: float = 0.75,
        min_displacement_meters: float = 0.8,
        ema_factor: float = 0.7,
    ):
        self.speed_limit = speed_limit_kmh
        self.max_plausible_kmh = max_plausible_kmh
        self.window_seconds = window_seconds
        self.min_displacement_meters = min_displacement_meters
        self.ema_factor = ema_factor

        self.tracks_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self.speeds: Dict[int, float] = {}
        self.speeding_history: Dict[int, float] = {}

    def update(
        self, track_id: int, current_time: float, meter_x: float, meter_y: float, px: int, py: int
    ) -> Optional[float]:
        """
        Record a vehicle's position at current_time and calculate its speed in km/h.
        """
        history = self.tracks_history[track_id]
        history.append((current_time, meter_x, meter_y, px, py))

        # Look back in history for a point approximately `window_seconds` ago
        target_time = current_time - self.window_seconds
        prev_point = None

        for pt in history:
            if pt[0] <= target_time:
                prev_point = pt
            else:
                break

        if prev_point is None:
            return self.speeds.get(track_id)

        t_prev, x_prev, y_prev, _, _ = prev_point
        dt = current_time - t_prev
        if dt < 0.1:
            return self.speeds.get(track_id)

        # Euclidean displacement in real-world meters
        displacement = np.hypot(meter_x - x_prev, meter_y - y_prev)

        if displacement < self.min_displacement_meters and track_id not in self.speeds:
            return None

        # Speed in km/h
        instant_speed = (displacement / dt) * 3.6

        # Outlier rejection: ignore tracking jumps that exceed physical thresholds
        if instant_speed > self.max_plausible_kmh:
            return self.speeds.get(track_id)

        # Apply exponential moving average (EMA) smoothing
        if track_id in self.speeds:
            smoothed = self.ema_factor * self.speeds[track_id] + (1 - self.ema_factor) * instant_speed
        else:
            smoothed = instant_speed

        self.speeds[track_id] = smoothed

        if smoothed > self.speed_limit:
            self.speeding_history[track_id] = max(smoothed, self.speeding_history.get(track_id, 0.0))

        return smoothed

    def clean_stale(self, active_track_ids: List[int], max_stale_frames: int = 60):
        """Purge tracking records for objects no longer visible."""
        current_active = set(active_track_ids)
        for tid in list(self.tracks_history.keys()):
            if tid not in current_active and len(self.tracks_history[tid]) > 0:
                if tid in self.speeds and len(self.tracks_history[tid]) > max_stale_frames:
                    del self.tracks_history[tid]
                    del self.speeds[tid]


class VehicleSpeedDetector:
    """
    Main pipeline combining YOLOv8 detection, ByteTrack tracking,
    perspective metric mapping, and HUD video visualization.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        speed_limit: float = 50.0,
        config_path: Optional[str] = "calibration_config.yaml",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.35,
        iou: float = 0.7,
        classes: Optional[List[int]] = None,
        window_seconds: float = 0.75,
        max_speed: float = 220.0,
    ):
        if YOLO is None:
            raise ImportError(
                "The 'ultralytics' library is required. Please install it using: pip install -r requirements.txt"
            )

        print(f"[INFO] Loading YOLOv8 model from {weights}...")
        self.model = YOLO(weights)
        self.tracker_name = tracker
        self.confidence_threshold = conf
        self.iou_threshold = iou
        self.vehicle_classes = classes if classes is not None else [2, 3, 5, 7]
        self.config_path = config_path

        self.transformer: Optional[PerspectiveTransformer] = None
        self.estimator = SpeedEstimator(
            speed_limit_kmh=speed_limit,
            max_plausible_kmh=max_speed,
            window_seconds=window_seconds,
        )

        self.speed_limit = speed_limit
        self.total_vehicles_seen = set()

    def _initialize_calibration(self, frame_width: int, frame_height: int):
        """Initialize perspective transformer once frame dimensions are known."""
        cfg = self.config_path or "calibration_config.yaml"
        self.transformer = PerspectiveTransformer.from_config(cfg, frame_width, frame_height)
        print(f"[INFO] Perspective calibration active (Target: {self.transformer.target_width}m x {self.transformer.target_length}m)")

    def process_frame(
        self, frame: np.ndarray, frame_index: int, fps: float
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Processes a single frame:
        1. Tracks vehicles using YOLOv8 + ByteTrack
        2. Warps bottom-center pixel to metric coordinates if inside ROI
        3. Estimates speed via video time (frame_index / fps)
        4. Draws polished HUD overlay
        """
        height, width = frame.shape[:2]
        if self.transformer is None:
            self._initialize_calibration(width, height)

        video_time = frame_index / max(fps, 1.0)

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_name,
            classes=self.vehicle_classes,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        active_ids = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().numpy()

                for tid, box, cls_id, conf in zip(track_ids, xyxy, classes, confs):
                    active_ids.append(tid)
                    self.total_vehicles_seen.add(tid)
                    x1, y1, x2, y2 = box

                    # Bottom-center ground contact point
                    cx = (x1 + x2) / 2.0
                    cy = float(y2)

                    # Only measure if vehicle ground point is within ROI
                    speed = None
                    if self.transformer.is_inside_roi(cx, cy):
                        mx, my = self.transformer.transform_point(cx, cy)
                        speed = self.estimator.update(tid, video_time, mx, my, int(cx), int(cy))
                    else:
                        speed = self.estimator.speeds.get(tid)

                    detections.append({
                        "id": tid,
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "cls": DEFAULT_VEHICLE_CLASSES.get(cls_id, "vehicle"),
                        "conf": float(conf),
                        "speed": speed,
                        "ground_pt": (int(cx), int(cy)),
                    })

        self.estimator.clean_stale(active_ids)

        annotated_frame = self.render_overlay(frame, detections, video_time, fps)
        return annotated_frame, detections

    def render_overlay(
        self, frame: np.ndarray, detections: List[Dict], video_time: float, fps: float
    ) -> np.ndarray:
        """Renders bounding boxes, trajectories, speed badges, and stats dashboard."""
        if self.transformer:
            self.transformer.draw_roi(frame, alpha=0.18)

        speeding_now = 0
        for det in detections:
            tid = det["id"]
            x1, y1, x2, y2 = det["box"]
            speed = det["speed"]
            cls_name = det["cls"]
            gx, gy = det["ground_pt"]

            is_speeding = speed is not None and speed > self.speed_limit
            if is_speeding:
                speeding_now += 1
                color = (40, 40, 240)    # Red for violation
                badge_bg = (30, 30, 200)
            elif speed is not None:
                color = (50, 210, 80)    # Green for compliant
                badge_bg = (30, 150, 50)
            else:
                color = (220, 180, 50)   # Yellow/Amber tracking
                badge_bg = (140, 110, 20)

            # Draw trajectory trail
            history = self.estimator.tracks_history[tid]
            if len(history) > 1:
                pts = [np.array([pt[3], pt[4]], dtype=np.int32) for pt in history]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), color, thickness, cv2.LINE_AA)

            # Draw bounding box & ground marker
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (gx, gy), 4, color, -1, cv2.LINE_AA)

            # Format label badge
            if speed is not None:
                speed_str = f"{speed:.1f} km/h"
                if is_speeding:
                    label = f"#{tid} {cls_name} | {speed_str} [ALERT]"
                else:
                    label = f"#{tid} {cls_name} | {speed_str}"
            else:
                label = f"#{tid} {cls_name} | Tracking..."

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.52
            thick = 1
            (lw, lh), _ = cv2.getTextSize(label, font, scale, thick)
            badge_y1 = max(0, y1 - lh - 10)
            badge_y2 = y1
            cv2.rectangle(frame, (x1, badge_y1), (x1 + lw + 12, badge_y2), badge_bg, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 6, badge_y2 - 5),
                font,
                scale,
                (255, 255, 255),
                thick,
                cv2.LINE_AA,
            )

        self._render_hud(frame, video_time, fps, speeding_now, len(detections))
        return frame

    def _render_hud(
        self, frame: np.ndarray, video_time: float, fps: float, speeding_now: int, active_count: int
    ):
        """Draws modern HUD banner at top of the video."""
        h, w = frame.shape[:2]
        banner_h = 52

        sub = frame[0:banner_h, 0:w]
        dark_rect = np.zeros(sub.shape, dtype=np.uint8)
        cv2.addWeighted(sub, 0.25, dark_rect, 0.75, 0, sub)
        frame[0:banner_h, 0:w] = sub
        cv2.line(frame, (0, banner_h), (w, banner_h), (0, 200, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        mins = int(video_time // 60)
        secs = video_time % 60
        time_str = f"Time: {mins:02d}:{secs:04.1f}"

        stats = [
            "SPEED DETECTOR v2.0",
            time_str,
            f"Limit: {self.speed_limit:.0f} km/h",
            f"Active: {active_count}",
            f"Total: {len(self.total_vehicles_seen)}",
            f"Violations: {len(self.estimator.speeding_history)}",
        ]

        x_offset = 16
        for stat in stats:
            color = (255, 255, 255)
            if "Violations" in stat and len(self.estimator.speeding_history) > 0:
                color = (60, 60, 255)
            elif "Limit" in stat:
                color = (0, 230, 255)
            elif "SPEED DETECTOR" in stat:
                color = (0, 255, 140)

            cv2.putText(frame, stat, (x_offset, 32), font, 0.52, color, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(stat, font, 0.52, 1)
            x_offset += tw + 20

        if speeding_now > 0:
            alert_text = f"WARNING: {speeding_now} VEHICLE(S) OVER SPEED LIMIT!"
            (aw, ah), _ = cv2.getTextSize(alert_text, font, 0.65, 2)
            alert_x = (w - aw) // 2
            cv2.rectangle(frame, (alert_x - 12, banner_h + 6), (alert_x + aw + 12, banner_h + 38), (0, 0, 200), -1)
            cv2.putText(
                frame,
                alert_text,
                (alert_x, banner_h + 29),
                font,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


def generate_synthetic_demo_video(output_path: str = "demo_clip.mp4", num_frames: int = 180, fps: int = 30):
    """
    Generates a realistic synthetic road video clip with vehicles driving at known ground-truth speeds.
    """
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Generating synthetic test video: {output_path} ({num_frames} frames @ {fps} fps)...")

    vehicles = [
        {"id": "Car A (Compliant)", "v_kmh": 42.0, "x_m": 2.2, "y_m": 36.0, "color": (190, 60, 40), "w_m": 1.8, "l_m": 4.2},
        {"id": "Car B (Speeding)", "v_kmh": 72.0, "x_m": 5.8, "y_m": 38.0, "color": (40, 80, 220), "w_m": 2.0, "l_m": 4.6},
    ]

    src = np.float32([
        [448, 324],   # TL
        [832, 324],   # TR
        [1152, 684],  # BR
        [128, 684],   # BL
    ])
    dst = np.float32([
        [0.0, 0.0],
        [8.0, 0.0],
        [8.0, 40.0],
        [0.0, 40.0],
    ])
    inv_matrix = cv2.getPerspectiveTransform(dst, src)

    def meter_to_pixel(xm: float, ym: float):
        pts = np.array([[[xm, ym]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, inv_matrix)
        return int(warped[0][0][0]), int(warped[0][0][1])

    for frame_idx in range(num_frames):
        t = frame_idx / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Grass background
        frame[:] = (45, 90, 45)
        # Road asphalt polygon
        road_poly = src.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [road_poly], (50, 50, 55))

        # Lane dividers
        for dash_y in np.linspace(2, 38, 12):
            p1 = meter_to_pixel(4.0, dash_y)
            p2 = meter_to_pixel(4.0, dash_y + 1.8)
            cv2.line(frame, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

        # Road boundaries
        p_tl, p_tr, p_br, p_bl = [tuple(pt) for pt in src.astype(int)]
        cv2.line(frame, p_tl, p_bl, (0, 220, 255), 3, cv2.LINE_AA)
        cv2.line(frame, p_tr, p_br, (0, 220, 255), 3, cv2.LINE_AA)

        for v in vehicles:
            v_ms = v["v_kmh"] / 3.6
            curr_y_m = v["y_m"] - v_ms * t
            if curr_y_m < -2.0:
                continue

            cx_m = v["x_m"]
            p_bottom = meter_to_pixel(cx_m, curr_y_m)
            p_top = meter_to_pixel(cx_m, curr_y_m + v["l_m"] * 0.7)
            p_left = meter_to_pixel(cx_m - v["w_m"] / 2.0, curr_y_m)
            p_right = meter_to_pixel(cx_m + v["w_m"] / 2.0, curr_y_m)

            box_w = max(24, abs(p_right[0] - p_left[0]))
            box_h = max(20, abs(p_bottom[1] - p_top[1]))
            x1 = p_bottom[0] - box_w // 2
            y1 = p_bottom[1] - box_h
            x2 = x1 + box_w
            y2 = p_bottom[1]

            cv2.rectangle(frame, (x1, y1), (x2, y2), v["color"], -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), 2)
            ws_y1 = y1 + int(box_h * 0.25)
            ws_y2 = y1 + int(box_h * 0.55)
            cv2.rectangle(frame, (x1 + 4, ws_y1), (x2 - 4, ws_y2), (180, 220, 240), -1)

            cv2.putText(
                frame,
                f"True: {v['v_kmh']:.0f} km/h",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        out.write(frame)

    out.release()
    print(f"[INFO] Synthetic demo video saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Detection using YOLOv8, ByteTrack & Perspective Calibration."
    )
    parser.add_argument("--video", type=str, default=None, help="Path to input video file or webcam index (0).")
    parser.add_argument("--limit", type=float, default=50.0, help="Speed limit in km/h (default: 50.0).")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video (MP4).")
    parser.add_argument("--config", type=str, default="calibration_config.yaml", help="Path to calibration YAML file.")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLOv8 model weights (default: yolov8n.pt).")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config (default: bytetrack.yaml).")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for detection.")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold for tracking.")
    parser.add_argument("--window", type=float, default=0.75, help="Temporal speed calculation window in seconds.")
    parser.add_argument("--max-speed", type=float, default=220.0, help="Maximum plausible speed for outlier rejection.")
    parser.add_argument("--view", dest="view", action="store_true", default=True, help="Display video window.")
    parser.add_argument("--no-view", dest="view", action="store_false", help="Run headlessly without GUI display.")
    parser.add_argument("--generate-demo", action="store_true", help="Generate a synthetic test clip with known speeds.")

    args = parser.parse_args()

    if args.generate_demo:
        generate_synthetic_demo_video("demo_clip.mp4")
        if args.video is None:
            args.video = "demo_clip.mp4"

    if args.video is None:
        # Check if sample_traffic.mp4 exists, else generate demo
        if Path("sample_traffic.mp4").exists():
            args.video = "sample_traffic.mp4"
        else:
            print("[INFO] No input video specified. Creating synthetic demo clip for immediate testing...")
            args.video = generate_synthetic_demo_video("demo_clip.mp4")

    if args.video.isdigit():
        video_src = int(args.video)
    else:
        video_src = args.video

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_src}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video opened: {video_src} ({width}x{height} @ {fps:.1f} fps, {total_frames} frames)")

    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"[INFO] Saving output video to: {args.output}")

    detector = VehicleSpeedDetector(
        weights=args.weights,
        speed_limit=args.limit,
        config_path=args.config,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        window_seconds=args.window,
        max_speed=args.max_speed,
    )

    frame_idx = 0
    start_wall_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, detections = detector.process_frame(frame, frame_idx, fps)

            if writer is not None:
                writer.write(processed_frame)

            if args.view:
                cv2.imshow("Vehicle Speed Detection - YOLOv8 & ByteTrack", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("\n[INFO] User requested termination.")
                    break

            frame_idx += 1
            if frame_idx % 30 == 0 and total_frames > 0:
                elapsed = time.time() - start_wall_time
                proc_fps = frame_idx / max(elapsed, 0.001)
                print(f"[PROGRESS] Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | Speed: {proc_fps:.1f} FPS")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.view:
            cv2.destroyAllWindows()

    total_time = time.time() - start_wall_time
    avg_fps = frame_idx / max(total_time, 0.001)
    print("\n" + "=" * 50)
    print("           DETECTION & SPEED SUMMARY")
    print("=" * 50)
    print(f"Total Frames Processed : {frame_idx}")
    print(f"Average Processing FPS : {avg_fps:.1f}")
    print(f"Total Unique Vehicles  : {len(detector.total_vehicles_seen)}")
    print(f"Speeding Violations    : {len(detector.estimator.speeding_history)}")
    if detector.estimator.speeding_history:
        print("Violation Details:")
        for tid, max_spd in detector.estimator.speeding_history.items():
            print(f"  - Vehicle #{tid}: Peak Speed = {max_spd:.1f} km/h (Limit: {args.limit} km/h)")
    if args.output:
        print(f"Output Video Saved to  : {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()

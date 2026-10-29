"""
Interactive Camera Perspective Calibration Tool for Vehicle Speed Detection
Author: Bharadwaj Patil

Allows users to click 4 road points on the first frame of a video,
specify real-world road width and length in metres, and export a
properly formatted calibration YAML file with reference resolution.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

POINT_LABELS = [
    "1. Top-Left (TL)",
    "2. Top-Right (TR)",
    "3. Bottom-Right (BR)",
    "4. Bottom-Left (BL)",
]

POINT_COLORS = [
    (0, 255, 255),  # Yellow
    (255, 255, 0),  # Cyan
    (0, 165, 255),  # Orange
    (0, 255, 0),    # Green
]


class Calibrator:
    def __init__(self, frame: np.ndarray, width_m: float, length_m: float, out_path: str):
        self.orig_frame = frame
        self.height, self.width = frame.shape[:2]
        self.width_m = float(width_m)
        self.length_m = float(length_m)
        self.out_path = Path(out_path)
        self.points: List[Tuple[int, int]] = []
        self.saved = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((int(x), int(y)))
                print(f"[CLICK] Point {len(self.points)}/4 {POINT_LABELS[len(self.points)-1]}: ({x}, {y})")

    def draw(self) -> np.ndarray:
        display = self.orig_frame.copy()

        # Draw already clicked points
        for idx, (px, py) in enumerate(self.points):
            color = POINT_COLORS[idx]
            cv2.circle(display, (px, py), 6, color, -1, cv2.LINE_AA)
            cv2.circle(display, (px, py), 8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(
                display,
                f"{POINT_LABELS[idx]}: ({px},{py})",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Draw connecting polygon if 4 points are selected
        if len(self.points) == 4:
            pts_arr = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts_arr], (0, 200, 255))
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.polylines(display, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Draw Top HUD instructions banner
        hud = display[0:60, 0:self.width]
        dark = np.zeros(hud.shape, dtype=np.uint8)
        cv2.addWeighted(hud, 0.25, dark, 0.75, 0, hud)
        display[0:60, 0:self.width] = hud
        cv2.line(display, (0, 60), (self.width, 60), (0, 200, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(self.points) < 4:
            curr_target = POINT_LABELS[len(self.points)]
            msg = f"Click point {len(self.points)+1}/4: {curr_target} | [R] Reset | [Q/ESC] Quit"
            color = (0, 255, 255)
        else:
            msg = f"4 Points Selected ({self.width_m}m x {self.length_m}m) | [ENTER/S] Save YAML | [R] Reset | [Q/ESC] Quit"
            color = (0, 255, 0)

        cv2.putText(display, msg, (16, 38), font, 0.65, color, 2, cv2.LINE_AA)
        return display

    def save_yaml(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        config_data = {
            "calibration": {
                "reference_resolution": [int(self.width), int(self.height)],
                "source_points": [[int(pt[0]), int(pt[1])] for pt in self.points],
                "target_dimensions": {
                    "width": float(self.width_m),
                    "length": float(self.length_m),
                },
            },
            "speed_estimation": {
                "speed_limit_kmh": 50.0,
                "max_plausible_kmh": 220.0,
                "window_seconds": 0.75,
                "min_displacement_meters": 0.8,
                "ema_smoothing_factor": 0.7,
            },
            "model": {
                "weights": "yolov8n.pt",
                "tracker": "bytetrack.yaml",
                "confidence": 0.35,
                "classes": [2, 3, 5, 7],
            },
        }

        with open(self.out_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

        print("\n" + "=" * 60)
        print("          CALIBRATION CONFIGURATION SAVED")
        print("=" * 60)
        print(f"Output File          : {self.out_path}")
        print(f"Reference Resolution : {self.width}x{self.height}")
        print(f"Source Points (px)   : {[[p[0], p[1]] for p in self.points]}")
        print(f"Target Dimensions (m): Width={self.width_m}m, Length={self.length_m}m")
        print("=" * 60)
        self.saved = True

    def run(self) -> bool:
        window_name = "Perspective Calibration - Click 4 Points [TL, TR, BR, BL]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse)

        print("=" * 65)
        print("   INTERACTIVE ROAD CALIBRATION TOOL")
        print("=" * 65)
        print("Instructions:")
        print("  1. Click 4 points on the road surface in order:")
        print("     - Point 1: Top-Left (TL)")
        print("     - Point 2: Top-Right (TR)")
        print("     - Point 3: Bottom-Right (BR)")
        print("     - Point 4: Bottom-Left (BL)")
        print("  2. Press 'r' or 'R' at any time to RESET points.")
        print("  3. Press 's', 'S', or ENTER to SAVE the calibration YAML.")
        print("  4. Press 'q', 'ESC' to exit without saving.")
        print("=" * 65)

        while True:
            display = self.draw()
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                print("[INFO] Calibration cancelled by user.")
                break
            elif key in (ord("r"), ord("R")):
                self.points.clear()
                print("[INFO] Points reset.")
            elif key in (ord("s"), ord("S"), 13):  # 's' or Enter
                if len(self.points) == 4:
                    self.save_yaml()
                    break
                else:
                    print(f"[WARNING] Please select all 4 points before saving (currently {len(self.points)}/4).")

        cv2.destroyAllWindows()
        return self.saved


def main():
    parser = argparse.ArgumentParser(
        description="Interactive road perspective calibration tool for vehicle speed detection."
    )
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--width", type=float, required=True, help="Real-world road width in metres across lane/road.")
    parser.add_argument("--length", type=float, required=True, help="Real-world length in metres along the road.")
    parser.add_argument(
        "--out",
        default="config/calibration_config.yaml",
        help="Output YAML configuration path (default: config/calibration_config.yaml).",
    )
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="Optional manual coordinates 'x1,y1 x2,y2 x3,y3 x4,y4' for non-interactive/headless environments.",
    )

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video}")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[ERROR] Failed to read first frame from {args.video}")
        sys.exit(1)

    calibrator = Calibrator(frame, width_m=args.width, length_m=args.length, out_path=args.out)

    if args.points:
        # Parse manual coordinates "x1,y1 x2,y2 x3,y3 x4,y4"
        raw_pts = args.points.strip().split()
        if len(raw_pts) != 4:
            print("[ERROR] Exactly 4 coordinate pairs required for --points (e.g. 'x1,y1 x2,y2 x3,y3 x4,y4').")
            sys.exit(1)
        parsed = []
        for pair in raw_pts:
            coords = pair.split(",")
            if len(coords) != 2:
                print(f"[ERROR] Invalid coordinate pair '{pair}'. Must be 'x,y'.")
                sys.exit(1)
            parsed.append((int(float(coords[0])), int(float(coords[1]))))
        calibrator.points = parsed
        calibrator.save_yaml()
        sys.exit(0)

    success = calibrator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

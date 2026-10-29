"""
Demo GIF Generator for Vehicle Speed Detection
Author: Bharadwaj Patil

Generates an annotated demonstration GIF from video footage for the README.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from speed import (
    VehicleSpeedDetector,
    generate_synthetic_demo_video,
)


def create_demo_gif(
    video_path: str = "demo_clip.mp4",
    config_path: str = "config/calibration_config.yaml",
    output_gif_path: str = "assets/demo.gif",
    start_time_s: float = 0.0,
    duration_s: float = 12.0,
    target_width: int = 720,
    target_fps: int = 12,
    speed_limit: float = 50.0,
):
    out_file = Path(output_gif_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not Path(video_path).exists():
        if video_path == "demo_clip.mp4":
            generate_synthetic_demo_video(video_path, num_frames=180, fps=30)
        else:
            raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time_s * in_fps)
    max_frames_to_record = int(duration_s * target_fps)
    step = max(1, int(round(in_fps / target_fps)))

    detector = VehicleSpeedDetector(
        weights="yolov8n.pt",
        speed_limit=speed_limit,
        config_path=config_path,
        tracker="bytetrack.yaml",
        conf=0.3,
        window_seconds=0.75,
    )

    frames = []
    frame_idx = 0
    speeding_frames_found = 0

    print(f"[INFO] Generating demo GIF from: {video_path}")
    print(f"[INFO] Time window: {start_time_s:.1f}s to {start_time_s + duration_s:.1f}s ({max_frames_to_record} GIF frames @ {target_fps} fps, width={target_width}px)")

    while cap.isOpened() and len(frames) < max_frames_to_record:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames starting from start_frame
        if frame_idx >= start_frame:
            processed_frame, detections = detector.process_frame(frame, frame_idx, in_fps)

            # Check for speeding alerts
            if any(d.get("speed") is not None and d["speed"] > speed_limit for d in detections):
                speeding_frames_found += 1

            if (frame_idx - start_frame) % step == 0:
                h, w = processed_frame.shape[:2]
                scale = target_width / w
                target_h = int(h * scale)
                resized = cv2.resize(processed_frame, (target_width, target_h), interpolation=cv2.INTER_AREA)

                # Convert BGR to RGB for PIL Image
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame))

        frame_idx += 1

    cap.release()

    if not frames:
        print("[ERROR] No frames captured for GIF generation.")
        return False

    print(f"[INFO] Captured {len(frames)} frames (with {speeding_frames_found} frames showing speeding vehicles).")
    print(f"[INFO] Quantizing and saving to {output_gif_path}...")

    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / target_fps),
        loop=0,
        optimize=True,
    )

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Demo GIF saved: {output_gif_path} ({size_mb:.2f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate demo GIF from video footage for README.")
    parser.add_argument("--video", default="demo_clip.mp4", help="Path to input video file.")
    parser.add_argument("--config", default="config/calibration_config.yaml", help="Path to calibration YAML file.")
    parser.add_argument("--output", default="assets/demo.gif", help="Output GIF path (default: assets/demo.gif).")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default: 0.0).")
    parser.add_argument("--duration", type=float, default=12.0, help="Duration in seconds (default: 12.0).")
    parser.add_argument("--width", type=int, default=720, help="Output GIF width in pixels (default: 720).")
    parser.add_argument("--fps", type=int, default=12, help="Output GIF frame rate (default: 12).")
    parser.add_argument("--limit", type=float, default=50.0, help="Speed limit in km/h (default: 50.0).")

    args = parser.parse_args()

    create_demo_gif(
        video_path=args.video,
        config_path=args.config,
        output_gif_path=args.output,
        start_time_s=args.start,
        duration_s=args.duration,
        target_width=args.width,
        target_fps=args.fps,
        speed_limit=args.limit,
    )


if __name__ == "__main__":
    main()

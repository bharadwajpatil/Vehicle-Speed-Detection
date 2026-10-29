"""
Demo GIF Generator for Vehicle Speed Detection
Author: Bharadwaj Patil

Generates an annotated sample GIF for the README demonstration.
"""

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
    output_gif_path: str = "assets/demo.gif",
    duration_frames: int = 150,
    target_width: int = 640,
    fps: int = 15,
):
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not Path(video_path).exists():
        generate_synthetic_demo_video(video_path, num_frames=180, fps=30)

    cap = cv2.VideoCapture(video_path)
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = VehicleSpeedDetector(
        weights="yolov8n.pt",
        speed_limit=50.0,
        config_path="config/calibration_config.yaml",
        tracker="bytetrack.yaml",
        conf=0.3,
        window_seconds=0.75,
    )

    frames = []
    frame_idx = 0
    step = max(1, int(in_fps / fps))

    print(f"[INFO] Generating demo GIF: {output_gif_path} (sampling every {step} frames)...")

    while cap.isOpened() and len(frames) < duration_frames:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, _ = detector.process_frame(frame, frame_idx, in_fps)

        if frame_idx % step == 0:
            h, w = processed_frame.shape[:2]
            scale = target_width / w
            target_h = int(h * scale)
            resized = cv2.resize(processed_frame, (target_width, target_h), interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))

        frame_idx += 1

    cap.release()

    if frames:
        print(f"[INFO] Saving {len(frames)} frames into {output_gif_path}...")
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        print(f"[SUCCESS] Demo GIF successfully created at {output_gif_path}")


if __name__ == "__main__":
    create_demo_gif()

import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import filedialog
import torch
import math
from collections import defaultdict


class VehicleSpeedDetector:
    def __init__(self, speed_limit=50.0, distance_calibration=10.0, fps=30):
        """
        Initialize the speed detector with parameters
        Args:
            speed_limit: Speed limit in km/h
            distance_calibration: Calibration factor for pixel to meter conversion
            fps: Frame rate of the video
        """
        # Load YOLOv5 model for vehicle detection
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Filter for vehicle classes (car, truck, bus, motorcycle)
        self.vehicle_classes = [2, 3, 5, 7]

        # Parameters
        self.speed_limit = speed_limit          # km/h
        self.distance_calibration = distance_calibration  # meters per 100 pixels
        self.fps = fps

        # Tracking variables
        self.vehicle_trackers = {}
        self.vehicle_speeds = {}
        self.next_vehicle_id = 0

        # Initialize centroid tracker
        self.disappeared = {}
        self.max_disappeared = 50  # frames

    def _calculate_centroid(self, box):
        """Calculate centroid of bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _calculate_speed(self, distance_pixels, time_diff):
        """
        Calculate speed in km/h
        Args:
            distance_pixels: Distance traveled in pixels
            time_diff: Time difference in seconds
        """
        # Convert pixels to meters using calibration factor
        distance_meters = distance_pixels * (self.distance_calibration / 100.0)
        # Calculate speed in m/s
        speed_ms = distance_meters / time_diff
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        return speed_kmh

    def _assign_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assign detections to existing trackers based on IOU
        """
        if len(trackers) == 0:
            return [], [], detections

        if len(detections) == 0:
            return [], trackers, []

        # Calculate IOU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d_idx, detection in enumerate(detections):
            for t_idx, tracker_id in enumerate(trackers):
                tracker = self.vehicle_trackers[tracker_id]
                iou_matrix[d_idx, t_idx] = self._calculate_iou(detection, tracker['box'])

        # Find matches
        matches = []
        unmatched_detections = []
        unmatched_trackers = list(trackers)

        # Find best matches
        while iou_matrix.size > 0 and np.max(iou_matrix) >= iou_threshold:
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            d_idx, t_idx = max_idx
            matches.append((detections[d_idx], trackers[t_idx]))

            # Remove matched detection and tracker from consideration
            iou_matrix = np.delete(iou_matrix, d_idx, 0)
            iou_matrix = np.delete(iou_matrix, t_idx, 1)
            unmatched_trackers.remove(trackers[t_idx])

            # Update remaining indices
            detections = np.delete(detections, d_idx, 0)
            trackers = np.delete(trackers, t_idx, 0)

        unmatched_detections = detections
        return matches, unmatched_trackers, unmatched_detections

    def _calculate_iou(self, box1, box2):
        """Calculate intersection over union for two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def process_frame(self, frame):
        """
        Process a video frame to detect vehicles and calculate speed
        Args:
            frame: Input video frame
        Returns:
            Processed frame with speed annotations
            List of vehicle IDs exceeding speed limit
        """
        height, width = frame.shape[:2]
        results = self.model(frame)

        # Get detections (filter for vehicles only)
        predictions = results.pandas().xyxy[0]
        current_trackers = list(self.vehicle_trackers.keys())
        current_detections = []

        for _, row in predictions.iterrows():
            if int(row['class']) in self.vehicle_classes and row['confidence'] > 0.5:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                current_detections.append((x1, y1, x2, y2))

        # Match detections with existing trackers
        matches, unmatched_trackers, unmatched_detections = self._assign_detections_to_trackers(
            np.array(current_detections), np.array(current_trackers)
        )

        # Update matched trackers
        current_time = time.time()
        for detection, tracker_id in matches:
            tracker = self.vehicle_trackers[tracker_id]
            prev_centroid = tracker['centroid']
            prev_time = tracker['timestamp']

            # Update tracker
            tracker['box'] = detection
            tracker['centroid'] = self._calculate_centroid(detection)
            tracker['timestamp'] = current_time

            # Reset disappeared counter
            self.disappeared[tracker_id] = 0

            # Calculate speed if enough time has passed
            time_diff = current_time - prev_time
            if time_diff > 0.1:  # Avoid division by very small numbers
                distance = self._calculate_distance(prev_centroid, tracker['centroid'])
                speed = self._calculate_speed(distance, time_diff)

                # Update speed using moving average
                if 'speed' in tracker:
                    tracker['speed'] = 0.7 * tracker['speed'] + 0.3 * speed
                else:
                    tracker['speed'] = speed

                self.vehicle_speeds[tracker_id] = tracker['speed']

        # Handle unmatched trackers (disappeared)
        for tracker_id in unmatched_trackers:
            self.disappeared[tracker_id] += 1

            # Remove tracker if disappeared for too long
            if self.disappeared[tracker_id] > self.max_disappeared:
                del self.vehicle_trackers[tracker_id]
                del self.disappeared[tracker_id]
                if tracker_id in self.vehicle_speeds:
                    del self.vehicle_speeds[tracker_id]

        # Create new trackers for unmatched detections
        for detection in unmatched_detections:
            new_tracker = {
                'box': detection,
                'centroid': self._calculate_centroid(detection),
                'timestamp': current_time
            }
            self.vehicle_trackers[self.next_vehicle_id] = new_tracker
            self.disappeared[self.next_vehicle_id] = 0
            self.next_vehicle_id += 1

        # Draw bounding boxes and speed information
        speeding_vehicles = []
        for vehicle_id, tracker in self.vehicle_trackers.items():
            x1, y1, x2, y2 = tracker['box']

            # Determine color based on speed
            color = (0, 255, 0)  # Green by default
            if 'speed' in tracker:
                speed = tracker['speed']
                if speed > self.speed_limit:
                    color = (0, 0, 255)  # Red for speeding
                    speeding_vehicles.append(vehicle_id)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Show speed
                speed_text = f"ID: {vehicle_id}, {speed:.1f} km/h"
                cv2.putText(frame, speed_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display speed limit on frame
        cv2.putText(frame, f"Speed Limit: {self.speed_limit} km/h", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return frame, speeding_vehicles


def calibrate_distance(frame_width, known_width_meters, fov_degrees=70):
    """
    Calculate calibration factor for pixel to meter conversion
    Args:
        frame_width: Width of the video frame in pixels
        known_width_meters: Known width of the road in meters
        fov_degrees: Camera field of view in degrees
    Returns:
        Calibration factor (meters per 100 pixels)
    """
    return (known_width_meters / frame_width) * 100


def select_video_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    root.destroy()
    return file_path


def main():
    # Prompt user to select video file
    video_path = select_video_file()
    if not video_path:
        print("No video selected. Exiting.")
        return

    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Calibrate distance (assume road width is 10 meters)
    calibration_factor = calibrate_distance(frame_width, known_width_meters=10)

    # Initialize speed detector
    detector = VehicleSpeedDetector(
        speed_limit=50.0,  # 50 km/h limit
        distance_calibration=calibration_factor,
        fps=fps
    )

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, speeding_vehicles = detector.process_frame(frame)

        # Show alert for speeding vehicles
        if speeding_vehicles:
            cv2.putText(processed_frame, "ALERT: Speeding Detected!",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print(f"Speeding vehicles detected: {speeding_vehicles}")

        # Display frame
        cv2.imshow("Vehicle Speed Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

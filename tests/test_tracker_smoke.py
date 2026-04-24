import io
import json
import unittest
from pathlib import Path

import numpy as np

from src.tracker import (
    TrackerState,
    generate_drone_commands,
    load_model_and_video,
    resolve_project_path,
    write_frame_data,
)


class FakeVideoWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)


class TrackerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "tracking": {
                "target_imgsz": 640,
                "tolerance_pixels": 20,
                "target_bbox_area_percentage": 0.15,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.7,
                "target_class_id": 0,
                "tracker_type": "bytetrack",
            }
        }

    def test_generate_drone_commands_centered_target(self):
        commands = generate_drone_commands(
            object_center=(960, 540),
            object_bbox_size=(560, 555),
            original_frame_width=1920,
            original_frame_height=1080,
            config=self.config,
        )

        self.assertEqual(
            commands,
            {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
        )

    def test_generate_drone_commands_offset_and_small_target(self):
        commands = generate_drone_commands(
            object_center=(800, 700),
            object_bbox_size=(100, 100),
            original_frame_width=1920,
            original_frame_height=1080,
            config=self.config,
        )

        self.assertEqual(commands["horizontal"], "LEFT")
        self.assertEqual(commands["vertical"], "DOWN")
        self.assertEqual(commands["distance"], "FORWARD")

    def test_load_model_and_video_raises_clear_error_for_missing_model(self):
        missing_model = Path("runs/missing-model.pt")
        existing_video = Path("resources/snowboard_quick_test.mp4")

        with self.assertRaises(FileNotFoundError) as ctx:
            load_model_and_video(missing_model, existing_video, self.config)

        self.assertIn(str(resolve_project_path(missing_model)), str(ctx.exception))

    def test_write_frame_data_logs_black_frame_when_target_was_never_seen(self):
        state = TrackerState()
        writer = FakeVideoWriter()
        log_file = io.StringIO()
        frame_data = {
            "frame": 1,
            "status": "Поиск",
            "commands": {"horizontal": "NONE", "vertical": "NONE", "distance": "NONE"},
            "confidence": None,
            "bbox_size_ratio": None,
            "tracked_id": None,
        }
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        write_frame_data(
            frame=frame,
            current_frame_data=frame_data,
            state=state,
            out=writer,
            log_file=log_file,
            config=self.config,
            frame_count=1,
            frame_width=1920,
            frame_height=1080,
        )

        self.assertEqual(len(writer.frames), 1)
        self.assertEqual(writer.frames[0].shape, (640, 640, 3))
        self.assertEqual(json.loads(log_file.getvalue()), frame_data)


if __name__ == "__main__":
    unittest.main()

"""Production entry point for the browser-camera realtime service."""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import realtime_server as runtime


def make_args():
    return SimpleNamespace(
        config=str(runtime.ROOT / "config_online.yaml"),
        source="web",
        camera_calibration=None,
        marker_layout=None,
        yolo_weights=None,
        phone_marker_map=str(runtime.ROOT / "phone_marker_map.json"),
        phone_config=str(runtime.ROOT / "phone_pose_config.json"),
        device=os.getenv("YOLO_DEVICE", "cpu"),
        yolo_conf=float(os.getenv("YOLO_CONF", "0.10")),
        yolo_imgsz=int(os.getenv("YOLO_IMGSZ", "640")),
        yolo_every=int(os.getenv("YOLO_EVERY", "3")),
        max_reprojection_error=float(os.getenv("MAX_REPROJECTION_ERROR", "8.0")),
        max_pose_jump_deg=float(os.getenv("MAX_POSE_JUMP_DEG", "45.0")),
        max_rpm=float(os.getenv("MAX_RPM", "3000")),
        output_dir=os.getenv("OUTPUT_DIR", "realtime_results"),
        save_video=None,
        output_fps=10.0,
        loop=False,
    )


def start_processing_worker():
    worker = threading.Thread(
        target=runtime.process_stream_safe,
        args=(make_args(),),
        daemon=True,
        name="pose-processing",
    )
    worker.start()
    return worker


app = runtime.app
processing_worker = start_processing_worker()

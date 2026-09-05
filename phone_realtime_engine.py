"""Real-time adapter for the v2.0.0 phone pose recognition core.

The offline recognizer can use future interpolation. This adapter deliberately
uses only current and previous frames, so live output remains causal.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import phone_pose_core as core


class PhoneRealtimeRecognizer:
    def __init__(self, width: int, height: int, marker_map_path: str, config_path: str):
        self.config = core.load_json(Path(config_path))
        self.marker_map = core.load_marker_map(Path(marker_map_path))
        dictionary_name = self.config["aruco_dictionary"]
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        self.detectors = core.build_detectors(dictionary)
        self.matrix, self.distortion = core.scaled_camera(self.config, width, height)
        self.max_rmse = float(self.config["max_reprojection_error_px"])
        self.max_age = int(self.config["max_track_age_frames"])
        self.max_step = float(self.config["max_rotation_step_deg"])
        self.fb_threshold = float(self.config["optical_flow_fb_threshold_px"])

        self.previous_gray = None
        self.previous_rotation = None
        self.previous_translation = None
        self.previous_time = None
        self.track_image = None
        self.track_object = None
        self.track_age = 0
        self.last_marker_ids = []

    def process(self, image: np.ndarray, time_s: float) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        groups, corners, ids = core.exhaustive_groups(
            gray, self.detectors, self.marker_map
        )
        pose = core.solve_pose(
            groups,
            self.marker_map,
            self.matrix,
            self.distortion,
            self.previous_rotation,
            self.previous_translation,
            self.max_step,
        )
        source = "direct" if pose is not None else "none"

        if groups and pose is not None and pose[2] <= self.max_rmse:
            self.track_image, self.track_object = core.dense_correspondences(
                groups, self.marker_map
            )
            self.track_age = 0
            self.last_marker_ids = [marker_id for marker_id, _ in groups]
        elif (
            self.previous_gray is not None
            and self.track_image is not None
            and self.track_age < self.max_age
        ):
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                self.track_image.reshape(-1, 1, 2),
                None,
            )
            if nxt is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    self.previous_gray,
                    nxt,
                    None,
                )
                fb_error = np.linalg.norm(
                    back.reshape(-1, 2) - self.track_image, axis=1
                )
                keep = (
                    status.ravel().astype(bool)
                    & back_status.ravel().astype(bool)
                    & (fb_error < self.fb_threshold)
                )
                self.track_image = nxt.reshape(-1, 2)[keep]
                self.track_object = self.track_object[keep]
                self.track_age += 1
                pose = core.solve_tracked(
                    self.track_object,
                    self.track_image,
                    self.matrix,
                    self.distortion,
                    self.previous_rotation,
                    self.previous_translation,
                    self.max_step,
                )
                source = "klt" if pose is not None else "none"
                if len(self.track_image) < 4:
                    self.track_image = self.track_object = None

        valid = pose is not None and pose[2] <= self.max_rmse
        result = {
            "corners": corners,
            "ids": ids,
            "success": bool(valid),
            "status": "valid" if valid else ("pnp_failed" if groups else "no_marker"),
            "source": source,
            "marker_count": len(groups) if groups else len(self.last_marker_ids),
            "used_marker_ids": list(self.last_marker_ids),
            "reprojection_error_px": np.nan,
            "rotation": None,
            "translation": None,
            "rvec": None,
            "tvec": None,
            "omega": np.full(3, np.nan),
            "rpm": np.nan,
            "quat": np.full(4, np.nan),
            "euler": np.full(3, np.nan),
        }
        if valid:
            rotation, translation, rmse = pose
            result["reprojection_error_px"] = rmse
            result["rotation"] = rotation
            result["translation"] = translation
            result["rvec"] = cv2.Rodrigues(rotation)[0]
            result["tvec"] = np.asarray(translation, np.float64).reshape(3, 1)
            result["quat"] = Rotation.from_matrix(rotation).as_quat()
            result["euler"] = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
            if self.previous_rotation is not None and self.previous_time is not None:
                dt = float(time_s - self.previous_time)
                if dt > 1e-6:
                    relative = Rotation.from_matrix(self.previous_rotation.T @ rotation)
                    result["omega"] = relative.as_rotvec() / dt
                    result["rpm"] = float(np.linalg.norm(result["omega"]) * 60.0 / (2.0 * np.pi))
            self.previous_rotation = rotation
            self.previous_translation = translation
            self.previous_time = time_s

        self.previous_gray = gray
        return result

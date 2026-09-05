from __future__ import annotations

import configparser
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> dict:
    # utf-8-sig accepts both ordinary UTF-8 and Windows files with a BOM.
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_export_fps(folder: str | Path) -> float | None:
    exp_path = Path(folder) / "export.exp"
    if not exp_path.exists():
        return None
    parser = configparser.ConfigParser()
    parser.read(exp_path, encoding="utf-8")
    if parser.has_option("CAMERA", "Framerate"):
        return parser.getfloat("CAMERA", "Framerate")
    return None


def find_experiments(data_root: str | Path, glob_pattern: str):
    root = Path(data_root)
    experiments = []
    for folder in sorted(root.glob(glob_pattern)):
        if not folder.is_dir():
            continue
        videos = sorted(folder.glob("*.avi")) + sorted(folder.glob("*.mp4"))
        if not videos:
            continue
        experiments.append({"name": folder.name, "folder": folder, "video": videos[0]})
    return experiments


def load_marker_layout(path: str | Path):
    data = load_json(path)
    layout = {}
    for marker_id, item in data["markers"].items():
        corners = np.asarray(item["corners_3d"], dtype=np.float32)
        if corners.shape != (4, 3):
            raise ValueError(f"Marker {marker_id} must have shape (4, 3)")
        layout[int(marker_id)] = corners
    return data.get("dictionary", "DICT_4X4_50"), layout


def load_calibration(path: str | Path | None, width: int, height: int):
    if path:
        data = load_json(path)
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(data.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
        return camera_matrix, dist_coeffs
    focal = 1.2 * max(width, height)
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def make_aruco_detector(dictionary_name: str):
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def collect_pnp_points(corners, ids, layout, offset_xy=(0.0, 0.0)):
    if ids is None:
        return None, None, []

    obj_points = []
    img_points = []
    used_ids = []
    offset = np.asarray(offset_xy, dtype=np.float32).reshape(1, 2)
    for detected_corners, marker_id in zip(corners, ids.reshape(-1)):
        marker_id = int(marker_id)
        if marker_id not in layout:
            continue
        obj_points.append(layout[marker_id])
        image_corners = np.asarray(detected_corners, dtype=np.float32).reshape(4, 2) + offset
        img_points.append(image_corners)
        used_ids.append(marker_id)

    if not obj_points:
        return None, None, used_ids
    return np.vstack(obj_points), np.vstack(img_points), used_ids


def quat_wxyz_from_rmat(r: np.ndarray):
    tr = float(np.trace(r))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def euler_xyz_deg_from_rmat(r: np.ndarray):
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-8:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def angular_velocity_body(prev_r: np.ndarray | None, cur_r: np.ndarray, dt: float):
    if prev_r is None or dt <= 0.0:
        return np.full(3, np.nan), np.nan, np.nan

    rel_r = cur_r @ prev_r.T
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(rel_r)) - 1.0) / 2.0)))
    if angle < 1e-10:
        omega_cam = np.zeros(3, dtype=np.float64)
    else:
        axis = np.array(
            [rel_r[2, 1] - rel_r[1, 2], rel_r[0, 2] - rel_r[2, 0], rel_r[1, 0] - rel_r[0, 1]],
            dtype=np.float64,
        ) / (2.0 * math.sin(angle))
        omega_cam = axis * angle / dt

    omega_body = prev_r.T @ omega_cam
    omega_mag = float(np.linalg.norm(omega_body))
    rpm = omega_mag * 60.0 / (2.0 * math.pi)
    return omega_body, omega_mag, rpm


def normalize_quaternion_sign(df):
    quat_cols = ["quat_w", "quat_x", "quat_y", "quat_z"]
    prev = None
    for idx, row in df.iterrows():
        q = row[quat_cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(q)):
            continue
        if prev is not None and float(np.dot(prev, q)) < 0.0:
            df.loc[idx, quat_cols] = -q
            q = -q
        prev = q
    return df

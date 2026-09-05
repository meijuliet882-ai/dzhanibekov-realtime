"""Offline ArUco rigid-body pose extraction for phone slow-motion video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


ROTATION_COLUMNS = [f"R_CB_{i}{j}" for i in range(3) for j in range(3)]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_marker_map(path: Path) -> dict[int, np.ndarray]:
    data = load_json(path)
    markers = {
        int(marker_id): np.asarray(value["corners_body_m"], np.float64)
        for marker_id, value in data["markers"].items()
    }
    if not markers or any(corners.shape != (4, 3) for corners in markers.values()):
        raise ValueError("marker_map 的每个 corners_body_m 必须是 4×3 数组。")
    return markers


def scaled_camera(config: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    reference_width, reference_height = config["reference_image_size"]
    matrix = np.asarray(config["camera_matrix"], np.float64).copy()
    matrix[0, :] *= width / float(reference_width)
    matrix[1, :] *= height / float(reference_height)
    matrix[2, :] = (0.0, 0.0, 1.0)
    distortion = np.asarray(config["distortion_coefficients"], np.float64)
    return matrix, distortion


def marker_groups(ids, corners, marker_map: dict[int, np.ndarray]):
    if ids is None:
        return []
    return [
        (int(marker_id), np.asarray(corner, np.float64).reshape(4, 2))
        for marker_id, corner in zip(ids.reshape(-1), corners)
        if int(marker_id) in marker_map
    ]


def preprocessing_variants(gray: np.ndarray) -> list[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
    clahe_blurred = cv2.GaussianBlur(clahe, (0, 0), 2.0)
    clahe_sharpened = cv2.addWeighted(clahe, 2.0, clahe_blurred, -1.0, 0)
    return [gray, clahe, sharpened, clahe_sharpened]


def build_detectors(dictionary) -> list[cv2.aruco.ArucoDetector]:
    detectors = [cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())]
    for adaptive_constant, polygon_accuracy, minimum_perimeter, error_correction in (
        (7, 0.05, 0.015, 1.0),
        (5, 0.10, 0.010, 1.5),
        (3, 0.15, 0.005, 2.0),
    ):
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 53
        parameters.adaptiveThreshWinSizeStep = 4
        parameters.adaptiveThreshConstant = adaptive_constant
        parameters.minMarkerPerimeterRate = minimum_perimeter
        parameters.maxMarkerPerimeterRate = 5.0
        parameters.polygonalApproxAccuracyRate = polygon_accuracy
        parameters.minCornerDistanceRate = 0.01
        parameters.minDistanceToBorder = 1
        parameters.perspectiveRemovePixelPerCell = 8
        parameters.maxErroneousBitsInBorderRate = 0.5
        parameters.errorCorrectionRate = error_correction
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 50
        detectors.append(cv2.aruco.ArucoDetector(dictionary, parameters))
    return detectors


def exhaustive_groups(gray, detectors, marker_map):
    """Run complementary detectors; conservative results take precedence."""
    corners, ids, _ = detectors[0].detectMarkers(gray)
    conservative = marker_groups(ids, corners, marker_map)
    if conservative:
        clean_ids = np.asarray([[marker_id] for marker_id, _ in conservative], np.int32)
        clean_corners = [points.reshape(1, 4, 2).astype(np.float32) for _, points in conservative]
        return conservative, clean_corners, clean_ids
    best: dict[int, np.ndarray] = {}
    for variant in preprocessing_variants(gray):
        for detector in detectors[1:]:
            corners, ids, _ = detector.detectMarkers(variant)
            for marker_id, points in marker_groups(ids, corners, marker_map):
                if marker_id not in best:
                    best[marker_id] = points
    groups = sorted(best.items())
    ids = None if not groups else np.asarray([[marker_id] for marker_id, _ in groups], np.int32)
    corners = [points.reshape(1, 4, 2).astype(np.float32) for _, points in groups]
    return groups, corners, ids


def solve_pose(groups, marker_map, matrix, distortion, previous_rotation=None,
               previous_translation=None, max_step_deg=25.0):
    if not groups:
        return None
    object_points = np.concatenate([marker_map[marker_id] for marker_id, _ in groups])
    image_points = np.concatenate([points for _, points in groups])
    candidates = []
    if len(groups) == 1:
        body = marker_map[groups[0][0]]
        center = body.mean(axis=0)
        right = body[1] - body[0]
        up = body[0] - body[3]
        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)
        rotation_body_marker = np.column_stack([right, up, np.cross(right, up)])
        half = 0.5 * np.linalg.norm(body[1] - body[0])
        local = np.array([[-half, half, 0], [half, half, 0],
                          [half, -half, 0], [-half, -half, 0]], np.float64)
        result = cv2.solvePnPGeneric(
            local, image_points, matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if result[0]:
            for rvec_cm, tvec_cm in zip(result[1], result[2]):
                rotation_camera_marker = cv2.Rodrigues(rvec_cm)[0]
                rotation = rotation_camera_marker @ rotation_body_marker.T
                translation = np.asarray(tvec_cm).reshape(3) - rotation @ center
                candidates.append((rotation, translation))
    else:
        use_guess = previous_rotation is not None and previous_translation is not None
        rvec = None if not use_guess else cv2.Rodrigues(previous_rotation)[0]
        tvec = None if not use_guess else np.asarray(previous_translation).reshape(3, 1)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, matrix, distortion, rvec, tvec,
            use_guess, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if ok:
            candidates.append((cv2.Rodrigues(rvec)[0], np.asarray(tvec).reshape(3)))

    scored = []
    for rotation, translation in candidates:
        if translation[2] <= 0:
            continue
        projected, _ = cv2.projectPoints(
            object_points, cv2.Rodrigues(rotation)[0], translation, matrix, distortion
        )
        residual = projected.reshape(-1, 2) - image_points
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        step = 0.0 if previous_rotation is None else float(np.degrees(
            Rotation.from_matrix(previous_rotation.T @ rotation).magnitude()
        ))
        if previous_rotation is not None and step > max_step_deg:
            continue
        scored.append((rmse + 0.08 * step, rotation, translation, rmse))
    return None if not scored else min(scored, key=lambda item: item[0])[1:]


def dense_correspondences(groups, marker_map, grid_size=4):
    image_points, object_points = [], []
    for marker_id, image_corners in groups:
        for v in np.linspace(0.08, 0.92, grid_size):
            for u in np.linspace(0.08, 0.92, grid_size):
                weights = np.array([(1-u)*(1-v), u*(1-v), u*v, (1-u)*v])
                image_points.append(weights @ image_corners)
                object_points.append(weights @ marker_map[marker_id])
    return np.asarray(image_points, np.float32), np.asarray(object_points, np.float32)


def solve_tracked(object_points, image_points, matrix, distortion,
                  previous_rotation, previous_translation, max_step_deg):
    if (object_points is None or len(object_points) < 4
            or previous_rotation is None or previous_translation is None):
        return None
    rvec = cv2.Rodrigues(previous_rotation)[0]
    tvec = np.asarray(previous_translation, np.float64).reshape(3, 1)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float64), image_points.astype(np.float64),
            matrix, distortion, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error:
        return None
    if not ok:
        return None
    rotation = cv2.Rodrigues(rvec)[0]
    translation = np.asarray(tvec).reshape(3)
    if translation[2] <= 0:
        return None
    step = float(np.degrees(Rotation.from_matrix(previous_rotation.T @ rotation).magnitude()))
    if step > max_step_deg:
        return None
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)
    residual = projected.reshape(-1, 2) - image_points
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return rotation, translation, rmse


def longest_invalid_run(valid: list[bool]) -> int:
    longest = current = 0
    for item in valid:
        current = 0 if item else current + 1
        longest = max(longest, current)
    return longest


def complete_trajectory(table: pd.DataFrame, physical_fps: float) -> pd.DataFrame:
    """Fill every frame while preserving which rows are real visual anchors."""
    result = table.copy()
    anchor_rows = result.index[result["visual_valid"].eq(1)].to_numpy(int)
    if len(anchor_rows) == 0:
        raise RuntimeError("整段视频没有可用视觉锚点，无法补全姿态。")
    anchor_frames = result.loc[anchor_rows, "frame"].to_numpy(float)
    anchor_rotations = Rotation.from_quat(
        result.loc[anchor_rows, ["qx", "qy", "qz", "qw"]].to_numpy(float)
    )
    all_frames = result["frame"].to_numpy(float)
    inside = (all_frames >= anchor_frames[0]) & (all_frames <= anchor_frames[-1])
    completed_quaternions = np.empty((len(result), 4), float)
    if len(anchor_rows) >= 3:
        completed_quaternions[inside] = RotationSpline(
            anchor_frames, anchor_rotations
        )(all_frames[inside]).as_quat()
    elif len(anchor_rows) == 2:
        completed_quaternions[inside] = Slerp(
            anchor_frames, anchor_rotations
        )(all_frames[inside]).as_quat()
    else:
        completed_quaternions[:] = anchor_rotations.as_quat()[0]
    def boundary_rate(rotations: Rotation, frames: np.ndarray, at_start: bool) -> np.ndarray:
        count = min(8, len(frames))
        subset = np.arange(count) if at_start else np.arange(len(frames) - count, len(frames))
        rates = []
        for left, right in zip(subset[:-1], subset[1:]):
            gap = frames[right] - frames[left]
            rates.append((rotations[left].inv() * rotations[right]).as_rotvec() / gap)
        return np.median(np.asarray(rates), axis=0) if rates else np.zeros(3)

    before = all_frames < anchor_frames[0]
    after = all_frames > anchor_frames[-1]
    start_rate = boundary_rate(anchor_rotations, anchor_frames, True)
    end_rate = boundary_rate(anchor_rotations, anchor_frames, False)
    if before.any():
        delta = all_frames[before] - anchor_frames[0]
        completed_quaternions[before] = (
            anchor_rotations[0] * Rotation.from_rotvec(delta[:, None] * start_rate)
        ).as_quat()
    if after.any():
        delta = all_frames[after] - anchor_frames[-1]
        completed_quaternions[after] = (
            anchor_rotations[-1] * Rotation.from_rotvec(delta[:, None] * end_rate)
        ).as_quat()
    # Exact anchors are restored after interpolation so measured values never drift.
    completed_quaternions[anchor_rows] = anchor_rotations.as_quat()

    translations = result.loc[anchor_rows, ["tx_m", "ty_m", "tz_m"]].to_numpy(float)
    completed_translation = np.column_stack([
        np.interp(all_frames, anchor_frames, translations[:, axis]) for axis in range(3)
    ])
    regression_count = min(8, len(anchor_frames))
    if len(anchor_frames) >= 2:
        for axis in range(3):
            start_slope, start_intercept = np.polyfit(
                anchor_frames[:regression_count], translations[:regression_count, axis], 1
            )
            end_slope, end_intercept = np.polyfit(
                anchor_frames[-regression_count:], translations[-regression_count:, axis], 1
            )
            completed_translation[before, axis] = start_slope * all_frames[before] + start_intercept
            completed_translation[after, axis] = end_slope * all_frames[after] + end_intercept
    completed_rotations = Rotation.from_quat(completed_quaternions)
    result[["qx", "qy", "qz", "qw"]] = completed_quaternions
    result[["tx_m", "ty_m", "tz_m"]] = completed_translation
    matrices = completed_rotations.as_matrix().reshape(-1, 9)
    result[ROTATION_COLUMNS] = matrices
    result["pose_available"] = 1
    result["estimate_kind"] = "interpolated"
    result.loc[before | after, "estimate_kind"] = "extrapolated"
    result.loc[anchor_rows, "estimate_kind"] = result.loc[anchor_rows, "pose_source"].map(
        {"direct": "measured_direct", "klt": "measured_klt"}
    ).fillna("measured")

    nearest_distance = np.min(np.abs(all_frames[:, None] - anchor_frames[None, :]), axis=1)
    result["confidence"] = np.exp(-nearest_distance / 12.0) * 0.65
    result.loc[result["estimate_kind"].eq("extrapolated"), "confidence"] *= 0.35
    measured_confidence = np.exp(
        -result.loc[anchor_rows, "reprojection_error_px"].to_numpy(float) / 8.0
    )
    result.loc[anchor_rows, "confidence"] = np.clip(measured_confidence, 0.25, 1.0)

    omega = np.zeros((len(result), 3), float)
    if len(result) > 1:
        relative = completed_rotations[:-1].inv() * completed_rotations[1:]
        omega[1:] = relative.as_rotvec() * physical_fps
        omega[0] = omega[1]
    result[["omega_body_x_rad_s", "omega_body_y_rad_s", "omega_body_z_rad_s"]] = omega
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="从手机视频提取 ArUco 刚体视觉姿态。")
    parser.add_argument("video", type=Path)
    parser.add_argument("--marker-map", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--annotated-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    marker_map = load_marker_map(args.marker_map)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    container_fps = float(capture.get(cv2.CAP_PROP_FPS))
    matrix, distortion = scaled_camera(config, width, height)
    dictionary_name = config["aruco_dictionary"]
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"未知 ArUco 字典：{dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detectors = build_detectors(dictionary)
    max_rmse = float(config["max_reprojection_error_px"])
    max_age = int(config["max_track_age_frames"])
    max_step = float(config["max_rotation_step_deg"])
    fb_threshold = float(config["optical_flow_fb_threshold_px"])
    physical_fps = float(config["physical_fps"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.annotated_video:
        writer = cv2.VideoWriter(
            str(args.output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
            container_fps if container_fps > 0 else 30.0, (width, height)
        )

    rows = []
    previous_gray = previous_rotation = previous_translation = None
    track_image = track_object = None
    track_age = 0
    frame_number = 0
    while not args.max_frames or frame_number < args.max_frames:
        ok, image = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        groups, corners, ids = exhaustive_groups(gray, detectors, marker_map)
        pose = solve_pose(groups, marker_map, matrix, distortion,
                          previous_rotation, previous_translation, max_step)
        source = "direct" if pose is not None else "none"
        if groups and pose is not None and pose[2] <= max_rmse:
            track_image, track_object = dense_correspondences(groups, marker_map)
            track_age = 0
        elif previous_gray is not None and track_image is not None and track_age < max_age:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, track_image.reshape(-1, 1, 2), None)
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, nxt, None)
            fb_error = np.linalg.norm(back.reshape(-1, 2) - track_image, axis=1)
            keep = status.ravel().astype(bool) & back_status.ravel().astype(bool) & (fb_error < fb_threshold)
            track_image = nxt.reshape(-1, 2)[keep]
            track_object = track_object[keep]
            track_age += 1
            pose = solve_tracked(track_object, track_image, matrix, distortion,
                                 previous_rotation, previous_translation, max_step)
            source = "klt" if pose is not None else "none"
            if len(track_image) < 4:
                track_image = track_object = None

        valid = pose is not None and pose[2] <= max_rmse
        row = {"frame": frame_number, "time_s": frame_number / physical_fps,
               "visual_valid": int(valid), "pose_source": source,
               "marker_count": len(groups),
               "marker_ids": " ".join(str(marker_id) for marker_id, _ in groups),
               "reprojection_error_px": np.nan}
        if valid:
            rotation, translation, rmse = pose
            quaternion = Rotation.from_matrix(rotation).as_quat()
            row.update({"reprojection_error_px": rmse,
                        "tx_m": translation[0], "ty_m": translation[1], "tz_m": translation[2],
                        "qx": quaternion[0], "qy": quaternion[1], "qz": quaternion[2], "qw": quaternion[3]})
            row.update({ROTATION_COLUMNS[index]: rotation.flat[index] for index in range(9)})
            previous_rotation, previous_translation = rotation, translation
        rows.append(row)
        if writer is not None:
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
            color = (0, 200, 0) if valid else (0, 0, 255)
            cv2.putText(image, f"frame={frame_number} source={source}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            writer.write(image)
        previous_gray = gray
        frame_number += 1
    capture.release()
    if writer is not None:
        writer.release()

    table = complete_trajectory(pd.DataFrame(rows), physical_fps)
    table.to_csv(args.output_dir / "poses.csv", index=False, encoding="utf-8-sig")
    valid_mask = table["visual_valid"].astype(bool).tolist()
    errors = table.loc[table.visual_valid.eq(1), "reprojection_error_px"]
    summary = {
        "status": "PASS" if any(valid_mask) else "REVIEW",
        "video": str(args.video.resolve()), "width": width, "height": height,
        "container_fps": container_fps, "physical_fps": physical_fps,
        "reported_total_frames": total_frames, "processed_frames": len(table),
        "valid_frames": int(sum(valid_mask)),
        "complete_pose_frames": int(table.pose_available.sum()),
        "all_frames_complete": bool(table.pose_available.all()),
        "visual_success_rate": float(np.mean(valid_mask)) if valid_mask else 0.0,
        "direct_frames": int(table.estimate_kind.eq("measured_direct").sum()),
        "klt_frames": int(table.estimate_kind.eq("measured_klt").sum()),
        "median_reprojection_error_px": None if errors.empty else float(errors.median()),
        "longest_invalid_run_frames": longest_invalid_run(valid_mask),
        "interpolated_frames": int(table.estimate_kind.eq("interpolated").sum()),
        "extrapolated_frames": int(table.estimate_kind.eq("extrapolated").sum()),
        "marker_ids_in_map": sorted(marker_map),
        "pose_convention": "X_camera = R_CB @ X_body + t_CB",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import (
    angular_velocity_body,
    euler_xyz_deg_from_rmat,
    find_experiments,
    load_calibration,
    load_marker_layout,
    load_yaml,
    make_aruco_detector,
    quat_wxyz_from_rmat,
    read_export_fps,
)


def load_yolo(weights):
    if not weights:
        return None
    from ultralytics import YOLO

    return YOLO(weights)


def make_strict_aruco_detectors(dictionary_name):
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    configurations = [
        # Conservative error correction is essential on blurred 6x6 markers.
        (7, 0.03, 0.010, 4, 0.60),
        (3, 0.04, 0.006, 6, 0.50),
        (11, 0.05, 0.004, 8, 0.40),
    ]
    detectors = []
    for adaptive_constant, polygon_rate, perimeter_rate, pixels_per_cell, correction in configurations:
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 53
        parameters.adaptiveThreshWinSizeStep = 4
        parameters.adaptiveThreshConstant = adaptive_constant
        parameters.minMarkerPerimeterRate = perimeter_rate
        parameters.maxMarkerPerimeterRate = 5.0
        parameters.polygonalApproxAccuracyRate = polygon_rate
        parameters.minCornerDistanceRate = 0.01
        parameters.minDistanceToBorder = 2
        parameters.perspectiveRemovePixelPerCell = pixels_per_cell
        parameters.perspectiveRemoveIgnoredMarginPerCell = 0.15
        parameters.maxErroneousBitsInBorderRate = 0.35
        parameters.errorCorrectionRate = correction
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 50
        detectors.append(cv2.aruco.ArucoDetector(dictionary, parameters))
    return detectors


def boxes_to_rois(
    boxes, frame, pad=60, footer_mask_px=24, full_frame_with_boxes=True
):
    rois = []
    for box in boxes:
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)
        if (y1 + y2) * 0.5 < frame.shape[0] - footer_mask_px:
            rois.append((x1, y1, x2, y2))
    if not rois or full_frame_with_boxes:
        rois.append((0, 0, frame.shape[1], frame.shape[0]))
    return rois


def yolo_rois(
    model,
    frame,
    conf,
    imgsz=960,
    footer_mask_px=24,
    precomputed_boxes=None,
    full_frame_with_boxes=True,
    roi_pad_px=60,
):
    if precomputed_boxes is not None:
        return boxes_to_rois(
            precomputed_boxes,
            frame,
            pad=roi_pad_px,
            footer_mask_px=footer_mask_px,
            full_frame_with_boxes=full_frame_with_boxes,
        )
    if model is None:
        return [(0, 0, frame.shape[1], frame.shape[0])]
    inference_frame = frame.copy()
    if footer_mask_px > 0:
        inference_frame[-footer_mask_px:, :] = 0
    result = model.predict(inference_frame, conf=conf, imgsz=imgsz, verbose=False, device="cpu")[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [(0, 0, frame.shape[1], frame.shape[0])]
    return boxes_to_rois(
        boxes.xyxy.cpu().numpy(),
        frame,
        pad=roi_pad_px,
        footer_mask_px=footer_mask_px,
        full_frame_with_boxes=full_frame_with_boxes,
    )


def _is_duplicate_marker(existing_corners, candidate, max_center_distance=8.0):
    candidate_center = candidate.reshape(4, 2).mean(axis=0)
    for old in existing_corners:
        old_center = old.reshape(4, 2).mean(axis=0)
        if float(np.linalg.norm(candidate_center - old_center)) < max_center_distance:
            return True
    return False


def _detect_aruco_multiscale(
    detector,
    roi_gray,
    min_consensus=2,
    max_upscale=4,
    force_upscale=False,
    upscale_factors=None,
):
    detectors = detector if isinstance(detector, (list, tuple)) else [detector]
    variants = [(roi_gray, 1.0)]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gray)
    variants.append((clahe, 1.0))
    blur = cv2.GaussianBlur(roi_gray, (0, 0), 1.2)
    sharpened = cv2.addWeighted(roi_gray, 1.8, blur, -0.8, 0.0)
    variants.append((sharpened, 1.0))
    if force_upscale or min(roi_gray.shape[:2]) < 900:
        requested = upscale_factors or [max(2, int(max_upscale))]
        scales = []
        for value in requested:
            scale = float(value)
            if scale > 1.0 and scale <= float(max_upscale) and all(
                abs(scale - old) > 1e-6 for old in scales
            ):
                scales.append(scale)
        for scale in scales:
            interpolation = cv2.INTER_CUBIC if scale <= 2.0 else cv2.INTER_LANCZOS4
            for image in (roi_gray, clahe, sharpened):
                variants.append(
                    (cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation), scale)
                )

    candidates = defaultdict(list)
    for variant_index, (image, scale) in enumerate(variants):
        for detector_index, current_detector in enumerate(detectors):
            corners, ids, _ = current_detector.detectMarkers(image)
            if ids is None:
                continue
            for c, marker_id in zip(corners, ids.reshape(-1)):
                c = np.asarray(c, dtype=np.float32) / scale
                candidates[int(marker_id)].append((c, (variant_index, detector_index)))

    merged_corners = []
    merged_ids = []
    required_support = min(max(1, int(min_consensus)), len(detectors) * len(variants))
    if len(detectors) == 1:
        required_support = 1
    for marker_id, marker_candidates in candidates.items():
        clusters = []
        for corners, strategy in marker_candidates:
            center = corners.reshape(4, 2).mean(axis=0)
            side = float(
                np.mean(
                    np.linalg.norm(
                        np.roll(corners.reshape(4, 2), -1, axis=0) - corners.reshape(4, 2),
                        axis=1,
                    )
                )
            )
            threshold = max(6.0, 0.25 * side)
            matching = None
            for cluster in clusters:
                if float(np.linalg.norm(center - cluster["center"])) <= threshold:
                    matching = cluster
                    break
            if matching is None:
                matching = {"center": center, "items": []}
                clusters.append(matching)
            matching["items"].append((corners, strategy))
            matching["center"] = np.mean(
                [item[0].reshape(4, 2).mean(axis=0) for item in matching["items"]], axis=0
            )
        best = max(
            clusters,
            key=lambda cluster: len({item[1] for item in cluster["items"]}),
        )
        support = len({item[1] for item in best["items"]})
        if support < required_support:
            continue
        stack = np.stack([item[0].reshape(4, 2) for item in best["items"]], axis=0)
        consensus = np.median(stack, axis=0)
        dispersion = float(np.median(np.linalg.norm(stack - consensus[None, :, :], axis=2)))
        side = float(np.mean(np.linalg.norm(np.roll(consensus, -1, axis=0) - consensus, axis=1)))
        if dispersion > max(2.0, 0.10 * side):
            continue
        merged_corners.append(consensus.astype(np.float32).reshape(1, 4, 2))
        merged_ids.append(marker_id)
    if not merged_ids:
        return [], None
    return merged_corners, np.asarray(merged_ids, dtype=np.int32).reshape(-1, 1)


def detect_aruco_in_rois(
    detector,
    frame,
    model=None,
    conf=0.35,
    imgsz=960,
    footer_mask_px=24,
    precomputed_boxes=None,
    min_consensus=2,
    max_upscale=4,
    full_frame_with_yolo=True,
    force_upscale=False,
    upscale_factors=None,
    roi_pad_px=60,
):
    all_corners = []
    all_ids = []
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    for x1, y1, x2, y2 in yolo_rois(
        model,
        frame,
        conf,
        imgsz,
        footer_mask_px,
        precomputed_boxes,
        full_frame_with_boxes=full_frame_with_yolo,
        roi_pad_px=roi_pad_px,
    ):
        roi = gray_full[y1:y2, x1:x2]
        corners, ids = _detect_aruco_multiscale(
            detector,
            roi,
            min_consensus=min_consensus,
            max_upscale=max_upscale,
            force_upscale=force_upscale,
            upscale_factors=upscale_factors,
        )
        if ids is None:
            continue
        for c, marker_id in zip(corners, ids.reshape(-1)):
            shifted = np.asarray(c, dtype=np.float32) + np.array([[[x1, y1]]], dtype=np.float32)
            if _is_duplicate_marker(all_corners, shifted):
                continue
            all_corners.append(shifted)
            all_ids.append(int(marker_id))
    if not all_ids:
        return [], None
    return all_corners, np.asarray(all_ids, dtype=np.int32).reshape(-1, 1)


def _marker_grid_points(corners, grid_size=5, margin=0.12):
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    tl, tr, br, bl = corners
    values = np.linspace(margin, 1.0 - margin, int(grid_size), dtype=np.float32)
    points = []
    for v in values:
        left = (1.0 - v) * tl + v * bl
        right = (1.0 - v) * tr + v * br
        for u in values:
            points.append((1.0 - u) * left + u * right)
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def _track_marker_quad_dense_lk(previous_gray, current_gray, corners, tracking_cfg):
    grid_size = int(tracking_cfg.get("dense_grid_size", 5))
    min_points = int(tracking_cfg.get("dense_min_points", 8))
    fb_limit = float(tracking_cfg.get("dense_forward_backward_error_px", 3.0))
    ransac_limit = float(tracking_cfg.get("dense_homography_error_px", 3.0))
    window = int(tracking_cfg.get("window_size", 41))
    max_level = int(tracking_cfg.get("max_pyramid_level", 4))
    previous_points = _marker_grid_points(corners, grid_size=grid_size)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    lk_kwargs = {
        "winSize": (window, window),
        "maxLevel": max_level,
        "criteria": criteria,
        "minEigThreshold": 1e-4,
    }
    current_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, previous_points, None, **lk_kwargs
    )
    if current_points is None or forward_status is None:
        return None
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, current_points, None, **lk_kwargs
    )
    if backward_points is None or backward_status is None:
        return None
    previous_xy = previous_points.reshape(-1, 2)
    current_xy = current_points.reshape(-1, 2)
    backward_xy = backward_points.reshape(-1, 2)
    fb_error = np.linalg.norm(backward_xy - previous_xy, axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(fb_error)
        & (fb_error <= fb_limit)
    )
    if forward_error is not None:
        valid &= np.isfinite(forward_error.reshape(-1))
    if int(valid.sum()) < min_points:
        return None
    homography, inlier_mask = cv2.findHomography(
        previous_xy[valid], current_xy[valid], cv2.RANSAC, ransac_limit
    )
    if homography is None or inlier_mask is None:
        return None
    inliers = inlier_mask.reshape(-1).astype(bool)
    if int(inliers.sum()) < min_points:
        return None
    warped = cv2.perspectiveTransform(
        np.asarray(corners, dtype=np.float32).reshape(1, 4, 2), homography
    ).reshape(4, 2)
    return warped.astype(np.float32), float(np.median(fb_error[valid]))


def track_marker_corners_lk(
    previous_gray,
    current_gray,
    track_state,
    directly_detected_ids,
    tracking_cfg,
):
    if previous_gray is None or not track_state:
        return [], None, []

    max_age = int(tracking_cfg.get("max_age_frames", 2))
    fb_limit = float(tracking_cfg.get("max_forward_backward_error_px", 1.5))
    lk_limit = float(tracking_cfg.get("max_lk_error", 25.0))
    min_area = float(tracking_cfg.get("min_marker_area_px2", 36.0))
    min_area_ratio = float(tracking_cfg.get("min_area_ratio", 0.55))
    max_area_ratio = float(tracking_cfg.get("max_area_ratio", 1.80))
    min_edge_ratio = float(tracking_cfg.get("min_edge_ratio", 0.50))
    max_edge_ratio = float(tracking_cfg.get("max_edge_ratio", 2.00))
    window = int(tracking_cfg.get("window_size", 31))
    max_level = int(tracking_cfg.get("max_pyramid_level", 3))
    dense_enabled = str(tracking_cfg.get("mode", "corner_lk")).lower() == "dense_lk"
    directly_detected_ids = set(directly_detected_ids)
    height, width = current_gray.shape[:2]

    tracked_corners = []
    tracked_ids = []
    next_states = []
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    for item in track_state:
        marker_id = int(item["marker_id"])
        age = int(item.get("age", 0))
        if marker_id in directly_detected_ids or age >= max_age:
            continue
        previous_points = np.asarray(item["corners"], dtype=np.float32).reshape(4, 1, 2)
        dense_result = None
        if dense_enabled:
            dense_result = _track_marker_quad_dense_lk(
                previous_gray,
                current_gray,
                previous_points.reshape(4, 2),
                tracking_cfg,
            )
        if dense_result is not None:
            dense_corners, dense_fb = dense_result
            current_points = dense_corners.reshape(4, 1, 2)
            forward_backward = np.full(4, dense_fb, dtype=np.float32)
        else:
            current_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                previous_points,
                None,
                winSize=(window, window),
                maxLevel=max_level,
                criteria=criteria,
            )
            if current_points is None or forward_status is None:
                continue
            backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                current_gray,
                previous_gray,
                current_points,
                None,
                winSize=(window, window),
                maxLevel=max_level,
                criteria=criteria,
            )
            if backward_points is None or backward_status is None:
                continue
            if not np.all(forward_status.reshape(-1)) or not np.all(backward_status.reshape(-1)):
                continue
            forward_backward = np.linalg.norm(
                backward_points.reshape(4, 2) - previous_points.reshape(4, 2), axis=1
            )
            if float(np.max(forward_backward)) > fb_limit:
                continue
            if forward_error is not None and float(np.median(forward_error)) > lk_limit:
                continue

        old = previous_points.reshape(4, 2)
        new = current_points.reshape(4, 2)
        if not np.all(np.isfinite(new)):
            continue
        if (
            np.min(new[:, 0]) < 2
            or np.max(new[:, 0]) >= width - 2
            or np.min(new[:, 1]) < 2
            or np.max(new[:, 1]) >= height - 2
        ):
            continue
        old_area = float(abs(cv2.contourArea(old)))
        new_area = float(abs(cv2.contourArea(new)))
        if old_area < min_area or new_area < min_area or not cv2.isContourConvex(new):
            continue
        area_ratio = new_area / max(old_area, 1e-6)
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            continue
        old_edges = np.linalg.norm(np.roll(old, -1, axis=0) - old, axis=1)
        new_edges = np.linalg.norm(np.roll(new, -1, axis=0) - new, axis=1)
        edge_ratios = new_edges / np.maximum(old_edges, 1e-6)
        if np.min(edge_ratios) < min_edge_ratio or np.max(edge_ratios) > max_edge_ratio:
            continue

        corners = new.astype(np.float32).reshape(1, 4, 2)
        tracked_corners.append(corners)
        tracked_ids.append(marker_id)
        next_states.append(
            {
                "marker_id": marker_id,
                "corners": new.astype(np.float32),
                "age": age + 1,
                "forward_backward_error_px": float(np.max(forward_backward)),
            }
        )
    if not tracked_ids:
        return [], None, []
    return (
        tracked_corners,
        np.asarray(tracked_ids, dtype=np.int32).reshape(-1, 1),
        next_states,
    )


def reprojection_errors(obj_points, img_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - img_points.reshape(-1, 2), axis=1)
    return float(np.mean(residuals)), float(np.max(residuals))


def collect_marker_observations(corners, ids, layout):
    observations = []
    if ids is None:
        return observations
    for detected_corners, marker_id in zip(corners, ids.reshape(-1)):
        marker_id = int(marker_id)
        if marker_id not in layout:
            continue
        observations.append(
            {
                "marker_id": marker_id,
                "obj_points": np.asarray(layout[marker_id], dtype=np.float32).reshape(4, 3),
                "img_points": np.asarray(detected_corners, dtype=np.float32).reshape(4, 2),
            }
        )
    return observations


def stack_observations(observations):
    if not observations:
        return None, None
    return (
        np.vstack([item["obj_points"] for item in observations]),
        np.vstack([item["img_points"] for item in observations]),
    )


def marker_reprojection_errors(
    observations, rvec, tvec, camera_matrix, dist_coeffs
):
    errors = {}
    for item in observations:
        projected, _ = cv2.projectPoints(
            item["obj_points"], rvec, tvec, camera_matrix, dist_coeffs
        )
        residuals = np.linalg.norm(
            projected.reshape(-1, 2) - item["img_points"], axis=1
        )
        errors[item["marker_id"]] = float(np.mean(residuals))
    return errors


def points_are_coplanar(obj_points):
    centered = np.asarray(obj_points, dtype=np.float64) - np.mean(obj_points, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return bool(
        len(singular_values) < 3
        or singular_values[-1] <= max(1e-8, singular_values[0] * 1e-5)
    )


def solve_body_pose(
    obj_points,
    img_points,
    camera_matrix,
    dist_coeffs,
    prev_rvec,
    prev_tvec,
    continuity_weight=0.03,
    translation_weight=15.0,
    temporal_gap=1,
):
    candidates = []
    use_guess = prev_rvec is not None and prev_tvec is not None

    def add_candidate(ok, rvec, tvec, method):
        if not ok or tvec is None or float(np.asarray(tvec).reshape(-1)[2]) <= 0.0:
            return
        mean_error, max_error = reprojection_errors(
            obj_points, img_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        continuity_penalty = 0.0
        gap = max(1, int(temporal_gap))
        if prev_rvec is not None:
            prev_rmat, _ = cv2.Rodrigues(prev_rvec)
            rmat, _ = cv2.Rodrigues(rvec)
            continuity_penalty = (
                continuity_weight * rotation_jump_deg(prev_rmat, rmat) / gap
            )
        translation_penalty = 0.0
        if prev_tvec is not None:
            translation_penalty = translation_weight * float(
                np.linalg.norm(np.asarray(tvec).reshape(3) - np.asarray(prev_tvec).reshape(3))
            ) / gap
        candidates.append(
            (
                mean_error + continuity_penalty + translation_penalty,
                mean_error,
                max_error,
                rvec,
                tvec,
                method,
            )
        )

    if use_guess:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                rvec=prev_rvec.copy(),
                tvec=prev_tvec.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            add_candidate(ok, rvec, tvec, "iterative_guided")
        except cv2.error:
            pass

    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        add_candidate(ok, rvec, tvec, "iterative_unguided")
    except cv2.error:
        pass

    if points_are_coplanar(obj_points):
        try:
            result = cv2.solvePnPGeneric(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if result[0]:
                for index, (rvec, tvec) in enumerate(zip(result[1], result[2])):
                    add_candidate(True, rvec, tvec, f"ippe_{index}")
        except cv2.error:
            pass

    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        add_candidate(ok, rvec, tvec, "sqpnp")
    except cv2.error:
        pass

    if len(obj_points) >= 8:
        try:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                rvec=prev_rvec,
                tvec=prev_tvec,
                useExtrinsicGuess=use_guess,
                iterationsCount=150,
                reprojectionError=5.0,
                confidence=0.995,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_points, img_points, camera_matrix, dist_coeffs, rvec, tvec
                )
                add_candidate(True, rvec, tvec, "ransac_lm")
        except cv2.error:
            pass

    if not candidates:
        return False, None, None, np.nan, np.nan, "none"
    _, mean_error, max_error, rvec, tvec, method = min(
        candidates, key=lambda item: item[0]
    )
    return True, rvec, tvec, mean_error, max_error, method


def solve_pose_with_marker_rejection(
    observations,
    camera_matrix,
    dist_coeffs,
    prev_rvec,
    prev_tvec,
    marker_error_limit,
    continuity_weight,
    translation_weight,
    temporal_gap=1,
):
    active = list(observations)
    rejected_ids = []
    final = (False, None, None, np.nan, np.nan, "none")
    final_marker_errors = {}
    all_marker_errors = {}

    while active:
        obj_points, img_points = stack_observations(active)
        final = solve_body_pose(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            prev_rvec,
            prev_tvec,
            continuity_weight=continuity_weight,
            translation_weight=translation_weight,
            temporal_gap=temporal_gap,
        )
        solved, rvec, tvec, _, _, _ = final
        if not solved:
            break
        final_marker_errors = marker_reprojection_errors(
            active, rvec, tvec, camera_matrix, dist_coeffs
        )
        all_marker_errors.update(final_marker_errors)
        worst_id, worst_error = max(final_marker_errors.items(), key=lambda item: item[1])
        if len(active) <= 1 or worst_error <= marker_error_limit:
            break
        rejected_ids.append(worst_id)
        active = [item for item in active if item["marker_id"] != worst_id]

    used_ids = [item["marker_id"] for item in active]
    obj_points, img_points = stack_observations(active)
    return final, used_ids, rejected_ids, all_marker_errors, obj_points, img_points


def rotation_jump_deg(prev_r, cur_r):
    if prev_r is None:
        return np.nan
    relative = cur_r @ prev_r.T
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def evaluate_pose_candidate(
    pose_result,
    prev_r,
    prev_tvec,
    prev_time,
    prev_frame,
    time_s,
    frame_index,
    max_mean_reprojection_error,
    max_pose_jump_per_frame,
    max_absolute_pose_jump_deg,
    max_translation_step_per_frame,
    max_rpm,
):
    solved, rvec, tvec, mean_error, _, _ = pose_result
    result = {
        "solved": bool(solved),
        "rmat": None,
        "pose_jump_deg": np.nan,
        "translation_step_m": np.nan,
        "omega": np.full(3, np.nan),
        "omega_mag": np.nan,
        "rpm": np.nan,
        "reasons": ["pnp_failed"] if not solved else [],
    }
    if not solved:
        return result

    rmat, _ = cv2.Rodrigues(rvec)
    result["rmat"] = rmat
    result["pose_jump_deg"] = rotation_jump_deg(prev_r, rmat)
    dt = 0.0 if prev_time is None else time_s - prev_time
    frame_gap = 1 if prev_frame is None else frame_index - prev_frame
    if prev_tvec is not None:
        result["translation_step_m"] = float(
            np.linalg.norm(tvec.reshape(3) - prev_tvec.reshape(3))
        )
    omega, omega_mag, rpm = angular_velocity_body(prev_r, rmat, dt)
    result["omega"] = omega
    result["omega_mag"] = omega_mag
    result["rpm"] = rpm

    reasons = []
    if tvec.reshape(-1)[2] <= 0.0:
        reasons.append("negative_depth")
    if mean_error > max_mean_reprojection_error:
        reasons.append("reprojection")
    if (
        np.isfinite(result["pose_jump_deg"])
        and result["pose_jump_deg"] > max_pose_jump_per_frame * frame_gap
    ):
        reasons.append("pose_jump")
    if (
        np.isfinite(result["pose_jump_deg"])
        and result["pose_jump_deg"] > max_absolute_pose_jump_deg
    ):
        reasons.append("absolute_pose_jump")
    if (
        np.isfinite(result["translation_step_m"])
        and result["translation_step_m"]
        > max_translation_step_per_frame * frame_gap
    ):
        reasons.append("translation_step")
    if np.isfinite(rpm) and rpm > max_rpm:
        reasons.append("rpm_limit")
    result["reasons"] = reasons
    return result


def load_precomputed_yolo_boxes(root, experiment_name):
    if not root:
        return None
    experiment_dir = Path(root) / experiment_name
    csv_paths = sorted(experiment_dir.glob("video_*_yolo_detections.csv"))
    if len(csv_paths) != 1:
        raise RuntimeError(
            f"Expected one precomputed YOLO CSV in {experiment_dir}, found {len(csv_paths)}"
        )
    boxes_by_frame = defaultdict(list)
    with csv_paths[0].open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            boxes_by_frame[int(row["frame"])].append(
                [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            )
    print(f"Loaded precomputed YOLO boxes: {csv_paths[0]}", flush=True)
    return boxes_by_frame


def process_one(exp, cfg, yolo_model, max_frames=0, output_root=None):
    video = exp["video"]
    out_dir = Path(output_root or cfg["output_root"]) / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "pose.csv"
    output_video = out_dir / "annotated.mp4"

    dictionary_name, layout = load_marker_layout(cfg["marker_layout"])
    detection_cfg = cfg.get("aruco_detection", {})
    if bool(detection_cfg.get("strict_ensemble", True)):
        detector = make_strict_aruco_detectors(
            dictionary_name or cfg["aruco_dictionary"]
        )
    else:
        detector = make_aruco_detector(dictionary_name or cfg["aruco_dictionary"])

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    container_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = read_export_fps(exp["folder"]) or container_fps
    camera_matrix, dist_coeffs = load_calibration(cfg.get("camera_calibration"), width, height)

    frame_limit = frame_count if max_frames <= 0 else min(frame_count, max_frames)
    writer_fps = float(cfg.get("annotated_video_fps", container_fps or 30.0))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), writer_fps, (width, height))
    rows = []
    prev_r = None
    prev_rvec = None
    prev_tvec = None
    prev_omega_for_prediction = None
    prev_time = None
    prev_frame = None
    previous_tracking_gray = None
    marker_track_state = []

    quality = cfg.get("quality", {})
    max_mean_reprojection_error = float(quality.get("max_reprojection_error_px", 6.0))
    max_rpm = float(quality.get("max_rpm", 2500.0))
    max_marker_error = float(quality.get("max_marker_reprojection_error_px", 3.0))
    max_pose_jump_per_frame = float(quality.get("max_pose_jump_deg_per_frame", 12.0))
    max_absolute_pose_jump_deg = float(quality.get("max_absolute_pose_jump_deg", 45.0))
    max_translation_step_per_frame = float(
        quality.get("max_translation_step_m_per_frame", 0.02)
    )
    max_temporal_gap_frames = int(quality.get("max_temporal_gap_frames", 4))
    reacquire_after_gap_frames = int(
        quality.get("reacquire_after_gap_frames", 0)
    )
    continuity_weight = float(quality.get("continuity_weight_px_per_deg", 0.03))
    translation_weight = float(quality.get("translation_weight_px_per_m", 15.0))
    use_motion_prediction = bool(quality.get("use_motion_prediction", False))
    motion_prediction_alpha = float(
        np.clip(quality.get("motion_prediction_omega_alpha", 0.65), 0.0, 1.0)
    )
    yolo_imgsz = int(cfg.get("yolo_imgsz", 960))
    footer_mask_px = int(cfg.get("footer_mask_px", 24))
    precomputed_boxes = load_precomputed_yolo_boxes(
        cfg.get("precomputed_yolo_root"), exp["name"]
    )
    min_detection_consensus = int(detection_cfg.get("min_consensus", 2))
    max_detection_upscale = float(detection_cfg.get("max_upscale", 4))
    force_detection_upscale = bool(detection_cfg.get("force_upscale", False))
    upscale_factors = detection_cfg.get("upscale_factors", [2.0])
    if not isinstance(upscale_factors, (list, tuple)):
        upscale_factors = [upscale_factors]
    upscale_factors = [float(value) for value in upscale_factors]
    full_frame_with_yolo = bool(
        detection_cfg.get("full_frame_with_yolo", True)
    )
    roi_pad_px = int(detection_cfg.get("roi_pad_px", 60))
    tracking_cfg = cfg.get("optical_flow_tracking", {})
    tracking_enabled = bool(tracking_cfg.get("enabled", False))

    for frame_index in range(frame_limit):
        ok, frame = cap.read()
        if not ok:
            break
        time_s = frame_index / fps
        current_tracking_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        direct_corners, direct_ids = detect_aruco_in_rois(
            detector,
            frame,
            yolo_model,
            cfg.get("yolo_conf", 0.35),
            yolo_imgsz,
            footer_mask_px,
            None if precomputed_boxes is None else precomputed_boxes.get(frame_index, []),
            min_consensus=min_detection_consensus,
            max_upscale=max_detection_upscale,
            full_frame_with_yolo=full_frame_with_yolo,
            force_upscale=force_detection_upscale,
            upscale_factors=upscale_factors,
            roi_pad_px=roi_pad_px,
        )
        direct_id_values = (
            [] if direct_ids is None else direct_ids.reshape(-1).astype(int).tolist()
        )
        tracked_corners, tracked_ids, tracked_state_candidates = ([], None, [])
        if tracking_enabled:
            tracked_corners, tracked_ids, tracked_state_candidates = track_marker_corners_lk(
                previous_tracking_gray,
                current_tracking_gray,
                marker_track_state,
                direct_id_values,
                tracking_cfg,
            )
        corners = list(direct_corners) + list(tracked_corners)
        all_id_values = list(direct_id_values)
        if tracked_ids is not None:
            all_id_values.extend(tracked_ids.reshape(-1).astype(int).tolist())
        ids = (
            None
            if not all_id_values
            else np.asarray(all_id_values, dtype=np.int32).reshape(-1, 1)
        )
        tracked_id_values = (
            [] if tracked_ids is None else tracked_ids.reshape(-1).astype(int).tolist()
        )
        tracked_id_set = set(tracked_id_values)
        frame_gap_from_last_pose = (
            0 if prev_frame is None else frame_index - prev_frame
        )
        if prev_frame is not None and (
            frame_gap_from_last_pose > max_temporal_gap_frames
            or (
                reacquire_after_gap_frames > 0
                and frame_gap_from_last_pose >= reacquire_after_gap_frames
            )
        ):
            prev_r = None
            prev_rvec = None
            prev_tvec = None
            prev_omega_for_prediction = None
            prev_time = None
            prev_frame = None

        direct_observations = collect_marker_observations(
            direct_corners, direct_ids, layout
        )
        tracked_observations = collect_marker_observations(
            tracked_corners, tracked_ids, layout
        )
        observations = direct_observations + tracked_observations
        detected_layout_ids = [marker_id for marker_id in direct_id_values if marker_id in layout]
        observed_layout_ids = [item["marker_id"] for item in observations]
        used_ids = list(observed_layout_ids)
        rejected_marker_ids = []
        marker_errors = {}
        obj_points, img_points = stack_observations(observations)

        success = False
        pnp_solved = False
        quality_rejected = False
        rvec = None
        tvec = None
        mean_reprojection_error = np.nan
        max_reprojection_error = np.nan
        pose_jump = np.nan
        translation_step = np.nan
        candidate_rpm = np.nan
        rejection_reason = ""
        pnp_method = "none"
        quat = np.full(4, np.nan)
        euler = np.full(3, np.nan)
        omega = np.full(3, np.nan)
        omega_mag = np.nan
        rpm = np.nan
        motion_prediction_used = False
        prediction_residual_deg = np.nan

        if observations:
            solve_gap = 1 if prev_frame is None else max(1, frame_index - prev_frame)
            solver_rvec = prev_rvec
            solver_tvec = prev_tvec
            solver_gap = solve_gap
            predicted_r = None
            if (
                use_motion_prediction
                and prev_r is not None
                and prev_time is not None
                and prev_omega_for_prediction is not None
                and np.all(np.isfinite(prev_omega_for_prediction))
            ):
                prediction_dt = time_s - prev_time
                predicted_delta, _ = cv2.Rodrigues(
                    np.asarray(prev_omega_for_prediction, dtype=np.float64) * prediction_dt
                )
                predicted_r = prev_r @ predicted_delta
                solver_rvec, _ = cv2.Rodrigues(predicted_r)
                solver_gap = 1
                motion_prediction_used = True
            if direct_observations and tracked_observations:
                direct_trial = solve_pose_with_marker_rejection(
                    direct_observations,
                    camera_matrix,
                    dist_coeffs,
                    solver_rvec,
                    solver_tvec,
                    marker_error_limit=max_marker_error,
                    continuity_weight=continuity_weight,
                    translation_weight=translation_weight,
                    temporal_gap=solver_gap,
                )
                direct_quality = evaluate_pose_candidate(
                    direct_trial[0],
                    prev_r,
                    prev_tvec,
                    prev_time,
                    prev_frame,
                    time_s,
                    frame_index,
                    max_mean_reprojection_error,
                    max_pose_jump_per_frame,
                    max_absolute_pose_jump_deg,
                    max_translation_step_per_frame,
                    max_rpm,
                )
                if direct_quality["solved"] and not direct_quality["reasons"]:
                    observations = direct_observations
            (
                pose_result,
                used_ids,
                rejected_marker_ids,
                marker_errors,
                obj_points,
                img_points,
            ) = solve_pose_with_marker_rejection(
                observations,
                camera_matrix,
                dist_coeffs,
                solver_rvec,
                solver_tvec,
                marker_error_limit=max_marker_error,
                continuity_weight=continuity_weight,
                translation_weight=translation_weight,
                temporal_gap=solver_gap,
            )
            (
                pnp_solved,
                rvec,
                tvec,
                mean_reprojection_error,
                max_reprojection_error,
                pnp_method,
            ) = pose_result
            if not pnp_solved:
                rejection_reason = "pnp_failed"
            if pnp_solved:
                quality_result = evaluate_pose_candidate(
                    pose_result,
                    prev_r,
                    prev_tvec,
                    prev_time,
                    prev_frame,
                    time_s,
                    frame_index,
                max_mean_reprojection_error,
                max_pose_jump_per_frame,
                max_absolute_pose_jump_deg,
                max_translation_step_per_frame,
                    max_rpm,
                )
                rmat = quality_result["rmat"]
                pose_jump = quality_result["pose_jump_deg"]
                prediction_residual_deg = rotation_jump_deg(predicted_r, rmat)
                translation_step = quality_result["translation_step_m"]
                candidate_omega = quality_result["omega"]
                candidate_omega_mag = quality_result["omega_mag"]
                candidate_rpm = quality_result["rpm"]
                reasons = quality_result["reasons"]
                rejection_reason = ";".join(reasons)
                quality_rejected = bool(reasons)
                success = not quality_rejected
            if success:
                quat = quat_wxyz_from_rmat(rmat)
                euler = euler_xyz_deg_from_rmat(rmat)
                omega, omega_mag, rpm = candidate_omega, candidate_omega_mag, candidate_rpm
                previous_r = prev_r
                if previous_r is not None and np.all(np.isfinite(candidate_omega)):
                    measured_current_body = rmat.T @ previous_r @ candidate_omega
                    if prev_omega_for_prediction is None:
                        prev_omega_for_prediction = measured_current_body
                    else:
                        transported_prediction = rmat.T @ previous_r @ prev_omega_for_prediction
                        prev_omega_for_prediction = (
                            motion_prediction_alpha * measured_current_body
                            + (1.0 - motion_prediction_alpha) * transported_prediction
                        )
                prev_r = rmat.copy()
                prev_rvec = rvec.copy()
                prev_tvec = tvec.copy()
                prev_time = time_s
                prev_frame = frame_index
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03, 2)
            elif pnp_solved:
                rvec = None
                tvec = None
        elif ids is not None:
            rejection_reason = "no_layout_marker"

        direct_next_state = []
        if success and direct_ids is not None:
            for marker_corners, marker_id in zip(direct_corners, direct_id_values):
                if marker_id in used_ids:
                    direct_next_state.append(
                        {
                            "marker_id": marker_id,
                            "corners": np.asarray(marker_corners, dtype=np.float32).reshape(4, 2),
                            "age": 0,
                        }
                    )
        marker_track_state = direct_next_state
        if success:
            direct_layout_set = {item["marker_id"] for item in direct_next_state}
            for item in tracked_state_candidates:
                marker_id = int(item["marker_id"])
                if marker_id in used_ids and marker_id not in direct_layout_set:
                    marker_track_state.append(item)
        previous_tracking_gray = current_tracking_gray

        if direct_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, direct_corners, direct_ids)
        for marker_corners, marker_id in zip(tracked_corners, tracked_id_values):
            polygon = np.rint(marker_corners.reshape(4, 2)).astype(np.int32)
            cv2.polylines(frame, [polygon], True, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"T{marker_id}",
                tuple(polygon[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
        status_color = (0, 255, 0) if success else (0, 180, 255)
        cv2.putText(frame, f"{exp['name']}  frame={frame_index}  t={time_s:.4f}s", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"markers={used_ids} tracked={tracked_id_values} drop={rejected_marker_ids} reproj={mean_reprojection_error:.2f}px", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"rpm={rpm:.1f}  omega=({omega[0]:.1f},{omega[1]:.1f},{omega[2]:.1f})", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        writer.write(frame)

        rv = np.full(3, np.nan) if rvec is None else rvec.reshape(-1)
        tv = np.full(3, np.nan) if tvec is None else tvec.reshape(-1)
        rows.append(
            {
                "experiment": exp["name"],
                "frame": frame_index,
                "time_s": time_s,
                "success": int(success),
                "pnp_solved": int(pnp_solved),
                "quality_rejected": int(quality_rejected),
                "rejection_reason": rejection_reason,
                "pnp_method": pnp_method,
                "detected_marker_ids": "" if direct_ids is None else " ".join(map(str, direct_id_values)),
                "detected_layout_marker_ids": " ".join(map(str, detected_layout_ids)),
                "tracked_marker_ids": " ".join(map(str, tracked_id_values)),
                "used_tracked_marker_ids": " ".join(
                    map(str, [marker_id for marker_id in used_ids if marker_id in tracked_id_set])
                ),
                "num_tracked_markers": len(
                    [marker_id for marker_id in used_ids if marker_id in tracked_id_set]
                ),
                "marker_ids": " ".join(map(str, used_ids)),
                "rejected_marker_ids": " ".join(map(str, rejected_marker_ids)),
                "marker_reprojection_errors_px": " ".join(
                    f"{marker_id}:{error:.3f}"
                    for marker_id, error in sorted(marker_errors.items())
                ),
                "num_markers": len(set(used_ids)),
                "num_corners": 0 if obj_points is None else len(obj_points),
                "mean_reprojection_error_px": float(mean_reprojection_error),
                "max_reprojection_error_px": float(max_reprojection_error),
                "pose_jump_deg": float(pose_jump),
                "translation_step_m": float(translation_step),
                "candidate_rpm": float(candidate_rpm),
                "motion_prediction_used": int(motion_prediction_used),
                "prediction_residual_deg": float(prediction_residual_deg),
                "rvec_x": float(rv[0]),
                "rvec_y": float(rv[1]),
                "rvec_z": float(rv[2]),
                "tvec_x_m": float(tv[0]),
                "tvec_y_m": float(tv[1]),
                "tvec_z_m": float(tv[2]),
                "quat_w": float(quat[0]),
                "quat_x": float(quat[1]),
                "quat_y": float(quat[2]),
                "quat_z": float(quat[3]),
                "roll_x_deg": float(euler[0]),
                "pitch_y_deg": float(euler[1]),
                "yaw_z_deg": float(euler[2]),
                "omega_body_x_rad_s": float(omega[0]),
                "omega_body_y_rad_s": float(omega[1]),
                "omega_body_z_rad_s": float(omega[2]),
                "omega_mag_rad_s": float(omega_mag),
                "rpm": float(rpm),
            }
        )
        if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_limit:
            good = sum(row["success"] for row in rows)
            print(f"PROGRESS {exp['name']} {frame_index + 1}/{frame_limit} pose={good}", flush=True)

    cap.release()
    writer.release()
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    good = sum(r["success"] for r in rows)
    print(f"{exp['name']}: {good}/{len(rows)} pose frames, fps={fps}, out={output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiment-glob", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-yolo", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    experiments = find_experiments(cfg["data_root"], args.experiment_glob or cfg["experiment_glob"])
    yolo_model = (
        None
        if args.no_yolo or cfg.get("precomputed_yolo_root")
        else load_yolo(cfg.get("yolo_weights"))
    )
    print(f"Found {len(experiments)} experiments")
    for exp in experiments:
        process_one(exp, cfg, yolo_model, max_frames=args.max_frames, output_root=args.output_root)


if __name__ == "__main__":
    main()




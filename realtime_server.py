from __future__ import annotations

import argparse
import csv
import importlib.util
import ipaddress
import json
import socket
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, request, send_from_directory, Response


ROOT = Path(__file__).resolve().parent
BATCH_PATH = ROOT / "01_batch_pose_yolo_aruco_pnp.py"
spec = importlib.util.spec_from_file_location("batch_pose", BATCH_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {BATCH_PATH}")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


app = Flask(__name__)
STATE_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
LATEST_JPEG = None
INPUT_CONDITION = threading.Condition()
LATEST_INPUT_FRAME = None
LATEST_INPUT_SEQUENCE = 0
LATEST_STATUS = {
    "connected": False,
    "frame": 0,
    "status": "starting",
}


@app.after_request
def add_cors_headers(response):
    """Allow the GitHub Pages frontend to call the local backend."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def wait_for_web_frame(last_sequence: int, timeout: float = 1.0):
    """Wait for a newer frame uploaded by the browser camera page."""
    with INPUT_CONDITION:
        INPUT_CONDITION.wait_for(
            lambda: LATEST_INPUT_SEQUENCE > last_sequence,
            timeout=timeout,
        )
        if LATEST_INPUT_SEQUENCE <= last_sequence or LATEST_INPUT_FRAME is None:
            return False, None, last_sequence
        return True, LATEST_INPUT_FRAME.copy(), LATEST_INPUT_SEQUENCE


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def load_scaled_calibration(path: str | None, width: int, height: int):
    if not path:
        return batch.load_calibration(None, width, height)
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    camera = np.asarray(data["camera_matrix"], dtype=np.float64).copy()
    source_width = float(data.get("image_width", width))
    source_height = float(data.get("image_height", height))
    camera[0, 0] *= width / source_width
    camera[0, 2] *= width / source_width
    camera[1, 1] *= height / source_height
    camera[1, 2] *= height / source_height
    dist = np.asarray(data.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
    return camera, dist


def load_model(weights: str | None):
    if not weights:
        return None
    from ultralytics import YOLO
    print(f"loading YOLO model: {weights}", flush=True)
    return YOLO(weights)


def run_yolo_boxes(model, frame, conf: float, imgsz: int, device: str):
    if model is None:
        return []
    result = model.predict(
        frame,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
        device=device,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    return result.boxes.xyxy.cpu().numpy()


def put_text(frame, text: str, x: int, y: int, color=(0, 255, 0), scale=0.72):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def process_stream(args):
    global LATEST_JPEG, LATEST_STATUS
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    layout_path = args.marker_layout or cfg["marker_layout"]
    dictionary_name, layout = batch.load_marker_layout(layout_path)
    detection_cfg = cfg.get("aruco_detection", {})
    if bool(detection_cfg.get("strict_ensemble", False)):
        detector = batch.make_strict_aruco_detectors(dictionary_name)
    else:
        detector = batch.make_aruco_detector(dictionary_name)

    weights = args.yolo_weights or cfg.get("yolo_weights")
    model = load_model(weights)
    web_source = isinstance(args.source, str) and args.source.lower() in {
        "web", "browser", "phone-web"
    }
    cap = None
    web_sequence = 0
    pending_frame = None
    if web_source:
        print("source=web, waiting for the phone browser camera ...", flush=True)
        ok, pending_frame, web_sequence = wait_for_web_frame(0, timeout=30.0)
        if not ok:
            raise RuntimeError("30秒内没有收到手机网页摄像头画面")
        height, width = pending_frame.shape[:2]
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {args.source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_path = args.camera_calibration or cfg.get("camera_calibration")
    camera_matrix, dist_coeffs = load_scaled_calibration(camera_path, width, height)
    print(f"source={args.source}, size={width}x{height}", flush=True)
    print(f"camera calibration={camera_path or 'fallback'}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "realtime_pose.csv"
    fieldnames = [
        "frame", "time_s", "success", "status", "marker_count", "used_marker_ids",
        "reprojection_error_px", "rpm", "omega_x", "omega_y", "omega_z",
        "quat_w", "quat_x", "quat_y", "quat_z", "roll_deg", "pitch_deg", "yaw_deg",
        "processing_fps",
    ]
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    prev_r = None
    prev_rvec = None
    prev_tvec = None
    prev_time = None
    cached_boxes = []
    frame_index = 0
    start_clock = time.perf_counter()
    max_reprojection = float(args.max_reprojection_error)
    max_pose_jump = float(args.max_pose_jump_deg)
    max_rpm = float(args.max_rpm)
    yolo_every = max(1, int(args.yolo_every))
    aruco_upscale = float(detection_cfg.get("max_upscale", 2.0))
    upscale_factors = detection_cfg.get("upscale_factors", [2.0])
    if not isinstance(upscale_factors, (list, tuple)):
        upscale_factors = [upscale_factors]

    video_writer = None
    if args.save_video:
        video_writer = cv2.VideoWriter(
            args.save_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.output_fps),
            (width, height),
        )

    try:
        while not STOP_EVENT.is_set():
            if web_source:
                if pending_frame is not None:
                    ok, frame = True, pending_frame
                    pending_frame = None
                else:
                    ok, frame, web_sequence = wait_for_web_frame(web_sequence, timeout=1.0)
            else:
                ok, frame = cap.read()
            if not ok:
                if web_source:
                    continue
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    prev_r = prev_rvec = prev_tvec = prev_time = None
                    continue
                break
            now = time.perf_counter()
            time_s = now - start_clock
            if frame_index % yolo_every == 0:
                cached_boxes = run_yolo_boxes(
                    model, frame, float(args.yolo_conf), int(args.yolo_imgsz), args.device
                )

            corners, ids = batch.detect_aruco_in_rois(
                detector,
                frame,
                model=None,
                conf=float(args.yolo_conf),
                imgsz=int(args.yolo_imgsz),
                footer_mask_px=int(cfg.get("footer_mask_px", 24)),
                precomputed_boxes=cached_boxes,
                min_consensus=int(detection_cfg.get("min_consensus", 1)),
                max_upscale=aruco_upscale,
                full_frame_with_yolo=False,
                force_upscale=bool(detection_cfg.get("force_upscale", True)),
                upscale_factors=[float(v) for v in upscale_factors],
                roi_pad_px=int(detection_cfg.get("roi_pad_px", 80)),
            )
            observations = batch.collect_marker_observations(corners, ids, layout)
            success = False
            status = "no_marker"
            mean_error = np.nan
            rvec = tvec = None
            rmat = None
            omega = np.full(3, np.nan)
            rpm = np.nan
            quat = np.full(4, np.nan)
            euler = np.full(3, np.nan)
            used_ids = []

            if observations:
                pose_result, used_ids, rejected_ids, marker_errors, obj_points, img_points = (
                    batch.solve_pose_with_marker_rejection(
                        observations,
                        camera_matrix,
                        dist_coeffs,
                        prev_rvec,
                        prev_tvec,
                        marker_error_limit=float(cfg.get("quality", {}).get(
                            "max_marker_reprojection_error_px", 6.0
                        )),
                        continuity_weight=float(cfg.get("quality", {}).get(
                            "continuity_weight_px_per_deg", 0.1
                        )),
                        translation_weight=float(cfg.get("quality", {}).get(
                            "translation_weight_px_per_m", 200.0
                        )),
                        temporal_gap=1,
                    )
                )
                solved, rvec, tvec, mean_error, max_error, method = pose_result
                if solved:
                    rmat, _ = cv2.Rodrigues(rvec)
                    pose_jump = batch.rotation_jump_deg(prev_r, rmat)
                    depth_ok = float(np.asarray(tvec).reshape(-1)[2]) > 0.0
                    jump_ok = not np.isfinite(pose_jump) or pose_jump <= max_pose_jump
                    if mean_error <= max_reprojection and depth_ok and jump_ok:
                        dt = 0.0 if prev_time is None else time_s - prev_time
                        omega, omega_mag, rpm = batch.angular_velocity_body(prev_r, rmat, dt)
                        if np.isfinite(rpm) and rpm > max_rpm:
                            status = "rpm_limit"
                        else:
                            success = True
                            status = "valid"
                            quat = batch.quat_wxyz_from_rmat(rmat)
                            euler = batch.euler_xyz_deg_from_rmat(rmat)
                    else:
                        status = "quality_rejected"
                else:
                    status = "pnp_failed"

            if success:
                prev_r, prev_rvec, prev_tvec, prev_time = rmat, rvec, tvec, time_s

            processing_fps = (frame_index + 1) / max(time.perf_counter() - start_clock, 1e-6)
            overlay = frame.copy()
            for box in cached_boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 180, 0), 2)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(overlay, corners, ids)
            if success and rvec is not None and tvec is not None:
                cv2.drawFrameAxes(overlay, camera_matrix, dist_coeffs, rvec, tvec, 0.03, 2)

            color = (0, 220, 0) if success else (0, 80, 255)
            put_text(overlay, f"{status} | markers={len(used_ids)} | fps={processing_fps:.1f}", 20, 35, color)
            put_text(overlay, f"RPM: {rpm:.2f}", 20, 70, color)
            put_text(overlay, f"omega: {omega[0]:.2f}, {omega[1]:.2f}, {omega[2]:.2f} rad/s", 20, 105, color)
            put_text(overlay, f"Euler: {euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f} deg", 20, 140, color)
            put_text(overlay, f"PnP error: {mean_error:.2f} px", 20, 175, color)

            ok_jpg, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok_jpg:
                with STATE_LOCK:
                    LATEST_JPEG = encoded.tobytes()
                    LATEST_STATUS = {
                        "connected": True,
                        "frame": frame_index,
                        "time_s": time_s,
                        "success": success,
                        "status": status,
                        "marker_count": len(used_ids),
                        "used_marker_ids": [int(v) for v in used_ids],
                        "reprojection_error_px": finite_or_none(mean_error),
                        "rpm": finite_or_none(rpm),
                        "omega_x": finite_or_none(omega[0]),
                        "omega_y": finite_or_none(omega[1]),
                        "omega_z": finite_or_none(omega[2]),
                        "quat_w": finite_or_none(quat[0]),
                        "quat_x": finite_or_none(quat[1]),
                        "quat_y": finite_or_none(quat[2]),
                        "quat_z": finite_or_none(quat[3]),
                        "roll_deg": finite_or_none(euler[0]),
                        "pitch_deg": finite_or_none(euler[1]),
                        "yaw_deg": finite_or_none(euler[2]),
                        "processing_fps": processing_fps,
                    }
                    row = dict(LATEST_STATUS)
                writer.writerow({key: row.get(key) for key in fieldnames})
                csv_file.flush()
            if video_writer is not None:
                video_writer.write(overlay)
            frame_index += 1
    finally:
        if cap is not None:
            cap.release()
        csv_file.close()
        if video_writer is not None:
            video_writer.release()
        with STATE_LOCK:
            LATEST_STATUS = {"connected": False, "frame": frame_index, "status": "stopped"}
    print(f"saved {csv_path}", flush=True)


def process_stream_safe(args):
    global LATEST_STATUS
    try:
        process_stream(args)
    except Exception as error:
        traceback.print_exc()
        with STATE_LOCK:
            LATEST_STATUS = {
                "connected": False,
                "frame": 0,
                "status": f"error: {error}",
            }


def source_value(value: str):
    try:
        return int(value)
    except ValueError:
        return value


@app.get("/")
def index():
    return send_from_directory(str(ROOT), "realtime_dashboard.html")


@app.get("/api/status")
def api_status():
    with STATE_LOCK:
        return jsonify(dict(LATEST_STATUS))


@app.post("/api/frame")
def api_frame():
    """Receive one JPEG frame from the phone browser camera."""
    global LATEST_INPUT_FRAME, LATEST_INPUT_SEQUENCE
    payload = request.get_data(cache=False)
    if not payload:
        return jsonify({"ok": False, "error": "empty frame"}), 400
    encoded = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "error": "invalid jpeg"}), 400
    with INPUT_CONDITION:
        LATEST_INPUT_FRAME = frame
        LATEST_INPUT_SEQUENCE += 1
        sequence = LATEST_INPUT_SEQUENCE
        INPUT_CONDITION.notify_all()
    return jsonify({"ok": True, "sequence": sequence})


@app.get("/video_feed")
def video_feed():
    def generate():
        while not STOP_EVENT.is_set():
            with STATE_LOCK:
                image = LATEST_JPEG
            if image:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(image)).encode("ascii") + b"\r\n\r\n" + image + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time YOLOv8 + ArUco + PnP dashboard")
    parser.add_argument("--config", default=str(ROOT / "config_new_body_phone.yaml"))
    parser.add_argument("--source", default="0", help="webcam index, video path, MJPEG URL, or web")
    parser.add_argument("--camera-calibration", default=None)
    parser.add_argument("--marker-layout", default=None)
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--device", default="cpu", help="cpu, 0, or cuda:0")
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-every", type=int, default=3)
    parser.add_argument("--max-reprojection-error", type=float, default=8.0)
    parser.add_argument("--max-pose-jump-deg", type=float, default=45.0)
    parser.add_argument("--max-rpm", type=float, default=3000.0)
    parser.add_argument("--output-dir", default=str(ROOT / "realtime_results"))
    parser.add_argument("--save-video", default=None)
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--https", action="store_true", help="serve HTTPS for browser camera permission")
    return parser.parse_args()


def main():
    args = parse_args()
    args.source = source_value(args.source)
    worker = threading.Thread(target=process_stream_safe, args=(args,), daemon=True)
    worker.start()
    scheme = "https" if args.https else "http"
    print(f"电脑本机打开: {scheme}://127.0.0.1:{args.port}", flush=True)
    lan_ips = sorted({info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    lan_ips = [ip for ip in lan_ips if not ip.startswith("127.")]
    private_ips = [ip for ip in lan_ips if ipaddress.ip_address(ip).is_private]
    if private_ips:
        lan_ips = private_ips
    if lan_ips:
        print("手机打开:", ", ".join(f"{scheme}://{ip}:{args.port}" for ip in lan_ips), flush=True)
    else:
        print(f"未发现局域网IP。请让电脑连接手机热点后重新运行，再访问电脑IPv4地址:{args.port}", flush=True)
    app.run(
        host="0.0.0.0",
        port=args.port,
        threaded=True,
        use_reloader=False,
        ssl_context="adhoc" if args.https else None,
    )


if __name__ == "__main__":
    main()

# Online realtime deployment

This package separates the public GitHub Pages frontend from the Python backend.
The backend runs YOLOv8, ArUco detection, PnP, quaternion/Euler pose, angular
velocity, and RPM estimation. The phone browser sends JPEG frames to the backend.

## Render deployment

1. Upload these files to the root of the GitHub repository:
   `Dockerfile`, `render.yaml`, `requirements-online.txt`, `.dockerignore`,
   `online_server.py`, `realtime_server.py`, `common.py`,
   `01_batch_pose_yolo_aruco_pnp.py`, `config_online.yaml`,
   `marker_layout_new_body.json`, and `yolo_rigid_body_best.pt`.
2. Sign in to Render and create a new Web Service from the GitHub repository.
3. Select Docker runtime and deploy.
4. Copy the HTTPS `onrender.com` service URL.
5. Open the GitHub Pages frontend and enter that URL as the Python backend.

The free plan is suitable for a connectivity test but may process only a few
frames per second on CPU. A paid CPU plan or GPU service is needed for higher
real-time throughput.

Do not upload private videos, passwords, API keys, or raw experimental data.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-online.txt ./
RUN pip install -r requirements-online.txt

COPY online_server.py realtime_server.py phone_realtime_engine.py phone_pose_core.py common.py 01_batch_pose_yolo_aruco_pnp.py ./
COPY config_online.yaml marker_layout_new_body.json yolo_rigid_body_best.pt ./
COPY phone_marker_map.json phone_pose_config.json ./

EXPOSE 10000

CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 0 online_server:app

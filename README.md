# 刚体实时识别开源工程

本工程使用手机摄像头采集画面，在电脑上运行 Python、YOLOv8、ArUco 和 PnP，网页实时显示手机原始画面、标注画面、姿态角、角速度和 RPM。

## 文件说明

- `index.html`：网页前端
- `realtime_server.py`：本地实时后端，接收手机浏览器上传的帧
- `phone_pose_core.py`：手机画面中的 ArUco/PnP 姿态解算核心
- `phone_realtime_engine.py`：实时识别适配器和连续角速度计算
- `01_batch_pose_yolo_aruco_pnp.py`：批处理模式的 YOLOv8、ArUco、PnP 和角速度计算
- `yolo_rigid_body_best.pt`：YOLOv8权重（批处理模式使用）
- `phone_marker_map.json`、`phone_pose_config.json`：手机识别参数
- `marker_layout_new_body.json`：刚体标记布局
- `config_online.yaml`：运行配置
- `start_public_realtime.bat`：一键启动本地后端和公网隧道

## 第一次使用

1. 安装 Python 3.11 或更高版本。
2. 双击 `start_public_realtime.bat`。第一次启动会自动创建 `.venv` 并安装依赖，可能需要几分钟。

   也可以手动在本目录打开 PowerShell，执行：

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements-online.txt
   ```

3. 准备 `cloudflared.exe`，放在项目根目录，或将它加入系统 PATH。
4. 从 Cloudflare 窗口复制 `https://xxxxx.trycloudflare.com` 地址。
5. 手机和电脑都打开 GitHub Pages 网页：

   ```text
   https://meijuliet882-ai.github.io/dzhanibekov-realtime/
   ```

6. 两边网页的后端地址栏都填写刚才复制的 Cloudflare 地址。
7. 手机点击“连接Python后端”和“启动手机摄像头”；电脑只点击“连接Python后端”。

## 注意事项

- Python 后端和 Cloudflare 窗口必须保持运行。
- 临时 Cloudflare 地址每次启动可能变化。
- 手机和电脑不需要连接同一个局域网；Cloudflare 窗口必须保持运行。
- 本项目使用CPU时处理帧率有限，GPU电脑会更流畅。
- 不要将私人视频、传感器数据、密码或API密钥上传到公开仓库。

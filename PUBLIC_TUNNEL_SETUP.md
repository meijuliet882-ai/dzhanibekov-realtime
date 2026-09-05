# GitHub 网页 + 电脑本地 Python + 手机摄像头

## 工作方式

```text
手机摄像头 -> GitHub Pages 网页 -> 公网 HTTPS 隧道 -> 电脑 Python
                                                       |
                                      YOLOv8 + ArUco + PnP
                                                       |
                                原始画面、标注画面、姿态角、角速度
```

GitHub Pages 仍然是网页，YOLOv8、ArUco 和 PnP 仍然在本机运行。公网隧道只负责把手机请求转发到本机 8000 端口。

## 使用步骤

1. 安装 `cloudflared`，下载 Windows 版本并重命名为 `cloudflared.exe`。
2. 将 `cloudflared.exe` 放到本项目根目录 `D:\dzhanibekov_pipeline`。
3. 双击 `start_public_realtime.bat`。
4. 在隧道窗口找到类似下面的地址：

   ```text
   https://xxxxx.trycloudflare.com
   ```

5. 手机和电脑都打开 GitHub Pages：

   ```text
   https://meijuliet882-ai.github.io/dzhanibekov-realtime/
   ```

6. 手机网页的后端地址填写隧道地址，点击连接，再启动手机摄像头。
7. 电脑网页的后端地址也填写同一个隧道地址，只点击连接，不启动电脑摄像头。

## 注意

- Python 后端窗口和隧道窗口必须一直保持打开。
- 隧道地址每次启动可能变化。
- 手机不再填写 `127.0.0.1` 或 `192.168.137.1`。
- GitHub 仓库只需要更新网页 `index.html`；Python 后端使用电脑本地文件。

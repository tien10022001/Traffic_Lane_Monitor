# ĐẶC TẢ KỸ THUẬT HỆ THỐNG GIÁM SÁT GIAO THÔNG MULTI-CAM THỜI GIAN THỰC
**Tên dự án:** `CCTV Traffic AI Service` (Windows Background Service)  
**Phiên bản tài liệu:** `v2.1` (Sẵn sàng giao cho Coding Agent)  
**Mục đích:** Đặc tả kỹ thuật toàn diện (PRD & Architecture Spec) để triển khai dịch vụ chạy nền 24/7 trên Windows, tích hợp mô hình **YOLO11s (8 lớp phương tiện Việt Nam)**, xử lý **Multi-Cam**, **OCR biển số bất đồng bộ**, **Local SQL (SQLite WAL)**, **Active Learning** và **Giám sát tài nguyên phần cứng theo thời gian thực**.

---

## 1. Mục tiêu & Nguyên tắc kỹ thuật bắt buộc (Coding Standards)

### 1.1. Mục tiêu hệ thống
* Chạy ổn định 24/7 dưới dạng **Windows Service**, hỗ trợ đồng thời **nhiều camera CCTV (Multi-Cam)** từ luồng RTSP thực tế hoặc file video MP4 (chế độ kiểm thử).
* Nhận diện & theo dõi (tracking) 8 nhóm phương tiện giao thông Việt Nam theo thời gian thực.
* Phát hiện tự động 3 nhóm vi phạm:
  1. **Lấn chiếm vỉa hè / lề đường** (`sidewalk_encroachment`).
  2. **Dừng đỗ trái phép vượt ngưỡng thời gian** (`no_parking`).
  3. **Đi sai làn / lấn làn** (`lane_roi`, `lane_line`).
* Tự động chọn **Best-shot** và chạy **OCR biển số Việt Nam** (biển 1 dòng & 2 dòng) bất đồng bộ.
* Cung cấp **Web Dashboard** trực quan: xem Multi-Cam Live, vẽ/cấu hình Multi-ROI & Line bằng chuột, quản lý & duyệt bằng chứng (Active Learning), và **theo dõi tài nguyên máy tính đang sử dụng** (CPU, RAM, GPU/VRAM, Ổ đĩa, FPS, độ sâu hàng đợi).

### 1.2. Nguyên tắc code bắt buộc cho Agent
* **Không hard-code:** Toàn bộ tham số (đường dẫn model, ngưỡng confidence chung và theo từng class, `iou`, `track_buffer`, ngưỡng dừng đỗ, cổng API, ngưỡng cảnh báo tài nguyên, chính sách xóa ảnh cũ) phải được nạp từ `config/config.yaml` và/hoặc SQLite, hỗ trợ **hot-reload** không cần khởi động lại service.
* **Xử lý lỗi chặt chẽ & Tự phục hồi:** Mọi kết nối RTSP, truy vấn SQLite, suy luận ONNX, đọc/ghi ảnh, đọc thông số phần cứng đều phải bọc `try/except`, có cơ chế auto-reconnect (exponential backoff) và retry an toàn.
* **Graceful Shutdown (Quan trọng cho Windows Service):** Bắt buộc xử lý tín hiệu dừng (`SIGINT`, `SIGTERM` từ NSSM/Windows) để đóng sạch toàn bộ `CameraReader`, xả hàng đợi `ocr_queue`, đóng kết nối SQLite WAL an toàn trước khi thoát tiến trình.
* **Bảo mật thông tin nhạy cảm:** Mọi URL RTSP chứa tài khoản/mật khẩu (`rtsp://user:pass@ip...`) bắt buộc phải được che (`rtsp://***:***@ip...`) khi ghi log hoặc trả về qua API.
* **Logging an toàn ổ đĩa:** Dùng `RotatingFileHandler` (mặc định `50MB x 5 file`), tách `logs/app.log` và `logs/error.log`.
* **Chuẩn Python 3.10+:** Type hints đầy đủ, tách biệt hoàn toàn các tầng `core`, `ai`, `rules`, `monitoring`, `db`, `api`.

---

## 2. Công nghệ sử dụng (Tech Stack)

| Thành phần | Công nghệ / Thư viện | Ghi chú triển khai |
| :--- | :--- | :--- |
| **Ngôn ngữ** | Python 3.10+ | Quản lý bằng `venv` |
| **AI Inference** | `onnxruntime-gpu` / `onnxruntime-directml` / `onnxruntime` | **1 Shared Session** cho `best.onnx` (YOLO11s), tự động fallback sang CPU |
| **Tracking** | ByteTrack / BoT-SORT | Khởi tạo **độc lập cho từng camera**, `track_buffer` 60–90 frames |
| **OCR Biển số** | PaddleOCR / FastPlateOCR (+ tùy chọn Plate Detector ONNX) | Hỗ trợ biển 1 dòng & 2 dòng VN, chạy Worker nền riêng |
| **Database** | SQLite 3 (`WAL` mode) | Local SQL, không cần cài server ngoài, hỗ trợ đọc/ghi đồng thời |
| **System Monitor** | `psutil` + `nvidia-ml-py` (`pynvml`) | Đọc CPU, RAM, Disk, Process RSS và GPU/VRAM (fallback an toàn nếu không có NVIDIA GPU) |
| **Backend & Stream** | FastAPI + Uvicorn + WebSockets / MJPEG | Bất đồng bộ, nhẹ, hỗ trợ nhiều client kết nối đồng thời |
| **Frontend** | HTML5 Canvas + Vanilla JS / React + Tailwind CSS | Giao diện Multi-Cam Grid, Canvas vẽ ROI/Line, Biểu đồ tài nguyên |
| **Windows Service** | NSSM (Non-Sucking Service Manager) | Đóng gói chạy nền, auto-start cùng Windows & auto-restart khi crash |

---

## 3. Kiến trúc hệ thống Multi-Cam & Mô hình đa luồng

Hệ thống chia thành **5 tầng xử lý độc lập**, giao tiếp qua `Queue` có giới hạn kích thước để đảm bảo không bao giờ nghẽn luồng chính hay tràn RAM/VRAM:

1. **Camera Manager & Reader Threads (`1 Thread / 1 Camera`):**
   * Đọc RTSP bằng OpenCV/FFmpeg, đẩy frame vào `frame_queue` (`maxsize=1`, luôn ghi đè frame cũ nếu AI xử lý chậm hơn để triệt tiêu độ trễ).
   * Tự động kết nối lại theo exponential backoff (`2s -> 5s -> 10s -> 30s`) khi rớt mạng.
   * **Chế độ Video Test (`.mp4`):** Tự động điều tiết tốc độ (FPS pacing) theo FPS gốc của video hoặc `fps_limit` để không làm drop hàng loạt frame; hỗ trợ `loop=True`.
   * Cập nhật trạng thái camera (`online`, `reconnecting`, `offline`, `current_fps`, `dropped_frames`).
2. **Shared AI Inference Engine + Per-Camera Context:**
   * **Shared Detector (`detector.py`):** Chỉ load **duy nhất 1 phiên `best.onnx`** vào VRAM/RAM, bảo vệ bằng Lock (hoặc gom micro-batch từ các camera đang hoạt động) để chạy Multi-Cam không bị tràn VRAM.
   * **Per-Camera Context (`camera_context.py`):** Mỗi `camera_id` sở hữu riêng 1 bộ: `Tracker` + `ClassSmoother` + `RuleEngine` + `StateManager` + `BestShotSelector`. Tuyệt đối không dùng chung trạng thái giữa các camera.
   * Khi xác nhận vi phạm: lưu 3 loại ảnh (`raw` gốc sạch, `annotated` có overlay, `crop` phương tiện), ghi bản ghi vào SQLite và đẩy job vào `ocr_queue`.
3. **Async OCR Worker Pool (`1–2 Threads`):**
   * Nhận job từ `ocr_queue` (`maxsize=100`, tự drop/log warning nếu quá tải để bảo vệ RAM) $\rightarrow$ detect/crop biển số $\rightarrow$ nhận diện ký tự (ghép dòng trên/dưới cho biển 2 dòng) $\rightarrow$ chuẩn hóa regex biển số VN.
   * Cập nhật `plate_number`, `ocr_conf`, `ocr_status` vào SQLite và đẩy sự kiện qua WebSocket tới Dashboard.
4. **System Resource Monitor Thread (`1 Background Thread`):**
   * Chạy định kỳ mỗi `interval_sec` (mặc định `2.0s`), thu thập tài nguyên toàn máy, tài nguyên riêng của tiến trình Windows Service, GPU/VRAM, dung lượng thư mục `data/evidence/` và telemetry của toàn bộ camera/queue.
   * Lưu vào Ring Buffer (mặc định `180` điểm = 6 phút gần nhất) để vẽ biểu đồ realtime và đẩy xuống Dashboard qua WebSocket `/ws/metrics`.
5. **FastAPI Server & Retention Scheduler:**
   * Phục vụ REST API, MJPEG Stream theo `camera_id`, WebSocket sự kiện & tài nguyên.
   * **Retention Worker:** Chạy lúc `02:00` hằng ngày, xóa thư mục ảnh cũ hơn `retention_days` (mặc định 30 ngày) và thực hiện bảo trì SQLite (`PRAGMA wal_checkpoint(TRUNCATE);`).

---

## 4. Cấu trúc thư mục dự án

```text
traffic_ai_service/
├── config/
│   └── config.yaml                 # Cấu hình động toàn hệ thống
├── models/
│   ├── best.onnx                   # Model YOLO11s 8 class đã huấn luyện
│   └── plate_detector.onnx         # (Tùy chọn) Model cắt biển số
├── data/
│   ├── traffic.db                  # SQLite Database (WAL mode)
│   └── evidence/                   # Ảnh bằng chứng: evidence/YYYY-MM-DD/cam_<id>/
├── logs/                           # app.log, error.log (RotatingFileHandler)
├── src/
│   ├── main.py                     # Entrypoint: FastAPI lifespan, khởi chạy & graceful shutdown
│   ├── core/
│   │   ├── config_loader.py        # Nạp, validate (Pydantic) & hot-reload config.yaml
│   │   ├── logger.py               # RotatingFileHandler + mask_rtsp_url()
│   │   ├── camera_reader.py        # Thread đọc RTSP/MP4 + auto-reconnect + FPS pacing
│   │   └── camera_manager.py       # Quản lý vòng đời Multi-Cam (add/remove/reload/health)
│   ├── ai/
│   │   ├── detector.py             # Shared ONNX Runtime YOLO11s (thread-safe)
│   │   ├── tracker.py              # ByteTrack / BoT-SORT wrapper (per-camera)
│   │   ├── class_smoother.py       # Majority voting theo track_id (chống nhảy truck/car)
│   │   ├── best_shot.py            # Chấm điểm chất lượng ảnh crop theo track_id
│   │   └── ocr_worker.py           # Async OCR Worker (biển 1 dòng & 2 dòng VN + regex)
│   ├── rules/
│   │   ├── geometry.py             # Chuẩn hóa tọa độ (0..1), point-in-polygon, line-crossing có hướng
│   │   ├── rule_engine.py          # Multi-ROI, phân làn, đếm giờ dừng đỗ, miễn trừ xe ưu tiên
│   │   └── state_manager.py        # Quản lý dwell_time bằng time.monotonic(), grace period & occlusion
│   ├── monitoring/
│   │   └── resource_monitor.py     # Thu thập CPU, RAM, GPU/VRAM, Disk, Process, FPS & Queue depth
│   ├── db/
│   │   ├── database.py             # Kết nối SQLite WAL + khởi tạo schema + thread-safe helper
│   │   └── repositories.py         # CRUD cameras, rules, violations + export Active Learning ZIP
│   └── api/
│       ├── routes_cameras.py       # CRUD camera + hot-reload
│       ├── routes_rules.py         # CRUD Multi-ROI & Line + hot-reload
│       ├── routes_violations.py    # Lọc, duyệt/báo sai, export YOLO feedback
│       ├── routes_system.py        # /api/health, /api/system/resources, /api/system/history
│       └── routes_stream.py        # MJPEG stream theo camera_id, WS /ws/events & WS /ws/metrics
├── web/                            # Giao diện Dashboard (HTML/CSS/JS hoặc React build)
├── scripts/
│   ├── install_service.bat         # Cài đặt Windows Service bằng NSSM
│   └── uninstall_service.bat       # Gỡ Windows Service
└── requirements.txt
```

---

## 5. Mẫu Cấu hình động (`config/config.yaml`)

Agent bắt buộc nạp toàn bộ tham số từ file này (validate bằng Pydantic khi khởi động):

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]

paths:
  db_path: "data/traffic.db"
  evidence_dir: "data/evidence"
  logs_dir: "logs"
  model_path: "models/best.onnx"
  plate_model_path: "models/plate_detector.onnx"

logging:
  level: "INFO"
  max_bytes: 52428800      # 50 MB
  backup_count: 5

ai:
  providers: ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
  imgsz: 640
  default_conf: 0.43       # Tối ưu giảm báo giả nền từ kết quả train
  iou_threshold: 0.45
  per_class_conf:
    0: 0.45                # car
    1: 0.40                # motorcycle
    2: 0.40                # bicycle
    3: 0.45                # bus
    4: 0.42                # truck
    5: 0.42                # tricycle
    6: 0.50                # ambulance
    7: 0.50                # fire_truck
  classes:
    0: "car"
    1: "motorcycle"
    2: "bicycle"
    3: "bus"
    4: "truck"
    5: "tricycle"
    6: "ambulance"
    7: "fire_truck"
  exempt_classes: [6, 7]   # Miễn trừ xe cứu thương & xe cứu hỏa

tracking:
  track_buffer: 75         # Giữ ID khi bị che khuất (~3-5s)
  class_smoothing_window: 8

rules:
  stationary_speed_px_per_sec: 15.0
  grace_period_sec: 1.5
  occlusion_iou_threshold: 0.65
  occlusion_max_gap_sec: 5.0
  wrong_lane_min_frames: 6
  violation_cooldown_sec: 60.0

ocr:
  workers: 1
  queue_maxsize: 100
  min_ocr_conf: 0.55
  vn_plate_regex: "^[0-9]{2}[A-Z]{1,2}[0-9]?-?[0-9]{4,5}$"

monitoring:
  interval_sec: 2.0
  history_size: 180        # Lưu 180 mẫu gần nhất (~6 phút)
  warn_cpu_percent: 85.0
  warn_ram_percent: 85.0
  warn_gpu_percent: 90.0
  warn_disk_free_gb: 5.0

retention:
  enabled: true
  run_at_hour: 2           # 02:00 hằng ngày
  retention_days: 30
```

---

## 6. Thiết kế cơ sở dữ liệu Local SQL (SQLite WAL)

Bắt buộc thực thi mỗi khi mở kết nối SQLite:
```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
```

### 6.1. Bảng `cameras`
```sql
CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source_url TEXT NOT NULL,         -- RTSP URL hoặc đường dẫn file MP4 test
    source_type TEXT DEFAULT 'rtsp',  -- 'rtsp' hoặc 'video_file'
    is_active INTEGER DEFAULT 1,
    fps_limit INTEGER DEFAULT 15,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 6.2. Bảng `rules`
```sql
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    rule_type TEXT NOT NULL,          -- 'no_parking', 'sidewalk_encroachment', 'lane_roi', 'lane_line'
    geometry_json TEXT NOT NULL,      -- JSON tọa độ chuẩn hóa [0.0..1.0]: [[x1, y1], [x2, y2], ...]
    direction TEXT DEFAULT 'both',    -- Dùng cho lane_line: 'both', 'left_to_right', 'right_to_left'
    allowed_classes_json TEXT,        -- JSON danh sách class_id được phép
    forbidden_classes_json TEXT,      -- JSON danh sách class_id bị cấm
    dwell_threshold_sec REAL DEFAULT 10.0,
    is_active INTEGER DEFAULT 1,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);
```

### 6.3. Bảng `violations` (Tích hợp đầy đủ Active Learning)
```sql
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    rule_id INTEGER,
    track_id INTEGER NOT NULL,
    violation_type TEXT NOT NULL,     -- 'no_parking', 'sidewalk_encroachment', 'wrong_lane'
    vehicle_class_id INTEGER NOT NULL,
    vehicle_class_name TEXT NOT NULL,
    detect_conf REAL NOT NULL,
    bbox_norm_json TEXT NOT NULL,     -- JSON [cx, cy, w, h] chuẩn hóa để xuất nhãn YOLO re-train
    dwell_duration_sec REAL DEFAULT 0.0,
    plate_number TEXT,
    ocr_conf REAL DEFAULT 0.0,
    ocr_status TEXT DEFAULT 'pending', -- 'pending', 'done', 'failed'
    raw_image_path TEXT NOT NULL,     -- Ảnh toàn cảnh GỐC (sạch, không overlay) để re-train
    annotated_image_path TEXT NOT NULL, -- Ảnh toàn cảnh CÓ overlay để hiển thị/đối soát
    crop_image_path TEXT NOT NULL,    -- Ảnh Best-shot phương tiện
    plate_image_path TEXT,            -- Ảnh crop biển số
    review_status TEXT DEFAULT 'pending', -- 'pending', 'confirmed', 'false_positive'
    corrected_class_name TEXT,
    corrected_plate TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_violations_cam_time ON violations(camera_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_violations_review ON violations(review_status);
```

---

## 7. Đặc tả chi tiết Logic AI & Rule Engine

### 7.1. Chống nhảy nhãn (`ClassSmoother`) & Miễn trừ xe ưu tiên
* Mỗi `track_id` duy trì một `deque(maxlen=class_smoothing_window)` lưu lịch sử `class_id` dự đoán. Nhãn cuối cùng của track là nhãn xuất hiện nhiều nhất (majority voting), giúp khắc phục triệt để hiện tượng xe tải nhẹ bị nhảy nhãn giữa `truck` và `car`.
* Nếu nhãn ổn định nằm trong `exempt_classes` (`ambulance`, `fire_truck`), bỏ qua toàn bộ kiểm tra vi phạm nhưng vẫn vẽ bounding box màu xanh riêng trên live stream.

### 7.2. Điểm xét vị trí (`bottom-center`) & Tọa độ chuẩn hóa
* Tọa độ ROI/Line luôn lưu trong `[0.0, 1.0]`, chỉ nhân với `(frame_w, frame_h)` khi suy luận.
* Điểm xét vị trí bắt buộc là **điểm giữa cạnh dưới của bounding box**:
  $$\text{anchor} = \left(\frac{x_1 + x_2}{2},\ y_2\right)$$
  để phản ánh đúng vị trí bánh xe chạm mặt đường, tránh hiện tượng nóc xe buýt/xe tải lẹm vào vỉa hè.

### 7.3. Logic phát hiện 3 nhóm vi phạm
1. **Dừng đỗ trái phép / Lấn chiếm vỉa hè (`no_parking`, `sidewalk_encroachment`):**
   * Đo thời gian bằng **`time.monotonic()`** (tuyệt đối không đếm bằng số frame).
   * Áp dụng **Grace Period (`1.5s`)**: nếu `anchor` rung nhẹ ra ngoài biên ROI trong thời gian ngắn rồi quay lại, không reset `dwell_time`.
   * **Chống reset khi bị che khuất:** Khi một `track_id` đang đứng yên biến mất, lưu trạng thái vào bộ đệm tạm trong `occlusion_max_gap_sec` (`5.0s`). Nếu xuất hiện `track_id` mới cùng class có $\text{IoU} \ge 0.65$ với vị trí cũ, kế thừa lại `dwell_time`.
   * Khi `dwell_time >= dwell_threshold_sec`, kích hoạt vi phạm **1 lần duy nhất** cho track đó (kèm `violation_cooldown_sec`).
2. **Đi sai làn / Lấn làn (`lane_roi`, `lane_line`):**
   * `lane_roi`: `anchor` nằm trong vùng cấm liên tục tối thiểu `wrong_lane_min_frames` (`6` frame).
   * `lane_line`: Kiểm tra đoạn thẳng nối `anchor(t-1)` $\rightarrow$ `anchor(t)` cắt vạch cấm theo đúng `direction` (`both`, `left_to_right`, `right_to_left`) bằng tích có hướng 2D.

### 7.4. Chấm điểm Best-shot & OCR bất đồng bộ
* Công thức chấm điểm chất lượng ảnh crop cho từng `track_id`:
  $$\text{Score} = w_1 \cdot \text{Area\_Norm} + w_2 \cdot \text{Sharpness\_Laplacian} + w_3 \cdot \text{Detect\_Conf} - w_4 \cdot \text{Edge\_Penalty}$$
* OCR Worker đọc ảnh Best-shot, hỗ trợ biển 1 dòng và biển 2 dòng (sắp xếp text box theo trục $Y$ rồi trục $X$ để ghép dòng trên + dòng dưới), làm sạch ký tự và kiểm tra theo `vn_plate_regex`.

---

## 8. Đặc tả Giám sát Tài nguyên Hệ thống (System Resource Monitoring)

Module `src/monitoring/resource_monitor.py` chạy nền liên tục để theo dõi sức khỏe máy chủ Windows đang chạy Service và hiển thị trực quan lên Web Dashboard.

### 8.1. Các nhóm chỉ số bắt buộc thu thập
1. **Tài nguyên toàn máy (Host System — qua `psutil`):**
   * `cpu_percent`: Tổng % CPU toàn hệ thống.
   * `ram_used_gb`, `ram_total_gb`, `ram_percent`: Bộ nhớ RAM đang dùng / tổng RAM.
   * `disk_used_gb`, `disk_free_gb`, `disk_percent`: Dung lượng phân vùng chứa thư mục `data/`.
   * `evidence_dir_size_mb`: Tổng dung lượng thực tế thư mục `data/evidence/` (tính định kỳ mỗi 60 giây để tránh tốn I/O).
2. **Tài nguyên riêng của Windows Service (Current Process — qua `psutil.Process(os.getpid())`):**
   * `service_cpu_percent`: % CPU riêng service đang tiêu thụ.
   * `service_ram_mb`: RAM (RSS) riêng service đang chiếm giữ.
   * `service_threads`: Số luồng (threads) đang hoạt động.
   * `uptime_sec`: Thời gian chạy liên tục của service.
3. **Tài nguyên GPU / VRAM (qua `pynvml`, tự động fallback nếu không có GPU NVIDIA):**
   * `gpu_available`: `true` / `false`.
   * `gpu_name`: Tên card (ví dụ: `NVIDIA GeForce GTX 1660 Ti` / `RTX 3060`).
   * `gpu_util_percent`: % tải nhân GPU.
   * `vram_used_mb`, `vram_total_mb`, `vram_percent`: VRAM đang dùng / tổng VRAM.
   * `gpu_temp_c`: Nhiệt độ GPU (°C).
   * `inference_provider`: Provider ONNX đang thực tế chạy (`CUDAExecutionProvider` / `DmlExecutionProvider` / `CPUExecutionProvider`).
4. **Hiệu năng Pipeline AI (Pipeline Telemetry):**
   * `active_cameras` / `total_cameras`.
   * `total_fps`: Tổng FPS xử lý của tất cả camera.
   * `ocr_queue_size` / `ocr_queue_maxsize`: Độ sâu hàng đợi OCR.
   * Danh sách chi tiết từng camera: `camera_id`, `name`, `status` (`online`/`reconnecting`/`offline`), `fps`, `active_tracks`.
5. **Cảnh báo tài nguyên (`alerts`):**
   * Tự động sinh cảnh báo (`warning`) nếu vượt ngưỡng cấu hình trong `config.yaml` (`warn_cpu_percent`, `warn_ram_percent`, `warn_gpu_percent`, `warn_disk_free_gb`).

### 8.2. Cấu trúc JSON mẫu cho `/api/system/resources` & `WS /ws/metrics`
```json
{
  "timestamp": "2026-11-19T23:30:00+07:00",
  "uptime_sec": 86420,
  "host": {
    "cpu_percent": 34.5,
    "ram_used_gb": 8.4,
    "ram_total_gb": 16.0,
    "ram_percent": 52.5,
    "disk_free_gb": 142.8,
    "disk_percent": 68.2,
    "evidence_dir_size_mb": 1240.5
  },
  "service_process": {
    "cpu_percent": 18.2,
    "ram_mb": 1450.3,
    "threads": 14
  },
  "gpu": {
    "available": true,
    "name": "NVIDIA GeForce GTX 1660 Ti",
    "provider": "CUDAExecutionProvider",
    "util_percent": 42.0,
    "vram_used_mb": 1820,
    "vram_total_mb": 6144,
    "vram_percent": 29.6,
    "temp_c": 61
  },
  "pipeline": {
    "active_cameras": 2,
    "total_cameras": 2,
    "total_fps": 28.4,
    "ocr_queue_size": 1,
    "ocr_queue_maxsize": 100,
    "cameras": [
      {"id": 1, "name": "Cam Ngã Tư A", "status": "online", "fps": 14.5, "active_tracks": 12},
      {"id": 2, "name": "Cam Vỉa Hè B", "status": "online", "fps": 13.9, "active_tracks": 7}
    ]
  },
  "alerts": []
}
```

---

## 9. Đặc tả API & Giao diện Web Dashboard

### 9.1. REST API & WebSocket Endpoints
* **Hệ thống & Tài nguyên:**
  * `GET /api/health`: Kiểm tra nhanh service còn sống (`200 OK`).
  * `GET /api/system/resources`: Lấy snapshot tài nguyên hiện tại.
  * `GET /api/system/history`: Lấy chuỗi lịch sử tài nguyên từ Ring Buffer (để vẽ biểu đồ đường CPU/RAM/GPU/FPS).
  * `WS /ws/metrics`: Push JSON tài nguyên định kỳ mỗi `2s` xuống Dashboard.
* **Camera & Luật (Hot-reload):**
  * `GET/POST/PUT/DELETE /api/cameras`: Quản lý Multi-Cam (ẩn mật khẩu RTSP trong response), tự động hot-reload `CameraManager`.
  * `GET /api/cameras/{id}/snapshot`: Lấy 1 ảnh tĩnh (JPEG) hiện tại của camera để làm nền vẽ ROI/Line.
  * `GET/POST/PUT/DELETE /api/cameras/{id}/rules`: Lưu/sửa Multi-ROI & Line, tự động hot-reload `RuleEngine` của camera đó.
* **Vi phạm & Active Learning:**
  * `GET /api/violations`: Phân trang, lọc theo ngày, `camera_id`, `violation_type`, `vehicle_class_name`, `review_status`.
  * `PATCH /api/violations/{id}/review`: Duyệt (`confirmed`), báo sai (`false_positive`), sửa `corrected_class_name` / `corrected_plate`.
  * `GET /api/violations/export-feedback`: Đóng gói file ZIP gồm ảnh `raw_image_path` + file `.txt` nhãn YOLO (dựa trên `bbox_norm_json` và nhãn đã sửa) để re-train.
* **Stream & Sự kiện:**
  * `GET /api/stream/{camera_id}`: MJPEG stream có overlay (bbox, track_id, ROI/Line, đồng hồ đếm giây dừng đỗ).
  * `WS /ws/events`: Push realtime sự kiện vi phạm mới và kết quả OCR vừa cập nhật.

### 9.2. Yêu cầu Giao diện Web Dashboard (`web/`)
1. **Thanh trạng thái Tài nguyên (System Monitor Bar / Tab):**
   * Hiển thị trực quan dạng thẻ (Cards) + biểu đồ đường realtime (Chart.js / Lightweight SVG): **CPU % (Host & Service)**, **RAM (GB & Service MB)**, **GPU/VRAM % & Nhiệt độ**, **Dung lượng ổ đĩa trống / Evidence**, **Tổng FPS & OCR Queue**.
   * Đổi màu cảnh báo (vàng/đỏ) khi bất kỳ chỉ số nào vượt ngưỡng `warn_*`.
2. **Màn hình Multi-Cam Live Grid:**
   * Chuyển đổi linh hoạt chế độ lưới (`1x1`, `2x2`, `3x3`), hiển thị badge trạng thái (`Online` / `Reconnecting` / `Offline`) và FPS từng camera.
3. **Trình vẽ Multi-ROI & Line (Canvas Editor):**
   * Đóng băng khung hình snapshot, click chuột để vẽ đa giác (ROI) hoặc đoạn thẳng (Line), chọn loại luật, chiều cắt, danh sách class áp dụng và ngưỡng giây dừng đỗ.
4. **Bảng quản lý Vi phạm:**
   * Cập nhật realtime, cho phép phóng to ảnh `annotated`, `crop`, `plate`, xem kết quả OCR và thao tác nhanh **Xác nhận / Báo sai / Sửa biển số**.

---

## 10. Triển khai & Vận hành Windows Service (NSSM)

Cung cấp 2 script trong `scripts/`:
* **`scripts/install_service.bat`:**
  * Kiểm tra quyền Administrator và sự tồn tại của `nssm.exe`.
  * Đăng ký service tên `TrafficAIService` trỏ tới:
    * `Application`: `%PROJECT_DIR%\venv\Scripts\python.exe`
    * `AppParameters`: `-m uvicorn src.main:app --host 0.0.0.0 --port 8000`
    * `AppDirectory`: `%PROJECT_DIR%`
  * Cấu hình tự động khởi động cùng Windows (`SERVICE_AUTO_START`), `AppRestartDelay 5000`, `AppStopMethodConsole 10000` (cho phép graceful shutdown 10s), và chuyển hướng `AppStdout`/`AppStderr` vào `logs/service_stdout.log` & `logs/service_stderr.log`.
* **`scripts/uninstall_service.bat`:** Dừng service và gỡ đăng ký sạch sẽ.

---

## 11. Lộ trình triển khai & Tiêu chí nghiệm thu cho Coding Agent

Agent triển khai cuốn chiếu theo **4 giai đoạn** và tự kiểm chứng tiêu chí nghiệm thu của từng giai đoạn trước khi sang bước tiếp theo:

1. **Giai đoạn 1 — Core, DB, Shared AI & Resource Monitor:**
   * Dựng `config_loader.py`, `logger.py` (kèm `mask_rtsp_url`), `database.py` (SQLite WAL), `repositories.py`, `detector.py` (Shared ONNX Runtime) và `resource_monitor.py`.
   * *Nghiệm thu:* Load thành công `best.onnx`, ghi/đọc SQLite WAL không lock, `/api/system/resources` trả đủ thông số CPU/RAM/GPU/Disk/Process.
2. **Giai đoạn 2 — Multi-Cam Pipeline & Rule Engine:**
   * Dựng `camera_reader.py` (RTSP + MP4 pacing), `camera_manager.py`, `tracker.py`, `class_smoother.py`, `geometry.py`, `rule_engine.py`, `state_manager.py`.
   * *Nghiệm thu:* Chạy song song 2 file video MP4 test cùng lúc; `track_id` độc lập giữa 2 cam; xác nhận đúng logic `bottom-center`, đếm giờ bằng `time.monotonic()`, không reset khi bị che khuất ngắn, và miễn trừ `ambulance`/`fire_truck`.
3. **Giai đoạn 3 — Best-shot, Evidence & Async OCR:**
   * Dựng `best_shot.py`, lưu đủ 3 ảnh (`raw`, `annotated`, `crop`) + `bbox_norm_json`, tích hợp `ocr_worker.py`.
   * *Nghiệm thu:* Luồng camera chính không bị tụt FPS khi OCR đang chạy; đọc và chuẩn hóa được biển số 1 dòng & 2 dòng; cập nhật ngược vào SQLite thành công.
4. **Giai đoạn 4 — FastAPI, Web Dashboard & Windows Service (NSSM):**
   * Dựng toàn bộ REST API, MJPEG Stream, WebSocket (`/ws/events`, `/ws/metrics`), giao diện Dashboard (Live Grid + Canvas vẽ ROI/Line + Bảng duyệt + Biểu đồ tài nguyên) và script NSSM.
   * *Nghiệm thu:* Vẽ ROI trên trình duyệt có hiệu lực ngay (hot-reload); bấm Export Feedback tải được ZIP ảnh `raw` + nhãn `.txt`; cài đặt và chạy ổn định dưới dạng Windows Service, dừng service bằng `nssm stop` thoát sạch không treo tiến trình.
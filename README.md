# CCTV Traffic AI Service (Traffic Lane Monitor)

Hệ thống giám sát vi phạm giao thông thông minh thời gian thực đa camera (Multi-Camera Surveillance), tích hợp nhận diện phương tiện AI YOLO11s, OCR biển số xe bất đồng bộ, lọc trùng lặp đa tầng và giao diện điều hành tương tác trực tiếp.

---

## 🌟 Tính Năng Nổi Bật

1. **Giám Sát Đa Camera & Bố Cục Windows 11 Snap Layouts**:
   - Quản lý đa luồng camera RTSP hoặc file video MP4.
   - Bố cục chia màn hình thông minh phong cách Windows 11:
     - Chia đôi 50/50
     - 1 Cam lớn (Tiêu điểm 70%) + 1 hoặc 2 Cam phụ
     - Lưới 4 Cams (2x2)
     - Toàn màn hình (1x1)
     - ✨ Tự Động Vừa Khít Màn Hình (Auto-Fit cho mọi số lượng camera)
   - Click đúp hoặc bấm `⛶ Phóng to` bất kỳ camera nào để xem toàn màn hình; nút `🔙 Trở về lưới đa cam` để quay lại.
   - Tab quản lý camera phong cách Google Chrome (đổi tên bằng icon `⚙️`, xóa bằng `✕`, thêm camera bằng `+`).

2. **Vẽ & Cấu Hình Luật Trực Tiếp Trên Màn Hình (Direct On-Screen Drawing)**:
   - Vẽ trực tiếp trên luồng video camera, loại bỏ hoàn toàn cửa sổ modal che màn hình.
   - Gán luật chính xác theo từng camera được chọn (`khi vẽ cho camera nào thì rule thuộc camera đó`).
   - Hỗ trợ các loại luật: Dừng đỗ trái phép (`no_parking`), Lấn chiếm vỉa hè (`sidewalk_encroachment`), Đi sai làn (`lane_roi`), Đè/Cắt vạch cấm (`lane_line`).
   - Hot-reload luật tức thì vào luồng xử lý AI mà không cần khởi động lại dịch vụ.

3. **Khử Trùng Lặp Cảnh Báo Đa Lớp (Alert Deduplication & Session Management)**:
   - **Lớp 1 (Plate Cooldown)**: Khử trùng theo Biển số + Vùng ROI + Loại vi phạm, hỗ trợ so khớp mờ Levenshtein distance `<= 1`.
   - **Lớp 2 (Spatial / IoU Lock)**: Khóa không gian theo vị trí xe đứng yên khi biển số bị mờ hoặc che khuất (`IoU >= 0.65`).
   - **Lớp 3 (Event Session)**: Quản lý vòng đời phiên vi phạm (`ACTIVE` / `CLOSED`), chỉ đóng phiên khi xe rời khỏi vùng ROI quá thời gian ân hạn (`exit_grace_sec`).
   - **Lớp 4 (UI Real-time Aggregate)**: Tự động cộng dồn thời gian vi phạm (`duration`) và số lần lặp lại (`repeat_count`) theo thời gian thực.

4. **Tối Ưu Hóa CPU & Tương Thích Triệt Để Với Ubuntu Linux**:
   - Cơ chế chạy an toàn (Safe CPU Mode): Giới hạn luồng tính toán song song, tránh nghẽn CPU và chống crash trên hệ thống không có GPU rời.
   - Tự động phát hiện và chuyển đổi giữa `CPUExecutionProvider` và DirectML.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Trên Windows
```powershell
# Tạo môi trường ảo
python -m venv venv
.\venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

# Khởi chạy dịch vụ
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 2. Trên Ubuntu Linux (Không card rời, CPU 4 nhân / 8GB RAM)
```bash
# Cập nhật và cài đặt thư viện hệ thống
sudo apt update && sudo apt install -y python3-venv python3-pip libgl1-mesa-glx libglib2.0-0

# Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Cài đặt các thư viện tối ưu cho Linux CPU (Headless OpenCV & CPU ONNX Runtime)
pip install -r requirements-linux.txt

# Khởi chạy dịch vụ
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Truy cập giao diện Web Dashboard tại: `http://localhost:8000`

---

## 📁 Cấu Trúc Dự Án
```
Traffic_Lane_Monitor/
├── config/                  # File cấu hình YAML hệ thống (config.yaml)
├── data/                    # Cơ sở dữ liệu SQLite & dữ liệu vi phạm
├── models/                  # Trọng số mô hình AI (YOLO11s ONNX & PaddleOCR)
├── src/
│   ├── ai/                  # AI Detector, Tracker, Async OCR Worker
│   ├── api/                 # FastAPI REST & WebSocket endpoints
│   ├── core/                # Camera Manager, Config Loader, Deduplication
│   ├── db/                  # SQLite Database & Repositories
│   ├── monitoring/          # Giám sát tài nguyên CPU, RAM, GPU
│   └── rules/               # Rule Engine hình học không gian & xử lý ROI
├── web/                     # Web Dashboard HTML/JS/Tailwind CSS
├── requirements.txt         # Thư viện cho môi trường Windows (DirectML)
└── requirements-linux.txt   # Thư viện cho môi trường Linux CPU (Headless)
```

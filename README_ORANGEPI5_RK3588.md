# Orange Pi 5 Plus (RK3588) DAMO-YOLO 部署與多 NPU 調用指南

本目錄包含在 **RK3588 (Orange Pi 5 Plus)** 上運行已轉換完成的 DAMO-YOLO RKNN 模型的完整推論程式與後處理模組。

支援動態選擇 **1 ~ 3 個 NPU 核心**，兼顧低延遲（單張多核協作）與極致吞吐量（多執行緒管線平行）。

---

## 目錄檔案說明

- **`my_damoyolo_T_640_rk3588.rknn`**：由 Colab 匯出並完成 INT8 量化的 RKNN 模型。
- **`model_info.json`**：模型元數據（輸入尺寸 640、DFL reg_max 16、strides [8, 16, 32]、21 個類別名稱）。
- **`labels.txt`**：21 個自訂類別清單。
- **`damo_rknn_utils.py`**：純 NumPy + OpenCV 的前處理（縮放/RGB）、解碼（DFL Softmax）、類別感知 NMS 與繪圖模組。
- **`board_infer.py`**：板端推論核心程式（支援 1~3 核心調用、多執行緒管線、FPS 計算、HUD 顯示、影片儲存與 Benchmark）。

---

## 1. Orange Pi 5 Plus 環境準備

在板端終端機（Ubuntu 22.04 / Debian 11/12）執行以下步驟：

### (1) 檢查 RK3588 NPU 驅動狀態
```bash
dmesg | grep -i rknpu
```
若正常會看到類似 `RKNPU: Initialized` 的核心日誌。

### (2) 安裝基礎依賴
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv python3-numpy ffmpeg
```

### (3) 安裝 RKNN-Toolkit-Lite2 Runtime
依據你的 Python 版本（例如 Python 3.10）：
```bash
# 下載官方 rknn-toolkit2 倉庫
git clone --depth=1 https://github.com/airockchip/rknn-toolkit2.git

# 安裝對應 Python 版本的 whl (以 Python 3.10 aarch64 為例)
cd rknn-toolkit2/rknn-toolkit-lite2/packages/
pip3 install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl
```
> **提示**：可使用 `python3 --version` 查看當前 Python 版本，並安裝對應的 `cp38`、`cp39`、`cp310` 或 `cp311` wheel 檔。

---

## 2. 複製資料夾至板端

在你的電腦（PC）上使用 SCP 將整個目錄傳輸到 Orange Pi 5 Plus：
```bash
scp -r my_damoyolo_T_export orangepi@<ORANGE_PI_IP>:~/
```
連線到板端後進入該目錄：
```bash
cd ~/my_damoyolo_T_export
```

---

## 3. 執行指令與參數說明

### (A) 單張圖片推論與 Benchmark 測試

#### 1. 調用 1 個核心推論 (預設 Core 0)
```bash
python3 board_infer.py --source test.jpg --cores 1 --save result_1core.jpg
```

#### 2. 調用 3 個核心推論 (Core 0 + Core 1 + Core 2 協作)
```bash
python3 board_infer.py --source test.jpg --cores 3 --save result_3core.jpg
```

#### 3. 測試延遲基準（Benchmark 模式，自動跑 100 次並統計延遲與等效 FPS）
```bash
# 測試 1 核心延遲
python3 board_infer.py --source test.jpg --cores 1 --benchmark

# 測試 3 核心多核協作延遲
python3 board_infer.py --source test.jpg --cores 3 --benchmark
```

---

### (B) 即時攝影機 / 串流 / 影片推論（多執行緒平行管線）

在影片或串流模式下，程式會自動啟用 **多執行緒管線（Worker Pool）**：
- `--cores 1`：建立 1 個 Worker 執行緒綁定 Core 0。
- `--cores 2`：建立 2 個 Worker 執行緒分別綁定 Core 0 與 Core 1。
- `--cores 3`：建立 3 個 Worker 執行緒分別獨立綁定 Core 0、Core 1、Core 2，**達到全負載最高 FPS**！

#### 1. USB 攝影機即時推論 (3 核心全開 + 螢幕顯示)
```bash
python3 board_infer.py --source 0 --cores 3 --show
```
*(若攝影機為 `/dev/video11`，傳入 `--source /dev/video11`)*

#### 2. 本地影片推論並輸出帶標註的 MP4 影片
```bash
python3 board_infer.py --source demo.mp4 --cores 3 --save output.mp4
```

#### 3. 指定單一特定核心（例如保留 Core 0 給其他程式，僅用 Core 2）
```bash
python3 board_infer.py --source 0 --core 2 --show
```

---

## 4. 參數詳細清單

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--source` | *(必填)* | 圖片路徑、影片路徑、攝影機編號（如 `0`）或裝置路徑（如 `/dev/video0`）、RTSP 串流 |
| `--cores` | `3` | 調用 NPU 核心數量：`1`、`2`、`3` |
| `--core` | `None` | 強制綁定單一特定核心：`0`、`1`、`2`（指定時會覆蓋 `--cores`） |
| `--conf` | `0.3` | 目標信心度閾值 |
| `--iou` | `0.45` | NMS 重疊 IoU 閾值 |
| `--save` | `None` | 儲存輸出標註檔案路徑（如 `result.jpg` 或 `output.mp4`） |
| `--show` | `False` | 是否開啟視窗即時預覽（按鍵盤 `q` 或 `ESC` 結束） |
| `--benchmark` | `False` | 啟動基準測試模式，輸出前處理/NPU/後處理精準耗時 |
| `--bench_loops`| `100` | Benchmark 測試循環次數 |
| `--warmup` | `10` | Benchmark 前預熱次數 |

---

## 5. 效能預估參考 (DAMO-YOLO-Tiny, 640x640, RK3588)

| 模式 | NPU 核心配置 | NPU 單張延遲 | 影片/串流吞吐量 |
| :--- | :--- | :--- | :--- |
| 單核心 | `--cores 1` (Core 0) | 約 8 ~ 11 ms | 約 **30 ~ 35 FPS** |
| 雙核心 | `--cores 2` (Core 0 + 1) | 約 7 ~ 9 ms | 約 **60 ~ 70 FPS** |
| **三核心** | **`--cores 3` (Core 0 + 1 + 2)** | **約 6 ~ 8 ms** | **約 85 ~ 100+ FPS** |

> **備註**：影片/串流模式下使用多執行緒（3 個 Worker 分工處理），NPU 核心完全獨立平行運算，整體 FPS 會接近線性倍增！

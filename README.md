# PyTorch Job Manager

一個用於在 Kubeflow 上管理 PyTorch 分散式訓練任務的 Python 套件。

## 功能特色

- 簡化 PyTorch 分散式訓練任務的建立和管理
- 支援 Master-Worker 架構
- 整合 Gradio UI 介面，方便監控訓練進度
- 自動處理 Kubernetes 資源配置
- 支援 GPU 資源分配
- 即時日誌顯示

## 安裝

使用 pip 安裝：

```bash
pip install pytorch-job-manager
```

從原始碼安裝：

```bash
git clone https://github.com/yourusername/pytorch-job-manager.git
cd pytorch-job-manager
pip install -e .
```

## 快速開始

### 基本使用

```python
from pytorch_job_manager import PyTorchJobManager

# 建立 PyTorchJobManager 實例
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="my-training-job",
    image="your-docker-image:tag",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0
)

# 建立訓練任務
manager.create_pytorch_job()

# 使用 Gradio UI 監控訓練進度
manager.wait_logs(ui=True)
```

### 不使用 UI 的方式

```python
# 建立訓練任務
manager.create_pytorch_job()

# 在終端機顯示日誌
manager.wait_logs(ui=False)

# 檢查任務是否成功
if manager.is_job_succeeded():
    print("訓練完成！")

# 刪除任務
manager.delete_job()
```

## 參數說明

- `namespace`: Kubernetes 命名空間
- `name`: 訓練任務名稱
- `image`: Docker 映像檔
- `command`: 執行命令（列表格式）
- `working_dir`: 工作目錄
- `cpu`: CPU 資源請求（預設: "4"）
- `memory`: 記憶體資源請求（預設: "8Gi"）
- `gpu`: GPU 資源請求（預設: "1"）
- `worker_replicas`: Worker 節點數量（預設: 4）

## 方法說明

### `create_pytorch_job()`
建立 PyTorch 分散式訓練任務。

### `get_job()`
取得當前任務的狀態資訊。

### `wait_for_job(wait_timeout=900)`
等待任務完成，可設定超時時間（秒）。

### `is_job_succeeded()`
檢查任務是否成功完成。

### `get_job_logs()`
取得任務的日誌。

### `delete_job()`
刪除訓練任務。

### `wait_logs(ui=True)`
等待並顯示訓練日誌。
- `ui=True`: 使用 Gradio UI 介面
- `ui=False`: 在終端機顯示日誌

## 需求

- Python >= 3.8
- gradio >= 3.0.0
- kubernetes >= 20.0.0
- kubeflow-training >= 1.7.0

## TODO
* 支援使用 affinity 選擇 GPU


## 授權

MIT License

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 作者

Yu-Chung Wang - wycca1@gmail.com

## 更新日誌

### 0.1.0 (2024-02-14)
- 初始版本發布
- 支援基本的 PyTorch 分散式訓練任務管理
- 整合 Gradio UI 介面

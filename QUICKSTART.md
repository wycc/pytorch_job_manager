# 快速開始指南

## 安裝套件

### 方法 1: 從本地 wheel 檔案安裝

```bash
pip install dist/pytorch_job_manager-0.1.0-py3-none-any.whl
```

### 方法 2: 從本地 tar.gz 檔案安裝

```bash
pip install dist/pytorch-job-manager-0.1.0.tar.gz
```

### 方法 3: 開發模式安裝（可編輯）

```bash
pip install -e .
```

## 驗證安裝

```bash
python -c "from pytorch_job_manager import PyTorchJobManager; print('安裝成功！版本:', PyTorchJobManager.__module__)"
```

## 基本使用

### 1. 匯入套件

```python
from pytorch_job_manager import PyTorchJobManager
```

### 2. 建立管理器實例

```python
manager = PyTorchJobManager(
    namespace="kubeflow-user",           # Kubernetes 命名空間
    name="my-training-job",              # 訓練任務名稱
    image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",  # Docker 映像
    command=["python", "train.py"],      # 執行命令
    working_dir="/home/jovyan/work",     # 工作目錄
    cpu="4",                             # CPU 資源
    memory="8Gi",                        # 記憶體資源
    gpu="1",                             # GPU 資源
    worker_replicas=2                    # Worker 節點數量
)
```

### 3. 啟動訓練任務

```python
# 建立訓練任務
manager.create_pytorch_job()

# 使用 Gradio UI 監控（推薦）
manager.wait_logs(ui=True)
```

### 4. 其他常用操作

```python
# 檢查任務狀態
status = manager.get_job()
print(status)

# 檢查是否完成
if manager.is_job_succeeded():
    print("訓練完成！")

# 取得日誌
logs = manager.get_job_logs()
print(logs)

# 刪除任務
manager.delete_job()
```

## 完整範例

```python
from pytorch_job_manager import PyTorchJobManager

# 建立管理器
manager = PyTorchJobManager(
    namespace="kubeflow-user",
    name="mnist-training",
    command=["python", "mnist_ddp.py", "--epochs", "10"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=2
)

# 啟動訓練並監控
try:
    manager.create_pytorch_job()
    manager.wait_logs(ui=True)  # 開啟 Gradio UI
    
    if manager.is_job_succeeded():
        print("✅ 訓練成功完成！")
    else:
        print("❌ 訓練失敗")
        
except Exception as e:
    print(f"發生錯誤: {e}")
    manager.delete_job()
```

## 單節點訓練（無分散式）

如果不需要分散式訓練，設定 `worker_replicas=0`：

```python
manager = PyTorchJobManager(
    namespace="kubeflow-user",
    name="single-node-training",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="8",
    memory="16Gi",
    gpu="2",
    worker_replicas=0  # 單節點訓練
)
```

## 在 Jupyter Notebook 中使用

```python
# 在 Notebook 中
from pytorch_job_manager import PyTorchJobManager

manager = PyTorchJobManager(
    namespace="kubeflow-user",
    name="notebook-training",
    command=["python", "train.py"],
    worker_replicas=2
)

manager.create_pytorch_job()
manager.wait_logs(ui=True)  # 會在 Notebook 中顯示 Gradio UI
```

## 常見問題

### Q: 如何指定自己的 Docker 映像？

A: 在建立 `PyTorchJobManager` 時設定 `image` 參數：

```python
manager = PyTorchJobManager(
    image="myregistry/my-pytorch-image:v1.0",
    ...
)
```

### Q: 如何調整資源配置？

A: 使用 `cpu`、`memory`、`gpu` 參數：

```python
manager = PyTorchJobManager(
    cpu="8",        # 8 個 CPU 核心
    memory="16Gi",  # 16GB 記憶體
    gpu="2",        # 2 個 GPU
    ...
)
```

### Q: 如何在訓練完成後自動刪除任務？

A: 使用 try-finally 結構：

```python
try:
    manager.create_pytorch_job()
    manager.wait_logs(ui=False)
finally:
    manager.delete_job()
```

## 更多資訊

- 完整文件：請參閱 [README.md](README.md)
- 安裝指南：請參閱 [INSTALL.md](INSTALL.md)
- 範例程式：請參閱 [examples/](examples/) 目錄
- 更新日誌：請參閱 [CHANGELOG.md](CHANGELOG.md)

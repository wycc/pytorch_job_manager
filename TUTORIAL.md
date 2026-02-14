# PyTorch Job Manager 使用教學

## 目錄
1. [準備 Kubeflow Notebook 環境](#1-準備-kubeflow-notebook-環境)
2. [不同節點指定方式的範例](#2-不同節點指定方式的範例)
3. [使用 Gradio UI 查看執行結果](#3-使用-gradio-ui-查看執行結果)
4. [使用命令列查看執行結果](#4-使用命令列查看執行結果)
5. [使用 kubectl logs 查看執行結果](#5-使用-kubectl-logs-查看執行結果)
6. [多節點分散式訓練範例](#6-多節點分散式訓練範例)

---

## 1. 準備 Kubeflow Notebook 環境

### 1.1 安裝 PyTorch Job Manager

在 Kubeflow Notebook 中執行以下命令安裝套件：

```bash
pip install pytorch-job-manager
```

或從原始碼安裝：

```bash
git clone https://github.com/yourusername/pytorch-job-manager.git
cd pytorch-job-manager
pip install -e .
```

### 1.2 準備訓練腳本

建立您的訓練腳本，例如 `train.py`：

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os

def main():
    print(f"Starting training on rank {os.environ.get('RANK', '0')}")
    print(f"World size: {os.environ.get('WORLD_SIZE', '1')}")
    
    # 您的訓練程式碼
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        # 訓練邏輯
        print(f"Epoch {epoch + 1}/10 completed")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
```

### 1.3 準備執行腳本（選用）

建立 shell 腳本 `run_training.sh`：

```bash
#!/bin/bash
cd /home/jovyan/work
python train.py
```

賦予執行權限：

```bash
chmod +x run_training.sh
```

### 1.4 確認檔案結構

確保您的 Notebook 工作目錄包含所有必要檔案：

```
/home/jovyan/
├── work/
│   ├── train.py          # 訓練腳本
│   ├── run_training.sh   # 執行腳本（選用）
│   ├── data/             # 訓練資料
│   └── models/           # 模型檔案
└── job/
    └── pytorch_job.py    # Job 管理腳本
```

---

## 2. 不同節點指定方式的範例

### 2.1 範例 1：不指定節點（自動分配）

讓 Kubernetes 自動選擇可用的節點：

```python
from pytorch_job_manager import PyTorchJobManager

# 建立 Job Manager（不指定 affinity）
manager = PyTorchJobManager(
    namespace="your-namespace",      # 替換為您的 namespace
    name="auto-assign-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0                # 單節點訓練
)

# 建立並執行任務
manager.create_pytorch_job()
manager.wait_logs(ui=False)

# 檢查是否成功
if manager.is_job_succeeded():
    print("訓練完成！")

# 清理資源
manager.delete_job()
```

### 2.2 範例 2：使用主機名稱指定節點

將任務指定到特定的節點（例如：`rtxpro6000`）：

```python
from pytorch_job_manager import PyTorchJobManager

# 使用 hostname 指定節點
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="hostname-affinity-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0,
    affinity="hostname:rtxpro6000"   # 指定到 rtxpro6000 節點
)

# 建立並執行任務
manager.create_pytorch_job()
manager.wait_logs(ui=False)

# 檢查是否成功
if manager.is_job_succeeded():
    print("訓練完成！")

# 清理資源
manager.delete_job()
```

**如何查詢可用的節點名稱：**

```bash
kubectl get nodes
```

### 2.3 範例 3：使用 GPU 類別指定節點

將任務指定到具有特定 GPU 型號的節點：

```python
from pytorch_job_manager import PyTorchJobManager

# 使用 GPU 產品型號指定節點
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="gpu-affinity-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0,
    affinity="gpu.product:NVIDIA-RTX-A5000"  # 指定 RTX A5000 GPU
)

# 建立並執行任務
manager.create_pytorch_job()
manager.wait_logs(ui=False)

# 檢查是否成功
if manager.is_job_succeeded():
    print("訓練完成！")

# 清理資源
manager.delete_job()
```

**常見的 GPU 型號：**
- `NVIDIA-RTX-A5000`
- `Tesla-V100-SXM2-16GB`
- `NVIDIA-GeForce-RTX-3090`
- `NVIDIA-A100-SXM4-40GB`

**如何查詢節點的 GPU 型號：**

```bash
kubectl get nodes -o json | grep -i "nvidia.com/gpu.product"
```

---

## 3. 使用 Gradio UI 查看執行結果

### 3.1 基本 Gradio UI 使用

使用 Gradio UI 提供友善的網頁介面來監控訓練進度：

```python
from pytorch_job_manager import PyTorchJobManager

# 建立 Job Manager
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="gradio-ui-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0
)

# 使用 Gradio UI 監控（會自動建立 job）
manager.wait_logs(ui=True)
```

### 3.2 Gradio UI 功能說明

執行後會啟動一個網頁介面，包含以下功能：

- **開始訓練**：建立並啟動 PyTorch Job
- **察看進度**：查看當前訓練狀態和日誌
- **清除日誌**：刪除 Job 並清理資源

### 3.3 完整的 Gradio UI 範例

```python
from pytorch_job_manager import PyTorchJobManager

# 建立 Job Manager
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="my-training-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="8",
    memory="16Gi",
    gpu="2",
    worker_replicas=0,
    affinity="gpu.product:NVIDIA-RTX-A5000"
)

# 啟動 Gradio UI
# 這會開啟一個網頁介面，您可以：
# 1. 點擊「開始訓練」按鈕來建立和啟動 job
# 2. 即時查看訓練日誌
# 3. 訓練完成後點擊「清除日誌」來刪除 job
manager.wait_logs(ui=True)
```

### 3.4 在 JupyterLab 中使用 Gradio

如果您在 JupyterLab 中執行，Gradio 會自動產生一個公開的 URL：

```python
# 執行後會顯示類似以下訊息：
# Running on local URL:  http://127.0.0.1:7860
# Running on public URL: https://xxxxx.gradio.live
```

點擊公開 URL 即可在瀏覽器中開啟介面。

---

## 4. 使用命令列查看執行結果

### 4.1 基本命令列使用

在終端機中即時顯示訓練日誌：

```python
from pytorch_job_manager import PyTorchJobManager

# 建立 Job Manager
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="cli-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0
)

# 建立任務
manager.create_pytorch_job()

# 在命令列顯示日誌（ui=False）
manager.wait_logs(ui=False)

# 檢查任務狀態
if manager.is_job_succeeded():
    print("✓ 訓練成功完成！")
else:
    print("✗ 訓練失敗")

# 清理資源
manager.delete_job()
```

### 4.2 手動控制執行流程

分步驟控制任務的建立、監控和刪除：

```python
from pytorch_job_manager import PyTorchJobManager
import time

# 建立 Job Manager
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="manual-control-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0
)

# 步驟 1：建立任務
print("正在建立 PyTorch Job...")
manager.create_pytorch_job()
print("Job 已建立")

# 步驟 2：等待任務啟動
print("等待 Job 啟動...")
time.sleep(5)

# 步驟 3：獲取任務狀態
job_info = manager.get_job()
print(f"Job 狀態: {job_info}")

# 步驟 4：即時顯示日誌
print("開始顯示訓練日誌...")
manager.wait_logs(ui=False)

# 步驟 5：檢查結果
if manager.is_job_succeeded():
    print("✓ 訓練成功完成！")
    
    # 可以在這裡加入後續處理
    # 例如：下載模型、發送通知等
else:
    print("✗ 訓練失敗或未完成")

# 步驟 6：清理資源
print("正在刪除 Job...")
manager.delete_job()
print("Job 已刪除")
```

### 4.3 只獲取日誌不等待完成

如果只想查看當前日誌而不等待完成：

```python
from pytorch_job_manager import PyTorchJobManager

manager = PyTorchJobManager(
    namespace="your-namespace",
    name="existing-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0
)

# 只獲取當前日誌
logs = manager.fetch_logs()
if logs:
    print("=== 當前訓練日誌 ===")
    print(logs)
else:
    print("尚無日誌或 Job 未啟動")

# 檢查狀態
if manager.is_job_succeeded():
    print("Job 已完成")
else:
    print("Job 仍在執行中")
```

---

## 5. 使用 kubectl logs 查看執行結果

### 5.1 查詢 Pod 名稱

首先找出 PyTorch Job 建立的 Pod：

```bash
# 列出所有 Pod
kubectl get pods -n your-namespace

# 篩選特定 Job 的 Pod
kubectl get pods -n your-namespace | grep my-training-job
```

輸出範例：
```
my-training-job-master-0   1/1     Running   0          2m
my-training-job-worker-0   1/1     Running   0          2m
my-training-job-worker-1   1/1     Running   0          2m
```

### 5.2 查看 Master Pod 日誌

```bash
# 查看 master pod 的日誌
kubectl logs -n your-namespace my-training-job-master-0

# 即時追蹤日誌（類似 tail -f）
kubectl logs -n your-namespace my-training-job-master-0 -f

# 查看最近 100 行日誌
kubectl logs -n your-namespace my-training-job-master-0 --tail=100

# 查看過去 1 小時的日誌
kubectl logs -n your-namespace my-training-job-master-0 --since=1h
```

### 5.3 查看 Worker Pod 日誌

```bash
# 查看特定 worker 的日誌
kubectl logs -n your-namespace my-training-job-worker-0

# 查看所有 worker 的日誌
kubectl logs -n your-namespace -l training.kubeflow.org/job-name=my-training-job,training.kubeflow.org/replica-type=worker
```

### 5.4 查看 Pod 詳細資訊

```bash
# 查看 Pod 詳細狀態
kubectl describe pod -n your-namespace my-training-job-master-0

# 查看 Pod 的事件
kubectl get events -n your-namespace --field-selector involvedObject.name=my-training-job-master-0
```

### 5.5 完整的監控腳本

建立一個 shell 腳本 `monitor_job.sh` 來監控訓練：

```bash
#!/bin/bash

NAMESPACE="your-namespace"
JOB_NAME="my-training-job"
MASTER_POD="${JOB_NAME}-master-0"

echo "=== 監控 PyTorch Job: $JOB_NAME ==="
echo ""

# 檢查 Pod 狀態
echo "1. Pod 狀態："
kubectl get pods -n $NAMESPACE | grep $JOB_NAME
echo ""

# 檢查 Master Pod 是否就緒
echo "2. 等待 Master Pod 就緒..."
kubectl wait --for=condition=Ready pod/$MASTER_POD -n $NAMESPACE --timeout=300s
echo ""

# 即時顯示日誌
echo "3. 開始顯示訓練日誌："
echo "================================"
kubectl logs -n $NAMESPACE $MASTER_POD -f
```

使用方式：

```bash
chmod +x monitor_job.sh
./monitor_job.sh
```

### 5.6 使用 Python 呼叫 kubectl

在 Python 中使用 subprocess 呼叫 kubectl：

```python
import subprocess
import time

def get_pod_logs(namespace, pod_name):
    """使用 kubectl 獲取 Pod 日誌"""
    cmd = f"kubectl logs -n {namespace} {pod_name}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def monitor_job_with_kubectl(namespace, job_name):
    """使用 kubectl 監控 Job"""
    master_pod = f"{job_name}-master-0"
    
    print(f"監控 Job: {job_name}")
    print("等待 Pod 啟動...")
    
    # 等待 Pod 就緒
    wait_cmd = f"kubectl wait --for=condition=Ready pod/{master_pod} -n {namespace} --timeout=300s"
    subprocess.run(wait_cmd, shell=True)
    
    print("Pod 已就緒，開始顯示日誌：")
    print("=" * 60)
    
    # 即時顯示日誌
    logs_cmd = f"kubectl logs -n {namespace} {master_pod} -f"
    subprocess.run(logs_cmd, shell=True)

# 使用範例
if __name__ == "__main__":
    monitor_job_with_kubectl("your-namespace", "my-training-job")
```

---

## 6. 多節點分散式訓練範例

### 6.1 基本多節點訓練

使用 1 個 Master 和 3 個 Worker 進行分散式訓練：

```python
from pytorch_job_manager import PyTorchJobManager

# 建立多節點訓練任務
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="multi-node-training",
    command=["python", "distributed_train.py"],
    working_dir="/home/jovyan/work",
    cpu="8",
    memory="16Gi",
    gpu="2",
    worker_replicas=3  # 3 個 worker 節點
)

# 建立並執行任務
manager.create_pytorch_job()

# 使用 Gradio UI 監控
manager.wait_logs(ui=True)

# 或使用命令列監控
# manager.wait_logs(ui=False)

# 檢查是否成功
if manager.is_job_succeeded():
    print("分散式訓練完成！")

# 清理資源
manager.delete_job()
```

### 6.2 分散式訓練腳本範例

建立 `distributed_train.py`：

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup():
    """初始化分散式環境"""
    # PyTorch Job 會自動設定這些環境變數
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "23456")
    
    print(f"Initializing process group: rank={rank}, world_size={world_size}")
    print(f"Master: {master_addr}:{master_port}")
    
    # 初始化 process group
    dist.init_process_group(
        backend="nccl",  # 使用 NCCL 後端（GPU）
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    
    # 設定當前 GPU
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    return rank, world_size

def cleanup():
    """清理分散式環境"""
    dist.destroy_process_group()

def train(rank, world_size):
    """分散式訓練函數"""
    print(f"[Rank {rank}] Starting training...")
    
    # 建立模型並移到 GPU
    model = nn.Linear(10, 1).cuda()
    
    # 使用 DDP 包裝模型
    ddp_model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    
    # 定義損失函數和優化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # 訓練迴圈
    for epoch in range(10):
        # 建立假資料
        inputs = torch.randn(32, 10).cuda()
        targets = torch.randn(32, 1).cuda()
        
        # 前向傳播
        outputs = ddp_model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:  # 只在 master 節點印出
            print(f"Epoch {epoch + 1}/10, Loss: {loss.item():.4f}")
    
    print(f"[Rank {rank}] Training completed!")

def main():
    # 設定分散式環境
    rank, world_size = setup()
    
    try:
        # 執行訓練
        train(rank, world_size)
    finally:
        # 清理
        cleanup()

if __name__ == "__main__":
    main()
```

### 6.3 大規模多節點訓練

使用 1 個 Master 和 7 個 Worker（總共 8 個節點）：

```python
from pytorch_job_manager import PyTorchJobManager

# 大規模分散式訓練
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="large-scale-training",
    command=["python", "distributed_train.py"],
    working_dir="/home/jovyan/work",
    cpu="16",
    memory="32Gi",
    gpu="4",
    worker_replicas=7  # 7 個 worker + 1 個 master = 8 節點
)

# 建立任務
manager.create_pytorch_job()

# 監控訓練
manager.wait_logs(ui=False)

# 檢查結果
if manager.is_job_succeeded():
    print("大規模分散式訓練完成！")
    print(f"總共使用了 {1 + 7} 個節點")
    print(f"總共使用了 {(1 + 7) * 4} 個 GPU")

# 清理
manager.delete_job()
```

### 6.4 多節點訓練 + 節點指定

將多節點訓練指定到特定 GPU 類型的節點：

```python
from pytorch_job_manager import PyTorchJobManager

# 多節點訓練，指定使用 RTX A5000 GPU
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="multi-node-a5000",
    command=["python", "distributed_train.py"],
    working_dir="/home/jovyan/work",
    cpu="8",
    memory="16Gi",
    gpu="2",
    worker_replicas=3,
    affinity="gpu.product:NVIDIA-RTX-A5000"  # 所有節點都使用 A5000
)

# 建立並執行
manager.create_pytorch_job()
manager.wait_logs(ui=True)

# 清理
if manager.is_job_succeeded():
    print("訓練完成！")
manager.delete_job()
```

### 6.5 監控多節點訓練

使用 kubectl 監控所有節點：

```bash
#!/bin/bash
# monitor_multi_node.sh

NAMESPACE="your-namespace"
JOB_NAME="multi-node-training"

echo "=== 多節點訓練監控 ==="
echo ""

# 顯示所有 Pod
echo "1. 所有節點狀態："
kubectl get pods -n $NAMESPACE | grep $JOB_NAME
echo ""

# 顯示 Master 日誌
echo "2. Master 節點日誌："
echo "================================"
kubectl logs -n $NAMESPACE ${JOB_NAME}-master-0 -f &
MASTER_PID=$!

# 等待一下
sleep 2

# 顯示 Worker 日誌（背景執行）
echo ""
echo "3. Worker 節點日誌："
for i in {0..2}; do
    echo "--- Worker $i ---"
    kubectl logs -n $NAMESPACE ${JOB_NAME}-worker-$i --tail=10
done

# 等待 Master 日誌
wait $MASTER_PID
```

### 6.6 完整的多節點訓練範例

```python
from pytorch_job_manager import PyTorchJobManager
import time

def run_distributed_training():
    """執行完整的分散式訓練流程"""
    
    # 配置參數
    config = {
        "namespace": "your-namespace",
        "name": "distributed-training-demo",
        "command": ["python", "distributed_train.py"],
        "working_dir": "/home/jovyan/work",
        "cpu": "8",
        "memory": "16Gi",
        "gpu": "2",
        "worker_replicas": 3,  # 3 個 worker
        "affinity": "gpu.product:NVIDIA-RTX-A5000"
    }
    
    # 建立 Manager
    manager = PyTorchJobManager(**config)
    
    # 清理舊的 Job（如果存在）
    try:
        print("清理舊的 Job...")
        manager.delete_job()
        time.sleep(5)
    except:
        pass
    
    # 建立新的 Job
    print(f"建立分散式訓練 Job: {config['name']}")
    print(f"配置: {config['worker_replicas'] + 1} 個節點, 每個節點 {config['gpu']} GPU")
    print(f"總 GPU 數: {(config['worker_replicas'] + 1) * int(config['gpu'])}")
    manager.create_pytorch_job()
    
    # 等待 Job 啟動
    print("等待 Job 啟動...")
    time.sleep(10)
    
    # 顯示 Job 資訊
    job_info = manager.get_job()
    print(f"Job 狀態: {job_info.status}")
    
    # 監控訓練（使用 Gradio UI）
    print("啟動監控介面...")
    manager.wait_logs(ui=True)
    
    # 檢查結果
    if manager.is_job_succeeded():
        print("✓ 分散式訓練成功完成！")
    else:
        print("✗ 訓練失敗或未完成")
    
    # 詢問是否刪除 Job
    response = input("是否刪除 Job？(y/n): ")
    if response.lower() == 'y':
        manager.delete_job()
        print("Job 已刪除")
    else:
        print("Job 保留，可使用 kubectl 查看")

if __name__ == "__main__":
    run_distributed_training()
```

---

## 總結

本教學涵蓋了 PyTorch Job Manager 的完整使用方式：

1. **環境準備**：如何在 Kubeflow Notebook 中準備所有必要的檔案和程式
2. **節點指定**：三種不同的節點指定方式（自動、主機名稱、GPU 類別）
3. **Gradio UI**：使用友善的網頁介面監控訓練
4. **命令列**：在終端機中查看訓練進度
5. **kubectl**：使用 Kubernetes 原生工具查看日誌
6. **多節點訓練**：設定和執行分散式訓練任務

根據您的需求選擇適合的方式來管理和監控您的 PyTorch 訓練任務！

---

## 常見問題

### Q1: 如何查看可用的節點？
```bash
kubectl get nodes
```

### Q2: 如何查看節點的 GPU 資訊？
```bash
kubectl describe node <node-name> | grep -i gpu
```

### Q3: Job 建立失敗怎麼辦？
```bash
# 查看 Job 狀態
kubectl get pytorchjobs -n your-namespace

# 查看詳細錯誤
kubectl describe pytorchjob <job-name> -n your-namespace
```

### Q4: 如何刪除卡住的 Job？
```python
manager.delete_job()
```
或
```bash
kubectl delete pytorchjob <job-name> -n your-namespace
```

### Q5: 如何調整資源配置？
修改 `cpu`、`memory`、`gpu` 參數：
```python
manager = PyTorchJobManager(
    cpu="16",      # 增加 CPU
    memory="32Gi", # 增加記憶體
    gpu="4",       # 增加 GPU
    ...
)
```

"""
PyTorch Job Manager - Affinity 使用範例

此範例展示如何使用 affinity 參數將 PyTorch Job 指定到特定的節點。

支援兩種 affinity 設定方式：
1. hostname:節點名稱 - 將 job 指定到特定的節點
2. gpu.product:GPU型號 - 將 job 指定到具有特定 GPU 型號的節點
"""

from pytorch_job_manager import PyTorchJobManager

# 範例 1: 使用 hostname 將 job 指定到特定節點
print("範例 1: 使用 hostname 指定節點")
manager1 = PyTorchJobManager(
    namespace="d000018238",
    name="pytorch-job-hostname-affinity",
    command=["/home/jovyan/job/test.sh"],
    working_dir="/home/jovyan",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0,
    affinity="hostname:rtxpro6000"  # 指定到 rtxpro6000 節點
)

# 建立並執行 job
manager1.create_pytorch_job()
manager1.wait_logs(ui=False)
manager1.delete_job()

print("\n" + "="*60 + "\n")

# 範例 2: 使用 gpu.product 將 job 指定到具有特定 GPU 的節點
print("範例 2: 使用 GPU 產品型號指定節點")
manager2 = PyTorchJobManager(
    namespace="d000018238",
    name="pytorch-job-gpu-affinity",
    command=["/home/jovyan/job/test.sh"],
    working_dir="/home/jovyan",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0,
    affinity="gpu.product:Tesla-V100-SXM2-16GB"  # 指定到有 RTX A5000 GPU 的節點
)

# 建立並執行 job
manager2.create_pytorch_job()
manager2.wait_logs(ui=False)
manager2.delete_job()

print("\n" + "="*60 + "\n")


"""
單節點訓練範例（無 Worker）
"""

from pytorch_job_manager import PyTorchJobManager
import time

# 建立單節點訓練任務（worker_replicas=0）

manager = PyTorchJobManager(
    namespace="d000018238",
    name="test",
    command=["/home/jovyan/job/test.sh"],
    working_dir="/home/jovyan",
    cpu="8",
    memory="16Gi",
    gpu="1",
    worker_replicas=0
)
# 建立訓練任務
try:
    manager.delete_job()
except:
    pass
time.sleep(5)
manager.create_pytorch_job()


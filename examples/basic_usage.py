"""
基本使用範例
"""

from pytorch_job_manager import PyTorchJobManager

# 建立 PyTorchJobManager 實例
manager = PyTorchJobManager(
    namespace="d000018238",
    name="my-training-job",
    command=["/home/jovyan/job/test.sh"],
    working_dir="/home/jovyan/",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0
)

try:
    manager.delete_job()
except:
    pass

# 建立訓練任務
manager.create_pytorch_job()

# 使用 Gradio UI 監控訓練進度
# manager.wait_logs(ui=True)

# 或者在終端機顯示日誌
# manager.wait_logs(ui=False)
manager.wait_logs(ui=False)
# 檢查任務是否成功
if manager.is_job_succeeded():
    print("訓練完成！")

# 刪除任務
manager.delete_job()

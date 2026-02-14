# PyTorch Job Manager Tutorial

## Table of Contents
1. [Preparing Kubeflow Notebook Environment](#1-preparing-kubeflow-notebook-environment)
2. [Examples for Different Node Selection Methods](#2-examples-for-different-node-selection-methods)
3. [Viewing Results with Gradio UI](#3-viewing-results-with-gradio-ui)
4. [Viewing Results from Command Line](#4-viewing-results-from-command-line)
5. [Viewing Results with kubectl logs](#5-viewing-results-with-kubectl-logs)
6. [Multi-Node Distributed Training Examples](#6-multi-node-distributed-training-examples)

---

## 1. Preparing Kubeflow Notebook Environment

### 1.1 Installing PyTorch Job Manager

Execute the following command in your Kubeflow Notebook to install the package:

```bash
pip install pytorch-job-manager
```

Or install from source:

```bash
git clone https://github.com/yourusername/pytorch-job-manager.git
cd pytorch-job-manager
pip install -e .
```

### 1.2 Preparing Training Script

Create your training script, for example `train.py`:

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os

def main():
    print(f"Starting training on rank {os.environ.get('RANK', '0')}")
    print(f"World size: {os.environ.get('WORLD_SIZE', '1')}")
    
    # Your training code
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        # Training logic
        print(f"Epoch {epoch + 1}/10 completed")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
```

### 1.3 Preparing Execution Script (Optional)

Create a shell script `run_training.sh`:

```bash
#!/bin/bash
cd /home/jovyan/work
python train.py
```

Grant execution permissions:

```bash
chmod +x run_training.sh
```

### 1.4 Verifying File Structure

Ensure your Notebook working directory contains all necessary files:

```
/home/jovyan/
├── work/
│   ├── train.py          # Training script
│   ├── run_training.sh   # Execution script (optional)
│   ├── data/             # Training data
│   └── models/           # Model files
└── job/
    └── pytorch_job.py    # Job management script
```

---

## 2. Examples for Different Node Selection Methods

### 2.1 Example 1: No Node Specification (Auto-Assignment)

Let Kubernetes automatically select available nodes:

```python
from pytorch_job_manager import PyTorchJobManager

# Create Job Manager (without specifying affinity)
manager = PyTorchJobManager(
    namespace="your-namespace",      # Replace with your namespace
    name="auto-assign-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0                # Single-node training
)

# Create and execute job
manager.create_pytorch_job()
manager.wait_logs(ui=False)

# Check if successful
if manager.is_job_succeeded():
    print("Training completed!")

# Clean up resources
manager.delete_job()
```

### 2.2 Example 2: Specifying Node by Hostname

Assign the job to a specific node (e.g., `rtxpro6000`):

```python
from pytorch_job_manager import PyTorchJobManager

# Use hostname to specify node
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="hostname-affinity-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0,
    affinity="hostname:rtxpro6000"   # Assign to rtxpro6000 node
)

# Create and execute job
manager.create_pytorch_job()
manager.wait_logs(ui=False)

# Check if successful
if manager.is_job_succeeded():
    print("Training completed!")

# Clean up resources
manager.delete_job()
```

**How to query available node names:**

```bash
kubectl get nodes
```

### 2.3 Example 3: Specifying Node by GPU Type

Assign the job to nodes with specific GPU models:

```python
from pytorch_job_manager import PyTorchJobManager

# Use GPU product model to specify node
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="gpu-affinity-job",
    command=["python", "train.py"],
    working_dir="/home/jovyan/work",
    cpu="4",
    memory="8Gi",
    gpu="1",
    worker_replicas=0,
    affinity="gpu.product:NVIDIA-RTX-A5000"  # Specify RTX A5000 GPU
)

# Create and execute job
manager.create_pytorch_job()
manager.wait_logs(ui=False)

# Check if successful
if manager.is_job_succeeded():
    print("Training completed!")

# Clean up resources
manager.delete_job()
```

**Common GPU models:**
- `NVIDIA-RTX-A5000`
- `Tesla-V100-SXM2-16GB`
- `NVIDIA-GeForce-RTX-3090`
- `NVIDIA-A100-SXM4-40GB`

**How to query GPU models on nodes:**

```bash
kubectl get nodes -o json | grep -i "nvidia.com/gpu.product"
```

---

## 3. Viewing Results with Gradio UI

### 3.1 Basic Gradio UI Usage

Use Gradio UI to provide a user-friendly web interface for monitoring training progress:

```python
from pytorch_job_manager import PyTorchJobManager

# Create Job Manager
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

# Monitor with Gradio UI (will automatically create job)
manager.wait_logs(ui=True)
```

### 3.2 Gradio UI Features

After execution, a web interface will launch with the following features:

- **Start Training**: Create and launch PyTorch Job
- **Check Progress**: View current training status and logs
- **Clear Logs**: Delete Job and clean up resources

### 3.3 Complete Gradio UI Example

```python
from pytorch_job_manager import PyTorchJobManager

# Create Job Manager
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

# Launch Gradio UI
# This will open a web interface where you can:
# 1. Click "Start Training" button to create and launch job
# 2. View training logs in real-time
# 3. Click "Clear Logs" after training to delete job
manager.wait_logs(ui=True)
```

### 3.4 Using Gradio in JupyterLab

If you're running in JupyterLab, Gradio will automatically generate a public URL:

```python
# After execution, you'll see messages like:
# Running on local URL:  http://127.0.0.1:7860
# Running on public URL: https://xxxxx.gradio.live
```

Click the public URL to open the interface in your browser.

---

## 4. Viewing Results from Command Line

### 4.1 Basic Command Line Usage

Display training logs in real-time in the terminal:

```python
from pytorch_job_manager import PyTorchJobManager

# Create Job Manager
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

# Create job
manager.create_pytorch_job()

# Display logs in command line (ui=False)
manager.wait_logs(ui=False)

# Check job status
if manager.is_job_succeeded():
    print("✓ Training completed successfully!")
else:
    print("✗ Training failed")

# Clean up resources
manager.delete_job()
```

### 4.2 Manual Control of Execution Flow

Control job creation, monitoring, and deletion step by step:

```python
from pytorch_job_manager import PyTorchJobManager
import time

# Create Job Manager
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

# Step 1: Create job
print("Creating PyTorch Job...")
manager.create_pytorch_job()
print("Job created")

# Step 2: Wait for job to start
print("Waiting for Job to start...")
time.sleep(5)

# Step 3: Get job status
job_info = manager.get_job()
print(f"Job status: {job_info}")

# Step 4: Display logs in real-time
print("Starting to display training logs...")
manager.wait_logs(ui=False)

# Step 5: Check results
if manager.is_job_succeeded():
    print("✓ Training completed successfully!")
    
    # You can add post-processing here
    # e.g., download model, send notifications, etc.
else:
    print("✗ Training failed or incomplete")

# Step 6: Clean up resources
print("Deleting Job...")
manager.delete_job()
print("Job deleted")
```

### 4.3 Fetching Logs Without Waiting for Completion

If you only want to view current logs without waiting for completion:

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

# Fetch current logs only
logs = manager.fetch_logs()
if logs:
    print("=== Current Training Logs ===")
    print(logs)
else:
    print("No logs available or Job not started")

# Check status
if manager.is_job_succeeded():
    print("Job completed")
else:
    print("Job still running")
```

---

## 5. Viewing Results with kubectl logs

### 5.1 Querying Pod Names

First, find the Pods created by the PyTorch Job:

```bash
# List all Pods
kubectl get pods -n your-namespace

# Filter Pods for specific Job
kubectl get pods -n your-namespace | grep my-training-job
```

Example output:
```
my-training-job-master-0   1/1     Running   0          2m
my-training-job-worker-0   1/1     Running   0          2m
my-training-job-worker-1   1/1     Running   0          2m
```

### 5.2 Viewing Master Pod Logs

```bash
# View master pod logs
kubectl logs -n your-namespace my-training-job-master-0

# Follow logs in real-time (similar to tail -f)
kubectl logs -n your-namespace my-training-job-master-0 -f

# View last 100 lines of logs
kubectl logs -n your-namespace my-training-job-master-0 --tail=100

# View logs from the past 1 hour
kubectl logs -n your-namespace my-training-job-master-0 --since=1h
```

### 5.3 Viewing Worker Pod Logs

```bash
# View specific worker logs
kubectl logs -n your-namespace my-training-job-worker-0

# View all worker logs
kubectl logs -n your-namespace -l training.kubeflow.org/job-name=my-training-job,training.kubeflow.org/replica-type=worker
```

### 5.4 Viewing Pod Details

```bash
# View detailed Pod status
kubectl describe pod -n your-namespace my-training-job-master-0

# View Pod events
kubectl get events -n your-namespace --field-selector involvedObject.name=my-training-job-master-0
```

### 5.5 Complete Monitoring Script

Create a shell script `monitor_job.sh` to monitor training:

```bash
#!/bin/bash

NAMESPACE="your-namespace"
JOB_NAME="my-training-job"
MASTER_POD="${JOB_NAME}-master-0"

echo "=== Monitoring PyTorch Job: $JOB_NAME ==="
echo ""

# Check Pod status
echo "1. Pod Status:"
kubectl get pods -n $NAMESPACE | grep $JOB_NAME
echo ""

# Check if Master Pod is ready
echo "2. Waiting for Master Pod to be ready..."
kubectl wait --for=condition=Ready pod/$MASTER_POD -n $NAMESPACE --timeout=300s
echo ""

# Display logs in real-time
echo "3. Starting to display training logs:"
echo "================================"
kubectl logs -n $NAMESPACE $MASTER_POD -f
```

Usage:

```bash
chmod +x monitor_job.sh
./monitor_job.sh
```

### 5.6 Using kubectl from Python

Call kubectl from Python using subprocess:

```python
import subprocess
import time

def get_pod_logs(namespace, pod_name):
    """Get Pod logs using kubectl"""
    cmd = f"kubectl logs -n {namespace} {pod_name}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def monitor_job_with_kubectl(namespace, job_name):
    """Monitor Job using kubectl"""
    master_pod = f"{job_name}-master-0"
    
    print(f"Monitoring Job: {job_name}")
    print("Waiting for Pod to start...")
    
    # Wait for Pod to be ready
    wait_cmd = f"kubectl wait --for=condition=Ready pod/{master_pod} -n {namespace} --timeout=300s"
    subprocess.run(wait_cmd, shell=True)
    
    print("Pod is ready, starting to display logs:")
    print("=" * 60)
    
    # Display logs in real-time
    logs_cmd = f"kubectl logs -n {namespace} {master_pod} -f"
    subprocess.run(logs_cmd, shell=True)

# Usage example
if __name__ == "__main__":
    monitor_job_with_kubectl("your-namespace", "my-training-job")
```

---

## 6. Multi-Node Distributed Training Examples

### 6.1 Basic Multi-Node Training

Use 1 Master and 3 Workers for distributed training:

```python
from pytorch_job_manager import PyTorchJobManager

# Create multi-node training job
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="multi-node-training",
    command=["python", "distributed_train.py"],
    working_dir="/home/jovyan/work",
    cpu="8",
    memory="16Gi",
    gpu="2",
    worker_replicas=3  # 3 worker nodes
)

# Create and execute job
manager.create_pytorch_job()

# Monitor with Gradio UI
manager.wait_logs(ui=True)

# Or monitor from command line
# manager.wait_logs(ui=False)

# Check if successful
if manager.is_job_succeeded():
    print("Distributed training completed!")

# Clean up resources
manager.delete_job()
```

### 6.2 Distributed Training Script Example

Create `distributed_train.py`:

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup():
    """Initialize distributed environment"""
    # PyTorch Job automatically sets these environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "23456")
    
    print(f"Initializing process group: rank={rank}, world_size={world_size}")
    print(f"Master: {master_addr}:{master_port}")
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend (GPU)
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    
    # Set current GPU
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    return rank, world_size

def cleanup():
    """Clean up distributed environment"""
    dist.destroy_process_group()

def train(rank, world_size):
    """Distributed training function"""
    print(f"[Rank {rank}] Starting training...")
    
    # Create model and move to GPU
    model = nn.Linear(10, 1).cuda()
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(10):
        # Create dummy data
        inputs = torch.randn(32, 10).cuda()
        targets = torch.randn(32, 1).cuda()
        
        # Forward pass
        outputs = ddp_model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:  # Only print on master node
            print(f"Epoch {epoch + 1}/10, Loss: {loss.item():.4f}")
    
    print(f"[Rank {rank}] Training completed!")

def main():
    # Setup distributed environment
    rank, world_size = setup()
    
    try:
        # Execute training
        train(rank, world_size)
    finally:
        # Cleanup
        cleanup()

if __name__ == "__main__":
    main()
```

### 6.3 Large-Scale Multi-Node Training

Use 1 Master and 7 Workers (total 8 nodes):

```python
from pytorch_job_manager import PyTorchJobManager

# Large-scale distributed training
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="large-scale-training",
    command=["python", "distributed_train.py"],
    working_dir="/home/jovyan/work",
    cpu="16",
    memory="32Gi",
    gpu="4",
    worker_replicas=7  # 7 workers + 1 master = 8 nodes
)

# Create job
manager.create_pytorch_job()

# Monitor training
manager.wait_logs(ui=False)

# Check results
if manager.is_job_succeeded():
    print("Large-scale distributed training completed!")
    print(f"Total nodes used: {1 + 7}")
    print(f"Total GPUs used: {(1 + 7) * 4}")

# Cleanup
manager.delete_job()
```

### 6.4 Multi-Node Training + Node Specification

Assign multi-node training to specific GPU types:

```python
from pytorch_job_manager import PyTorchJobManager

# Multi-node training, specify RTX A5000 GPUs
manager = PyTorchJobManager(
    namespace="your-namespace",
    name="multi-node-a5000",
    command=["python", "distributed_train.py"],
    working_dir="/home/jovyan/work",
    cpu="8",
    memory="16Gi",
    gpu="2",
    worker_replicas=3,
    affinity="gpu.product:NVIDIA-RTX-A5000"  # All nodes use A5000
)

# Create and execute
manager.create_pytorch_job()
manager.wait_logs(ui=True)

# Cleanup
if manager.is_job_succeeded():
    print("Training completed!")
manager.delete_job()
```

### 6.5 Monitoring Multi-Node Training

Monitor all nodes using kubectl:

```bash
#!/bin/bash
# monitor_multi_node.sh

NAMESPACE="your-namespace"
JOB_NAME="multi-node-training"

echo "=== Multi-Node Training Monitor ==="
echo ""

# Display all Pods
echo "1. All Node Status:"
kubectl get pods -n $NAMESPACE | grep $JOB_NAME
echo ""

# Display Master logs
echo "2. Master Node Logs:"
echo "================================"
kubectl logs -n $NAMESPACE ${JOB_NAME}-master-0 -f &
MASTER_PID=$!

# Wait a moment
sleep 2

# Display Worker logs (run in background)
echo ""
echo "3. Worker Node Logs:"
for i in {0..2}; do
    echo "--- Worker $i ---"
    kubectl logs -n $NAMESPACE ${JOB_NAME}-worker-$i --tail=10
done

# Wait for Master logs
wait $MASTER_PID
```

### 6.6 Complete Multi-Node Training Example

```python
from pytorch_job_manager import PyTorchJobManager
import time

def run_distributed_training():
    """Execute complete distributed training workflow"""
    
    # Configuration parameters
    config = {
        "namespace": "your-namespace",
        "name": "distributed-training-demo",
        "command": ["python", "distributed_train.py"],
        "working_dir": "/home/jovyan/work",
        "cpu": "8",
        "memory": "16Gi",
        "gpu": "2",
        "worker_replicas": 3,  # 3 workers
        "affinity": "gpu.product:NVIDIA-RTX-A5000"
    }
    
    # Create Manager
    manager = PyTorchJobManager(**config)
    
    # Clean up old Job (if exists)
    try:
        print("Cleaning up old Job...")
        manager.delete_job()
        time.sleep(5)
    except:
        pass
    
    # Create new Job
    print(f"Creating distributed training Job: {config['name']}")
    print(f"Configuration: {config['worker_replicas'] + 1} nodes, {config['gpu']} GPU per node")
    print(f"Total GPUs: {(config['worker_replicas'] + 1) * int(config['gpu'])}")
    manager.create_pytorch_job()
    
    # Wait for Job to start
    print("Waiting for Job to start...")
    time.sleep(10)
    
    # Display Job information
    job_info = manager.get_job()
    print(f"Job status: {job_info.status}")
    
    # Monitor training (using Gradio UI)
    print("Launching monitoring interface...")
    manager.wait_logs(ui=True)
    
    # Check results
    if manager.is_job_succeeded():
        print("✓ Distributed training completed successfully!")
    else:
        print("✗ Training failed or incomplete")
    
    # Ask whether to delete Job
    response = input("Delete Job? (y/n): ")
    if response.lower() == 'y':
        manager.delete_job()
        print("Job deleted")
    else:
        print("Job retained, can be viewed using kubectl")

if __name__ == "__main__":
    run_distributed_training()
```

---

## Summary

This tutorial covers the complete usage of PyTorch Job Manager:

1. **Environment Preparation**: How to prepare all necessary files and programs in Kubeflow Notebook
2. **Node Specification**: Three different node selection methods (auto, hostname, GPU type)
3. **Gradio UI**: Use a user-friendly web interface to monitor training
4. **Command Line**: View training progress in the terminal
5. **kubectl**: Use Kubernetes native tools to view logs
6. **Multi-Node Training**: Configure and execute distributed training jobs

Choose the appropriate method based on your needs to manage and monitor your PyTorch training jobs!

---

## Frequently Asked Questions

### Q1: How to view available nodes?
```bash
kubectl get nodes
```

### Q2: How to view GPU information on nodes?
```bash
kubectl describe node <node-name> | grep -i gpu
```

### Q3: What to do if Job creation fails?
```bash
# View Job status
kubectl get pytorchjobs -n your-namespace

# View detailed errors
kubectl describe pytorchjob <job-name> -n your-namespace
```

### Q4: How to delete a stuck Job?
```python
manager.delete_job()
```
or
```bash
kubectl delete pytorchjob <job-name> -n your-namespace
```

### Q5: How to adjust resource configuration?
Modify the `cpu`, `memory`, `gpu` parameters:
```python
manager = PyTorchJobManager(
    cpu="16",      # Increase CPU
    memory="32Gi", # Increase memory
    gpu="4",       # Increase GPU
    ...
)
```

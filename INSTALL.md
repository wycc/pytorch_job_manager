# 安裝指南

## 從 PyPI 安裝（發布後）

```bash
pip install pytorch-job-manager
```

## 從原始碼安裝

### 1. 克隆儲存庫

```bash
git clone https://github.com/yourusername/pytorch-job-manager.git
cd pytorch-job-manager
```

### 2. 安裝套件

#### 使用 pip（推薦）

```bash
pip install -e .
```

#### 使用 pip 安裝開發依賴

```bash
pip install -e ".[dev]"
```

或

```bash
pip install -r requirements-dev.txt
```

## 從本地建置安裝

### 1. 建置套件

```bash
python -m build
```

這會在 `dist/` 目錄下產生：
- `pytorch_job_manager-0.1.0-py3-none-any.whl` (wheel 檔案)
- `pytorch-job-manager-0.1.0.tar.gz` (source distribution)

### 2. 安裝建置的套件

```bash
pip install dist/pytorch_job_manager-0.1.0-py3-none-any.whl
```

或

```bash
pip install dist/pytorch-job-manager-0.1.0.tar.gz
```

## 發布到 PyPI

### 1. 安裝發布工具

```bash
pip install build twine
```

### 2. 建置套件

```bash
python -m build
```

### 3. 檢查套件

```bash
twine check dist/*
```

### 4. 上傳到 TestPyPI（測試用）

```bash
twine upload --repository testpypi dist/*
```

### 5. 上傳到 PyPI（正式發布）

```bash
twine upload dist/*
```

## 驗證安裝

安裝完成後，可以執行以下命令驗證：

```python
python -c "from pytorch_job_manager import PyTorchJobManager; print('安裝成功！')"
```

## 解除安裝

```bash
pip uninstall pytorch-job-manager
```

## 系統需求

- Python >= 3.8
- pip >= 20.0
- 可存取 Kubernetes 叢集（用於執行訓練任務）
- Kubeflow Training Operator（用於 PyTorch 分散式訓練）

## 依賴套件

套件會自動安裝以下依賴：

- gradio >= 3.0.0
- kubernetes >= 20.0.0
- kubeflow-training >= 1.7.0

## 疑難排解

### 問題：無法連接到 Kubernetes 叢集

確保您的環境已正確配置 Kubernetes 認證：

```bash
kubectl config view
```

### 問題：找不到 Kubeflow Training Operator

確保您的 Kubernetes 叢集已安裝 Kubeflow Training Operator：

```bash
kubectl get crd pytorchjobs.kubeflow.org
```

### 問題：權限錯誤

確保您的 Kubernetes 服務帳戶有足夠的權限建立和管理 PyTorchJob 資源。

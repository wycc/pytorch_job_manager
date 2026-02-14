# 更新日誌

所有重要的變更都會記錄在此檔案中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，
並且此專案遵循 [語義化版本](https://semver.org/lang/zh-TW/)。

## [未發布]

## [0.1.0] - 2024-02-14

### 新增
- 初始版本發布
- `PyTorchJobManager` 類別，用於管理 PyTorch 分散式訓練任務
- 支援 Master-Worker 架構的分散式訓練
- 整合 Gradio UI 介面，提供即時日誌監控
- 自動處理 Kubernetes 資源配置
- 支援 GPU 資源分配
- 支援單節點訓練（worker_replicas=0）
- 提供完整的任務生命週期管理：
  - `create_pytorch_job()` - 建立訓練任務
  - `get_job()` - 取得任務狀態
  - `wait_for_job()` - 等待任務完成
  - `is_job_succeeded()` - 檢查任務是否成功
  - `get_job_logs()` - 取得任務日誌
  - `delete_job()` - 刪除任務
  - `wait_logs()` - 等待並顯示日誌
- 自動從 Notebook 環境取得 PVC 配置
- 支援自訂 CPU、記憶體和 GPU 資源配置
- 提供範例程式碼

### 文件
- 完整的 README.md 說明文件
- 安裝指南 (INSTALL.md)
- 使用範例 (examples/)
- API 文件註解

### 開發工具
- 設定 pyproject.toml 和 setup.py
- 提供開發依賴配置
- 加入 .gitignore
- MIT 授權條款

[未發布]: https://github.com/yourusername/pytorch-job-manager/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/pytorch-job-manager/releases/tag/v0.1.0

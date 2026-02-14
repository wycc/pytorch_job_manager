import gradio as gr
import time
import textwrap
import os
import kubernetes

from kubernetes.client import V1ResourceRequirements, V1VolumeMount, V1Volume, V1PersistentVolumeClaimVolumeSource, V1Container, V1PodSpec, V1PodTemplateSpec, V1ObjectMeta, V1Affinity, V1NodeAffinity, V1NodeSelector, V1NodeSelectorTerm, V1NodeSelectorRequirement
from kubeflow.training import TrainingClient
from kubeflow.training.models import KubeflowOrgV1ReplicaSpec, KubeflowOrgV1PyTorchJob, KubeflowOrgV1PyTorchJobSpec, KubeflowOrgV1RunPolicy
from kubeflow.training.constants import constants

class PyTorchJobManager:
    def __init__(self, namespace=None, name="pytorch-dist-mnist-gloo", image=None, command=None, working_dir=None,
                 cpu="4", memory="8Gi", gpu="1", worker_replicas=4, affinity=None):
        self.namespace = namespace
        self.name = name
        self.container_name = "pytorch"
        self.training_client = TrainingClient(namespace=namespace)
        self.job_running = False
        self.image = image
        self.command = command
        self.working_dir = working_dir
        self.last_logs = ""
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu
        self.worker_replicas = worker_replicas
        self.affinity = affinity

    # 解析 affinity 參數並建立 V1Affinity 物件
    def _parse_affinity(self):
        """
        解析 affinity 參數，支援兩種格式：
        1. hostname:rtxpro6000 - 指定到特定節點
        2. gpu.product:NVIDIA-RTX-A5000 - 根據 GPU 產品指定
        """
        if not self.affinity:
            return None
        
        try:
            # 解析參數格式 "key:value"
            if ':' not in self.affinity:
                raise ValueError(f"Invalid affinity format: {self.affinity}. Expected format: 'hostname:value' or 'gpu.product:value'")
            
            affinity_type, affinity_value = self.affinity.split(':', 1)
            
            # 根據類型設定對應的 key
            if affinity_type == 'hostname':
                key = 'kubernetes.io/hostname'
            elif affinity_type == 'gpu.product':
                key = 'nvidia.com/gpu.product'
            else:
                raise ValueError(f"Unsupported affinity type: {affinity_type}. Supported types: 'hostname', 'gpu.product'")
            
            # 建立 NodeAffinity
            node_affinity = V1NodeAffinity(
                required_during_scheduling_ignored_during_execution=V1NodeSelector(
                    node_selector_terms=[
                        V1NodeSelectorTerm(
                            match_expressions=[
                                V1NodeSelectorRequirement(
                                    key=key,
                                    operator='In',
                                    values=[affinity_value]
                                )
                            ]
                        )
                    ]
                )
            )
            
            return V1Affinity(node_affinity=node_affinity)
        
        except Exception as e:
            print(f"Error parsing affinity: {e}")
            return None

    # 建立 PyTorch 工作
    def create_pytorch_job(self):
        volume_claim_name = self._get_volume_claim_name()
        image = self.image
        command = self.command
        working_dir = self.working_dir
        volume_mount = V1VolumeMount(
            name="model-volume",
            mount_path="/home/jovyan/",
        )

        volume = V1Volume(
            name="model-volume",
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=volume_claim_name)
        )

        # Master 容器使用原始資源配置
        master_container = V1Container(
            name=self.container_name,
            image=image,
            command=command,
            working_dir=working_dir,
            resources=V1ResourceRequirements(
                requests={
                    "cpu": self.cpu,
                    "memory": self.memory,
                    "nvidia.com/gpu": self.gpu
                },
                limits={
                    "cpu": self.cpu,
                    "memory": self.memory,
                    "nvidia.com/gpu": self.gpu
                }
            ),
            volume_mounts=[volume_mount],
        )

        # Worker 容器：當 worker 數量是 0 時，設定較低的資源配置
        if self.worker_replicas == 0:
            worker_cpu = "1"
            worker_memory = "1Gi"
            worker_gpu = "0"
        else:
            worker_cpu = self.cpu
            worker_memory = self.memory
            worker_gpu = self.gpu
        print(self.worker_replicas)
        worker_container = V1Container(
            name=self.container_name,
            image=image,
            command=command,
            working_dir=working_dir,
            resources=V1ResourceRequirements(
                requests={
                    "cpu": worker_cpu,
                    "memory": worker_memory,
                    "nvidia.com/gpu": worker_gpu
                },
                limits={
                    "cpu": worker_cpu,
                    "memory": worker_memory,
                    "nvidia.com/gpu": worker_gpu
                }
            ),
            volume_mounts=[volume_mount],
        )

        # 解析 affinity 設定
        pod_affinity = self._parse_affinity()

        replica_spec = KubeflowOrgV1ReplicaSpec(
            replicas=self.worker_replicas,
            restart_policy="OnFailure",
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(
                    name=self.name,
                    namespace=self.namespace,
                    annotations={
                        "sidecar.istio.io/inject": "false"
                    }
                ),
                spec=V1PodSpec(
                    containers=[worker_container],
                    volumes=[V1Volume(
                        name="model-volume",
                        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=volume_claim_name)
                    )],
                    affinity=pod_affinity
                )
            )
        )

        master_replica_spec = KubeflowOrgV1ReplicaSpec(
            replicas=1,
            restart_policy="OnFailure",
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(
                    name=self.name,
                    namespace=self.namespace,
                    annotations={
                        "sidecar.istio.io/inject": "false"
                    }
                ),
                spec=V1PodSpec(
                    containers=[master_container],
                    volumes=[V1Volume(
                        name="model-volume",
                        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name=volume_claim_name)
                    )],
                    affinity=pod_affinity
                )
            )
        )

        pytorchjob = KubeflowOrgV1PyTorchJob(
            api_version=constants.API_VERSION,
            kind=constants.PYTORCHJOB_KIND,
            metadata=V1ObjectMeta(name=self.name, namespace=self.namespace),
            spec=KubeflowOrgV1PyTorchJobSpec(
                run_policy=KubeflowOrgV1RunPolicy(clean_pod_policy="None"),
                pytorch_replica_specs={
                    "Master": master_replica_spec,
                    "Worker": replica_spec
                },
            ),
        )

        self.training_client.create_job(pytorchjob)
        self.job_running = True

    # 獲取卷聲明名稱
    def _get_volume_claim_name(self):
        notebook_name = os.environ.get('HOSTNAME', 'notebook').split('-')[0]
        kubernetes.config.load_incluster_config()
        crd_api = kubernetes.client.CustomObjectsApi()
        crd_group = 'kubeflow.org'
        crd_version = 'v1alpha1'
        crd_plural = 'notebooks'
        notebook = crd_api.get_namespaced_custom_object(crd_group, crd_version, self.namespace, crd_plural, notebook_name)
        pvc_name = notebook['spec']['template']['spec']['volumes'][1]['persistentVolumeClaim']['claimName']
        if self.image == None:
            try:
                self.image = notebook['spec']['template']['spec']['containers'][0]['image']
            except:
                print(notebook)
        return pvc_name

    # 獲取工作
    def get_job(self):
        return self.training_client.get_job(self.name, job_kind=constants.PYTORCHJOB_KIND)

    # 等待工作完成
    def wait_for_job(self, wait_timeout=900):
        return self.training_client.wait_for_job_conditions(
            name=self.name,
            job_kind=constants.PYTORCHJOB_KIND,
            wait_timeout=wait_timeout,
        )

    # 檢查工作是否成功
    def is_job_succeeded(self):
        return self.training_client.is_job_succeeded(name=self.name, job_kind=constants.PYTORCHJOB_KIND)

    # 獲取工作日誌
    def get_job_logs(self):
        return self.training_client.get_job_logs(name=self.name, job_kind=constants.PYTORCHJOB_KIND)
    
    # 刪除工作
    def delete_job(self):
        result = self.training_client.delete_job(self.name)
        self.job_running = False
        return result
    
    def fetch_logs(self):
        try:
            logs, _ = self.get_job_logs()
            return logs.get(f'{self.name}-master-0', None)
        except Exception as e:
            return f"Error while fetching logs: {e}"

    def display_logs(self):
        while not self.is_job_succeeded():
            log_content = self.fetch_logs()
            yield log_content if log_content is not None else "No logs available."
            time.sleep(1)
        yield "Completed"
    
    # 顯示 Gradio 介面
    def show_gradio_interface(self):
        def run_and_display():
            log_content = self.fetch_logs()
            if log_content is None:
                if not self.job_running:
                    self.create_pytorch_job()
                for log in self.display_logs():
                    yield log
            elif self.is_job_succeeded():
                yield "Completed"
            
        def clear_and_delete():
            log_content = self.fetch_logs()
            if log_content is None:
                yield "需先點擊開始訓練"
            elif log_content == 'Terminating':
                yield "Terminating"
            else:
                self.delete_job()
                yield "Terminating"

        def check_status():
            log_content = self.fetch_logs()
            if log_content is None:
                yield "需先點擊開始訓練"
            elif self.is_job_succeeded():
                yield "Completed"
            else:
                yield log_content if log_content else "No logs available."
            
        # 自定義JavaScript邏輯，實現自動滾動
        custom_js = """
        <script>
        function scrollToBottom(id) {
            var textbox = document.getElementById(id);
            if (textbox) {
                textbox.scrollTop = textbox.scrollHeight;
            }
        }

        function addAutoScroll(textbox_id) {
            var textbox = document.getElementById(textbox_id);
            if (textbox) {
                var isAutoScroll = true;
                textbox.addEventListener('scroll', function() {
                    if (textbox.scrollTop + textbox.clientHeight < textbox.scrollHeight) {
                        isAutoScroll = false;
                    } else {
                        isAutoScroll = true;
                    }
                });

                setInterval(function() {
                    if (isAutoScroll) {
                        scrollToBottom(textbox_id);
                    }
                }, 100);
            }
        }

        // 在文本框加載完成後初始化自動滾動
        document.addEventListener("DOMContentLoaded", function() {
            addAutoScroll('training_logs_textbox');
        });
        </script>
        """

        # CSS to fix the height of the Textbox
        custom_css = """
        <style>
        #training_logs_textbox {
            height: 300px;
            overflow: auto;
        }
        </style>
        """

        with gr.Blocks() as demo:
            gr.Markdown("點擊按鈕開始訓練模型並顯示日誌")
            
            # 使用 gr.Textbox
            out = gr.Textbox(
                label="Training Logs",
                elem_id="training_logs_textbox",
                lines=13,  # 設置固定的行數
            )  
            with gr.Row():
                btn_start = gr.Button("開始訓練")
                btn_look = gr.Button("察看進度")
                btn_clear = gr.Button("清除日誌")

            btn_start.click(fn=run_and_display, inputs=[], outputs=out)
            btn_look.click(fn=check_status, inputs=[], outputs=out)
            btn_clear.click(fn=clear_and_delete, inputs=[], outputs=out)  # 清除日誌功能

            gr.HTML(custom_css)  # 插入自定義CSS
            gr.HTML(custom_js)   # 插入自定義JavaScript

        demo.launch(share=True)
        
    # 等待並顯示日誌
    def wait_logs(self, ui=True):
        if ui:
            self.show_gradio_interface()
        else:
            try:
                while not self.is_job_succeeded():
                    log_content = self.fetch_logs()
                    if log_content == None:
                        log_content = ''
                    if log_content != self.last_logs:
                        if self.last_logs:
                            new_logs = log_content.replace(self.last_logs, '') if log_content else ''
                        else:
                            new_logs = log_content

                        wrapped_logs = textwrap.fill(new_logs, width=80)
                        print(f"\n--- New Logs at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                        print(wrapped_logs)
                        self.last_logs = log_content
                    time.sleep(1)
                print("Completed")
            except Exception as e:
                print(f"發生錯誤: {e}")
                raise

# AffectTrack 情绪诱发与 EEG 情绪识别系统

一个本机运行的情绪诱发实验系统，包含离线视频诱发采集、在线情绪演示看板、EEG 离线分析流水线，以及基于实时 EEG 的 Valence/Arousal 在线预测服务。

## 功能概览

- 离线采集：按随机顺序播放 `positive / neutral / negative` 视频，播放结束后记录 Valence 和 Arousal 评分，并立即写入 CSV。
- 在线演示：左侧播放视频，右侧展示 Valence/Arousal 动态进度条、最近 30 秒曲线和二维情绪坐标。
- EEG 分析：支持 BDF 审计、预处理、PSD/DE 特征、传统机器学习模型、深度学习模型、LOSO 和单被试实验。
- 在线 EEG 推理：使用全部已对齐数据训练 Valence/Arousal 二分类部署模型，连接 Neuracle 数据流后实时推送结果到在线看板。

## 目录结构

- `app/`：FastAPI 后端、API、数据保存、视频目录扫描、实时情绪流和在线 EEG 服务。
- `templates/`：离线采集页和在线演示页。
- `static/`：CSS、原生 JavaScript、本地 ECharts。
- `video/`：实验视频目录，只识别规范命名的视频。
- `pics/`：在线看板使用的情绪图标。
- `online/`：实时 EEG 设备接收相关代码，当前默认复用 Neuracle TCP 数据流。
- `analysis/`：EEG 审计、预处理、特征提取、模型注册、训练与部署训练代码。
- `scripts/`：常用 EEG 实验脚本。
- `data/`：离线评分 CSV 和 EEG 数据目录，默认不提交到 Git。
- `models/emotion_online/`：在线 Valence/Arousal 部署模型输出目录。
- `outputs/`：EEG 实验输出目录，默认不提交到 Git。
- `tests/`：单元测试和 API 测试。

## 快速开始

### 1. 准备视频

把实验视频放入 `video/`，并使用以下命名规则：

```text
positive_1.mp4
positive_2.mp4
neutral_1.mp4
negative_1.mp4
```

系统只接受：

```text
^(positive|neutral|negative)_(数字).mp4$
```

例如 `1.mp4`、`happy.mp4` 这类文件不会进入正式实验列表，这是为了避免类别误配。

### 2. 创建网页系统环境

```powershell
conda env create -f environment.yml
conda activate emotion-induction
```

也可以直接使用 pip：

```powershell
pip install -r requirements.txt
```

### 3. 启动网页系统

```powershell
uvicorn app.main:app --reload
```

启动后访问：

- 离线采集页：`http://127.0.0.1:8000/offline`
- 在线演示页：`http://127.0.0.1:8000/online`
- 健康检查：`http://127.0.0.1:8000/api/health`

## 离线数据采集

1. 打开 `http://127.0.0.1:8000/offline`。
2. 输入受试者编号，例如 `sub3`。
3. 点击开始实验，前端会拉取视频列表并随机打乱播放顺序。
4. 视频播放阶段会进入沉浸式全屏，隐藏控制条和干扰信息。
5. 每个视频结束后填写 Valence 和 Arousal，范围为 `1-5` 整数。
6. 两个滑块都被实际触摸后才能提交。
7. 提交成功后进入 10 秒休息页，然后自动播放下一段视频。
8. 全部视频结束后显示实验结束。

评分数据写入：

```text
data/offline_records.csv
```

CSV 包含：

```text
subject_id,video_name,category,start_time,end_time,valence,arousal,saved_at
```

## 在线演示

打开：

```text
http://127.0.0.1:8000/online
```

在线页面支持两种模式：

- `mock`：后端自动生成平滑模拟 Valence/Arousal 数据，适合展示和前端调试。
- `live`：接收真实模型或在线 EEG 服务推送的数据。

切换模式的 API：

```http
GET /api/emotion_mode
POST /api/emotion_mode
```

请求示例：

```json
{"mode": "live"}
```

也可以手动向后端推送一帧实时情绪数据：

```http
POST /api/emotion_frame
```

```json
{
  "timestamp": "19:35:25",
  "valence": 3.45,
  "arousal": 2.10
}
```

浏览器统一通过 WebSocket 订阅：

```text
/ws/emotion_stream
```

## 在线 EEG 推理

在线 EEG 推理分两步：先训练部署模型，再在在线页面启动 EEG 服务。

### 1. 创建 EEG 分析环境

```powershell
conda env create -f environment_eeg.yml
conda activate affect-track-eeg
```

如果使用当前 miniconda 环境，并且需要 GPU，请确认 PyTorch 是 CUDA 版：

```powershell
C:\Users\15043\miniconda3\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

### 2. 训练在线部署模型

使用全部有效受试者数据训练 Valence 和 Arousal 两个二分类模型。评分为 `3` 的中性样本会被剔除；`1/2` 映射为低/负向类，`4/5` 映射为高/正向类。

```powershell
C:\Users\15043\miniconda3\python.exe -m analysis.online_training train --config analysis/eeg_config.yaml --network FBSTCNet --device cuda --output-dir models/emotion_online
```

输出文件：

```text
models/emotion_online/valence_fbstcnet.pt
models/emotion_online/arousal_fbstcnet.pt
models/emotion_online/metadata.json
models/emotion_online/report.md
```

模型输出逻辑：

```text
probability = sigmoid(logit)
score = 1 + 4 * probability
```

因此模型最终显示为 `1-5` 范围的连续 Valence/Arousal 分数。

### 3. 启动实时 EEG 推理

默认设备参数：

```text
设备协议：Neuracle
Host：127.0.0.1
Port：8712
通道数：32
原始采样率：1000 Hz
```

后端每秒取最近 4 秒 EEG，执行：

```text
average reference
50 Hz notch
1-45 Hz bandpass
resample to 200 Hz
per-window per-channel z-score
FBSTCNet Valence/Arousal inference
```

在在线页面点击“启动 EEG”即可启动。也可以使用 API：

```http
POST /api/online_eeg/start
POST /api/online_eeg/stop
GET /api/online_eeg/status
```

如果模型文件不存在、设备未连接、窗口信号异常或通道数不匹配，服务会停留在可诊断状态，不会向看板发布伪结果。

## EEG 离线分析

EEG 数据默认放在：

```text
data/eeg_data/
```

评分标签默认读取：

```text
data/offline_records.csv
```

### 数据审计

```powershell
C:\Users\15043\miniconda3\python.exe -m analysis.eeg_pipeline audit --config analysis/eeg_config.yaml
```

审计会检查 BDF、受试者、标签时间、视频开始/结束时间是否能对齐。如果标签无法对齐，训练会被拒绝，避免标签错配。

### 跨个体 LOSO 实验

传统特征模型：

```powershell
C:\Users\15043\miniconda3\python.exe -m analysis.eeg_pipeline run --config analysis/eeg_config.yaml --task category --split-mode loso --model features --feature-kind all --classifier all
```

深度模型：

```powershell
C:\Users\15043\miniconda3\python.exe -m analysis.eeg_pipeline run --config analysis/eeg_config.yaml --task category --split-mode loso --model deep --deep-network FBSTCNet --protocol supervised --input-kind auto --device cuda
```

### 单被试 leave-one-trial-out 实验

例如只跑 `sub3`：

```powershell
C:\Users\15043\miniconda3\python.exe -m analysis.eeg_pipeline run --config analysis/eeg_config.yaml --task category --split-mode subject_dependent --subject-key sub3 --model deep --deep-network EEGNet --protocol supervised --input-kind auto --device cuda
```

`subject_dependent` 当前是严格 leave-one-trial-out：每折留出一个完整视频 trial 做测试，其余 trial 训练。

### 多受试者子集实验

例如只包含 `sub3 sub4 sub5` 做 LOSO：

```powershell
C:\Users\15043\miniconda3\python.exe -m analysis.eeg_pipeline run --config analysis/eeg_config.yaml --task category --split-mode loso --subject-keys sub3 sub4 sub5 --model deep --deep-network TSception --protocol supervised --input-kind auto --device cuda
```

### Valence/Arousal 二分类实验

```powershell
C:\Users\15043\miniconda3\python.exe -m analysis.eeg_pipeline run --config analysis/eeg_config.yaml --task valence_binary --split-mode loso --model features --feature-kind de --classifier rbf_svm
```

```powershell
C:\Users\15043\miniconda3\python.exe -m analysis.eeg_pipeline run --config analysis/eeg_config.yaml --task arousal_binary --split-mode loso --model deep --deep-network FBSTCNet --protocol supervised --input-kind auto --device cuda
```

`valence_binary` 和 `arousal_binary` 会自动剔除评分为 `3` 的样本。

## 常用脚本

仓库提供了一些便捷脚本：

```powershell
python scripts\eeg_audit.py
python scripts\eeg_loso_features.py
python scripts\eeg_loso_deep.py
python scripts\eeg_loso_de_rbf_svm.py
python scripts\eeg_loso_eegnet.py
python scripts\eeg_loso_dgcnn.py
python scripts\eeg_loso_bidann.py
python scripts\eeg_subject_dependent_features.py
python scripts\eeg_sub3_subject_dependent_features.py
python scripts\eeg_sub3_window_kfold_features.py
```

更推荐在正式实验中使用 `python -m analysis.eeg_pipeline ...`，因为参数更明确、结果目录更容易追踪。

## 预处理默认配置

配置文件：

```text
analysis/eeg_config.yaml
```

当前默认值：

- `50 Hz notch`
- `1-45 Hz bandpass`
- `200 Hz` 降采样
- `average reference`
- 丢弃 trial 开头 `30s`
- 丢弃 trial 结尾 `10s`
- 保留中间全部可用片段
- 切成 `4s` 非重叠窗口
- PSD/DE 特征使用 `delta/theta/alpha/beta/gamma` 五个频带

PSD 和 DE 特征提取前，会对每个受试者的每个 EEG 通道做 z-score。在线部署模型则使用更适合实时场景的“每个 4 秒窗口内每通道 z-score”。

## 测试

运行全部测试：

```powershell
C:\Users\15043\miniconda3\python.exe -m pytest -q
```

当前测试覆盖：

- 视频列表扫描和非法文件忽略
- 离线评分保存和请求体校验
- WebSocket 实时帧广播
- EEG 标签解析、时间戳解析、BDF 审计
- PSD/DE 特征、split 防泄露
- Torch 模型注册和基础 forward
- 在线 EEG 预处理、二分类映射和在线状态接口

## 数据与 Git 注意事项

默认不建议提交以下内容：

- `data/offline_records.csv`
- `data/eeg_data/`
- `outputs/`
- 大型 BDF/FIF/cache 文件

在线部署模型位于 `models/emotion_online/`。如果是私有仓库且需要复现实验演示，可以提交这些模型；如果担心模型包含受试者数据分布信息，也可以只提交训练代码，不提交权重文件。

## 第三方说明

部分深度模型结构参考 LibEER，相关许可说明见：

```text
THIRD_PARTY_NOTICES.md
```


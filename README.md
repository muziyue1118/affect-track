# 情感诱发系统

一个基于 FastAPI 的本机单机情绪诱发系统，包含离线数据采集页和在线情绪演示页。

## 目录说明

- `app/`: 后端入口、数据模型与服务层
- `templates/`: 离线采集页和在线演示页
- `static/`: CSS、原生 JavaScript 和本地前端资源
- `video/`: 实验视频目录，只识别 `positive_1.mp4` 这类命名规范
- `data/`: 离线评分 CSV 输出目录
- `tests/`: 单元测试和 API 测试

## 运行前准备

1. 将 `video/` 内视频统一重命名为以下形式：
   - `positive_1.mp4`
   - `neutral_1.mp4`
   - `negative_1.mp4`
2. 当前仓库中的 `1.mp4`、`2.mp4`、`3.mp4` 不会被系统识别，这是有意为之，用来避免错误分类。
3. 本地版 `ECharts` 已经放在 `static/vendor/echarts.min.js`，在线演示页无需再依赖公网 CDN。

## Conda 环境

```powershell
conda env create -f environment.yml
conda activate emotion-induction
```

如果你更习惯直接安装：

```powershell
pip install -r requirements.txt
```

## 启动方式

```powershell
uvicorn app.main:app --reload
```

启动后访问：

- 离线采集页: `http://127.0.0.1:8000/offline`
- 在线演示页: `http://127.0.0.1:8000/online`
- 健康检查: `http://127.0.0.1:8000/api/health`

## 实时接口

### 保存离线评分

`POST /api/save_score`

```json
{
  "subject_id": "SUBJ_001",
  "video_name": "positive_1.mp4",
  "start_time": "E202604091935251282",
  "end_time": "E202604091940251283",
  "valence": 4,
  "arousal": 3
}
```

### 推送实时情绪帧

`POST /api/emotion_frame`

```json
{
  "timestamp": "19:35:25",
  "valence": 3.45,
  "arousal": 2.10
}
```

### 切换在线演示模式

- `GET /api/emotion_mode`
- `POST /api/emotion_mode` with `{"mode":"mock"}` or `{"mode":"live"}`

### 在线 EEG 推理

先用所有已对齐 EEG 数据训练 Valence/Arousal 两个部署模型，评分为 `3` 的中性样本会被剔除，`1/2` 映射为低/负向类，`4/5` 映射为高/正向类：

```powershell
C:\Users\15043\miniconda3\python.exe -m analysis.online_training train --config analysis/eeg_config.yaml --network FBSTCNet --device cuda --output-dir models/emotion_online
```

训练完成后会生成：

- `models/emotion_online/valence_fbstcnet.pt`
- `models/emotion_online/arousal_fbstcnet.pt`
- `models/emotion_online/metadata.json`

在线页面点击“启动 EEG”后，后端默认连接 Neuracle `127.0.0.1:8712`、32 通道、1000 Hz 数据流；每次取最近 4 秒 EEG，执行 50 Hz notch、1-45 Hz bandpass、200 Hz 重采样、average reference 和每通道窗口 Z-score，然后输出 `1-5` 范围的 Valence/Arousal 并推送到 `/ws/emotion_stream`。

控制接口：

- `POST /api/online_eeg/start`
- `POST /api/online_eeg/stop`
- `GET /api/online_eeg/status`

## 测试

```powershell
pytest
```

## EEG 分析流水线

采集完成后，可以用独立的 `analysis/` 流水线审计 BDF 与评分标签，并在标签可对齐时运行 EEG 情绪分类。

建议单独创建 EEG 科学计算环境：

```powershell
conda env create -f environment_eeg.yml
conda activate affect-track-eeg
```

先运行审计，确认 `data/eeg_data/sub*_E*/data.bdf` 与 `data/offline_records.csv` 能正确对齐：

```powershell
python scripts\eeg_audit.py
```

审计通过后运行跨个体分类：

```powershell
python scripts\eeg_loso_features.py
```

只运行跨个体 `DE + RBF SVM`：

```powershell
python scripts\eeg_loso_de_rbf_svm.py
```

也可以通过参数只跑某一种特征和某一个分类器，避免每次都重新计算 `PSD + DE` 并遍历全部模型：

```powershell
python -m analysis.eeg_pipeline run --config analysis/eeg_config.yaml --task category --split-mode loso --model features --feature-kind de --classifier rbf_svm
```

运行不跨个体分类：

```powershell
python scripts\eeg_subject_dependent_features.py
```

只运行 `sub3` 的不跨个体分类：

```powershell
python scripts\eeg_sub3_subject_dependent_features.py
```

只运行 `sub3` 的窗口级随机打乱 10 折分类：

```powershell
python scripts\eeg_sub3_window_kfold_features.py
```

运行跨个体端到端 deep baseline：

```powershell
python scripts\eeg_loso_deep.py
```

深度模型同样支持通过参数选择网络。默认只跑一个指定模型，不会自动遍历全部 LibEER 模型：

```powershell
python -m analysis.eeg_pipeline run --config analysis/eeg_config.yaml --task category --split-mode loso --model deep --deep-network shallow_convnet
```

GPU 运行说明：

- `analysis/eeg_config.yaml` 中的 `models.deep_device: auto` 会在 CUDA 可用时自动使用 GPU，否则回退 CPU。
- 命令行可以覆盖配置，例如追加 `--device cuda`、`--device cuda:0` 或 `--device cpu`。
- 如果当前安装的是 CPU 版 PyTorch，强制 `--device cuda` 会直接报错；需要先安装 CUDA 版 PyTorch。

LibEER 风格模型已经接入统一入口，`Net.py` 负责注册，具体实现位于 `analysis/libeer_models/`。常见运行方式如下：

```powershell
# 原始 EEG 窗口模型
python scripts\eeg_loso_eegnet.py

# DE 图特征模型
python scripts\eeg_loso_dgcnn.py

# 需要测试域无标签 X 的 transductive domain adaptation 模型
python scripts\eeg_loso_bidann.py
```

已注册的 Torch 模型包括 `shallow_convnet`、`EEGNet`、`TSception`、`ACRNN`、`FBSTCNet`、`DGCNN`、`GCBNet`、`GCBNet_BLS`、`CDCN`、`DBN`、`HSLT`、`RGNN`、`RGNN_official`、`STRNN`、`CoralDgcnn`、`DannDgcnn`、`BiDANN`、`R2GSTNN`、`MsMDA`、`NSAL_DGAT`、`PRRL`。

协议含义：

- `supervised`: 普通监督训练；按当前设置，deep 模型 early stopping 使用每折测试集作为验证集，报告会显式标注。
- `source_dg`: 只使用训练 subjects 的 domain label，不读取测试 subject 特征。
- `transductive_da`: 训练时允许读取测试 subject 的无标签 EEG 特征 `X` 做域对齐，但不读取测试标签 `y`；这类结果不能和严格 supervised LOSO 混为同一种结论。

流水线默认采用 50 Hz notch、1-45 Hz bandpass、200 Hz 降采样、average reference、丢弃 trial 开头 30 秒和结尾 10 秒，并保留中间全部可用片段，再切成 4 秒非重叠窗口。提取 PSD 和 DE 前，会对每个受试者的每个 EEG 通道单独做 z-score 归一化；PSD 和 DE 特征使用 delta/theta/alpha/beta/gamma 五个频带。deep 路径可选择 ShallowConvNet 或 LibEER 风格模型，且 supervised deep 模型的 early stopping 按当前设置直接使用每折测试集作为验证集。所有输出写入 `outputs/eeg_runs/`，原始 BDF 和输出目录默认不会提交到 Git。

如果标签缺少 `end_time`、时间超出 BDF 记录范围，或者没有足够有效 trial，`run` 会拒绝训练，只生成审计报告，避免标签错配和数据泄露。

预处理和特征默认值主要参考 DEAP、SEED-IV 和 DREAMER 这类公开情绪 EEG 数据集的常见设置：DEAP 使用视频诱发和 valence/arousal 评分，SEED-IV 提供 200 Hz、4 秒片段、PSD/DE 五频带特征说明，DREAMER 使用视频诱发后的 valence/arousal/dominance 自评。

## 说明

- 离线采集端会记录每段视频的开始时间和结束时间，并在视频结束后立即保存评分，避免中途退出导致全部数据丢失。
- 前端时间戳使用 `E + YYYYMMDDHHMMSS + 4位子秒标记` 生成；后端保留 Python 版本生成函数作为格式校验基准。
- 在线演示页只消费 `/ws/emotion_stream`，未来模型协议变化时建议只改后端适配层。



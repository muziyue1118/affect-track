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

## 测试

```powershell
pytest
```

## 说明

- 离线采集端会记录每段视频的开始时间和结束时间，并在视频结束后立即保存评分，避免中途退出导致全部数据丢失。
- 前端时间戳使用 `E + YYYYMMDDHHMMSS + 4位子秒标记` 生成；后端保留 Python 版本生成函数作为格式校验基准。
- 在线演示页只消费 `/ws/emotion_stream`，未来模型协议变化时建议只改后端适配层。



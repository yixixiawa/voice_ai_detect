# Django Backend API 文档

本项目后端提供音频真伪检测、训练任务管理、数据归档上传能力。

## 1. 启动

在项目根目录执行：

```powershell
python backend/manage.py runserver 0.0.0.0:8000
```

默认服务地址：`http://127.0.0.1:8000`

## 2. 通用约定

- 响应格式统一为 JSON。
- 成功通常包含 `ok: true`。
- 失败通常包含 `ok: false` 与 `message`。
- 支持音频后缀：`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`

## 3. 接口列表

### 3.1 健康检查

- **Method**: `GET`
- **Path**: `/api/health/`

响应示例：

```json
{
  "ok": true,
  "message": "backend is running"
}
```

### 3.2 音频检测（test_archive）

- **Method**: `POST`
- **Path**: `/api/test-archive/`
- **Content-Type**: `multipart/form-data`
- **参数**:
  - `file`: 单个音频文件（可选）
  - `files`: 多个音频文件（可选）

说明：`file` 与 `files` 二选一，推荐统一用 `files`。

响应示例：

```json
{
  "ok": true,
  "count": 2,
  "results": [
    {
      "filename": "a.mp3",
      "result": "AI合成",
      "confidence": 0.9123,
      "pred_idx": 1
    },
    {
      "filename": "b.wav",
      "result": "真人",
      "confidence": 0.9567,
      "pred_idx": 0
    }
  ]
}
```

### 3.3 启动训练

- **Method**: `POST`
- **Path**: `/api/train/`

响应示例（启动成功）：

```json
{
  "ok": true,
  "started": true,
  "message": "训练任务已启动",
  "pid": 12345,
  "log": "backend/train.log"
}
```

响应示例（已在运行）：

```json
{
  "ok": false,
  "started": false,
  "message": "训练任务已在运行中",
  "pid": 12345
}
```

### 3.4 训练状态查询

- **Method**: `GET`
- **Path**: `/api/train/status/`

响应示例：

```json
{
  "ok": true,
  "running": true,
  "pid": 12345,
  "message": "训练任务运行中",
  "log": "backend/train.log"
}
```

### 3.5 停止训练

- **Method**: `POST`
- **Path**: `/api/train/stop/`

响应示例：

```json
{
  "ok": true,
  "stopped": true,
  "pid": 12345,
  "message": "训练任务已停止",
  "detail": "SUCCESS: The process with PID 12345 has been terminated.",
  "log": "backend/train.log"
}
```

### 3.6 训练日志尾部读取

- **Method**: `GET`
- **Path**: `/api/train/log/`
- **Query 参数**:
  - `n`: 返回最后 N 行日志（可选，默认 `100`，范围自动限制为 `1~2000`）

请求示例：

```text
GET /api/train/log/?n=200
```

响应示例：

```json
{
  "ok": true,
  "exists": true,
  "log": "backend/train.log",
  "lines": 200,
  "content": [
    "Epoch 2/10 [训练]: ...",
    "Epoch 2/10 [验证]: ..."
  ],
  "message": "ok"
}
```

### 3.7 音频归档上传（自动分组）

- **Method**: `POST`
- **Path**: `/api/archive/upload/`
- **Content-Type**: `multipart/form-data`
- **参数**:
  - `label`: `fake` 或 `real`（必填）
  - `file`: 单文件（可选）
  - `files`: 多文件（可选）

存储规则：

- 根目录：`data/fake` 或 `data/real`
- 子目录：`group_001`, `group_002`, ...
- 每组最多 500 个音频，超出自动新建下一组。

响应示例：

```json
{
  "ok": true,
  "count": 2,
  "saved": [
    {
      "filename": "sample1.mp3",
      "saved_as": "sample1_a1b2c3d4.mp3",
      "label": "fake",
      "group": "group_003",
      "path": "fake/group_003/sample1_a1b2c3d4.mp3"
    },
    {
      "filename": "sample2.wav",
      "saved_as": "sample2_d4c3b2a1.wav",
      "label": "fake",
      "group": "group_003",
      "path": "fake/group_003/sample2_d4c3b2a1.wav"
    }
  ]
}
```

## 4. 前端轮询建议（训练监控）

建议前端每 2~5 秒轮询以下接口：

1. `GET /api/train/status/`
2. `GET /api/train/log/?n=200`

典型流程：

1. 调 `POST /api/train/` 启动训练。
2. 轮询状态与日志刷新页面。
3. 用户点击停止时调 `POST /api/train/stop/`。
4. 停止后继续拉一次日志用于收尾展示。

## 5. 关键文件

- 路由：`backend/api/urls.py`
- 视图：`backend/api/views.py`
- 服务：`backend/api/services.py`
- 配置：`backend/backend_app/settings.py`

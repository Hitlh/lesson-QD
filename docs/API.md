# 前后端接口约定

启动后可在 `/docs` 查看自动生成的完整字段约束。时间均为原视频全局秒数，窗口偏移由后端处理，前端无需二次换算。

| 方法 / 路径 | 输入 | 输出 / 行为 |
| --- | --- | --- |
| `GET /api/health` | 无 | 模型加载状态、默认后处理、上传限制、窗口和缓存配置；`ready` 主要表示必要文件存在，完整加载可用启动器 `--check --warmup` 验证 |
| `POST /api/videos` | multipart `file`；可选 `duration` | 201，`video_id`、真实时长、`stream_url` 等；时长使用 FFprobe，浏览器传值不作为校验依据 |
| `GET /api/videos/{video_id}` | 上传返回的 ID | 视频状态与进度 |
| `GET /api/videos/{video_id}/stream` | ID；可选 Range 请求头 | 视频文件，支持 206 分段播放 |
| `POST /api/search` | JSON，见下文 | 候选片段、高光分数、后处理统计、窗口信息、缓存命中及耗时 |

搜索最小请求：

```json
{"video_id":"<上传返回的ID>","query":"A person cuts vegetables.","top_k":3}
```

`query` 长度 1–160，当前模型适用英文描述；`top_k` 为 1–5。可选 `nms_method` 为 `none` / `hard` / `soft_linear`，`nms_threshold` 为 0–1，`nms_min_score` 非负。默认 Hard NMS、阈值 0.5。

结果主要字段：

```text
moments: [{start, end, score, raw_score}, ...]
saliency_scores: 两秒特征网格上的高光分数数组
clip_duration: 2.0
duration: 原视频真实时长
windowing: {enabled, window_count, feature_clip_count, windows, ...}
window_ranking: {enabled, method, strength?}
postprocessing: {method, iou_threshold, candidates_before, candidates_after, returned_count, ...}
video_feature_cache_hit: 是否复用了视频外观特征
```

候选可能少于请求的 Top-K，甚至为空；前端应正常展示这些状态。Hard NMS 后仍允许 tIoU 不超过阈值的小范围交集。`raw_score` 是该候选经过 NMS 之前的分数；若启用窗口评分器，它已经是融合后的排序分数，不能解释成原始 QD-DETR 概率。

`saliency_scores[i]` 对应从 `2*i` 秒开始的时间格，最后一格播放边界裁剪至 `duration`。重叠窗口的高光值取均值，不能直接与候选 `score` 混用。

常见错误：404 表示 ID 不存在或后端已重启，需要重新上传；413 为文件过大；415 为格式不支持；422 为时长或参数不满足约束；500 为模型或特征提取失败，返回 `detail`。前端使用 `frontend/src/services/search.ts`，Vite 代理配置在 `frontend/vite.config.ts`；同源部署可不设置 API 基础地址。

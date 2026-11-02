# 模型资产

权重不上传 Git。可使用课程交付中另行提供的配套文件，放在本目录，或通过仓库根目录 `.env` 设置通用配置项。

| 文件名 | 用途 | 配置项 |
| --- | --- | --- |
| `qv_model_best.ckpt` | 本项目 QVHighlights 复现权重，默认模式 | `QD_DETR_CHECKPOINT` |
| `tacos_model_best.ckpt` | 本项目 TACoS 微调权重，可选模式 | `QD_DETR_CHECKPOINT` |
| `window_ranker.pt` | 与 TACoS 权重配套的窗口评分器 | `QD_DETR_WINDOW_RANKER` |
| `ViT-B-32.pt` | OpenAI CLIP 图像/文本编码 | `QD_DETR_CLIP_MODEL` |
| `SLOWFAST_8x8_R50.pkl` | Kinetics 预训练的 SlowFast | `SLOWFAST_CHECKPOINT` |

CLIP 公开下载来源由 [OpenAI CLIP 官方模型表](https://github.com/openai/CLIP/blob/main/clip/clip.py) 的 `ViT-B/32` 条目给出。本项目保留的 token 级编码实现及分词词表在 `QD-DETR/run_on_video/clip/`。SlowFast 权重来源见 [Lighthouse 官方说明](https://github.com/line/lighthouse)。QD-DETR 原始权重、数据与特征来源见 [QD-DETR 官方仓库](https://github.com/wjun0830/QD-DETR)。本项目自训练权重未提供公开下载地址，需另行交付，不能假定与上游任意 checkpoint 等价。

`manifest.example.json` 记录已验证的自训练权重 SHA256。可以用 `sha256sum models/文件名`，或 Windows `Get-FileHash models/文件名 -Algorithm SHA256` 比对。检查点包含上游实验配置，运行时会读取这类元信息；请使用来源可信且配套的文件。

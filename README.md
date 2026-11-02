# FrameSeek：基于 QD-DETR 的视频文本片段检索

课程项目代码归档。输入视频和英文自然语言描述，返回视频中的候选时间片段、分数和高光曲线。

模型与后端工作包括 QD-DETR 复现实验、时间片段去重、长视频滑动窗口推理、TACoS 微调与跨窗口排序。`frontend/` 为组内前端同学提供的新版 Vue 页面，本仓库保留与后端配套的接口适配。

## 目录

| 目录 / 文件 | 用途 |
| --- | --- |
| `backend/` | FastAPI 接口、时间 NMS、滑动窗口、分块提特征、CPU 特征缓存、窗口评分器 |
| `QD-DETR/` | 上游模型、训练和评测源码，以及本项目推理适配 |
| `lighthouse/` | 原视频 SlowFast 特征提取所需的上游代码子集 |
| `frontend/` | Vue 3 / TypeScript 前端，已匹配上传、搜索和视频流接口 |
| `experiments/tacos/` | TACoS 离线评测、微调和窗口排序训练 |
| `experiments/metrics/` | 已完成实验的汇总指标，不含数据样本或模型权重 |
| `scripts/` | 实际视频接口验收、提交内容检查 |
| `docs/` | 接口、训练评测和验证说明 |
| `models/` | 外部权重放置说明与校验清单 |
| `start_backend.py` | 跨平台后端启动入口 |

数据集、预提取特征、模型权重、视频、虚拟环境、运行缓存和报告不放进 Git。克隆代码后需要安装依赖并提供模型文件，不能仅靠源码启动真实推理。

## 环境和依赖

已验证：Python 3.11、PyTorch 2.7.1、torchvision 0.22.1、FFmpeg/FFprobe。前端建议 Node.js 20.19+ 或 22 LTS。NVIDIA GPU 推荐用于视频特征提取；CPU 后端支持运行，但提特征较慢。研究训练/离线实验脚本使用 CUDA。

```bash
git clone https://github.com/Hitlh/lesson-QD.git
cd lesson-QD
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell：.venv\Scripts\Activate.ps1

# 先按机器的驱动环境安装匹配的 torch / torchvision：
# https://pytorch.org/get-started/locally/
# 仅 CPU 环境的示例：
# pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

系统还需要可执行的 `ffmpeg` 和 `ffprobe`，例如 Ubuntu 使用 `sudo apt install ffmpeg`；其他系统使用对应包管理器，并将程序加入 PATH。`ffmpeg-python` 只是 Python 调用库。

## 准备模型

将另外提供的配套权重放在以下位置，来源和 SHA256 见 [models/README.md](models/README.md)：

```text
models/
  qv_model_best.ckpt       # 本项目复现得到的默认 QVHighlights 模型
  ViT-B-32.pt              # OpenAI CLIP ViT-B/32
  SLOWFAST_8x8_R50.pkl      # SlowFast 特征提取器
  tacos_model_best.ckpt    # 可选：本项目 TACoS 微调模型
  window_ranker.pt         # 可选：与 TACoS 模型严格配套的窗口评分器
```

默认 `qv` 模式只需前三个文件。`tacos` 模式需 CLIP、SlowFast、TACoS 模型和配套评分器。也可在 `.env` 中指定权重位置；相对路径始终从仓库根目录解析。配置示例保留通用回环地址，未写入开发者电脑或训练服务器地址。

## 启动与前后端连接

```bash
cp .env.example .env
python start_backend.py --check
python start_backend.py
```

默认后端：`http://127.0.0.1:8000`，接口文档：`http://127.0.0.1:8000/docs`。首次搜索才加载模型并提取视频特征；可先用 `python start_backend.py --check --warmup` 验证模型加载。也可使用 `./start_backend.sh` 或 `start_backend.bat`，默认调用当前激活环境的 Python；`FRAMESEEK_PYTHON` 可指定其他解释器。

开发前端（另开终端）：

```bash
cd frontend
npm ci
npm run dev
```

打开终端提示的前端地址，通常是 `http://localhost:5173`。页面请求 `/api`，由 Vite 转发至后端。端口被占用时，可运行 `python start_backend.py --port 8001`，并将 `frontend/.env` 中 `VITE_API_PROXY_TARGET` 设为 `http://127.0.0.1:8001`，重启前端。

也可以构建前端后由后端统一托管：

```bash
cd frontend
npm ci
npm run build
cd ..
python start_backend.py
```

访问后端地址即可打开页面。`frontend/dist` 需在后端启动前生成。部署到另一台机器时，通过 `--host 0.0.0.0`、`--port`、`CORS_ORIGINS` 和前端代理目标配置，不需要修改源码。请只运行一个后端 worker，当前上传登记和特征缓存保存在单个服务进程内。

启用 TACoS 模型和两级排序：

```bash
python start_backend.py --profile tacos --check --warmup
python start_backend.py --profile tacos
```

`.env` 中的 `QD_DETR_CHECKPOINT` / `QD_DETR_WINDOW_RANKER` 若被显式设置，会覆盖模式默认路径。评分器会验证基础模型的 SHA256，不能与任意权重混用。

## 实现范围和限制

- 时间去重：先处理非法边界，汇总全部窗口候选，再按分数做时间 Hard NMS，默认 tIoU 阈值 0.5，最后取 Top-K。阈值表示删除高度重叠的候选，输出仍可能有较小交集。
- 长视频：默认 150 秒窗口、30 秒重叠；各窗口采用局部时间编码，预测边界换算成原视频秒数，重叠区域高光分数取平均。服务默认接受不超过约 600 秒、256 MiB 的视频。这是当前工程配置，QD-DETR 的正弦位置编码没有架构上固定 150 秒的上限。
- 内存：长视频按默认 60 秒块解码和提取特征，外观 embedding 保留在 CPU，单次只将当前窗口送入模型。600 秒的 FP32 2816 维特征约 3.22 MiB；这不包含解码帧、模型和中间激活。短视频保留原整段提特征路径。
- 缓存：进程内 LRU，默认最多 32 个视频，可调 `FEATURE_CACHE_SIZE`。没有磁盘特征缓存和按字节预算；服务重启后需重新上传和提取。上传原文件不随 LRU 淘汰自动删除，演示结束后可停服务并清理 `storage/`。
- 稳定性：模型推理串行、上传大小/时长校验、失败提特征不写缓存，输出进度日志。尚无持久任务队列、请求排队上限和自动上传文件清理。
- 跨窗口排序：默认按候选分数；可选 TACoS 窗口相关性评分器，将窗口 logit 与候选 logit 加权融合。分数是排序依据，未经概率校准。

## 验证与实验

```bash
pip install -r requirements-test.txt
python run_tests.py
python scripts/audit_submission.py
# 启动服务后，使用自行准备的视频进行真实推理验收：
python scripts/smoke_test.py data/example.mp4 --base-url http://127.0.0.1:8000
```

回归测试无需权重和数据集；真实推理验收需要上述资源。验收覆盖上传、全局时间边界、NMS、长视频多窗口、再次搜索命中特征缓存、视频 Range 播放。具体结果见 [验证记录](docs/VALIDATION.md)。

已有实验的百分比结果如下。TACoS 为官方预提取特征离线评测，完整测试集含超过 600 秒的视频，不能当作原视频服务的吞吐量或上线范围证明。

| 实验 | 结果 / 解释 |
| --- | --- |
| QVHighlights 复现 | R1@0.5 = 62.77，平均 mAP = 41.39 |
| Hard NMS 0.5 | 平均 mAP 41.39 → 40.51；R3@0.5 77.419 → 78.065，去重不保证所有精度指标提高 |
| TACoS 原模型直接滑窗 | 4001 条测试查询，R1@0.5 = 3.674，R5@0.5 = 15.996 |
| TACoS 微调后 | R1@0.5 = 12.147，R5@0.5 = 29.518 |
| 微调 + 窗口评分器 | R1@0.5 = 12.972，R5@0.5 = 30.467 |

原模型直接滑窗的总体 R1@0.5 低于只看前 150 秒的 5.474，说明扩展覆盖范围后仍需解决跨窗口排序。汇总 CSV 保存在 `experiments/metrics/`；训练、数据格式、命令与适用范围见 [实验说明](docs/EXPERIMENTS.md)。

## 来源与分工

模型基于 [QD-DETR](https://github.com/wjun0830/QD-DETR)，特征提取使用 [OpenAI CLIP](https://github.com/openai/CLIP) 与 [Lighthouse](https://github.com/line/lighthouse) 的相应源码。上游许可已随对应目录保留，具体版本和修改见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目不将这些模型算法或组员的前端页面声明为个人原创。

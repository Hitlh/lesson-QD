# 复现、TACoS 微调与窗口排序

以下命令均从仓库根目录运行，使用激活环境的 Python。先安装匹配的 CUDA PyTorch，然后安装 `requirements-train.txt`；后端 HTTP 对齐检查还需要 `requirements-test.txt`。不随代码上传训练数据、预提取特征和检查点。

## QVHighlights 复现

按 [QD-DETR 官方说明](https://github.com/wjun0830/QD-DETR) 获取 QVHighlights 标注、SlowFast / CLIP 视频特征及 CLIP 文本特征。

```text
data/qvhighlights/highlight_train_release.jsonl
data/qvhighlights/highlight_val_release.jsonl
features/qvhighlights/slowfast_features/<vid>.npz
features/qvhighlights/clip_features/<vid>.npz
features/qvhighlights/clip_text_features/<qid>.npz
```

```bash
bash QD-DETR/qd_detr/scripts/train.sh --seed 2018 --n_epoch 200
# 完成后将实际输出的 checkpoint 路径替换下方占位值：
bash QD-DETR/qd_detr/scripts/inference.sh outputs/qvhighlights/<实验目录>/model_best.ckpt val
```

可通过 `QV_DATA_DIR`、`FEATURE_ROOT`、`RESULTS_ROOT`、`EXP_ID`、`FRAMESEEK_PYTHON` 覆盖位置/解释器，包含空格的路径应加引号。评测入口显式传入特征路径，避免复用 checkpoint 中保存的原训练机器目录。训练默认拼接 2304 维 SlowFast 和 512 维 CLIP，再添加 2 维 TEF，文本维度 512；保存模型和官方评测结果的原逻辑保留。

本项目复现最佳 epoch 为 54；存量记录 R1@0.5 = 62.77、平均 mAP = 41.39。`experiments/metrics/temporal_nms.csv` 是已保存预测上的后处理对照，非本次整理重新训练的结果。Hard NMS .5 优先减少高度重复候选，不能据此声明所有指标均提升。

对已有 QVHighlights 验证预测复核重叠率与不同 NMS 阈值：

```bash
python experiments/temporal_nms/audit.py \
  --predictions outputs/qvhighlights/<实验目录>/best_hl_val_preds.jsonl \
  --annotations data/qvhighlights/highlight_val_release.jsonl \
  --output outputs/temporal_nms_audit
```

预测文件需为官方输出格式，按分数降序，每条至少包含 3 个候选，以支持 Top-3 重叠统计。可传 `--baseline-metrics` 核对保存的官方指标。脚本输出重叠统计、示例及 NMS 指标对照，时间交集只能证明时间冗余，不能直接证明语义相同。

## TACoS 输入格式

使用 [Lighthouse](https://github.com/line/lighthouse) 发布的 TACoS 特征与同一划分的标注。先按原始分发包格式整理为：

```text
data/tacos/{train,val,test}.jsonl
features/tacos/slowfast/<vid>.npz       # features: [T,2304]
features/tacos/clip/<vid>.npz           # features: [T,512]
features/tacos/clip_text/<qid>.npz      # last_hidden_state: [L,512]，L <= 32
```

标注每行是一条查询，例如以下结构示意（数值仅说明字段）：

```json
{"qid":1,"vid":"video_identifier","query":"A person cuts vegetables.","duration":240.0,"relevant_windows":[[170.0,180.0]]}
```

时间需从原始标注的帧坐标按数据提供方的帧率换算为秒，不能把标注帧号直接当秒。`qid` 在所有划分间应保持唯一，与文本特征文件名一致；此训练脚本针对官方划分中的单事件查询，要求 train 9790 条、val 4436 条，测试为 4001 条，且 train/val 视频不重叠。它会验证两秒视频特征网格、维度、有限值和标注边界。不同来源的变体数据需要先核对划分与网格，不能靠放宽断言得到可比较结果。

## 第一步：基线离线评测

需要先保存完整验证集的原模型滑窗预测，用于微调前后路径对齐。`evaluate_tacos.py` 的默认筛选是 150–600 秒；以下显式改为整个验证集，以满足微调对齐检查。

```bash
python experiments/tacos/evaluate_tacos.py \
  --qd-root QD-DETR --checkpoint models/qv_model_best.ckpt \
  --features features/tacos --annotations data/tacos/val.jsonl \
  --split val --min-duration 0 --max-duration 1000000 \
  --output outputs/tacos_baseline_val
```

输出包括 `predictions.jsonl`、`metrics.json` 和运行清单。该路径注入预提取的视觉/文本特征，使用后端滑窗及后处理；不测原视频解码速度。`--limit` 仅用于快速排错，正式实验不要截断验证集。

## 第二步：微调主模型

```bash
python experiments/tacos/finetune.py \
  --qd-root QD-DETR --checkpoint models/qv_model_best.ckpt \
  --features features/tacos --train data/tacos/train.jsonl \
  --validation data/tacos/val.jsonl \
  --baseline-predictions outputs/tacos_baseline_val/predictions.jsonl \
  --output outputs/tacos_finetune --epochs 8 --patience 3 \
  --query-batch-size 16 --lr 2e-5
```

添加 `--smoke` 可先做少量反向传播和权重恢复检查，输出为 `smoke.ckpt`，不会完成正式训练。正式最佳模型为 `model_best.ckpt`，可另行复制到 `models/tacos_model_best.ckpt`。

每条查询每轮随机抽取一个覆盖目标至少 50% 的正窗口（不存在时取最大覆盖窗口），以及存在时一个与标注无交集的负窗口；不将部分重叠窗口直接当成负样本。训练按精确视频窗口长度分组，避免上游混合长度 attention mask 的影响。使用 Hungarian 匹配后的分类、L1/GIoU 边界损失与辅助层损失；TACoS 没有人工高光评分，不伪造该标签，关闭对应训练损失并冻结高光头。特征提取器不参与这一微调。

每轮使用完整视频滑窗检索验证，按验证集 R1@0.5 选择，平局比较 R5@0.5；测试集不参与选模型。微调改善总体检索不代表各个子集均提高，尤其需检查晚于 150 秒的目标。

## 第三步：窗口相关性评分器

冻结微调模型，先分别提取 train / val / test 的窗口表征和候选：

```bash
for split in train val test; do
  python experiments/tacos/extract_windows.py \
    --qd-root QD-DETR --checkpoint models/tacos_model_best.ckpt \
    --features features/tacos --annotations "data/tacos/$split.jsonl" \
    --split "$split" --output "outputs/window_features/$split"
done
```

提取输出为 `windows.jsonl`、`features.npy`、`manifest.json`。测试集只在冻结模型后提取、评测，不能参与参数选择。

评分器训练还需要微调主模型在完整验证集的预测。先将第一步评测的 checkpoint 换为 `models/tacos_model_best.ckpt`，保存到 `outputs/tacos_finetuned_val`。然后：

```bash
python experiments/tacos/train_ranker.py \
  --extracted-dir outputs/window_features \
  --baseline outputs/tacos_finetuned_val/predictions.jsonl \
  --output outputs/window_ranker
```

输出 `window_ranker.pt`，可复制到 `models/window_ranker.pt`。评分器输入是全局 token、窗口局部表示的均值/最大值、5 项候选统计量，共 773 维；不输入窗口在原视频中的全局位置。使用同视频的零重叠难负窗口、BCE 与成对排序损失训练，只更新小型评分头；强度与轮次由验证集选择，包括强度 0 的回退。

候选概率 p 和窗口相关性 logit r 融合为：

```text
s = sigmoid((logit(p) + λ*r) / (1+λ))
```

同一窗口加上相同偏置，窗口内顺序保持；不同窗口可重新排序。它依赖“局部模型表示包含查询与整个窗口的匹配信息”这一可实验检验的假设，不保证任意数据集有效。λ=0 时精确绕过融合，单窗口也绕过评分器。部署校验基础 checkpoint SHA256，防止评分器与表示空间不匹配。

## 第四步：与真实后端推理路径对齐

```bash
python experiments/tacos/evaluate_backend.py \
  --qd-root QD-DETR --checkpoint models/tacos_model_best.ckpt \
  --ranker models/window_ranker.pt --features features/tacos \
  --annotations data/tacos/test.jsonl --pack outputs/window_features/test \
  --output outputs/tacos_backend_test
```

该脚本使用预提取特征运行真实后端逻辑，并与窗口包离线排序结果逐条检查一致性。`--http` 可额外通过测试客户端验证接口；此模式保留 160 字请求限制，会跳过超长查询，因此不是完整测试集统计。原视频服务验收另用 `scripts/smoke_test.py`。

完整测试集 R1@0.5：原模型滑窗 3.674，微调后 12.147，加窗口评分器 12.972。超过 600 秒的部分仅属离线实验，默认演示服务仍以 600 秒为上限。详细子集结果见 `experiments/metrics/`，保留下降项与提高项，避免只汇报有利结果。

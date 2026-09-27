# BanjirKita AI

**Community Edge Flood Intelligence Network**

Privacy-preserving, offline-first flood risk detection for Malaysia's monsoon-prone communities, built for the Malaysia AI Impact Festival 2026 — in conjunction with the Intel® AI Global Impact Festival.

[English](#english) · [中文](#中文)

---

<a id="english"></a>
## English

### The Problem

In the November 2024–January 2025 monsoon season alone, Malaysia's National Disaster Management Agency (NADMA) reported over **137,000 people affected** and **40,900+ families displaced** — the consequence of aging drainage systems and increasingly intense precipitation.

### The Solution

BanjirKita AI turns residents' own smartphones into privacy-preserving edge sensors. An on-device model fuses local photos, text reports, and rainfall data to estimate flood risk **offline** — no raw images or location data ever leave the device.

### Key Features

- **On-device inference** — Intel OpenVINO-optimized MobileNetV2, runs entirely offline
- **Intel Arc A770 GPU acceleration** — confirmed for both training (via PyTorch's native XPU backend) and inference (OpenVINO GPU device), not just claimed
- **INT8 quantization via NNCF** — with real (not synthetic) calibration images
- **Explicit, disclosed risk-fusion logic** — rainfall + visual flood probability → LOW/MEDIUM/HIGH, not a black box (see `classify_risk()` in `banjirkita_infer_ex.py`)
- **Built-in dataset integrity tooling** — see [Honest Results](#honest-results-and-known-limitations) below; this isn't cosmetic, it materially changed the reported numbers

### Tech Stack

| Layer | Technology |
|---|---|
| Model | MobileNetV2 (transfer learning), binary flood/non-flood classifier |
| Training | PyTorch, Intel Arc A770 via native `torch.xpu` backend |
| Optimization | Intel OpenVINO Toolkit (ONNX → IR conversion, `PERFORMANCE_HINT: LATENCY`), NNCF INT8 quantization |
| Inference | OpenVINO Runtime, GPU device (Arc A770), automatic CPU fallback |
| Target deployment | Intel Core Ultra NPU (architecture target — **not yet benchmarked on physical NPU hardware**, stated honestly rather than implied) |

### Honest Results and Known Limitations

We're documenting this in detail because we think **catching and disclosing a data problem is worth more than a suspiciously perfect number.**

An early version of the flood/non-flood image classifier ([Kaggle Flood Classification Dataset](https://www.kaggle.com/datasets/dhawalsrivastava2583/flood-classification-dataset)) hit **~99.97% F1 after a single epoch** — which immediately raised a red flag.

1. **~4,150 near-duplicate images** (44.6% of the flood class) — likely reposted/duplicated photos in the scraped dataset, causing train/validation leakage. **Fixed** via perceptual-hash deduplication.
2. **A structural resolution mismatch between classes** — every `non_flood` image was uniformly 224×224px (consistent with a pre-processed source), while `flood` images retained varied native resolutions. A dimension-only classifier could separate them perfectly.

**Mitigation applied:** every training image (regardless of class) is forced through an identical resolution/compression-degrading transform before use (`_resolution_debias()` in `train_flood_classifier_ex.py`), evening the playing field.

**Current result:** Accuracy 96.7%, F1 97.1%, False-Negative Rate 3.3%, on our internal validation split.

**What we can't yet claim:** the resolution asymmetry at the *source-image level* is not fully eliminated — a dimension-only classifier still achieves 100% separation on raw file metadata. We can't retrain the source dataset; we can only debias the pipeline going forward.

### Getting Started

```bash
pip install openvino onnx nncf torch torchvision scikit-learn pillow opencv-python-headless

# 1. (Optional) audit your dataset for the issues above
python check_dataset_sanity.py --data_dir ./data

# 2. (Optional) deduplicate the flood class first
python dedupe_flood_class.py --data_dir ./data --apply --backup_dir ./data/flood_removed

# 3. Train
python train_flood_classifier_ex.py --data_dir ./data --epochs 10

# 4. Run inference (auto-detects Intel GPU, falls back to CPU)
python banjirkita_infer_ex.py
```

See `MANIFEST.md` for the full file breakdown, including which scripts are current vs. superseded by later iterations.

### Model Training & Inference Results

Below are real examples of the model training process and live inference outputs:

**Training Metrics & Performance:**
<div align="center">
  <img src="https://github.com/Rueded/BanjirKita-AI/blob/main/screenshots/Screenshot%20(11).png?raw=true" alt="Model training metrics showing accuracy and loss curves" width="500">
</div>

**Model Optimization via Intel OpenVINO:**
<div align="center">
  <img src="https://github.com/Rueded/BanjirKita-AI/blob/main/screenshots/Screenshot%20(21).png?raw=true" alt="OpenVINO model optimization and quantization process" width="500">
</div>

**Live Inference Output - Flood Detection:**
<div align="center">
  <img src="https://github.com/Rueded/BanjirKita-AI/blob/main/screenshots/Screenshot%20(22).png?raw=true" alt="Real-time flood detection inference results" width="500">
</div>

**Risk Classification Dashboard:**
<div align="center">
  <img src="https://github.com/Rueded/BanjirKita-AI/blob/main/screenshots/Screenshot%20(26).png?raw=true" alt="Flood risk classification dashboard showing LOW/MEDIUM/HIGH risk levels" width="500">
</div>

### Responsible AI

Principles applied, per [Intel's Responsible AI guidelines](https://www.intel.com/content/www/us/en/artificial-intelligence/responsible-ai-principles.html):

- **Human Oversight** — no alert is ever auto-broadcast; every high-confidence prediction is routed to a local disaster committee for human review first
- **Privacy by Design** — on-device inference; raw images/text/location never transmitted; only anonymized model updates would be shared under the planned federated-learning architecture
- **Transparency & Explainability** — the risk-fusion rule is a single, disclosed formula, not a black box; every prediction includes its underlying evidence
- **Data Integrity & Bias Mitigation** — see [Honest Results](#honest-results-and-known-limitations) above; this is the section we're proudest of

### Development Note

This project's architecture, code (OpenVINO pipeline, NNCF integration, PyTorch training script), and documentation were developed with substantial AI assistance (Claude). Direction on problem framing, technical decisions, and key research are our own.

### Sources

- NADMA, via ReliefWeb — Nov 2024–Jan 2025 flood impact data: https://reliefweb.int/disaster/fl-2024-000218-mys
- Department of Statistics Malaysia (DOSM), via NADMA/Bernama — 2024 flood economic losses: https://www.nadma.gov.my/bi/media-en/news/6320-flood-losses-ease-malaysia-s-damage-bill-drops-from-rm933-4
- Asian Disaster Reduction Center (ADRC) — 2014 flood reference data: https://www.adrc.asia/nationinformation.php?NationCode=458&Lang=en&NationNum=16
- Flood Classification Dataset, Kaggle (dhawalsrivastava2583): https://www.kaggle.com/datasets/dhawalsrivastava2583/flood-classification-dataset

### License

MIT — see `LICENSE`. 

---

<a id="中文"></a>
## 中文

### 问题背景

仅 2024 年 11 月至 2025 年 1 月这一波季风洪灾，马来西亚国家灾难管理局（NADMA）就统计出**超过 13.7 万人受灾**、**逾 4.09 万户被迫撤离**——是 2014 年以来最严重的一次。陈旧的排水系统和日益强烈的降水，使这个问题愈演愈烈。

### 解决方案

BanjirKita AI 把居民自己的手机变成保护隐私的边缘传感器。设备端模型融合本地照片、文字报告和雨量数据，**完全离线**估算洪水风险——原始图像和位置数据永远不离开用户设备。

### 核心特性

- **端侧推理**——Intel OpenVINO 优化的 MobileNetV2，完全离线运行
- **Intel Arc A770 GPU 加速**——训练（PyTorch 原生 XPU 后端）和推理（OpenVINO GPU 设备）都经过实测确认，不是空口宣称
- **NNCF INT8 量化**——使用真实（非合成）图片做校准
- **公开披露的风险融合逻辑**——雨量+视觉洪水概率 → 低/中/高，不是黑箱（见 `banjirkita_infer_ex.py` 中的 `classify_risk()`）
- **内置数据集完整性检测工具**——详见下方[诚实的结果说明](#诚实的结果说明与已知局限)，这不是装饰，它实实在在改变了最终上报的数字

### 技术栈

| 层级 | 技术 |
|---|---|
| 模型 | MobileNetV2（迁移学习），洪水/非洪水二分类器 |
| 训练 | PyTorch，Intel Arc A770，通过原生 `torch.xpu` 后端 |
| 优化 | Intel OpenVINO Toolkit（ONNX → IR 转换，`PERFORMANCE_HINT: LATENCY`），NNCF INT8 量化 |
| 推理 | OpenVINO Runtime，GPU 设备（Arc A770），自动回退 CPU |
| 目标部署 | Intel Core Ultra NPU（架构目标——**尚未在真实 NPU 硬件上测试过**，如实说明，不暗示已完成） |

### 诚实的结果说明与已知局限

我们详细记录这一段，是因为我们相信**发现并披露一个数据问题，比一个高得可疑的数字更有价值。**

洪水/非洪水图像分类器（[Kaggle Flood Classification Dataset](https://www.kaggle.com/datasets/dhawalsrivastava2583/flood-classification-dataset)）的早期版本，**仅训练一轮就达到 ~99.97% F1**——这个数字立即敲响了警钟。

1. **约 4,150 张近乎重复的图片**（占flood类的 44.6%）——很可能是爬取数据集时混入的转发/重复照片，导致训练集/验证集之间泄漏。**已修复**：通过感知哈希去重。
2. **两个类别之间存在结构性的分辨率不匹配**——所有 `non_flood` 图片全部统一是 224×224 像素（明显来自某个已预处理的数据来源），而 `flood` 图片保留了各种原生分辨率。单纯用图片尺寸就能 100% 区分两类。

**已采取的缓解措施：** 每张训练图片（不分类别）都会强制经过完全相同的分辨率/压缩降质变换（`train_flood_classifier_ex.py` 中的 `_resolution_debias()`），消除这个不公平优势。

**目前结果：** 内部验证集上，准确率 96.7%，F1 97.1%，漏报率（False-Negative Rate）3.3%。

**我们目前还不能宣称的：** 源图片层面的分辨率不对称问题并未被完全消除——仅用维度信息训练的分类器在原始文件元数据上依然能达到 100% 分离。我们无法重新训练源数据集；我们能做的只是在未来的数据处理管道中继续去偏差。

### 快速开始

```bash
pip install openvino onnx nncf torch torchvision scikit-learn pillow opencv-python-headless

# 1.（可选）先审计数据集，检查上述问题
python check_dataset_sanity.py --data_dir ./data

# 2.（可选）先给flood类去重
python dedupe_flood_class.py --data_dir ./data --apply --backup_dir ./data/flood_removed

# 3. 训练
python train_flood_classifier_ex.py --data_dir ./data --epochs 10

# 4. 运行推理（自动检测Intel GPU，找不到则回退CPU）
python banjirkita_infer_ex.py
```

完整文件说明见 `MANIFEST.md`，里面标注了哪些脚本是当前在用的、哪些已被后续版本取代。

### 模型训练与推理结果

以下是模型训练过程和实时推理输出的真实示例：

**训练指标与性能曲线：**
<div align="center">
  <img src="https://github.com/Rueded/BanjirKita-AI/blob/main/screenshots/Screenshot%20(11).png?raw=true" alt="模型训练指标展示准确度和损失曲线" width="500">
</div>

**Intel OpenVINO 模型优化：**
<div align="center">
  <img src="https://github.com/Rueded/BanjirKita-AI/blob/main/screenshots/Screenshot%20(21).png?raw=true" alt="OpenVINO 模型优化和量化过程" width="500">
</div>

**实时推理输出 - 洪水检测：**
<div align="center">
  <img src="https://github.com/Rueded/BanjirKita-AI/blob/main/screenshots/Screenshot%20(22).png?raw=true" alt="实时洪水检测推理结果" width="500">
</div>

**风险分类仪表板：**
<div align="center">
  <img src="https://github.com/Rueded/BanjirKita-AI/blob/main/screenshots/Screenshot%20(26).png?raw=true" alt="洪水风险分类仪表板展示低/中/高风险等级" width="500">
</div>

### 负责任 AI

依据 [Intel 负责任 AI 准则](https://www.intel.com/content/www/us/en/artificial-intelligence/responsible-ai-principles.html) 应用的原则：

- **人类监督**——警报从不自动广播；每一条高置信度预测都会先送到当地灾害应变委员会人工审核
- **隐私优先设计**——端侧推理；原始图像/文字/位置从不上传；未来联邦学习架构下只会共享匿名化的模型更新
- **透明与可解释**——风险融合规则是单一、公开的公式，不是黑箱；每条预测都附带其判断依据
- **数据完整性与偏差缓解**——详见上方[诚实的结果说明](#诚实的结果说明与已知局限)，这是我们最自豪的一部分

### 开发说明

本项目的架构、代码（OpenVINO 推理管线、NNCF 集成、PyTorch 训练脚本）和文档，均在 AI 工具（Claude）的大量协助下完成。问题定义、技术决策方向，以及关键研究均来自我们自己的思考。

### 数据来源

- NADMA，经 ReliefWeb 转载——2024年11月至2025年1月洪灾影响数据：https://reliefweb.int/disaster/fl-2024-000218-mys
- 马来西亚统计局（DOSM），经 NADMA/Bernama 转载——2024年洪灾经济损失：https://www.nadma.gov.my/bi/media-en/news/6320-flood-losses-ease-malaysia-s-damage-bill-drops-from-rm933-4
- 亚洲减灾中心（ADRC）——2014年洪灾参考数据：https://www.adrc.asia/nationinformation.php?NationCode=458&Lang=en&NationNum=16
- Flood Classification Dataset，Kaggle（dhawalsrivastava2583）：https://www.kaggle.com/datasets/dhawalsrivastava2583/flood-classification-dataset

### 许可证

MIT——见 `LICENSE` 文件。

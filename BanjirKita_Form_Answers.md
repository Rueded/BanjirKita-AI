# BanjirKita AI — Official Submission Form Draft Answers

Copy-paste into the Google Form.

**Status as of this version:**
1. Dataset quality audit complete — deduplication applied (9,296 → 5,149 flood images), resolution-debiasing applied during training, real diagnostic numbers filled into the Dataset section below.
2. Real MobileNetV2 model trained on Arc A770 (via XPU) and exported to ONNX — confirmed via terminal output (96.68% val accuracy, believable gradual training curve, not instant saturation).
3. Pipeline now uses `train_flood_classifier_ex.py` / `banjirkita_infer_ex.py` (MobileNetV2 + real calibration images + real demo photo) rather than the earlier toy-CNN scripts.

---

## Project Title *(max 10 words)*

> **BanjirKita AI: Community Edge Flood Intelligence Network**

(7 words)

---

## Project Synopsis *(max 150 words)*

> In the November 2024–January 2025 monsoon season alone, NADMA reported over 137,000 people affected and 40,900+ families displaced — Malaysia's worst flooding since 2014, with Kelantan and Terengganu hit hardest. Rural communities often receive under an hour of warning, as official sensors are sparse and connectivity fails when it's needed most. BanjirKita AI turns residents' smartphones into privacy-preserving edge sensors, fusing local photos, text reports, and rainfall data to estimate flood risk offline. The inference pipeline runs on an Intel Arc A770 GPU with OpenVINO, targeting deployment on low-power Intel NPU hardware in the field — no raw images or locations ever leave the device. Devices periodically share only model updates via federated learning, improving accuracy without centralizing sensitive data. High-confidence alerts are reviewed by local disaster committees before broadcast. BanjirKita AI is designed to deliver hours of extra warning time to Malaysia's most flood-vulnerable, connectivity-poor communities.

(149 words)

---

## Intel® Digital Readiness Program participation → **No**

Not currently enrolled, and no confirmed quick sign-up flow was found in time to complete one before submission. Selecting "No" honestly — this is a verifiable fact (Intel has enrollment records), not something to misrepresent for the 2-3 rubric points, and a false "Yes" risks disqualification if it comes up in a Stage 3 conversation. Leave the "If Yes, specify the name of the program" field blank.

---

## Specify GenAI tool usage → **GenAI is the primary development method**

**Explain the usage of GenAI, if used:**
> The project's architecture design, Python codebase (OpenVINO inference pipeline, NNCF INT8 quantization integration, PyTorch training script for the flood image classifier), Responsible AI framework, and submission materials were drafted with the assistance of AI tools (Claude). The developer's role was directing the project's problem framing and technical decisions, running and verifying all code on their own hardware (Intel Core i5-10400F + Arc A770 GPU), confirming actual OpenVINO device detection and GPU-based inference through direct testing, and making final decisions on scope, data sourcing, and what claims are honestly supportable given the prototype's current stage.

---

## Target audience

> Rural, flood-prone communities in Malaysia's east-coast and interior states (e.g., Kelantan, Terengganu, Pahang), particularly households with limited or no access to real-time flood monitoring infrastructure or reliable internet connectivity during monsoon season. Secondary users: local disaster management committees (JKKK/APM) who receive and act on prioritized alerts.

---

## Did you use any datasets? → **Yes**

**If yes, describe (150–200 words):**

> Historical water-level/flood-extent records (DID Malaysia, NADMA) and rainfall data (MetMalaysia) inform the numeric risk model. For visual classification, we use Kaggle's Flood Classification Dataset (dhawalsrivastava2583; ground-level flood/non-flood images matching resident-submitted photos). A dataset audit found (1) ~4,150 near-duplicate flood images, removed via perceptual-hash deduplication before training (near-duplicate pairs in sampled testing dropped from 789/1,500 to 4/1,500 post-dedup), and (2) a resolution asymmetry — non_flood images uniformly pre-resized to 224×224 vs. flood images at native resolution — risking a non-content shortcut (confirmed: a dimension-only classifier hits 100% separation). Training applies forced resolution/compression debiasing to mitigate this; a targeted blur-shortcut test now shows near-chance separability on real held-out images (0.50 before, 0.55 after), and the training curve shows gradual, realistic improvement (94.2%→96.3% over 10 epochs) rather than instant saturation. The resolution asymmetry at the source-image level is not fully eliminated and is disclosed as a limitation; reported metrics (Accuracy 96.7%, F1 97.1%, False-Negative Rate 3.3%) should be read with this caveat.

The two blanks are filled in with real numbers from the actual diagnostic run on your data — don't need further edits unless you re-run the audit and get different numbers.

---

## Project stage → **A working prototype**

Confidently resolved: real trained model (MobileNetV2, 96.68% val accuracy on deduplicated real data), real GPU inference and training on Arc A770 (confirmed via terminal output), full pipeline from photo input to risk classification runs end-to-end. This is a prototype running on your own dev machine, not deployed anywhere beyond it — so "A working prototype" is the honest choice, not "Deployment in test/controlled environment" (that would imply it's running somewhere other than your own machine) or "Full-scale live deployment" (not remotely true).

---

## Does your project use Intel technologies? → **Yes**

**Confirmed hardware:** Intel Core i5-10400F (dev machine) + Intel Arc A770 GPU. No NPU on hand.

**What's actually verified vs. planned — keep this distinction in the answer:**
- **Verified:** the inference pipeline runs on the Arc A770. Confirmed via direct testing — OpenVINO's `available_devices` reports `['CPU', 'GPU']` on this machine, and inference is confirmed running on `'GPU'` (real terminal output, not simulated).
- **Verified:** model training also runs on the Arc A770. Confirmed via `torch.xpu.is_available()` and real terminal output showing `[HARDWARE] Training will run on: xpu` during an actual training run — this uses PyTorch's native Intel XPU backend, not CPU.
- **Not yet tested:** NPU deployment is a stated architecture target, not benchmarked on physical NPU hardware.

> Intel Arc A770 GPU for both on-device inference acceleration and model training — both confirmed via direct testing: OpenVINO device detection consistently reports inference running on `'GPU'`, and PyTorch's native XPU backend (`torch.xpu`) confirms training runs on the same Arc A770, not CPU. Intel OpenVINO Toolkit is used for model optimization and format conversion (ONNX → OpenVINO IR), with `PERFORMANCE_HINT: LATENCY` set for low-latency edge inference, and NNCF is integrated for INT8 quantization with real (not synthetic) calibration images. The optimized model is designed for eventual deployment on Intel Core Ultra NPU hardware for low-power, offline field use — this is stated as an architecture/deployment target, not yet tested on physical NPU hardware.

**If asked in Stage 3:** "Why NPU if you haven't tested it?" → answer honestly: "OpenVINO's IR format is portable across Intel CPU/GPU/NPU — we developed, trained, and validated the pipeline on Arc A770, confirmed via direct device detection for both training and inference, and the same optimized model is designed to run on NPU hardware for power-constrained field deployment. We haven't benchmarked on physical NPU hardware yet." This is a normal, defensible engineering answer — don't dress it up as more than it is.

---

## Responsible AI principles — which apply

- ☑ **Enable Human Oversight** — high-confidence alerts are reviewed by local disaster committees (JKKK/APM) before broadcast; the system never auto-dispatches
- ☑ **Design for Privacy** — all photo/text analysis on-device; raw images, text, and precise location never transmitted; only federated model weight updates are shared
- ☑ **Advance Security, Safety, and Reliability** — confidence scores shown alongside every prediction; conservative thresholds to avoid false confidence
- ☑ **Enable Transparency and Explainability** — every alert includes the evidence/confidence behind it, and the risk-fusion rule (rainfall + visual flood probability → LOW/MEDIUM/HIGH) is an explicit, disclosed formula, not a black box
- ☑ **Promote Equity and Inclusion** — sourced datasets span multiple flood-affected states, not one region; designed for low-connectivity households specifically excluded by cloud-only alternatives
- ☐ **Protect the Environment** — weaker fit now that dev is on Arc A770 (a discrete GPU, not low-power); leave unchecked unless you have a specific efficiency argument you can defend
- ☐ **Respect Human Rights** — optional/weaker fit here; skip unless you have a specific angle

---

## Ethical/privacy measures elaboration

> Privacy is designed in at the architecture level: all photo and text analysis runs on-device via Intel OpenVINO-optimized models, and raw images, text, and precise location data never leave the user's phone. Only anonymized model weight updates are shared during federated learning rounds, so the system improves without centralizing personal data. To address bias, the sourced datasets span multiple flood-affected states and rainfall patterns rather than a single region, and training will be validated for consistent performance across regions before deployment; confidence scores are displayed alongside every risk estimate so users and reviewers can gauge reliability rather than treat outputs as certain. For safety, the system is explicitly advisory: no alert is broadcast to a community automatically — every high-confidence prediction is routed to a local disaster committee (JKKK/APM) for human review before dissemination, keeping a human decision-maker in the loop for any action affecting evacuation.

---

## SDGs (max 3)

> - SDG 11: Sustainable Cities and Communities
> - SDG 13: Climate Action
> - SDG 10: Reduced Inequalities

---

## Sources/references/citations

> - National Disaster Management Agency (NADMA), via ReliefWeb: approximately 137,410 people affected and 40,922 families displaced by the Nov 2024–Jan 2025 monsoon floods, described as Malaysia's worst flooding since 2014, with Kelantan and Terengganu hardest hit. https://reliefweb.int/disaster/fl-2024-000218-mys
> - Department of Statistics Malaysia (DOSM), via NADMA/Bernama: flood-related losses of RM933.4 million recorded in 2024. https://www.nadma.gov.my/bi/media-en/news/6320-flood-losses-ease-malaysia-s-damage-bill-drops-from-rm933-4m-in-2024-to-rm636-9m-in-2025
> - Asian Disaster Reduction Center (ADRC), on the 2014 monsoon flood (Malaysia's previous worst): approximately 400,000 people displaced, RM6.1 billion in losses. https://www.adrc.asia/nationinformation.php?NationCode=458&Lang=en&NationNum=16
> - Flood Classification Dataset (Kaggle, dhawalsrivastava2583): 9,296 flood + 3,748 non-flood ground-level images as sourced; 5,149 flood + 3,748 non-flood (8,897 total) actually used for training after perceptual-hash deduplication removed ~4,150 near-duplicate flood images. https://www.kaggle.com/datasets/dhawalsrivastava2583/flood-classification-dataset

---

## Video

Already drafted separately — see `BanjirKita_Video_Script.md`.

---

### Final checklist before submitting

1. ~~Intel Digital Readiness Program~~ — resolved: **No**.
2. ~~GenAI usage~~ — resolved: **"GenAI is the primary development method,"** explanation text filled in above.
3. ~~Project stage~~ — resolved: **"A working prototype."**
4. ~~Dataset quality audit~~ — resolved: real numbers from your actual diagnostic run are filled into the Dataset section above.
5. **One thing left to actually do:** re-run `banjirkita_infer_ex.py` with the real trained model (the one that produced the 96.68% result) to get a final, current terminal-output screenshot/recording for the video demo — the earlier screenshots in this conversation were from an intermediate model, not this final deduplicated + debiased one.
6. Read through the whole document once, top to bottom, before pasting into the actual Google Form — small numbers (word counts, percentages) are easy to lose track of after this many edits.

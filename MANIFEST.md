# BanjirKita AI — File Manifest

## Active code — this is what actually produced your real results

| File | Role |
|---|---|
| `train_flood_classifier_ex.py` | **Current training script.** MobileNetV2 transfer learning, trains on Intel Arc A770 via `torch.xpu`, applies resolution-debiasing to mitigate the dimension shortcut, saves a checkpoint *before* attempting ONNX export (so a failed export doesn't lose a completed training run), and supports `--export-only <checkpoint>` to re-export without retraining. This produced your 96.68% accuracy / 97.08% F1 result. |
| `banjirkita_infer_ex.py` | **Current inference/demo script.** Loads the MobileNetV2 ONNX model, runs on Arc A770 GPU via OpenVINO (`PERFORMANCE_HINT: LATENCY`), uses a real photo from `./data` for the demo (not random noise), and uses real images for NNCF INT8 calibration (not random noise) when `./data` is available. |
| `check_dataset_sanity.py` | **Diagnostic tool.** Checks for near-duplicate images, dimension-based shortcuts, a raw-pixel "dumb baseline," and — the newest addition — `blur_shortcut_baseline_check()`, which measures whether the resolution-debiasing mitigation actually closes the shortcut on your real data (confirmed: 0.500 → 0.550, i.e. mitigation is working). |
| `dedupe_flood_class.py` | **One-time utility**, already run. Removed ~4,150 near-duplicate images from the flood class via perceptual hashing before training. Kept here in case you want to re-run or verify. |

**Correct order to reproduce everything from scratch:**
```
python dedupe_flood_class.py --data_dir ./data --apply --backup_dir ./data/flood_removed
python check_dataset_sanity.py --data_dir ./data
python train_flood_classifier_ex.py --data_dir ./data --epochs 10
python banjirkita_infer_ex.py
```

---

## Superseded — kept for reference only, do not use for the actual submission

| File | Why it's superseded |
|---|---|
| `train_flood_classifier.py` | Early version: a small random-architecture CNN trained from scratch, no MobileNetV2 transfer learning, no resolution-debiasing, no checkpoint-before-export safety. Replaced by `train_flood_classifier_ex.py`. |
| `banjirkita_infer.py` | Early version: paired with the toy CNN above, no real-demo-photo logic, no real-calibration-image logic (those fixes exist only in `banjirkita_infer_ex.py`). |
| `README_inference_demo.md` | Written for the two files above; its instructions no longer match the current pipeline. |

These aren't wrong, exactly — they were honest, working versions of an earlier, less-developed pipeline. But if you run them now instead of the `_ex` versions, you'll get the old toy-model behavior, not the real MobileNetV2 results this submission is based on. Safe to ignore or delete.

# BanjirKita AI — Inference Demo: How to Run It

`banjirkita_infer.py` is tested and runs end-to-end (verified in a sandbox with CPU only). It does exactly what the video script's Demonstration section describes: feed in a photo + rainfall reading → get a risk level + confidence back, via a real OpenVINO compile-and-run pass.

## What's real vs. what's a placeholder

- **Real:** the full pipeline — ONNX model construction, conversion to OpenVINO IR, device selection, compiled inference, softmax, rainfall fusion. This is the actual mechanism, tested and working.
- **Placeholder:** the CNN weights are random/untrained. This is disclosed in the script's docstring. Swap in your trained model before the real demo — nothing else in the pipeline needs to change.

## Running it on your machine (to actually use the Arc A770)

```bash
pip install openvino onnx numpy
python banjirkita_infer.py
```

The `available_devices` line in the output will tell you what OpenVINO sees. On this sandbox it printed `['CPU']` because there's no GPU here. On your machine, if the Intel GPU plugin is set up correctly, it should print `['CPU', 'GPU']` and the script will automatically prefer `'GPU'` — that's the Arc A770 being picked.

**If `'GPU'` doesn't show up in `available_devices` on your machine:** it usually means the Intel Compute Runtime (OpenCL driver for the GPU plugin) isn't installed. On Linux that's the `intel-opencl-icd` package; on Windows it's usually already included with the standard Intel Arc graphics driver — worth double-checking before you rely on it for the demo.

## Recording the video demo (Section 4 of the script)

Run the script, screen-record the terminal output showing the risk result — that satisfies "real AI inference" for the video. If `'GPU'` shows in the device list, even better: it visibly proves the Arc A770 is doing the work, not just a CPU fallback.

## Next steps (after Stage 1, if you have time)

- Replace `build_toy_classifier_onnx()` with your actual trained model (once you have real photo/rainfall training data)
- NNCF INT8 quantization is now wired in and tested (`quantize_model()`), along with an FP32-vs-INT8 latency benchmark — this directly covers the "Optimized frameworks & toolkits used?" line in the Intel Technology rubric.

  **Important honesty note:** at this toy model's scale (sub-millisecond inference), the measured speedup is noisy and swings between roughly 0.8x and 2.7x run to run — system/interpreter noise dominates at that timescale. **Do not quote a specific speedup number from the toy model in the submission.** Once the real trained model (more layers, realistic input size) replaces the placeholder, re-run the same benchmark — the code doesn't need to change, and the result at that point should be stable enough to actually report.
- Wire the text/report input in, not just photos, to match the "multimodal" claim in the pitch

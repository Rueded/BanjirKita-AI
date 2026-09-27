"""
BanjirKita AI - OpenVINO Inference Pipeline (v3 - MobileNetV2 Optimized)
"""

import os
import time
import numpy as np
import onnx
from onnx import helper, TensorProto
import openvino as ov
import nncf

TRAINED_MODEL_PATH = "flood_risk_model_trained.onnx"
IMG_SIZE = 224

def build_toy_classifier_onnx(path="flood_risk_model.onnx"):
    input_tensor = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [1, 3, IMG_SIZE, IMG_SIZE]
    )
    output_tensor = helper.make_tensor_value_info(
        "risk_logits", TensorProto.FLOAT, [1, 2]
    )

    conv_w = (np.random.randn(8, 3, 3, 3) * 0.1).astype(np.float32)
    conv_w_init = helper.make_tensor("conv_w", TensorProto.FLOAT, conv_w.shape, conv_w.flatten().tolist())
    fc_w = (np.random.randn(2, 8) * 0.1).astype(np.float32)
    fc_w_init = helper.make_tensor("fc_w", TensorProto.FLOAT, fc_w.shape, fc_w.flatten().tolist())

    conv_node = helper.make_node("Conv", ["image", "conv_w"], ["conv_out"], kernel_shape=[3, 3])
    relu_node = helper.make_node("Relu", ["conv_out"], ["relu_out"])
    gap_node = helper.make_node("GlobalAveragePool", ["relu_out"], ["pooled"])
    flatten_node = helper.make_node("Flatten", ["pooled"], ["flat"], axis=1)
    gemm_node = helper.make_node("Gemm", ["flat", "fc_w"], ["risk_logits"], transB=1)

    graph = helper.make_graph(
        [conv_node, relu_node, gap_node, flatten_node, gemm_node],
        "flood_risk_toy_model", [input_tensor], [output_tensor], initializer=[conv_w_init, fc_w_init],
    )
    model = helper.make_model(graph, producer_name="banjirkita-ai", opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return path

def load_model_path():
    if os.path.exists(TRAINED_MODEL_PATH):
        return TRAINED_MODEL_PATH, True
    print("=" * 70)
    print("WARNING: flood_risk_model_trained.onnx not found. Using fallback.")
    print("=" * 70)
    return build_toy_classifier_onnx(), False

def convert_to_openvino(onnx_path):
    return ov.convert_model(onnx_path)

def compile_and_run(ov_model, image_array, device_preference=("GPU", "CPU")):
    core = ov.Core()
    available = core.available_devices
    chosen = next((d for d in device_preference if d in available), "CPU")
    core.set_property(chosen, {"PERFORMANCE_HINT": "LATENCY"})
    compiled_model = core.compile_model(ov_model, chosen)
    results = compiled_model(image_array)
    logits = results[compiled_model.output(0)]
    return logits, chosen, available

def classify_risk(logits, rainfall_mm):
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    flood_probability = float(probs[0][1])

    if rainfall_mm > 150 or flood_probability > 0.80:
        risk_level = "HIGH"
    elif rainfall_mm > 60 or flood_probability > 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_level": risk_level,
        "visual_flood_probability": round(flood_probability, 3),
        "rainfall_mm": rainfall_mm,
    }

def load_real_calibration_images(data_dir="./data", n=50, img_size=IMG_SIZE):
    """
    Loads up to n real images from data_dir/flood and data_dir/non_flood
    for NNCF calibration, using the same ImageNet normalization as
    train_flood_classifier_ex.py. Returns None if unavailable.
    """
    from pathlib import Path
    try:
        from PIL import Image
    except ImportError:
        return None

    root = Path(data_dir)
    paths = []
    for cls in ("flood", "non_flood"):
        cls_dir = root / cls
        if cls_dir.is_dir():
            paths.extend(sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")))
    if not paths:
        return None

    rng = np.random.default_rng(42)
    if len(paths) > n:
        idx = rng.choice(len(paths), size=n, replace=False)
        paths = [paths[i] for i in idx]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    samples = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB").resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)
            samples.append(arr[np.newaxis, ...].astype(np.float32))
        except Exception:
            pass
    return samples if samples else None


def quantize_model(ov_model, calibration_samples=None, num_calibration_samples=50):
    """
    HONESTY NOTE: if calibration_samples isn't provided, falls back to
    random noise -- proves the NNCF pipeline runs, but does NOT produce
    a meaningfully calibrated INT8 model. Printed loudly, not silent.
    """
    if calibration_samples is None:
        print(
            "WARNING: no real calibration images provided -- falling "
            "back to random noise. Pass real held-out images (see "
            "load_real_calibration_images()) before trusting any INT8 result."
        )
        calibration_samples = [
            np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
            for _ in range(num_calibration_samples)
        ]
    calibration_dataset = nncf.Dataset(calibration_samples, transform_func=lambda x: x)
    quantized_model = nncf.quantize(
        ov_model, calibration_dataset, subset_size=len(calibration_samples)
    )
    return quantized_model

def benchmark_latency(ov_model, image_array, device, n_runs=100, n_warmup=20):
    core = ov.Core()
    core.set_property(device, {"PERFORMANCE_HINT": "LATENCY"})
    compiled_model = core.compile_model(ov_model, device)

    for _ in range(n_warmup):
        compiled_model(image_array)

    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        compiled_model(image_array)
        timings.append(time.perf_counter() - start)

    return {
        "device": device,
        "runs": n_runs,
        "avg_ms": round(float(np.mean(timings)) * 1000, 3),
        "min_ms": round(float(np.min(timings)) * 1000, 3),
        "max_ms": round(float(np.max(timings)) * 1000, 3),
    }

def load_demo_image(data_dir="./data", img_size=IMG_SIZE, prefer_class="flood"):
    """
    Loads one real image from the dataset to use for the demo run,
    instead of random noise. A random-noise input produces a
    meaningless output regardless of how good the model is -- this
    makes the printed 'Flood risk result' actually demonstrate
    something. Falls back to random noise (with a loud warning) only
    if no real images are found.
    """
    from pathlib import Path
    try:
        from PIL import Image
    except ImportError:
        return None, None

    cls_dir = Path(data_dir) / prefer_class
    if cls_dir.is_dir():
        candidates = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        if candidates:
            p = candidates[0]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = Image.open(p).convert("RGB").resize((img_size, img_size))
            arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
            arr = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            return arr, p.name
    return None, None


if __name__ == "__main__":
    model_path, is_trained = load_model_path()
    ov_model = convert_to_openvino(model_path)

    np.random.seed(42)
    sample_image, demo_image_name = load_demo_image()
    if sample_image is None:
        print(
            "WARNING: no real image found in ./data for the demo -- falling "
            "back to random noise. The 'Flood risk result' below is "
            "MEANINGLESS in that case, regardless of model quality."
        )
        sample_image = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    else:
        print(f"Using real demo image: {demo_image_name}")
    sample_rainfall_mm = 95.0

    logits, device_used, available_devices = compile_and_run(ov_model, sample_image)
    result = classify_risk(logits, sample_rainfall_mm)

    print(f"\nModel in use: {model_path} ({'TRAINED MobileNetV2' if is_trained else 'UNTRAINED PLACEHOLDER'})")
    print(f"Available OpenVINO devices: {available_devices}")
    print(f"Inference ran on: {device_used}")
    print(f"Flood risk result: {result}")

    print("\n--- FP32 vs INT8 (NNCF) Latency Benchmark ---")
    fp32_stats = benchmark_latency(ov_model, sample_image, device_used)
    print(f"FP32 (Baseline):  {fp32_stats}")

    print("\n[INFO] Starting NNCF INT8 Quantization...")
    real_calib = load_real_calibration_images()
    if real_calib:
        print(f"Using {len(real_calib)} real images for NNCF calibration.")
    quantized_model = quantize_model(ov_model, calibration_samples=real_calib)
    int8_stats = benchmark_latency(quantized_model, sample_image, device_used)
    print(f"INT8 (Quantized): {int8_stats}")

    speedup = fp32_stats["avg_ms"] / int8_stats["avg_ms"] if int8_stats["avg_ms"] > 0 else float("nan")
    print(f"\nSpeedup: {speedup:.2f}x on {device_used} (FP32 -> INT8)")

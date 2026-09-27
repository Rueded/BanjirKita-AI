import os
import time
import numpy as np
import onnx
from onnx import helper, TensorProto
import openvino as ov
import nncf


TRAINED_MODEL_PATH = "flood_risk_model_trained.onnx"


def build_toy_classifier_onnx(path="flood_risk_model.onnx"):
    input_tensor = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    output_tensor = helper.make_tensor_value_info(
        "risk_logits", TensorProto.FLOAT, [1, 2]
    )

    conv_w = (np.random.randn(8, 3, 3, 3) * 0.1).astype(np.float32)
    conv_w_init = helper.make_tensor(
        "conv_w", TensorProto.FLOAT, conv_w.shape, conv_w.flatten().tolist()
    )
    fc_w = (np.random.randn(2, 8) * 0.1).astype(np.float32)
    fc_w_init = helper.make_tensor(
        "fc_w", TensorProto.FLOAT, fc_w.shape, fc_w.flatten().tolist()
    )

    conv_node = helper.make_node("Conv", ["image", "conv_w"], ["conv_out"], kernel_shape=[3, 3])
    relu_node = helper.make_node("Relu", ["conv_out"], ["relu_out"])
    gap_node = helper.make_node("GlobalAveragePool", ["relu_out"], ["pooled"])
    flatten_node = helper.make_node("Flatten", ["pooled"], ["flat"], axis=1)
    gemm_node = helper.make_node("Gemm", ["flat", "fc_w"], ["risk_logits"], transB=1)

    graph = helper.make_graph(
        [conv_node, relu_node, gap_node, flatten_node, gemm_node],
        "flood_risk_toy_model", [input_tensor], [output_tensor],
        initializer=[conv_w_init, fc_w_init],
    )
    model = helper.make_model(
        graph, producer_name="banjirkita-ai",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def load_model_path():
    if os.path.exists(TRAINED_MODEL_PATH):
        return TRAINED_MODEL_PATH, True
    print("=" * 70)
    print("WARNING: flood_risk_model_trained.onnx not found.")
    print("Falling back to an UNTRAINED placeholder model (random weights).")
    print("=" * 70)
    return build_toy_classifier_onnx(), False


def convert_to_openvino(onnx_path):
    return ov.convert_model(onnx_path)


def compile_and_run(ov_model, image_array, device_preference=("GPU", "CPU")):
    core = ov.Core()
    available = core.available_devices
    chosen = next((d for d in device_preference if d in available), "CPU")
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


def load_real_calibration_images(data_dir="./data", n=20, img_size=64):
    """
    Loads up to n real images from data_dir/flood and data_dir/non_flood
    (the same layout train_flood_classifier.py expects) to use as NNCF
    calibration data. Returns None if the folder doesn't exist or has
    no images -- caller must then decide how to handle that honestly.
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

    paths = paths[:n]
    samples = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((img_size, img_size))
        arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        samples.append(arr[np.newaxis, ...])
    return samples


def quantize_model(ov_model, calibration_samples=None, num_calibration_samples=20):
    """
    Post-training INT8 quantization via NNCF.

    HONESTY NOTE: calibration data determines how well-calibrated the
    INT8 model actually is. If calibration_samples isn't provided, this
    falls back to random noise -- which proves the NNCF pipeline
    mechanics run end-to-end, but does NOT produce a meaningfully
    calibrated INT8 model, even if the model being quantized is
    otherwise a real trained classifier. This is printed loudly below,
    not left as a silent assumption.
    """
    if calibration_samples is None:
        print(
            "WARNING: quantize_model() called with no real calibration "
            "images -- falling back to random noise. This proves the "
            "NNCF pipeline runs, but produces a poorly-calibrated INT8 "
            "model. Pass real held-out flood/non_flood images (see "
            "load_real_calibration_images()) before quoting any INT8 "
            "result as meaningful."
        )
        calibration_samples = [
            np.random.rand(1, 3, 64, 64).astype(np.float32)
            for _ in range(num_calibration_samples)
        ]
    calibration_dataset = nncf.Dataset(calibration_samples, transform_func=lambda x: x)
    quantized_model = nncf.quantize(
        ov_model, calibration_dataset, subset_size=len(calibration_samples)
    )
    return quantized_model


def benchmark_latency(ov_model, image_array, device, n_runs=50, n_warmup=5):
    core = ov.Core()
    compiled_model = core.compile_model(ov_model, device)
    for _ in range(n_warmup):
        compiled_model(image_array)
    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        compiled_model(image_array)
        timings.append(time.perf_counter() - start)
    return {
        "device": device, "runs": n_runs,
        "avg_ms": round(float(np.mean(timings)) * 1000, 3),
        "min_ms": round(float(np.min(timings)) * 1000, 3),
        "max_ms": round(float(np.max(timings)) * 1000, 3),
    }


if __name__ == "__main__":
    model_path, is_trained = load_model_path()
    ov_model = convert_to_openvino(model_path)

    np.random.seed(42)
    sample_image = np.random.rand(1, 3, 64, 64).astype(np.float32)
    sample_rainfall_mm = 95.0

    logits, device_used, available_devices = compile_and_run(ov_model, sample_image)
    result = classify_risk(logits, sample_rainfall_mm)

    print(f"\nModel in use: {model_path} ({'TRAINED' if is_trained else 'UNTRAINED PLACEHOLDER'})")
    print(f"Available OpenVINO devices in this environment: {available_devices}")
    print(f"Inference ran on: {device_used}")
    print(f"Flood risk result: {result}")

    print("\n--- FP32 vs INT8 (NNCF) latency comparison ---")
    fp32_stats = benchmark_latency(ov_model, sample_image, device_used)
    print(f"FP32 (baseline):  {fp32_stats}")

    real_calib_images = load_real_calibration_images()
    if real_calib_images:
        print(
            f"Found {len(real_calib_images)} images in ./data for NNCF "
            f"calibration -- verify these are the real Kaggle dataset, "
            f"not synthetic placeholders, before trusting the INT8 result."
        )
    quantized_model = quantize_model(ov_model, calibration_samples=real_calib_images)
    int8_stats = benchmark_latency(quantized_model, sample_image, device_used)
    print(f"INT8 (NNCF):      {int8_stats}")

    speedup = fp32_stats["avg_ms"] / int8_stats["avg_ms"] if int8_stats["avg_ms"] > 0 else float("nan")
    print(f"Speedup: {speedup:.2f}x on {device_used}")

import argparse
import copy
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from PIL import Image, ImageFilter

# 升级到 224x224 以适配预训练的 MobileNetV2
IMG_SIZE = 224

CHECKPOINT_PATH = "flood_model_checkpoint.pth"


class FloodImageDataset(Dataset):
    """加载真实的洪水/非洪水二分类数据集"""
    def __init__(self, root_dir, img_size=IMG_SIZE, apply_resolution_debias=True, seed=42):
        self.img_size = img_size
        self.samples = []
        self.apply_resolution_debias = apply_resolution_debias
        self._rng = np.random.default_rng(seed)
        root = Path(root_dir)

        for cls_name, label in [("flood", 1), ("non_flood", 0)]:
            cls_dir = root / cls_name
            if not cls_dir.is_dir():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((p, label))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found under '{root_dir}/flood' or '{root_dir}/non_flood'."
            )

        n_flood = sum(1 for _, l in self.samples if l == 1)
        n_nonflood = len(self.samples) - n_flood
        print(f"Loaded dataset: {n_flood} flood, {n_nonflood} non-flood "
              f"({len(self.samples)} total).")
        if apply_resolution_debias:
            print(
                "Resolution/compression de-biasing is ON: every image goes through a "
                "random downsample-upsample + blur + JPEG re-compress pass before "
                "training. This mitigates -- but per sandbox testing does NOT fully "
                "eliminate -- a diagnosed shortcut where flood images (native high-res "
                "sources) and non_flood images (already ~224x224, minimal resize) carry "
                "systematically different blur signatures. Verify with "
                "check_dataset_sanity.py's blur_shortcut_baseline_check() on YOUR real "
                "data, both before and after enabling this, before trusting any accuracy "
                "number from this dataset."
            )

    def __len__(self):
        return len(self.samples)

    def _resolution_debias(self, img):
        """
        HONESTY NOTE: this reduces but does not prove-eliminate the
        flood/non_flood resolution+compression shortcut diagnosed via
        check_dataset_sanity.py's dimension_only_baseline_check (which
        hit a hard 1.0000 on the real data).

        Sandbox testing on synthetic reproductions compared two versions:
          - random intermediate size (56-200px): shortcut still separable
            at ~0.875 accuracy using blur alone (weak mitigation)
          - FIXED small bottleneck (48px) for every image regardless of
            class or native resolution: reduced to ~0.708 (meaningfully
            better, still not proof of zero shortcut)
        This uses the fixed-bottleneck version since it tested stronger.
        The synthetic test is a proxy, not a guarantee for your real
        images -- re-run check_dataset_sanity.py's
        blur_shortcut_baseline_check() on your actual data, both with
        this disabled and enabled, before trusting any accuracy number.
        """
        # 1. FORCE every image through the same small bottleneck resolution,
        #    regardless of native size -- this is what actually destroys the
        #    "started at native high-res" advantage. A random *range* (tried
        #    first) was measurably weaker because a 900x700 image resized to
        #    e.g. 180px still retains much more real detail than a 224x224
        #    image resized to that same 180px -- the starting resolution
        #    still leaks through unless the bottleneck is small AND fixed.
        bottleneck = 48
        img = img.resize((bottleneck, bottleneck), Image.BILINEAR)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # 2. random Gaussian blur (most of the time, not always)
        if self._rng.random() < 0.7:
            radius = float(self._rng.uniform(0.3, 1.2))
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 3. random JPEG re-compression -- targets the filesize gap seen in
        #    the real data (non_flood ~11KB vs flood ~821KB average), which
        #    implies a compression-artifact shortcut on top of the resolution one
        quality = int(self._rng.integers(40, 95))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.apply_resolution_debias:
            img = self._resolution_debias(img)
        else:
            img = img.resize((self.img_size, self.img_size))

        arr = np.asarray(img, dtype=np.float32) / 255.0

        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(arr), label


def get_pretrained_model():
    """使用轻量级预训练模型 MobileNetV2 替代旧版 FloodCNN"""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # 修改最后一层分类器以适应我们的二分类任务 (0: non_flood, 1: flood)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model


def train(data_dir, epochs=8, batch_size=32, lr=1e-4, val_split=0.2, seed=42):
    # 1. 自动检测硬件 (XPU 优先)
    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n🚀 [HARDWARE] Training will run on: {device}\n")

    torch.manual_seed(seed)
    dataset = FloodImageDataset(data_dir)

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    # 开启 4 线程读取，并使用 pin_memory 加速数据送入显存
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_pretrained_model().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metrics = {}

    print(f"Training on {n_train} images, validating on {n_val} images, {epochs} epochs.\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Training Loop
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            opt.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / n_train

        # Validation Loop
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)

                # 将数据移回 CPU 用于计算指标
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 竞赛级评估指标计算
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        # 计算假阴性率 (False Negative Rate) - 漏报洪水的概率，非常重要！
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.3f} | F1: {f1:.3f} | Recall: {recall:.3f} | FNR: {fnr:.3f}")

        # 仅保存在验证集上 F1 表现最好的模型
        if f1 > best_f1:
            best_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            best_metrics = {
                "Accuracy": acc, "Precision": precision,
                "Recall": recall, "F1": f1, "FNR": fnr, "CM": cm
            }

    # 恢复最佳权重
    model.load_state_dict(best_model_wts)

    # 【关键修复】训练一结束立刻落盘，不要等到 export_to_onnx() 才第一次保存。
    # 上一次运行就是在这里之后的 ONNX 导出步骤崩溃（缺 onnx/onnxscript 包），
    # 导致训练完的权重只存在内存里、进程一崩溃就全部丢失，10 个 epoch 白跑。
    # 现在训练结果先落盘，后面 export_to_onnx() 哪怕再失败，也可以从这个
    # checkpoint 直接重新导出，不需要重新训练。
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\n💾 Checkpoint saved to: {CHECKPOINT_PATH} (safe even if ONNX export fails below)")

    print("\n" + "="*50)
    print("🏆 Best Model Quantitative Results:")
    print(f"Accuracy:  {best_metrics['Accuracy']:.4f}")
    print(f"Precision: {best_metrics['Precision']:.4f}")
    print(f"Recall:    {best_metrics['Recall']:.4f} (High recall is crucial for safety!)")
    print(f"F1 Score:  {best_metrics['F1']:.4f}")
    print(f"False-Negative Rate (Missed Floods): {best_metrics['FNR']:.4%}")
    print("Confusion Matrix:\n", best_metrics['CM'])
    print("="*50 + "\n")

    return model, device


def export_to_onnx(model, device, path="flood_risk_model_trained.onnx"):
    model.eval()
    # 模拟真实输入的张量大小 (Batch=1, Channels=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["image"],
        output_names=["risk_logits"],
        opset_version=13,  # 13 以保证更广的 OpenVINO 兼容性
    )
    print(f"✅ Successfully exported best model to: {path}")
    return path


def export_only_from_checkpoint(checkpoint_path, output_path):
    """
    独立的补救入口：如果训练已经完成、checkpoint 已经存在，但 ONNX 导出
    这一步失败了（比如这次缺包），不需要重新训练——直接从 checkpoint
    加载权重，重新跑一次导出即可。

    用法（训练已完成、只是导出失败时）：
        python train_flood_classifier_ex.py --export-only flood_model_checkpoint.pth
    """
    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading checkpoint from {checkpoint_path} onto {device}...")
    model = get_pretrained_model().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    print("Checkpoint loaded successfully. Re-attempting ONNX export...")
    export_to_onnx(model, device, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BanjirKita's flood image classifier.")
    parser.add_argument("--data_dir", default="./data", help="Folder containing flood/ and non_flood/")
    parser.add_argument("--epochs", type=int, default=10)  # 建议跑 10 轮
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="flood_risk_model_trained.onnx")
    parser.add_argument("--export-only", default=None, metavar="CHECKPOINT_PATH",
                         help="Skip training entirely; load an existing .pth checkpoint "
                              "and just (re-)run the ONNX export step. Use this if training "
                              "already succeeded but export failed for an unrelated reason "
                              "(e.g. missing onnx/onnxscript package).")
    args = parser.parse_args()

    if args.export_only:
        export_only_from_checkpoint(args.export_only, args.output)
    else:
        # 执行训练与导出
        model, device = train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
        export_to_onnx(model, device, args.output)

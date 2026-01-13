# CIFAR-10 CNN（PyTorch Lightning）使用说明

本仓库提供一个基于 **PyTorch Lightning** 的 CIFAR-10 训练/测试脚本（默认模型：**ResNet-18 CIFAR 适配版**，属于 CNN）。

## 目录结构（你只需要关心这些）

- `train.py`：训练入口（训练结束会自动用 best checkpoint 跑测试集并输出 `test_acc`）
- `src/`：模型、数据、LightningModule
- `_data/`：数据集目录（可复用，避免重复下载）
- `_outputs/`：训练输出目录（日志、checkpoint）

## 环境准备（服务器推荐）

建议使用 GPU 环境，并确保 `torch.cuda.is_available()` 为 True。

安装训练依赖（不包含 torch/torchvision；它们应由你的环境/conda 提供）：

```bash
pip install -r requirements.docker.txt
```

> 如果你的环境里没有 torch/torchvision：请先安装 GPU 版 PyTorch，再执行上面的 pip 安装。

## 训练（正式跑分）

在仓库根目录执行（默认训练 200 epochs）：

```bash
python train.py \
  --data_dir ./_data \
  --output_dir ./_outputs \
  --download \
  --accelerator gpu \
  --devices 1 \
  --max_epochs 200 \
  --batch_size 256 \
  --num_workers 8
```

- 训练会从训练集切分验证集：`--val_size 5000`（默认），避免用测试集做模型选择。
- 训练结束会打印测试集指标：`test_acc`、`test_loss`。

## 快速自检（不追分）

用于验证代码/数据/依赖是否正常：

```bash
python train.py \
  --data_dir ./_data \
  --output_dir ./_outputs \
  --download \
  --accelerator gpu \
  --devices 1 \
  --max_epochs 1 \
  --limit_train_batches 2 \
  --limit_val_batches 2 \
  --limit_test_batches 2 \
  --batch_size 64 \
  --num_workers 2
```

## 结果与产物在哪里

- **Checkpoint**：`_outputs/<experiment>/checkpoints/`（默认 experiment 为 `resnet18`）
- **TensorBoard 日志**：`_outputs/<experiment>/tb/`

## 提交用的“测试证明”（建议提交到 GitHub）

为避免把大文件（数据集、checkpoint）提交到 GitHub，建议提交**小文本**证明你达标：

1) 用 checkpoint 重新跑一次测试并输出 json：

```bash
python eval.py \
  --data_dir ./_data \
  --ckpt_path ./_outputs/resnet18/checkpoints/<your_best.ckpt> \
  --accelerator gpu \
  --devices 1
```

它会写入：`reports/test_metrics.json`

2) 把终端输出保存成日志文件（可提交）：

```bash
python eval.py \
  --data_dir ./_data \
  --ckpt_path ./_outputs/resnet18/checkpoints/<your_best.ckpt> \
  --accelerator gpu \
  --devices 1 | tee reports/test_log.txt
```

查看 TensorBoard：

```bash
tensorboard --logdir ./_outputs --bind_all --port 6006
```


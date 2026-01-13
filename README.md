# CIFAR-10 CNN (PyTorch Lightning)

目标：用 **PyTorch + PyTorch Lightning** 训练一个 **CNN**（默认 ResNet-18 CIFAR 适配版），在 CIFAR-10 测试集达到 **Top-1 Accuracy > 93%**。

## 目录结构

- `train.py`: 训练/测试入口
- `src/`
  - `data.py`: CIFAR-10 DataModule
  - `model.py`: ResNet-18（CIFAR 适配）模型
  - `lit_module.py`: LightningModule（loss/metrics/optim/scheduler）

## 环境安装（本地或服务器）

建议 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练（默认配置）

下面示例把数据与输出都放在你指定的服务器目录下（建议分开子目录）：

```bash
python train.py \
  --data_dir /root/autodl-tmp/cifar10-cnn/data \
  --output_dir /root/autodl-tmp/cifar10-cnn/outputs \
  --max_epochs 200 \
  --batch_size 256 \
  --num_workers 8
```

训练结束后会自动用 **best checkpoint** 在测试集评估并打印 `test/acc`。  
验证集默认从训练集切分 `--val_size 5000`，避免把测试集用于模型选择（防止“成绩泄漏”）。

## Docker（GPU，4090）

镜像可读性优先：基于官方 PyTorch CUDA runtime 镜像（内部自带 torch/torchvision），额外只装 Lightning 等训练依赖。

### 本地构建镜像

```bash
cd cifar10-cnn
docker build -t yuanextra7/cifar10-lightning-cnn:v0.1 .
```

### 推送到 Docker Hub（public repo 可自动创建）

```bash
docker push yuanextra7/cifar10-lightning-cnn:v0.1
```

### 服务器拉取并训练（挂载数据/输出目录）

> 服务器侧需要 Docker + NVIDIA Container Toolkit 可用，并支持 `--gpus all`。

```bash
mkdir -p /root/autodl-tmp/cifar10-cnn/data /root/autodl-tmp/cifar10-cnn/outputs

docker pull yuanextra7/cifar10-lightning-cnn:v0.1
docker run --rm --gpus all \
  -v /root/autodl-tmp/cifar10-cnn/data:/data \
  -v /root/autodl-tmp/cifar10-cnn/outputs:/outputs \
  yuanextra7/cifar10-lightning-cnn:v0.1 \
  --data_dir /data \
  --output_dir /outputs \
  --download \
  --accelerator gpu \
  --devices 1 \
  --max_epochs 200 \
  --batch_size 256 \
  --num_workers 8
```

### AutoDL 容器里没有 Docker（`docker: command not found`）怎么办？

你贴的环境里 PID 1 是 `bash`（无 systemd），并且没有 `/var/run/docker.sock`，说明当前容器里 **默认用不了 Docker**。

你可以尝试在容器内启动一个 `dockerd`（类似 docker-in-docker，**可能被平台限制**）：

```bash
cd /root/autodl-tmp/cifar10-cnn   # 任选一个目录
# 把仓库里的脚本内容拷贝过去执行，或直接在你代码目录里执行：
bash scripts/autodl_install_docker_dind.sh
```

如果 `docker info` 仍失败，请把：
- `tail -n 200 /var/log/dockerd.log`

贴给我，我们再决定是否需要换方案（例如不用 Docker 在服务器里直接跑训练，或用平台提供的宿主机 Docker 能力）。

## 只跑一次快速自检（不追分）

```bash
python train.py \
  --data_dir ./_data \
  --output_dir ./_outputs \
  --max_epochs 1 \
  --limit_train_batches 2 \
  --limit_val_batches 2 \
  --limit_test_batches 2 \
  --batch_size 64
```

## 常见问题

- **会不会每次都下载 CIFAR-10？**  
  不会。只要 `--data_dir` 目录下已有数据文件，`download=True` 也会跳过下载。

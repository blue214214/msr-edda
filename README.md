# MSR-EDDA

面向滚动轴承跨工况故障诊断的大创项目。模型使用多尺度一维残差卷积提取振动特征，通过双证据头输出类别概率和不确定性，并用三步对抗训练进行域适应。

## 当前状态

| 内容 | 当前用途 |
| --- | --- |
| JNU 原始数据和加载器 | 可读取原始信号，并通过训练入口做跨工况调试。 |
| HIT 原始数据 | 完整的 NPY 可直接读取。`configs/hit.yaml` 是占位配置，训练入口还需接入 HIT 加载器。 |
| 模型和训练 | 已有 MS-ResCNN、双证据头、域判别器、三步训练引擎和合成数据模式。 |
| 对照实验 | `source_only` 参数已预留，训练器还需支持无目标域批次的情况。MCDDFA 基线和 A5/A6 消融也待实现。 |

## 数据

| 数据集 | 文件 | 内容 |
| --- | --- | --- |
| [JNU](data/JNU) | 12 个单列 CSV，约 78.6 MB | 50 kHz 振动信号。正常、内圈故障、外圈故障、滚动体故障各有 600、800、1000 r/min 三种工况，每个类别与工况组合对应一个文件。 |
| [HIT](data/HIT/full) | 5 个 NPY，约 2.87 GB，使用 Git LFS | 航发中介轴承完整数据。各数组形状为 `(片段数, 8, 20480)`，其中 8 个字段含振动或位移信号及元数据。 |

各文件的来源和 SHA-256 校验值记录在 [JNU 清单](data/JNU/SHA256.json) 和 [HIT 清单](data/HIT/full/SHA256.json)。[JNU 发布页](https://github.com/ClarkGableWang/JNU-Bearing-Dataset) · [HIT 发布页](https://github.com/HouLeiHIT/HIT-dataset) · [HIT 数据论文](https://doi.org/10.37965/jdmd.2023.314)

## 开始使用

安装 [Git LFS](https://git-lfs.com/) 后，执行以下命令。已克隆仓库时，进入仓库根目录，从 `git lfs pull` 开始。

```bash
git clone https://github.com/blue214214/msr-edda.git
cd msr-edda
git lfs pull
python -m pip install -r requirements.txt
python tools/selfcheck.py
```

可在 Python 中直接读取原始数据。HIT 示例用内存映射，按需读取数组。

```python
import numpy as np

jnu = np.loadtxt("data/JNU/n600_3_2.csv", dtype=np.float32)
hit = np.load("data/HIT/full/data1.npy", mmap_mode="r", allow_pickle=False)
print(jnu.shape, hit.shape)  # (1501500,) (504, 8, 20480)
```

下面的命令可用于调试训练流程。基础配置默认训练 200 个 epoch，短跑时可在传给 `--config` 的 YAML 中把 `train.epochs` 调小。

```bash
python train.py --config configs/base.yaml --mode synthetic --device cpu
python train.py --config configs/jnu.yaml --src 0 --tgt 1 --seed 0 --device cpu
```

JNU 的 `0/1/2` 分别对应 `600/800/1000 r/min`。配置快照、评测指标和最佳权重写入 `runs/`。模型、损失和指标代码位于 [src/](src)，数据与训练配置位于 [configs/](configs)。

已在 Windows CPU 上分别完成合成数据和 JNU 600 → 800 r/min 的一轮短跑，验证了训练、评估和产物写入。三步梯度流向自检通过。自检中主干实测 104,576 参数，与脚本预期的约 251,000 有出入，单批损失下降也未达到脚本阈值。

## 正式实验前需要修正

- JNU 加载器先滑窗再按比例划分，默认设置下目标域适配集和测试集的边界窗口会共享原始采样点。应先划分原始信号区间，再分别切窗。
- 训练循环按目标域测试集准确率选择最佳权重。应设独立验证集或采用预先固定的训练轮数，最后再评估目标域测试集。
- 当前风险–覆盖率函数优先保留低置信度样本。正式报告 AURC 前，应修正排序并核对指标计算。

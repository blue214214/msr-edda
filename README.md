# MSR-EDDA

大创项目研究变工况下的滚动轴承故障诊断。目前仓库用于保留实验数据快照。

## 数据

- `data/JNU/` 保存 JNU 的 12 个原始 CSV，覆盖 4 类轴承状态和 600、800、1000 r/min 三种转速。上游提交和各文件校验值见 [JNU SHA256.json](data/JNU/SHA256.json)。[JNU 数据来源](https://github.com/ClarkGableWang/JNU-Bearing-Dataset)
- `data/HIT/full/` 保存 Hou 等人发布的完整 HIT 航发中介轴承数据，包括 `data1.npy` 至 `data5.npy`。文件以 Git LFS 跟踪，校验值见 [HIT SHA256.json](data/HIT/full/SHA256.json)。[HIT 数据来源](https://github.com/HouLeiHIT/HIT-dataset) · [原论文](https://doi.org/10.37965/jdmd.2023.314)

## 获取完整数据

安装 [Git LFS](https://git-lfs.com/) 后克隆仓库，再在仓库目录运行 `git lfs pull`。HIT 的 5 个 NPY 文件合计约 2.87 GB。

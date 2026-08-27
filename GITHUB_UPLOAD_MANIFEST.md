# GitHub 上传清单

## 当前模拟暂存结果

- 共 124 个候选文件；精确文件清单见 `GITHUB_UPLOAD_FILELIST.txt`。
- 没有单文件超过 100 MiB。
- 任务 01–07 的候选文件数分别为 22、3、23、20、39、1、12；另有 4 个根目录文档。
- 模拟结果中没有 `data/`、FITS、HDF5、模型权重或训练运行目录。

## 建议上传

- 根目录：`README.md`、`.gitignore`、本清单。
- 每个任务：`requirements.txt` 或 `environment.yml`；数据与运行说明统一保存在根目录 `README.md`。
- 项目实现：`src/`、`configs/`、精选 `notebooks/`。
- 结果展示：体积较小的最终图、汇总指标和方法说明文档。
- 任务 06：只上传我们的 `baseline/tutorial_inference_with_DINGO-T1.ipynb`；运行所需的上游代码、模型和事件配置由 notebook 获取。
- 任务 07：`workspace/notebooks/` 与必要说明；不包含查询结果表和 FITS cutout。

## 不上传

- 所有 `data/` 内容、FITS/HDF5/NumPy 数据和生成星表。
- `.pt`、`.pth`、`.joblib` 等模型权重。
- `runs/`、预测表、缓存、checkpoint、临时文件和本地环境。
- 任务 06/07 中克隆的上游仓库副本。
- 任务 06 的官方教程副本、事件目录、配置副本、采样结果、PDF 和图片。
- `archive/` 目录；任务 04 的 DR3 notebooks 已移到 `notebooks/`，因此会与 DR4/DR5 notebook 一同上传。

## 上传前仍需人工决定

- 选择仓库许可证后添加根目录 `LICENSE`。

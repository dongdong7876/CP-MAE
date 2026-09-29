# `exp/` — B4b / B5 实验脚本

本目录位于 CP-MAE 仓库内部（`CP-MAE/exp/`），随仓库一起上传到 GPU 机器即可直接运行。

**不改动仓库既有源码的任何一行**：`--repo` 默认解析为 `exp/` 的父目录，梯级消融所需的
行为差异一律在运行时以 monkey-patch 完成，L7 的统一编码器作为独立模块注入。
`exp/.gitignore` 已排除 `results/` 与 `__pycache__/`，实验产物不会进版本库。

## 执行顺序

**推荐用 `sh/` 下的一键脚本**（可断点续跑、自动记录耗时与日志）：

```bash
cd CP-MAE/exp/sh
export DATASET_ROOT=/path/to/dataset   # 仅在自动探测失败时需要
./run_all.sh                            # 或 ./run_all.sh 02 03 只跑指定阶段
```

只有需要单独调试时才用下面的逐条命令。

```bash
cd CP-MAE/exp

# 0) 先验证向量化掩码与原实现等价，再做任何计时
python fast_masks.py --self-test
python fast_masks.py --bench            # 看看原实现慢多少

# 1) 审计干净基底（不写数据）
python inject_contamination.py --path ../../dataset --data_name all --audit

# 2) 生成受控污染数据集（E0）
python inject_contamination.py --path ../../dataset --data_name all \
       --rates 0.0,0.05,0.10,0.20,0.30,0.40 --seeds 0,1,2,3,4

# 3) 成本剖析（E3，最便宜，可先做）
python profile_cost.py --dataset WADI --scaling \
       --out results/cost.csv

# 4) 消融梯级（E2）
for ds in SMD SWaT LTDB WADI PSM; do
  python run_ladder.py --dataset $ds \
         --rungs L0,L1,L2,L3,L4,L5,L6 --seeds 0,1,2,3,4 --out results/ladder.csv
done

# 5) 受控污染实验（E1）—— 需先接好污染数据通路，见下节
#    跑完把结果按 §6 schema 写入 results/contamination.csv

# 6) 转储推理统计量（E4），每个模型一次
python dump_stats.py --dataset SMD --seed 0 \
       --K 16 --tag clean --out results/npz/SMD_s0_clean.npz

# 7) 离线分析（E5/E6/E8），零重训
python score_analysis.py gamma   --npz "results/npz/*.npz"
python score_analysis.py calib   --npz "results/npz/*.npz"
python score_analysis.py corr    --npz "results/npz/*.npz"
python score_analysis.py failure --npz "results/npz/*.npz"

# 8) 生成 LaTeX 表体
python aggregate_tables.py ladder --csv results/ladder.csv
python aggregate_tables.py cost   --csv results/cost.csv
```

## 接通污染数据通路（E1 唯一需要你动手的地方）

`CP-MAE/data_factory/data_loader.py` 目前只读原始训练文件，**完全忽略**驱动脚本导出的
`CPMAE_CONTAM_TRAIN`。不打补丁，五个污染率会训练在同一份干净数据上，指标逐位相同
（2026-08-31 的 E1 就是这样报废的）。

一条命令打补丁，幂等，自动留 `.orig` 备份：

```bash
cd CP-MAE/exp
python3 patch_data_loader.py --check     # 只报告状态
python3 patch_data_loader.py             # 应用
python3 patch_data_loader.py --revert    # 回滚
```

它在模块顶部插入 `_maybe_contaminate()`，并在五个 `*SegLoader` 的
`self.scaler.fit(data_train)` 之前各插一行 `data_train = _maybe_contaminate(data_train)`。
验证集与测试集不受影响，标准化器仍然只在训练数据上拟合。

打完补丁**必须**跑验证，PASS 之后才开实验：

```bash
python3 verify_contam_patch.py --dataset SMD --dataset-root ../dataset --rate 0.40
```

⚠️ **不要用 `pd.read_csv(contam).iloc[:, 1:]` 自己手写**。WADI 的污染文件带索引列**和**尾部
标签列，PSM/LTDB 带首列元信息，直接切片会多带 1～2 个"通道"，训练时报
`The size of tensor a (129) must match the size of tensor b (127)`。补丁内部调用的
`contam_io.load_contaminated` 用期望通道数反解布局，五个数据集通用。

然后用环境变量驱动：

```bash
CPMAE_CONTAM_TRAIN=../../dataset/SMD/SMD_train_contam_r0.10_s0.npy \
  python main.py --dataset SMD --seed 0
```

这样主实验的代码路径完全不受影响，污染实验只是多一个环境变量。

## 各脚本一句话说明

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `fast_masks.py` | 向量化覆盖感知掩码 + 等价性自检 + 计时对比 | torch |
| `inject_contamination.py` | 受控污染生成，含实测污染率与逐段类型/时长清单 | numpy, pandas |
| `run_ladder.py` | L0–L7 受控消融梯级，复用项目自身 Solver | torch + 项目 |
| `unified_encoder.py` | L7 统一时空编码器（回答 R1-C15），单独运行可打印注意力预算 | torch + 项目 |
| `memorization_ratio.py` | Memorization Ratio 探针（unmasked 与 masked 两种） | torch + 项目 |
| `record_run.py` | 把 main.py 的输出转成统一 tidy schema | pandas |
| `profile_cost.py` | 参数量/FLOPs/显存/训练时长/延迟/吞吐 + 标定曲线 | torch + 项目 |
| `dump_stats.py` | 转储逐点 μ/σ 与标签，把 γ、校准、分层分析变成离线 | torch + 项目 |
| `score_analysis.py` | γ 扫描、ECE/Brier/可靠性、σ–误差相关、按时长的失败分析 | numpy |
| `aggregate_tables.py` | 结果 CSV → LaTeX 表体 | numpy, pandas |

## 并行：多个窗口分担数据集

E2 和 E2b 的脚本接受数据集名作为参数，**每个数据集写自己的结果文件**，因此并发窗口
之间不共享任何可写文件：

```bash
# 窗口 1                  窗口 2                  窗口 3
./03_ladder.sh LTDB       ./03_ladder.sh WADI     ./03_ladder.sh PSM
```

多卡时用 `GPU` 环境变量把每个窗口钉在不同设备上：

```bash
GPU=0 ./03_ladder.sh LTDB
GPU=1 ./03_ladder.sh WADI
GPU=2 ./03_ladder.sh PSM
```

跑完把所有结果文件合并成表体（通配符会同时收进旧的共用 `ladder.csv`）：

```bash
python3 aggregate_tables.py ladder --csv "results/ladder*.csv"
```

并发安全性说明：日志、`.done` 标记、检查点目录（`cpt_ladder_<rung>/<dataset>_checkpoint.pth`）
和结果文件都按数据集区分，互不覆盖。唯一的共享资源是显存——单卡上并发三个训练时，
WADI（127 通道，batch 512）是最容易触发 OOM 的一个。

## 断点续跑的两级粒度

| 级别 | 机制 | 作用 |
|---|---|---|
| 阶段/数据集 | `results/.done/<tag>` 标记文件 | 已完成的数据集整体跳过，秒级 |
| 单元格 | `run_ladder.py` 读回 `ladder.csv` 中已有的 `(dataset, rung, seed)` | 数据集跑到一半崩溃时，从崩掉的那一格继续，不重跑前面的 |

失败时 `set -e` 会在 `mark` 之前退出，所以失败的数据集不会留下标记，重跑该阶段即可继续：

```bash
./run_all.sh 03          # 已完成的数据集打印 skip，未完成的从断点继续
```

强制重做某个数据集：删掉它的标记；强制重做某一格：从 `ladder.csv` 里删掉对应行，
或整体加 `--no-resume`。

**若崩溃前的代码有 bug，务必先删掉受影响的行**，否则续跑会把错误结果当成已完成：

```bash
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("results/ladder.csv")
df[df.dataset != "LTDB"].to_csv("results/ladder.csv", index=False)   # 换成受影响的数据集
EOF
```

`aggregate_tables.py` 在读入时会按单元格去重并保留最后一次写入，因此崩溃留下的重复行
不会污染最终表格，但它无法分辨"重复"与"错误"——受 bug 影响的行必须手动删除。

## 已知的环境陷阱（已修）

| 现象 | 原因 | 处置 |
|---|---|---|
| `bash: ./05_contamination_runs.sh: Permission denied` | 文件传到服务器时执行位丢失（scp / 拖拽 / 挂载盘常见） | `chmod +x sh/*.sh`，或直接 `bash 05_contamination_runs.sh`；`00_preflight.sh` 现在会自动补回 |
| `AttributeError: module 'importlib' has no attribute 'util'` | `importlib.util` 是子模块，裸 `import importlib` 不保证可访问；部分环境恰好预导入了它 | 改为直接 `__import__` 逐个试导入，并顺带打印版本号与显存 |
| `main.py` 找不到 `config/` 或数据集 | `main.py` 以 **相对路径** 解析 `config/<ds>.conf`、`dataset/<ds>/` 与 `results/`，而 05/06 从 `exp/` 目录调用它 | 05/06 改为 `bash -c "cd $REPO && ... python3 main.py ..."` |
| `declare: -A: invalid option` | macOS 自带 bash 3.2 不支持关联数组 | `config.sh` 加版本护栏，低于 bash 4 立即报错并给出提示 |

## 已自检 / 未自检

- **已在本机自检**：`inject_contamination.py`（速率精确、跨种子有差异、可复现、不重叠、三类异常与三档时长齐备）、`score_analysis.py`（四个子命令在合成数据上全部跑通）、`aggregate_tables.py`（语法）。
- **已在本机自检**（补充）：全部 10 个 shell 脚本 `bash -n` 通过；`config.sh` 的路径探测、`already()`/`mark()` 断点续跑标记、`timed()` 计时与日志均实测可用；`unified_encoder.py` 的注意力预算报告可在无 torch 环境下运行。
- **已做端到端连线测试**：用桩 `main.py` 与假数据集在沙箱中实跑 E1 的 CP-MAE 分支，验证了工作目录、`CPMAE_CONTAM_TRAIN` 环境变量传递、`record_run.py` 的指标映射与**实测污染率回填**、以及断点标记生效；并用合成结果实测 `aggregate_tables.py` 的 Table 7 / Table 8 表体输出。
- **未自检**：`fast_masks.py`、`run_ladder.py`、`profile_cost.py`、`dump_stats.py`、`memorization_ratio.py`、`unified_encoder.py` 的建模部分 —— 本机无 torch，且没有数据集与 GPU。请务必先跑 `sh/00_preflight.sh`，它会依次检查环境、执行掩码等价性门禁、审计干净基底并打印 L7 预算；通过后再用单数据集单种子试跑 `03_ladder.sh`，确认无误后批量提交。

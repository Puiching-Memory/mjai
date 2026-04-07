# mjai

针对 [mjai.app](https://mjai.app) 麻将 AI 竞赛的强化学习研究仓库。

训练用 PyTorch，部署用 ONNX + Rust/tract，提交物是预构建的原生二进制 + 模型产物。

## 快速开始

```bash
# 环境要求：Python 3.14、Rust 1.94、uv
uv sync                  # 基础依赖（部署）
uv sync --extra train    # 训练依赖（torch, mjai, onnx, rich）
```

两条核心命令：

```bash
uv run python tools/cli.py train --preset single   # 训练
uv run python tools/cli.py infer --seat 0           # 推理
```

不带子命令直接运行 `uv run python tools/cli.py` 会打开 Rich 命令总览页。

## 项目结构

```
bot.py                  部署 bot 入口（BasicMahjongBot + ONNX 推理）
rust_mjai_bot.py        Rust 子进程通信层（SubprocessJsonClient、RustMjaiBot、game utils）
rust_mjai_engine.py     mjai-log 引擎适配器（DockerMjaiLogEngine、InProcessMjaiBotEngine）
rust_mjai_arena.py      mjai Rust arena 的 Python 封装

train/
  inference_spec.py     特征规格常量（INPUT_DIM=1281、ACTION_DIM=14）
  policy_net.py         策略网络定义
  async_self_play.py    异步 actor-learner 自对弈（共享内存 + batched inference）
  self_play.py          同步 ProcessPoolExecutor 自对弈（用于评测）
  async_training_bot.py 异步自对弈 bot
  training_bot.py       同步自对弈 bot
  training_config.py    训练配置与奖励函数
  checkpoints.py        checkpoint 加载与保存
  evaluation.py         评测逻辑
  profiling.py          NVTX profiling 标注
  training_ui.py        Rich UI 组件

tools/
  cli.py                统一 CLI 入口
  train_reinforce.py    底层 async trainer
  export_onnx.py        导出 ONNX + 元数据
  init_policy_checkpoint.py  生成初始 checkpoint
  evaluate_checkpoint.py     独立评测脚本
  profile_async_training.py  Nsight Systems profiling 封装
  validate_competition_image.py  比赛镜像闭环验证
  mjai_oracle.py        Python mjai 包的 oracle 探针（测试用）

native_runtime/         Rust crate
  src/bin/
    mjai-bot-decision   状态机 + 候选编译 + 特征编码（单进程）
    mjai-tract-runtime  ONNX 推理（tract 后端）
    mjai-rust-oracle    Rust 侧 oracle（合约测试用）
```

## 架构

```
┌──────────────────────────── 部署 ────────────────────────────┐
│                                                               │
│  mjai 事件流 → mjai-bot-decision → candidates + features      │
│                (Rust 子进程)         ↓                         │
│                                mjai-tract-runtime → action    │
│                                (Rust 子进程, ONNX)            │
│                                                               │
│  bot.py: 规则动作（和牌/立直/九种九牌）                        │
│        + 原生动作头（弃牌/副露选择）                           │
└───────────────────────────────────────────────────────────────┘

┌──────────────────────────── 训练 ────────────────────────────┐
│                                                               │
│  CPU actor 进程 → 共享内存 → batched inference server (GPU)   │
│  (mjai arena self-play)         ↓                             │
│                            权重同步                           │
│  replay buffer ←── episode tensors                            │
│       ↓                                                       │
│  learner (PPO-style actor-critic)                             │
│       ↓                                                       │
│  周期性后台评测 → best checkpoint 更新                        │
└───────────────────────────────────────────────────────────────┘
```

## 构建原生二进制

```bash
cargo build --release --manifest-path native_runtime/Cargo.toml
```

同一次构建产出 `mjai-bot-decision` 和 `mjai-tract-runtime`。

静态 musl 版本（用于比赛提交）：

```bash
docker build -f native_runtime/Dockerfile.musl -t mjai-tract-runtime-build native_runtime
docker create --name tmp mjai-tract-runtime-build
docker cp tmp:/mjai-tract-runtime artifacts/mjai-tract-runtime
docker rm tmp
```

## 训练

```bash
uv run python tools/cli.py train --preset single
```

未识别参数透传给底层 trainer：

```bash
uv run python tools/cli.py train --preset single --total-learner-steps 200 --actor-processes 12
```

底层 trainer 直接调用：

```bash
uv run python tools/train_reinforce.py \
  --checkpoint artifacts/policy.pt \
  --best-checkpoint artifacts/policy.best.pt \
  --learner-device cuda:0 \
  --inference-device cuda:0 \
  --actor-processes 8 \
  --total-learner-steps 1000 \
  --warmup-steps 4096 \
  --minibatch-size 2048 \
  --evaluation-matches 8
```

### 关键参数

| 参数                     | 说明                                                |
| ------------------------ | --------------------------------------------------- |
| `--actor-processes`      | 常驻 self-play actor 数量                           |
| `--learner-device`       | 优化器所在设备                                      |
| `--inference-device`     | batched policy inference 所在设备                   |
| `--warmup-steps`         | learner 启动前 replay 需积累的最少决策步数          |
| `--policy-sync-interval` | 权重发布到 inference server 的频率                  |
| `--max-policy-lag`       | 防止 learner 消费过旧样本                           |
| `--inference-timeout-ms` | inference 聚合窗口（actor 多时放宽提高 batch size） |
| `--evaluation-matches`   | >0 时周期性后台评测并更新 best checkpoint           |

### 产物

- `artifacts/policy.pt` — 最新 checkpoint
- `artifacts/policy.best.pt` — 评测最优 checkpoint
- `artifacts/training_metrics.jsonl` — 训练指标日志

## 评测

```bash
# 单 checkpoint 自对战
uv run python tools/evaluate_checkpoint.py \
  --checkpoint artifacts/policy.pt --matches 8 --workers 4

# candidate vs best
uv run python tools/evaluate_checkpoint.py \
  --checkpoint artifacts/policy.pt \
  --baseline-checkpoint artifacts/policy.best.pt \
  --matches 16 --workers 4
```

## 导出部署产物

```bash
# 生成初始 checkpoint（调试用）
uv run python tools/init_policy_checkpoint.py --output artifacts/policy.pt --hidden-dims 32 32

# 导出 ONNX + 元数据
uv run python tools/export_onnx.py --checkpoint artifacts/policy.pt --onnx artifacts/policy.onnx
```

## 二进制查找规则

| 组件               | 查找路径                                           | 环境变量覆盖               |
| ------------------ | -------------------------------------------------- | -------------------------- |
| mjai-tract-runtime | `artifacts/` → `native_runtime/target/release/`    | `MJAI_NATIVE_RUNTIME_BIN`  |
| policy.onnx        | `artifacts/policy.onnx`                            | `MJAI_NATIVE_RUNTIME_ONNX` |
| policy.json        | `artifacts/policy.json`                            | `MJAI_NATIVE_RUNTIME_META` |
| mjai-bot-decision  | `artifacts/` → `target/release/` → `target/debug/` | `MJAI_BOT_DECISION_BIN`    |

## Profiling

仓库在训练主循环、actor 请求、batched inference、learner update 上已有 NVTX 标注：

```bash
uv run python tools/profile_async_training.py \
  --output artifacts/nsys_profile \
  --cuda-visible-devices 0 \
  --cuda-memory-usage \
  -- \
  --checkpoint artifacts/policy.pt \
  --best-checkpoint artifacts/policy.best.pt \
  --learner-device cpu \
  --inference-device cuda:0 \
  --actor-processes 4 \
  --total-learner-steps 20 \
  --warmup-steps 512 \
  --minibatch-size 256 \
  --evaluation-matches 0
```

产出 `*.nsys-rep`（GUI 时间线）、`*.sqlite`（中间库）、`*.summary.txt`（摘要）。关闭 NVTX 标注：`MJAI_ENABLE_NVTX=0`。

## 比赛镜像验证

```bash
uv run python tools/validate_competition_image.py
```

在 `docker.io/smly/mjai-client:v3` 中执行：生成提交 ZIP → 解压检查 runtime → 回放固定事件流。

比赛镜像约束：Ubuntu 22.04 / glibc 2.35 / Python 3.10 / Torch 2.0.1 / 无 CUDA / mjai 0.1.9 / AVX2。

```bash
# 指定 runtime 路径
uv run python tools/validate_competition_image.py --runtime-path path/to/mjai-tract-runtime

# 使用已有 ZIP
uv run python tools/validate_competition_image.py --submission-zip path/to/submission.zip
```

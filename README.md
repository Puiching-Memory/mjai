# mjai
一项针对mjai.app麻将AI竞赛的强化学习研究

## 环境

本仓库使用 uv 管理 Python 环境，当前固定为 Python 3.12。

> `mjai==0.2.1` 目前无法在 Python 3.14 下构建，因此不要用 3.14 创建虚拟环境。

## 比赛运行环境

比赛提交实际运行在 `docker.io/smly/mjai-client:v3` 中。该镜像已经实测，不应仅依赖上游仓库当前 `docker/` 目录里的版本声明来推断环境。

实测关键结论：

- 平台为 `linux/amd64`，系统为 Ubuntu 22.04 系列，glibc 为 2.35。
- Python 版本为 `3.10.12`。
- PyTorch 实际版本为 `2.0.1+cu117`，不是上游仓库当前 Dockerfile 中声明的 2.2.x。
- `torch.cuda.is_available()` 为 `False`，因此比赛执行应按 CPU 推理设计。
- Torch CPU 后端可用 MKL、MKLDNN、OpenMP，构建目标明确启用了 AVX2。
- `mjai` 在镜像中的实际版本为 `0.1.9`，不是本地开发环境中的 `0.2.1`。

镜像内实测 Python 包：

- `torch==2.0.1+cu117`
- `triton==2.0.0`
- `numpy==1.25.2`
- `mjai==0.1.9`
- `mahjong==1.2.1`
- `loguru==0.7.0`
- `requests==2.31.0`

镜像内实测工具链：

- `gcc==11.4.0`
- `g++==11.4.0`
- `cmake==3.22.1`
- `rustc==1.73.0-nightly`
- `cargo==1.73.0-nightly`
- `protoc==3.17.3`

镜像中可见的原生库特征：

- `mjai` 带有本地扩展 `mlibriichi.cpython-310-x86_64-linux-gnu.so`
- Torch 自带 `libtorch_cpu.so`、`libtorch.so`、`libc10.so`、`libtorch_cuda.so` 等动态库
- `libtorch_cpu.so` 依赖 `libgomp`、`libstdc++`、`libc10` 和 `libcudart`

对部署的直接约束：

- 如果训练在 CUDA 上进行，部署产物仍应以 CPU 推理为主。
- 如果提交固定模型的原生推理库，编译目标应对齐到 `linux/amd64`、Python 3.10、glibc 2.35、GCC 11、AVX2。
- 需要以比赛镜像实测版本为准，不要假设本地 `uv` 环境中的 `mjai==0.2.1` 或更高版本在比赛环境中可用。

安装依赖：

```powershell
uv sync
```

## 运行

启动机器人：

```powershell
uv run python bot.py 0
```

## 当前策略

bot.py 当前实现的是一个混合控制流机器人，主要行为如下：

- 能和牌时立即和牌。
- 能立直时立即立直。
- 满足九种九牌时直接选择流局。
- Pon 候选只在役牌或已经副露推进手牌时送入原生动作头。
- Chi 候选只在已经副露或能明显推进向听时送入原生动作头。
- 常规弃牌、立直声明后的弃牌、以及 pass/pon/chi 选择都由原生 tract 运行时负责，不再回退到规则动作。
- 如果原生运行时、模型文件或元数据缺失，bot 会直接报错而不是降级运行。

最小可运行示例（PowerShell）：

```powershell
@'
[{"type":"start_game","names":["0","1","2","3"],"id":0}]
[{"type":"start_kyoku","bakaze":"E","dora_marker":"2s","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["E","6p","9m","8m","C","2s","7m","S","6m","1m","S","3s","8m"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]]},{"type":"tsumo","actor":0,"pai":"1m"}]
'@ | uv run python bot.py 0
```

## 第一档原生部署骨架

当前仓库已经补上了一套第一档方案的最小骨架：

- Python 侧继续用 Torch 训练和导出。
- 导出产物为 `model.onnx` 和 `model.json`。
- 部署侧使用 `native_runtime/` 下的 Rust + tract 二进制执行推理。
- 推理二进制通过标准输入读取 JSON，请求里只包含特征向量和合法动作掩码。

Python 侧安装训练与导出依赖：

```powershell
uv sync --extra train
```

生成一个最小可验证 checkpoint：

```powershell
uv run python tools/init_policy_checkpoint.py --output artifacts/policy.pt --hidden-dims 32 32
```

导出 ONNX 与元数据：

```powershell
uv run python tools/export_onnx.py --checkpoint artifacts/policy.pt --onnx artifacts/policy.onnx
```

本机构建 tract 推理器：

```powershell
cargo build --release --manifest-path native_runtime/Cargo.toml
```

如果要拿到更适合比赛镜像直接运行的静态 Linux 二进制，可以用 Docker 产出 `musl` 版本：

```powershell
docker build -f native_runtime/Dockerfile.musl -t mjai-tract-runtime-build native_runtime
docker create --name mjai-tract-runtime-out mjai-tract-runtime-build
docker cp mjai-tract-runtime-out:/mjai-tract-runtime artifacts/mjai-tract-runtime
docker rm mjai-tract-runtime-out
```

推理请求结构说明：

- `features` 是长度为 [train/inference_spec.py](train/inference_spec.py) 中 `INPUT_DIM` 的 `float[]`。
- `legal_actions` 是长度为 [train/inference_spec.py](train/inference_spec.py) 中 `ACTION_DIM` 的 `bool[]`。

本地 smoke test：

```powershell
uv run python -c "import json; from train.inference_spec import ACTION_DIM, INPUT_DIM; print(json.dumps({'features': [0.0] * INPUT_DIM, 'legal_actions': [index == 0 for index in range(ACTION_DIM)]}))" | .\native_runtime\target\release\mjai-tract-runtime.exe artifacts/policy.onnx artifacts/policy.json
```

当前这套骨架已经和 [bot.py](bot.py) 做了联动：

- 原生运行时接管常规弃牌、立直声明后的弃牌、以及 pass/pon/chi 的选择。
- 和牌、荣和、立直声明本身、九种九牌流局仍由规则逻辑处理。
- 可选鸣牌会先经过基础规则过滤，再进入统一动作头参与选择。
- 固定输入维度和动作槽位定义在 [train/inference_spec.py](train/inference_spec.py)。
- 原生运行时缺失、启动失败、通信失败或返回非法动作时会直接失败，不会回退到规则动作。

现阶段解决的是三件事：

- 训练侧与部署侧彻底解耦。
- 部署侧不需要 Torch 和 onnxruntime。
- 可以直接产出一个可执行文件随提交一起带入比赛环境。

bot 会自动查找这些默认路径：

- `artifacts/policy.onnx`
- `artifacts/policy.json`
- `artifacts/mjai-tract-runtime(.exe)`
- `native_runtime/target/release/mjai-tract-runtime(.exe)`

这些文件至少要满足一组可用，否则 [bot.py](bot.py) 会在启动时直接失败。

也可以显式指定环境变量：

```powershell
$env:MJAI_NATIVE_RUNTIME_BIN = (Resolve-Path .\native_runtime\target\release\mjai-tract-runtime.exe)
$env:MJAI_NATIVE_RUNTIME_ONNX = (Resolve-Path .\artifacts\policy.onnx)
$env:MJAI_NATIVE_RUNTIME_META = (Resolve-Path .\artifacts\policy.json)
```

桥接层端到端验证：

```powershell
@'
[{"type":"start_game","names":["0","1","2","3"],"id":0}]
[{"type":"start_kyoku","bakaze":"E","dora_marker":"2s","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["E","6p","9m","8m","C","2s","7m","S","6m","1m","S","3s","8m"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]]},{"type":"tsumo","actor":0,"pai":"1m"}]
'@ | uv run python bot.py 0
```

## 训练与评测

当前仓库已经具备一条最小可运行的强化学习闭环：

- 本地用 `mjai==0.2.1` 的进程内对局做 self-play 与评测。
- 学习器使用 Torch 做 REINFORCE 更新。
- 训练后继续导出为 ONNX，再交给 Rust + tract 原生运行时部署。

当前保留了监督预训练接口，但默认明确禁用：

- 没有整理好的对局数据前，不应启用 `--enable-supervised-pretrain`。
- 这个开关目前会直接抛出 `NotImplementedError`，目的是保留命令行接口而不是偷偷走空实现。

单 checkpoint 自对战评测：

```powershell
uv run python tools/evaluate_checkpoint.py --checkpoint artifacts/policy.pt --matches 8 --workers 4
```

candidate 对 best checkpoint 评测：

```powershell
uv run python tools/evaluate_checkpoint.py --checkpoint artifacts/policy.pt --baseline-checkpoint artifacts/policy.best.pt --matches 16 --workers 4
```

评测输出当前包含这些指标：

- `average_rank`
- `average_score`
- `top1_rate`
- `last_rate`
- `average_reward`
- `average_decisions`

最小 REINFORCE 训练命令：

```powershell
uv run python tools/train_reinforce.py --checkpoint artifacts/policy.pt --best-checkpoint artifacts/policy.best.pt --iterations 10 --matches-per-iteration 8 --evaluation-matches 8 --workers 4 --device auto
```

训练脚本每轮会执行：

- 先做 self-play 采样。
- 再按回合奖励做一轮 REINFORCE 更新。
- 然后拿当前 checkpoint 对 `best_checkpoint` 做对局评测。
- 如果平均名次更优，或平均名次相同但平均分更高，就覆盖 `best_checkpoint`。

训练产物默认包括：

- `artifacts/policy.pt`
- `artifacts/policy.best.pt`
- `artifacts/training_metrics.jsonl`

设备建议：

- self-play worker 始终是 CPU 进程。
- 学习器可以用 `--device cpu` 或 `--device cuda`，默认 `auto` 会在有 GPU 时自动切到 `cuda`。
- 如果只有 CPU，先把 `workers` 开起来通常比盲目增大模型更划算。
- 如果有单卡 GPU，推荐继续用 CPU 跑 self-play，把 GPU 留给 learner 做梯度更新。

训练完成后，记得重新导出部署产物：

```powershell
uv run python tools/export_onnx.py --checkpoint artifacts/policy.pt --onnx artifacts/policy.onnx
```

## 比赛镜像内闭环验证

如果本机已经安装 Docker，并且你已经在开发环境中产出了 Linux `amd64` 可执行二进制，可以直接用下面这条命令按“比赛提交 ZIP”的方式做闭环验证：

```powershell
uv run python tools/validate_competition_image.py
```

运行前需要确保 Docker daemon 已经启动，并且当前处于 Linux containers 模式。

这个脚本会按顺序执行三件事：

- 从当前工作区生成一个与比赛提交格式一致的单一 ZIP，默认输出到 `artifacts/submission.zip`。
- 创建一个临时比赛容器，把这个 ZIP 复制进去，并在容器里用 `python3 -m zipfile -e` 解压。
- 在解压后的提交目录里直接运行原生 runtime，验证 `policy.onnx` 和 `policy.json` 与当前协议维度一致。
- 在解压后的提交目录里用 `python3 bot.py <seat>` 回放固定 mjai 事件流，覆盖一个 `call-choice` 分支和一个 `riichi-discard` 分支。

默认回放的样例文件在：

- [tools/fixtures/competition_call_choice.jsonl](tools/fixtures/competition_call_choice.jsonl)
- [tools/fixtures/competition_riichi_discard.jsonl](tools/fixtures/competition_riichi_discard.jsonl)

如果你的预构建 Linux 二进制不在默认路径，也可以显式指定：

```powershell
uv run python tools/validate_competition_image.py --runtime-path path/to/mjai-tract-runtime
```

如果你想把生成的提交 ZIP 放到其他位置，也可以指定：

```powershell
uv run python tools/validate_competition_image.py --submission-zip path/to/submission.zip
```

如果你只想跳过直接 runtime smoke test、保留 bot 事件流回放，可以这样跑：

```powershell
uv run python tools/validate_competition_image.py --skip-runtime-smoke
```

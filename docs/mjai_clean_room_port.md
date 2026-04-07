# mjai Clean-Room Port

目标不是逐行转写第三方 `mjai` 源码，而是在 Rust 中做等价 public behavior 的 clean-room 实现，并用差分测试把行为钉死到当前 Python 包。

## 范围

当前项目直接依赖以下公开入口：

- `mjai.Bot`
- `mjai.engine.DockerMjaiLogEngine`
- `mjai.game.Simulator`
- `mjai.game.to_rank`
- `mjai.player.MjaiPlayerClient`
- `mjai.verify.Verification`
- `mjai.mlibriichi.arena.Match`

因此 Rust 迁移必须至少覆盖三层：

1. 事件流与 reaction 协议：`react()` / `start()` / engine 批处理语义。
2. 对局状态与候选查询：`can_*` flags、手牌/牌河视图、`find_improving_tiles`、`find_pon_candidates`、`find_chi_candidates`。
3. 整局/整场行为：比赛运行、排名与分数、验证器输出。

## 差分测试框架

### Python oracle

- `tools/mjai_oracle.py`
- 输入：一个 JSONL fixture；每行是一批 mjai events。
- 输出：逐步 transcript，包含：
  - 输入 events
  - `Bot.react()` 返回值
  - capabilities snapshot
  - state snapshot
  - query snapshot

这个脚本直接调用当前 `.venv` 中安装的 Python `mjai` 包，是 Rust 迁移期间唯一可信的 oracle。

### Rust adapter contract

未来 Rust `mjai` 实现需要提供一个可执行适配器，并满足：

- 接收 `--fixture <path>`
- 输出与 `tools/mjai_oracle.py` 完全一致的 JSON transcript

只要两边 transcript 全等，就说明 Rust 实现对该 fixture 的公开行为与 Python 版一致。

### 现有测试入口

- `cargo test --manifest-path native_runtime/Cargo.toml --test mjai_contract`

当前实现状态：

- `native_runtime` 现在通过 `default-features = false` 依赖 vendored `mjai.app` Rust core，不再需要为 contract test 注入 `libpython` 的 link/runtime 环境变量。
- vendored `mjai.app` 保留 `python-bindings` feature，Python 扩展导出层和纯 Rust core 已经拆开；当前仓库里的 transcript diff 走的是纯 Rust core。

默认行为：

- 总是运行 Python oracle 并校验 transcript 结构合法。
- 如果设置了 `MJAI_RUST_ORACLE_BIN`，则额外执行 Rust/Python 全量 transcript diff。
- 直接通过 `cargo test` 运行时，`mjai-rust-oracle` 测试二进制会被自动发现，因此默认就会执行 Rust/Python transcript diff。

可选环境变量：

- `MJAI_PYTHON_BIN`：覆盖 Python oracle 使用的解释器，默认是 `.venv/bin/python`。
- `MJAI_RUST_ORACLE_BIN`：Rust adapter 可执行文件路径；设置后才执行真正的差分对比。

## 迁移顺序

建议按下面顺序替换，不保留长期兼容层：

1. 先在 Rust 中重建 `Bot` 状态机与候选查询，并让 adapter 通过 transcript diff。
2. 再重建 `engine` / `game` 层，纳入整局 transcript 与 match 级结果对比。
3. 最后切掉 Python `mjai` 依赖，把训练、推理、自对局都指向 Rust 实现。

## fixture 扩展

当前框架先使用仓库已有场景：

- `tools/fixtures/competition_call_choice.jsonl`
- `tools/fixtures/competition_riichi_discard.jsonl`

后续每发现一个 Rust/Python 行为分歧，都应把最小复现事件流固化成新的 JSONL fixture，再让 transcript diff 覆盖它。
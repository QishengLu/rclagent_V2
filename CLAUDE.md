# RCLAgent V2

把 [RCLAgent-V2](https://github.com/LLM4AIOps/RCLAgent-V2)（fork 自 commit `3052e7a`，v1.1，MIT，arXiv 2605.14866）接到 harness 上。`upstream/` 是上游代码，未改动，只没带 `sample_data/`、`README.md`、`assets/`；完整 commit 在 `UPSTREAM_COMMIT`。运行时的适配都在 `agent_runner.py` 和 `native_data.py` 里，用 monkeypatch 实现，不编辑 `upstream/`。

方法：按错误 trace 建调用树（`build_trace_graph`），每个 span 一个 Dedicated Agent。`self_state_verification` 用工具 `search_logs`、`search_fluctuating_metrics` 查证，`consolidation` 自底向上合并成局部根因，Global Evidence Graph 汇总。

## 相对上游的改动

- 上游每条 trace 的最终调用（`print_results` 排名）不发送。全部 trace 跑完后发**一次** case 级调用，带上每条 trace 的证据块原文，必须调用 `output_causal_graph`；模型的图就是答案，无投票，名字规范化到服务（`canonical_graph`）。
- `tool_server.py` 在进程内由 werkzeug 线程提供服务，coordinator 仍走 HTTP。
- LLM 经 3 次尝试仍失败时进程以退出码 3 结束（基础设施失败）；`RCLV2_FAIL_FAST=0` 恢复上游的吞异常行为。
- 系统提示词只替换服务列表（RCA100 再替换系统名），`KNOWN_SERVICES` 在 import `coordinator` 前注入。原文和替换后都记入 artifact `system_prompt_patched`。
- `search_fluctuating_metrics` 的窗口：regular 为 abnormal 开始前 ts ± 1200 s，current 为开始后 ts ± 600 s（`install_metric_windows`）。上游的两个范围在约 8 分钟的 case 上完全重合，均值永远相等。

## Harness 如何驱动

配置 `rclagent-v2-rcabench`、`rclagent-v2-rca100`：cwd `agents/rclagent_v2`，命令 `agents/_rca_shared/.venv/bin/python agent_runner.py`，超时 7200 s，`RCLV2_MAX_AGENT_PARALLEL=8`、`RCLV2_TOP_N_TRACES=3`、`RCLV2_LOG_RESULT_CHAR_BUDGET=2000`。

- stdin：JSON payload，只读 `question` 和 `data_dir`；harness 的提示词被忽略。
- stdout：一行 JSON，键为 `output`、`trajectory`、`usage`、`rclagent_v2`（traces、final、LLM 调用数、`log_result_guard` 等）。其余进 stderr。
- 事件：`$RCA_EVENTS_DIR/events.jsonl`，role 为 `rclv2:trace-<k>:span-<id>:verification|consolidation|fallback`、`rclv2:case:final`。

## 环境

共用 `agents/_rca_shared/.venv`，构建方式见 `../_rca_shared/CLAUDE.md`（`uv pip install --python agents/_rca_shared/.venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r agents/_rca_shared/requirements.txt`，`requirements.txt` 指向同一份）。必须与 `_rca_shared` 同级。`cp .env.example .env` 后填 `OPENAI_API_KEY`，也可用 `_rca_shared/.env` 或 `RCA_ENV_FILE`。

| 变量 | 默认 | 含义 |
|---|---|---|
| `RCLV2_TOP_N_TRACES` | 5（配置 3） | 每 case 错误 trace 数（Nezha 规则）；0 为上游 AIOps 规则 |
| `RCLV2_SPAN_COMPACTION` / `RCLV2_DEDUP_TRACES` | 1 / 1 | span 压缩 / 相同调用树只留一条 |
| `RCLV2_LOG_RESULT_CHAR_BUDGET` | 0（配置 2000） | `search_logs` 结果上限；0 为上游行为 |
| `RCLV2_FINAL_PROMPT_CHAR_BUDGET` | 600000 | case 级提示词超限时压缩最大 trace 的证据图摘要 |
| `RCLV2_NATIVE_DIR` | `<RCA_EVENTS_DIR 或 tmp>/rclv2_native/<case_key>` | 转换缓存（fingerprint 变化时重建） |

## 数据集

`native_data.py` 通过 `_rca_shared/cases.py` 把 case 转成 V2 的 canonical 布局（`trace/all/trace_jaeger-span.csv`、`metric/all/metrics.csv`（10 s 桶）、`log/all/logs.csv`、`error_traces.txt`、pod/node 映射 pkl）。不写 `groundtruth.csv`。详细列定义见其 docstring。

| 项 | RCABench | RCA100 |
|---|---|---|
| `DATASET_TYPE` | `nezha` | `re2ob`（OTel Demo 是 Online Boutique 的 fork） |
| 系统提示词 | nezha，替换服务列表 | re2ob，替换服务列表和系统名（改为 OpenTelemetry Demo (Astronomy Shop)） |
| 错误 trace 窗口 | `env.json` ABNORMAL，否则 `abnormal_traces` 时间范围 | question 的 alert window |
| 指标 | container / node / service 行 | container 行为带 node 的 `apm.instance`；service 行为 `apm.service.legacy`、`apm.metric.*`、`apm.operation` |

Span 压缩是因为 RCABench 的自动埋点记录每个方法和每次重复调用：丢弃有父节点的 INTERNAL span 并把子节点重新挂接；相同的兄弟子树合并，保留最长的一条并写 `repeat_count`。root span 不动，root 到 span 的路径集合不变（smoke 校验）。

`search_logs` 上限：同 pod、同模板的行合并为首行原文加 `repeat_count`；仍超限时按 error、warn、其他取舍，保持时间序，末行注明省略数。未超限时与上游逐字节相同。

## Smoke（不调用真实 LLM）

```bash
cd agents/rclagent_v2
../_rca_shared/.venv/bin/python tests/smoke_fake_llm.py [<converted dir>]
../_rca_shared/.venv/bin/python tests/smoke_fake_llm.py --dataset rca100 [<RCA100_CONV>/t001]
```

RCABench 默认 case 为 `${RCABENCH_ROOT:-<repo>/datasets/RCABench-500}/ts0-mysql-container-kill-9t6n24/converted`。RCA100 的转换方式见 `../_rca_shared/CLAUDE.md`。smoke 起一个本地假 OpenAI HTTP server，成功时最后一行是 `ALL SMOKE ASSERTIONS PASSED`。`RCLV2_SAMPLE_DATA` 指向上游 `sample_data/` 时还会统计上游调用数，否则跳过。只估成本不调 LLM：`estimate_cost.py`。

## 注意

- 上游 `search_fluctuating_metrics` 按 (kpi_name, node_id, service_name) 分组，pandas 默认丢弃含 NaN 的组，所以 service 级和 node 级行从不被检验，只测 pod 级序列。
- Dedicated Agent 只查本 span 的 pod；mysql、rabbitmq 没有 span，只能经模型主动查询或错误信息间接触达。
- RCA100 上最长 root span 多为 3-span 的 frontend-proxy → frontend，部分 case 只分析一条这样的 trace。这是上游规则的结果，规则未改。

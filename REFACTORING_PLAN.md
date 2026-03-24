# WebRAGent 开源重构方案

> 版本：v1.0 | 日期：2026-03-22  
> 目标：面向开源发布，在**不改变任何核心算法输出**的前提下，对代码结构、规范性、文档和健壮性进行全面整理。

---

## 目录

1. [项目现状分析](#1-项目现状分析)
   - 1.1 [核心功能与架构](#11-核心功能与架构)
   - 1.2 [当前目录结构](#12-当前目录结构)
   - 1.3 [模块依赖关系](#13-模块依赖关系)
   - 1.4 [现存问题清单](#14-现存问题清单)
2. [重构方案：代码结构](#2-重构方案代码结构)
3. [重构方案：规范性](#3-重构方案规范性)
4. [重构方案：文档](#4-重构方案文档)
5. [重构方案：健壮性](#5-重构方案健壮性)
6. [优先级与执行路线图](#6-优先级与执行路线图)

---

## 1. 项目现状分析

### 1.1 核心功能与架构

WebRAGent 是一个基于 RAG（检索增强生成）的 **Web 自动化 Agent** 研究框架，核心能力：

- **多模式浏览器自动化**：支持 DOM、Operator（OpenAI CUA）、Vision、DOM+Vision 等观测模式
- **RAG 增强规划**：通过语义检索历史轨迹（text-description / vision 两路）为 LLM 规划提供 few-shot 示例
- **多 LLM 后端**：GPT-4o、Claude、Gemini、TogetherAI、OpenAI Operator
- **在线评测**：基于 Online-Mind2Web 基准的逐步奖励评估（GlobalReward）
- **智能停止机制**：TaskCompletionManager + EndJudge + ConfirmationLoopDetector 三层停止策略

核心数据流：

```
任务输入 → Planning（LLM + RAG）→ ActionParser → Environment（Playwright）
                                                      ↓
                                        GlobalReward（评分） → TaskCompletionManager（停止判断）
                                                      ↓
                                        结果写入 JSON + 截图
```

---

### 1.2 当前目录结构

```
WebRAGent/
├── eval.py                    # DOM/Vision 模式单任务/批量评测入口
├── eval_op.py                 # Operator 模式评测入口（1750 行！）
├── batch_eval.py              # DOM 批量评测（异步并发）
├── batch_eval_op.py           # Operator 批量评测（异步并发）
├── operator_demo.py           # Operator 快速演示脚本
├── experiment_results.py      # 结果统计与汇总
├── evaluate.py                # 评测器 re-export 入口
├── logs.py                    # 全局日志初始化（硬编码路径）
├── configs/
│   ├── setting.toml           # 主实验配置
│   └── embedding.toml         # Embedding 模型配置
├── agent/
│   ├── Environment/
│   │   └── html_env/
│   │       ├── async_env.py   # DOM 环境（Playwright 封装）
│   │       ├── operator_env.py # Operator 环境
│   │       ├── operator_actions.py
│   │       ├── actions.py     # DOM 动作执行
│   │       ├── build_tree.py  # DOM 树构建（含 TODO）
│   │       └── active_elements.py
│   ├── LLM/
│   │   ├── openai.py
│   │   ├── claude.py
│   │   ├── gemini.py
│   │   ├── operator.py        # OpenAI CUA 专用客户端
│   │   ├── togetherai.py
│   │   ├── llm_instance.py    # LLM 工厂函数
│   │   ├── token_calculator.py
│   │   └── token_utils.py
│   ├── Memory/
│   │   ├── retriever.py       # ChromaDB 向量检索
│   │   ├── base_trace.py
│   │   ├── long_memory/
│   │   └── short_memory/
│   │       └── history.py     # 短期历史记忆
│   ├── Plan/
│   │   ├── planning.py        # 核心规划类（1327 行）
│   │   └── action.py          # ActionParser
│   ├── Prompt/
│   │   ├── prompt_constructor.py   # 所有 Prompt 构造器（4015 行！）
│   │   ├── base_prompts.py
│   │   ├── operator_prompts.py
│   │   └── ...（其余 prompt 文件）
│   ├── Reward/
│   │   └── global_reward.py   # 全局奖励评估
│   ├── Tool/
│   │   └── base_tools.py
│   └── Utils/
│       ├── utils.py
│       ├── task_completion_manager.py
│       ├── end_judge.py
│       ├── confirmation_loop_detector.py
│       ├── rag_logger.py
│       ├── prompt_logger.py
│       └── log_filter.py
├── evaluate/
│   ├── evaluate_utils.py      # run_task 核心函数（890 行）
│   ├── step_score.py
│   ├── task_score.py
│   └── step_score_js.py
├── data/
│   ├── dataset_io.py
│   └── raw_data_processor.py
├── tools/
│   └── description_generator.py
├── utils/                     # 顶层工具（与 agent/Utils 职责重叠）
│   ├── dataset_process.py
│   ├── log_processor.py
│   ├── operator_*.py
│   └── utils.py
├── OM2W_Benchmarking/         # 离线基准测试子模块
│   └── src/methods/           # 多种评估方法实现
├── Embedding/                 # 嵌入模型（含 VLM2Vec-pro 大型子目录）
└── requirements.txt
```

---

### 1.3 模块依赖关系

```
eval.py / eval_op.py
    ├── evaluate.evaluate_utils.run_task()
    │       ├── agent.Plan.planning.Planning.plan()
    │       │       ├── agent.Plan.planning.{DomMode, OperatorMode, ...}.execute()
    │       │       │       └── agent.Prompt.prompt_constructor.{*Constructor}.construct()
    │       │       │               └── agent.Memory.retriever.TestOnlyRetriever.retrieve()
    │       │       └── agent.LLM.llm_instance.create_llm_instance()
    │       ├── agent.Environment.html_env.async_env.AsyncHTMLEnvironment
    │       └── agent.Reward.global_reward.GlobalReward.evaluate()
    └── experiment_results.get_evaluate_result()
```

关键依赖问题：
- `eval_op.py` 直接内联了 `run_operator_task()` 大函数，绕过了 `evaluate_utils.run_task()` 的统一入口
- `agent/Reward/global_reward.py` 中存在一份独立的 `InteractionMode` 副本，与 `agent/Plan/planning.py` 中的同名类形成**代码重复**
- `logs.py` 在模块被导入时立即创建文件和目录，存在副作用

---

### 1.4 现存问题清单

#### 🔴 严重问题

| # | 位置 | 问题描述 |
|---|------|---------|
| S1 | `eval_op.py` | 单文件 1750 行，`run_operator_task()` 函数超过 400 行，包含了规划、奖励、日志、错误恢复等多种逻辑，严重违反单一职责原则 |
| S2 | `agent/Prompt/prompt_constructor.py` | 单文件 4015 行，25 个 Prompt 类混杂，严重影响可读性与维护性 |
| S3 | `agent/Plan/planning.py` | 1327 行，`OperatorMode.execute()` 函数及内部辅助函数过度耦合 |
| S4 | `logs.py` | 模块导入即触发文件系统操作（创建 `test/exp/logs/` 目录和日志文件），是全局副作用，会在 import 时污染用户目录 |
| S5 | `eval_op.py` 内大量硬编码字符串 | 如 `"flightaware"`, `"student.com"` 等域名白名单，混杂在业务逻辑中 |

#### 🟡 中等问题

| # | 位置 | 问题描述 |
|---|------|---------|
| M1 | `eval.py` / `eval_op.py` | `ExperimentConfig` dataclass 完全重复定义，字段有细微差别但没有共享基类 |
| M2 | `agent/Utils/` vs `utils/` | 顶层 `utils/` 目录与 `agent/Utils/` 职责重叠，`utils/utils.py` 与 `agent/Utils/utils.py` 并存 |
| M3 | `agent/Memory/retriever.py` | `get_task_workflow_description()` 使用硬编码的相对路径 `"data/Online-Mind2Web/generated_steps/..."` |
| M4 | `evaluate/evaluate_utils.py` | `read_file()` 函数返回值与实际使用方式不一致（返回 `List[List]` 而调用处已切换为 `read_json_file()`），形成死代码 |
| M5 | `agent/Plan/planning.py` | `Planning.plan()` 静态方法内部硬编码了 `gpt-3.5-turbo` 和 `gpt-4-turbo` 实例，不遵从配置 |
| M6 | 多处 `print()` 与 `logger` 混用 | `planning.py`、`prompt_constructor.py` 等多处直接使用 `print()` 输出调试信息 |
| M7 | `eval_op.py` | 大量注释掉的代码块（含 `# TODO`、`# debug`、旧逻辑等），增加阅读噪音 |
| M8 | `agent/LLM/llm_instance.py` | 模型路由采用字符串 `in` 匹配，模型名扩展性差（如增加新 OpenAI 模型需修改多处条件） |

#### 🟢 轻微问题

| # | 位置 | 问题描述 |
|---|------|---------|
| L1 | 全局 | 缺少 `pyproject.toml` 或 `setup.py`，无法作为 Python 包安装 |
| L2 | 全局 | `requirements.txt` 包含 CUDA 相关包（`flash_attn`, `nvidia-*`），不适合非 GPU 环境，缺乏分层依赖文件 |
| L3 | 全局 | 没有 `.env.example` 文件，API Key 配置不清晰 |
| L4 | 全局 | 缺少 `tests/` 目录，核心逻辑（ActionParser、Retriever）没有单元测试 |
| L5 | `agent/Prompt/` | 部分 Prompt 类（如 `VisionDisc1PromptConstructor`）仅用于早期实验，需标注弃用或移除 |
| L6 | `OM2W_Benchmarking/` | 子模块未提供独立说明文档 |

---

## 2. 重构方案：代码结构

### 2.1 目标目录结构

重构后的目录结构按**关注点分离**原则重新组织：

```
WebRAGent/
├── pyproject.toml              # 【新增】包管理与元数据
├── .env.example                # 【新增】API Key 配置模板
├── README.md                   # 【更新】
├── CHANGELOG.md                # 【新增】
├── configs/
│   ├── setting.toml
│   └── embedding.toml
│
├── webragent/                  # 【重命名】agent/ → webragent/（与项目同名，更Pythonic）
│   ├── __init__.py
│   │
│   ├── environment/            # 浏览器环境封装
│   │   ├── __init__.py
│   │   ├── base.py             # 【新增】抽象基类 BaseEnvironment
│   │   ├── dom_env.py          # 原 async_env.py
│   │   ├── operator_env.py
│   │   ├── actions.py
│   │   ├── operator_actions.py
│   │   └── dom_builder.py      # 原 build_tree.py
│   │
│   ├── llm/                    # LLM 后端
│   │   ├── __init__.py
│   │   ├── base.py             # 【新增】抽象基类 BaseLLM
│   │   ├── openai_llm.py       # 原 openai.py
│   │   ├── claude_llm.py
│   │   ├── gemini_llm.py
│   │   ├── operator_llm.py
│   │   ├── togetherai_llm.py
│   │   ├── factory.py          # 原 llm_instance.py
│   │   └── token_counter.py    # 合并 token_calculator.py + token_utils.py
│   │
│   ├── memory/                 # 记忆与检索
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── history.py          # 原 short_memory/history.py
│   │   └── trace.py            # 原 base_trace.py + long_memory/
│   │
│   ├── planning/               # 规划与动作
│   │   ├── __init__.py
│   │   ├── modes/              # 【拆分】各观测模式独立文件
│   │   │   ├── __init__.py
│   │   │   ├── base_mode.py    # InteractionMode 基类
│   │   │   ├── dom_mode.py
│   │   │   ├── operator_mode.py
│   │   │   ├── vision_mode.py
│   │   │   └── mixed_modes.py  # DomVDesc, VisionToDom, DV 等复合模式
│   │   ├── planner.py          # 原 Planning 静态类 → 重构为 Planner
│   │   └── action_parser.py    # 原 action.py
│   │
│   ├── prompt/                 # Prompt 构造（按用途拆分）
│   │   ├── __init__.py
│   │   ├── base.py             # BasePromptConstructor
│   │   ├── dom_prompts.py      # DOM 规划相关
│   │   ├── vision_prompts.py   # Vision 相关
│   │   ├── operator_prompts.py # Operator 相关
│   │   ├── reward_prompts.py   # Reward 相关
│   │   ├── rag/                # RAG 构造器
│   │   │   ├── __init__.py
│   │   │   ├── description_rag.py
│   │   │   ├── vision_rag.py
│   │   │   └── dom_vision_rag.py
│   │   └── templates/          # Jinja2 模板文件（从 Python 字符串中分离）
│   │
│   ├── reward/                 # 奖励评估
│   │   ├── __init__.py
│   │   └── global_reward.py
│   │
│   └── utils/                  # 内部工具
│       ├── __init__.py
│       ├── io.py               # 文件 I/O 工具（原 utils.py 相关部分）
│       ├── image.py            # 图像/截图工具
│       ├── logging.py          # 日志配置工厂（不再有全局副作用）
│       ├── task_completion.py  # 原 task_completion_manager.py
│       ├── end_judge.py
│       ├── loop_detector.py    # 原 confirmation_loop_detector.py
│       ├── rag_logger.py
│       └── prompt_logger.py
│
├── evaluate/                   # 评测框架
│   ├── __init__.py
│   ├── runners/                # 【新增】任务执行器
│   │   ├── __init__.py
│   │   ├── base_runner.py      # 【新增】抽象基类
│   │   ├── dom_runner.py       # 从 eval.py 中提取
│   │   └── operator_runner.py  # 从 eval_op.py 中提取 run_operator_task()
│   ├── evaluators.py           # 原 step_score.py + task_score.py
│   ├── utils.py                # 原 evaluate_utils.py（精简）
│   └── metrics.py              # 原 experiment_results.py
│
├── scripts/                    # 命令行入口脚本
│   ├── eval.py                 # 原 eval.py（精简为 CLI 入口，<100 行）
│   ├── eval_op.py              # 原 eval_op.py（精简为 CLI 入口，<100 行）
│   ├── batch_eval.py
│   └── batch_eval_op.py
│
├── data/
│   ├── __init__.py
│   ├── io.py                   # 原 dataset_io.py
│   └── processor.py            # 原 raw_data_processor.py
│
├── tools/
│   └── description_generator.py
│
├── OM2W_Benchmarking/          # 保持不变（独立子模块）
│   └── ...
│
├── Embedding/                  # 保持不变（预训练模型）
│
└── tests/                      # 【新增】单元测试
    ├── unit/
    │   ├── test_action_parser.py
    │   ├── test_retriever.py
    │   └── test_prompt_constructors.py
    └── integration/
        └── test_dom_runner.py
```

### 2.2 关键结构重构详情

#### 重构 R-1：拆分 `eval_op.py`（S1）

**问题**：`eval_op.py` 将 CLI 解析、实验配置、任务循环、规划调用、奖励评估、截图管理、错误恢复全混在一起。

**方案**：按职责拆分为三层：

```
scripts/eval_op.py          ← 纯 CLI 层（argparse + asyncio.run，<80 行）
    ↓ 调用
evaluate/runners/operator_runner.py  ← 任务执行层（run_operator_task 逻辑）
    ↓ 调用
webragent/planning/modes/operator_mode.py  ← 规划层（已有）
```

`OperatorTaskRunner` 类封装：

```python
class OperatorTaskRunner(BaseTaskRunner):
    async def run_single_task(self, task: TaskConfig, env: BaseEnvironment) -> TaskResult:
        ...
    async def run_batch(self, tasks: list[TaskConfig]) -> list[TaskResult]:
        ...
```

#### 重构 R-2：拆分 `prompt_constructor.py`（S2）

**问题**：4015 行单文件包含 25 个类，按用途区分：

| 新文件 | 包含的类 |
|--------|---------|
| `prompt/dom_prompts.py` | `PlanningPromptConstructor`, `VisionDisc*`, `ObservationVision*`, `VisionToDomPromptConstructor`, `D_V*`, `VisionObservation*` |
| `prompt/reward_prompts.py` | `RewardPromptConstructor`, `CurrentRewardPromptConstructor`, `VisionRewardPromptConstructor`, `JudgeSearchbarPromptConstructor`, `SemanticMatchPromptConstructor` |
| `prompt/rag/description_rag.py` | `PlanningPromptRetrievalConstructor`, `PlanningPromptDescriptionRetrievalConstructor`, `OperatorPromptDescriptionRetrievalConstructor`, `OperatorDescRAGConstructor` |
| `prompt/rag/vision_rag.py` | `PlanningPromptVisionRetrievalConstructor`, `OperatorPromptVisionRetrievalConstructor`, `OperatorVisionRAGConstructor` |
| `prompt/rag/dom_vision_rag.py` | `DOMVisionRAGConstructor` |
| `prompt/operator_prompts.py` | `OperatorPromptConstructor`, `OperatorPromptRAGConstructor` |

**保持向后兼容**：`prompt/__init__.py` 继续导出全部类名，外部代码无需修改。

#### 重构 R-3：消除重复的 `InteractionMode`（M1, S3）

`agent/Reward/global_reward.py` 中存在一份只用于 reward 计算的 `InteractionMode`，与 `planning.py` 中的基类同名但不同。

**方案**：
- `global_reward.py` 中的 `InteractionMode` 重命名为 `RewardEvaluator`，并独立为 `reward/evaluator.py`
- 移除对 `planning.py` 中 `InteractionMode` 的继承依赖

#### 重构 R-4：统一 `utils` 目录（M2）

删除顶层 `utils/` 目录，将其内容归并：

| 原文件 | 迁移目标 |
|--------|---------|
| `utils/utils.py` | 合并入 `webragent/utils/io.py` |
| `utils/dataset_process.py` | 合并入 `data/processor.py` |
| `utils/log_processor.py` | 合并入 `evaluate/metrics.py` |
| `utils/operator_*.py` | 合并入 `evaluate/runners/operator_runner.py` |
| `utils/parser.py` | 合并入 `webragent/planning/action_parser.py` |

#### 重构 R-5：修复 `logs.py` 全局副作用（S4）

**现状**：

```python
# logs.py — 模块导入即执行，立即创建 test/exp/logs/ 目录
log_folder = "test/exp/logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_file_name = os.path.join(log_folder, time.strftime(...) + ".log")
logger = logging.getLogger()
```

**方案**：重构为工厂函数，调用方显式初始化：

```python
# webragent/utils/logging.py
import logging
import os
import colorlog

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """配置并返回全局 logger，不产生任何副作用直到被显式调用。"""
    os.makedirs(log_dir, exist_ok=True)
    ...
    return logging.getLogger("webragent")

# 向后兼容
logger = logging.getLogger("webragent")
```

CLI 入口脚本在 `main()` 中第一行调用 `setup_logging(args.log_dir)`。

---

## 3. 重构方案：规范性

### 3.1 代码风格统一

#### 工具链配置

新增 `pyproject.toml`：

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]  # 超长行由 ruff format 处理

[tool.ruff.isort]
known-first-party = ["webragent", "evaluate"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
```

#### 具体规范点

**N-1：消除全局 `print()` 调试语句（M6）**

全局搜索并替换：

```python
# 替换前（planning.py, prompt_constructor.py 等多处）
print(f"\033[36mvision_disc_response:\n{vision_desc_response}")
print(f"\033[32mplanning_request: {planning_request}")

# 替换后
logger.debug("vision_disc_response: %s", vision_desc_response)
logger.debug("planning_request: %s", planning_request)
```

**N-2：统一类型注解**

补充缺失的函数签名类型注解，优先补充公共 API：

```python
# 现状
def create_llm_instance(model, json_mode=False, all_json_models=None):

# 目标
def create_llm_instance(
    model: str,
    json_mode: bool = False,
    all_json_models: list[str] | None = None,
) -> BaseLLM:
```

**N-3：LLM 工厂函数改用注册表模式（M8）**

```python
# 现状：字符串 in 判断，扩展性差
if "operator" in model or "computer-use-preview" in model:
    ...
elif any(keyword in model for keyword in ["gpt", "o1", "o3-mini", "o4-mini"]):
    ...

# 目标：注册表 + 前缀匹配
_LLM_REGISTRY: dict[str, type[BaseLLM]] = {}

def register_llm(prefix: str):
    def decorator(cls: type[BaseLLM]) -> type[BaseLLM]:
        _LLM_REGISTRY[prefix] = cls
        return cls
    return decorator

@register_llm("gpt")
@register_llm("o1")
@register_llm("o3-mini")
@register_llm("o4-mini")
class GPTGenerator(BaseLLM): ...
```

**N-4：硬编码常量集中管理（S5）**

新增 `webragent/constants.py`：

```python
# 域名白名单（用于特殊等待逻辑）
SLOW_LOAD_DOMAINS: frozenset[str] = frozenset([
    "flightaware",
    "student.com",
    "booking.com",
])

# 坐标边界
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 720

# 默认步数
DEFAULT_MAX_STEPS = 80
DEFAULT_LOOP_THRESHOLD = 5
```

**N-5：清理注释掉的代码（M7）**

- 删除 `eval_op.py`、`eval.py`、`evaluate_utils.py` 中所有被注释掉的代码块
- 保留注释 `# TODO`，转换为 GitHub Issues 后再删除
- 保留关键的说明性注释

**N-6：统一异常处理风格**

现状多处使用 `exit()` 退出：

```python
# 现状（evaluate_utils.py, eval.py 等多处）
logger.error("batch_tasks_file_path not exist!")
exit()
```

改为抛出异常，由顶层 CLI 捕获：

```python
# 目标
raise FileNotFoundError(f"batch_tasks_file_path not found: {path}")

# CLI 入口
if __name__ == "__main__":
    try:
        asyncio.run(main(...))
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
```

### 3.2 Pythonic 改进

**P-1：dataclass 共享基类（M1）**

```python
# webragent/config.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class BaseExperimentConfig:
    mode: str
    global_reward_mode: str
    planning_text_model: str
    global_reward_text_model: str
    config: dict[str, Any]
    write_result_file_path: str
    record_time: str
    rag_enabled: bool
    rag_path: str
    rag_mode: str = "description"
    rag_cache_dir: str = ""
    end_judge_mode: str = "disabled"
    end_judge_confidence_threshold: float = 0.8
    end_judge_min_steps: int = 2
    consecutive_error_threshold: int = 2

@dataclass
class DOMExperimentConfig(BaseExperimentConfig):
    """DOM 模式专用配置"""
    ground_truth_mode: bool = False
    ground_truth_data: dict | None = None
    file: list = field(default_factory=list)
    single_task_name: str = ""
    rag_log_dir: str | None = None

@dataclass
class OperatorExperimentConfig(BaseExperimentConfig):
    """Operator 模式专用配置"""
    single_task_id: str = ""
    screenshot_base_dir: str = ""
    rag_logging_enabled: bool = False
    rag_log_dir: str = ""
    prompt_logging_enabled: bool = False
    prompt_log_dir: str = ""
```

**P-2：消除 `hasattr` 防御性检查（eval_op.py 多处）**

```python
# 现状：说明 dataclass 字段不稳定
if hasattr(experiment_config, 'rag_cache_dir') else None

# 目标：在 dataclass 中提供默认值，直接访问
experiment_config.rag_cache_dir  # 字段必然存在
```

**P-3：上下文管理器管理 Environment 生命周期**

```python
# 目标：AsyncHTMLEnvironment 实现 __aenter__/__aexit__
async with create_html_environment(mode) as env:
    await run_task(env=env, ...)
# 自动 close，不需要手动 del env
```

---

## 4. 重构方案：文档

### 4.1 README.md 重写

当前 README 缺少：关键章节补充如下：

```markdown
## WebRAGent

**WebRAGent** 是一个基于检索增强生成（RAG）的 Web 自动化 Agent 研究框架，
支持 DOM、Vision 和 OpenAI Operator 等多种观测模式，在 Online-Mind2Web 基准上进行评测。

## 核心特性
- 多模式浏览器自动化（DOM / Operator / Vision）
- RAG 增强规划（文本检索 / 视觉检索）
- 多 LLM 后端（GPT-4o、Claude、Gemini、OpenAI CUA）
- 基于 Online-Mind2Web 的在线评测

## 快速开始
### 环境准备
### 配置 API Keys
### 运行单任务评测
### 运行批量评测

## 架构概述
## 配置说明（configs/setting.toml 字段详解）
## 引用（BibTeX）
```

### 4.2 代码内文档

**D-1：为所有公共函数/类添加 docstring**

格式统一使用 Google 风格：

```python
def create_llm_instance(
    model: str,
    json_mode: bool = False,
    all_json_models: list[str] | None = None,
) -> BaseLLM:
    """根据模型名称创建对应的 LLM 实例。

    Args:
        model: 模型标识符，如 "gpt-4o"、"claude-3-5-sonnet"、
               "computer-use-preview-2025-03-11"。
        json_mode: 是否启用 JSON 输出模式。
        all_json_models: 支持 JSON 模式的模型列表，从配置文件读取。

    Returns:
        对应 LLM 后端的实例。

    Raises:
        ValueError: 若请求 JSON 模式但模型不支持。
    """
```

**D-2：`configs/setting.toml` 添加详细字段注释**

将所有配置字段加注释说明类型、取值范围和用途：

```toml
[rag]
# 是否启用 RAG 检索增强
enabled = true

# RAG 索引数据路径（包含 qry_embed_path 等子路径的目录）
rag_path = "data/rag_data/"
```

**D-3：新增 `docs/` 目录**

```
docs/
├── architecture.md     # 架构说明与模块依赖图
├── rag_modes.md        # 四种 RAG 模式说明（description/vision/vision_rag/description_rag）
├── evaluation.md       # 评测框架说明（GlobalReward、StepScore、TaskScore）
├── adding_llm.md       # 如何接入新 LLM 后端
└── dataset_format.md   # Online-Mind2Web 数据格式说明
```

**D-4：`.env.example`**

```bash
# OpenAI API
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，用于代理

# Anthropic API
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini API
GOOGLE_API_KEY=...

# TogetherAI API
TOGETHER_API_KEY=...
```

### 4.3 CHANGELOG.md

新增变更日志，记录开源版本与实验版本的差异，便于社区追踪演进。

---

## 5. 重构方案：健壮性

### 5.1 输入验证与错误处理

**H-1：统一配置验证层**

将 `eval.py` 和 `eval_op.py` 中分散的 `validate_config()` 合并，并补充完整的 schema 验证：

```python
# webragent/config.py
from pydantic import BaseModel, field_validator, model_validator

class AppConfig(BaseModel):
    """使用 Pydantic 对 setting.toml 进行完整验证"""

    class BasicConfig(BaseModel):
        task_mode: Literal["batch_tasks", "single_task"]

    class RagConfig(BaseModel):
        enabled: bool
        rag_path: str

        @field_validator("rag_path")
        @classmethod
        def rag_path_must_exist_when_enabled(cls, v, info):
            if info.data.get("enabled") and not os.path.exists(v):
                raise ValueError(f"rag_path '{v}' does not exist")
            return v

    basic: BasicConfig
    rag: RagConfig
    # ...
```

**H-2：修复硬编码路径（M3）**

```python
# 现状（retriever.py）
with open("data/Online-Mind2Web/generated_steps/generated_task_steps.json", "r") as f:

# 目标：接受路径参数，设置合理默认值
class TestOnlyRetriever:
    def __init__(self, path: dict, data_root: str = "."):
        self.data_root = Path(data_root)
        ...

    def get_task_workflow_description(self, task_id: str) -> str:
        steps_file = self.data_root / "data/Online-Mind2Web/generated_steps/generated_task_steps.json"
        ...
```

**H-3：完善 `ActionParser` 的容错（action.py）**

当前 `except:` 裸异常捕获会掩盖真实错误：

```python
# 现状
try:
    result_action = re.findall("```(.*?)```", message, re.S)[0]
    result_action = self.parse_action(message)
except:  # ← 裸 except
    ...

# 目标
except (IndexError, json5.JSONDecodeError, KeyError) as e:
    logger.warning("Action parsing failed with primary parser: %s", e)
    result_action = self.parse_action_with_re(message)
```

**H-4：异步任务取消与超时**

`eval_op.py` 中单个任务运行时间可能无限长，添加任务级超时：

```python
# evaluate/runners/operator_runner.py
async def run_single_task_with_timeout(
    self,
    task: TaskConfig,
    env: BaseEnvironment,
    timeout_seconds: int = 600,
) -> TaskResult:
    try:
        return await asyncio.wait_for(
            self.run_single_task(task, env),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Task %s timed out after %ds", task.task_id, timeout_seconds)
        return TaskResult.from_timeout(task)
```

**H-5：环境资源泄漏防护**

当前 `eval_op.py` 异常路径有多处 `env` 未被正确关闭：

```python
# 目标：使用 try/finally 确保关闭
async def run_experiment(...):
    for task_index in task_range:
        env = create_html_environment(mode)
        try:
            await run_single_task(env, ...)
        except Exception as e:
            logger.error("Task %d failed: %s", task_index, e)
        finally:
            await env.close()
```

### 5.2 依赖管理

**H-6：分层 requirements 文件（L2）**

```
requirements/
├── base.txt          # 核心运行依赖（playwright, openai, chromadb, etc.）
├── vision.txt        # Vision 模式额外依赖（torch, transformers, etc.）
├── dev.txt           # 开发依赖（ruff, pytest, mypy）
└── full.txt          # 完整依赖（-r base.txt -r vision.txt）
```

**H-7：可选依赖标注（`pyproject.toml`）**

```toml
[project.optional-dependencies]
vision = ["torch>=2.0", "transformers>=4.40", "timm", "flash_attn"]
dev = ["ruff", "pytest", "pytest-asyncio", "mypy"]
```

### 5.3 测试覆盖

**H-8：核心模块单元测试**

优先级由高到低：

| 测试文件 | 覆盖内容 |
|---------|---------|
| `tests/unit/test_action_parser.py` | ActionParser 对各种 LLM 输出格式的解析正确性 |
| `tests/unit/test_config_validation.py` | AppConfig pydantic 校验的正向/负向用例 |
| `tests/unit/test_llm_factory.py` | create_llm_instance 路由正确性 |
| `tests/unit/test_task_completion.py` | TaskCompletionManager 各停止条件 |
| `tests/integration/test_dom_runner.py` | DOM 模式端到端（mock Playwright）|

**H-9：CI 配置**

新增 `.github/workflows/ci.yml`：

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest tests/unit/ -v
```

---

## 6. 优先级与执行路线图

### Phase 1 — 基础整理（1～2 天，风险极低）

> 不涉及逻辑修改，纯清理与组织

| 任务 | 涉及文件 | 对应问题 |
|------|---------|---------|
| 删除注释掉的代码块 | `eval.py`, `eval_op.py`, `evaluate_utils.py` | M7 |
| 清理 `utils/` 顶层目录 | `utils/*.py` | M2 |
| 新增 `.env.example` | 根目录 | L3 |
| 新增 `pyproject.toml`（仅元数据部分） | 根目录 | L1 |
| 修复 `logs.py` 为工厂函数 | `logs.py` | S4 |
| 硬编码常量提取到 `constants.py` | `eval_op.py` 等 | S5 |

### Phase 2 — 规范统一（2～3 天）

> 代码行为不变，改善可读性

| 任务 | 涉及文件 | 对应问题 |
|------|---------|---------|
| 全局替换 `print()` → `logger` | 多处 | M6 |
| 补充类型注解（公共 API 优先） | 所有模块 | N-2 |
| 统一异常处理（`exit()` → raise） | `eval*.py`, `evaluate_utils.py` | N-6 |
| `ExperimentConfig` 共享基类 | `eval.py`, `eval_op.py` | M1 |
| LLM 工厂改注册表模式 | `llm_instance.py` | M8 |

### Phase 3 — 结构重构（3～5 天）

> 涉及目录迁移，需要更新所有 import

| 任务 | 涉及文件 | 对应问题 |
|------|---------|---------|
| 拆分 `prompt_constructor.py` 为 6 个文件 | `agent/Prompt/` | S2 |
| 拆分 `eval_op.py` 为 runner + CLI | `eval_op.py` | S1 |
| 拆分 `planning.py` modes 为独立文件 | `agent/Plan/` | S3 |
| `agent/` 目录重命名为 `webragent/` | 所有 import | 结构规范 |
| Environment 实现上下文管理器 | `async_env.py` | P-3 |

### Phase 4 — 健壮性与文档（2～3 天）

| 任务 | 涉及文件 | 对应问题 |
|------|---------|---------|
| Pydantic 配置校验 | 新增 `config.py` | H-1 |
| 修复硬编码路径 | `retriever.py` | H-3, M3 |
| 添加任务超时机制 | `operator_runner.py` | H-4 |
| 分层 requirements | `requirements/` | H-6 |
| 编写单元测试（优先 ActionParser） | `tests/` | H-8 |
| README 重写 + docs/ 补充 | 文档 | D-1 ~ D-4 |
| `configs/setting.toml` 字段注释 | 配置文件 | D-2 |

---

## 附录：重构约束备忘

1. **核心算法不变**：GlobalReward 评分逻辑、RAG 检索流程、ActionParser 解析规则不得修改
2. **配置文件向后兼容**：`configs/setting.toml` 和 `configs/embedding.toml` 的字段名保持不变
3. **命令行接口向后兼容**：`eval.py`、`eval_op.py` 的 argparse 参数保持不变（仅将实现移至内部）
4. **循序渐进**：每个 Phase 完成后均可独立运行评测，不引入 broken state
5. **git 提交粒度**：每个独立重构任务对应一个 commit，便于 code review 和回滚

# VISTA：LIBERO-Pro 首版代码

VISTA 让 Meta-agent 根据执行记录、主动环境探索和代码修改持续开发机器人 Harness。
每次提交的代码版本被冻结后，由执行 Agent 自主评估，底层模型参数保持固定。

本版提供 LIBERO-Pro 的完整开发流程、环境适配、初始 H₀，以及 T/S 实验配置。
RoboCasa、RoboTwin 的整合后续继续。

请从[英文 README](README.md) 的安装和单任务示例开始：

1. 安装源码及 LIBERO-Pro、Pi0.5、SAM3 所需依赖和外部资源。
2. 将 `configs/env.example.sh` 复制为 `configs/env.local.sh`，填写本机路径和服务凭据。
3. 用 `tools/configure_run.py` 选择可用模型，生成本地配置。
4. 用 `tools/check_config.py` 检查配置与 H₀，然后运行 `tools/run-rsi.sh`。

小样例从 Goal-t09 的 H₀ 开始，先评估基线，再进行一轮开发和候选评估。
默认保留 RGB、动作和状态记录，不保存密集深度／几何数组；实时感知工具仍然可用。

`meta_harness/` 是开发与评估流程，`rpent/` 是执行端，`robots/libero/` 是平台适配，
`harnesses/libero_pro_h0_min/` 是初始 Harness。实验差异通过 `configs/` 表达。

当前源码包含 H₀，论文逐任务选定的演化 Harness 和实验记录作为独立研究产物整理。
首版已通过 142 项测试、22 项子测试和六份配置检查。
离线验证使用受控模型和环境，不等同于完整模型、GPU、模拟器环境的端到端验证。
具体实验配置与复现边界见[协议说明](docs/protocols.md)。

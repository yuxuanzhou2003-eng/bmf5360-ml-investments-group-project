# AI supply-chain taxonomy v2.0

冻结日期：2026-09-10。该文件是 AI 股票池扩展的设计层 taxonomy；它不替代已有 `AI_POOL_SPEC.md` v1.1 的历史分部 PIT 成员表。候选注册表中的 49 个 RIC 是供应链研究候选，只有取得形成日前可追溯的公开证据后才可写入正式 PIT membership。

## 观察单位和范围

- 观察单位为 literal `RIC`，不把名称、ticker 猜测或当前公司名称用于替换/合并 RIC。
- 候选池不按收益、波动、规模、流动性、模型预测或 ETF 表现筛选。
- 仅从当前 781-RIC universe 选出供应链角色候选；未选 RIC 保留在全量 screening 表，状态为 `not_selected_design_candidate`，不是被物理删除的证券。
- 退市 RIC 继续保留；退市状态和可交易覆盖期单独记录，退市本身不是排除理由。

## 冻结的七组

一家公司可以有一个 primary group 和若干 secondary group；注册表按 primary group 计数，类别之间不做重复去重以外的权重处理。

| group_id | 中文 | 直接证据要求 | 仅有以下词时的处理 |
|---|---|---|---|
| `gpu_accelerator` | GPU / AI 加速器 | GPU、AI accelerator、accelerated computing、tensor/inference chip、data-center processor 等产品或业务分部 | 单独的“technology”“chip”不足以入组 |
| `ai_semiconductor` | AI 芯片及配套半导体 | AI/ML SoC、CPU/ASIC、edge inference、networking/analog/power silicon 明确服务 AI 或数据中心工作负载 | 单独的“semiconductor”或“electronics”标记 ambiguous |
| `memory_hbm_storage` | 存储 / HBM / 数据中心内存 | HBM、high bandwidth memory、AI/data-center memory、enterprise SSD/NAND、data-center storage | 单独的 memory、storage、NAND、flash 不足以入组 |
| `server_network` | 服务器 / 网络 / 光互连 | AI/accelerated server、HPC、data-center networking、Ethernet switch/adapter、optical interconnect、服务器 OEM | 泛化的 IT hardware 或 telecom 不足以入组 |
| `cloud_software` | 云 / AI 软件 / EDA 平台 | AI/ML/generative-AI platform、intelligent cloud、AI services、AI design/simulation/developer platform | 单独的 software、cloud 或 consulting 标记 ambiguous |
| `data_center_power_cooling` | 数据中心电力 / 散热 | data-center power/cooling、liquid cooling、UPS、电气配电、热管理或明确披露的数据中心负荷供电 | 泛化 utility、HVAC 或 construction 不足以入组 |
| `robotics_autonomy` | 机器人 / 自动驾驶 / 自主系统 | robotics、industrial automation/vision、autonomous driving/systems、明确的机器人测试设备 | 普通汽车、工业或航空业务不单独入组 |

## 证据优先级和 PIT 规则

1. `local_lseg_orig_segment`：项目已有 LSEG `segments_fy0_*` 的 `Orig` 历史分部名称；通过固定词典命中并能按 `Instrument + period_end` 连接 `is_dates_*` 原始公告日时，才可生成 PIT 候选形成日期。形成日期采用 Orig announcement calendar date；若公告时间不可用，只能声明日级 cutoff 限制。
2. `local_identity_span`：`eligible_spans_2015_2026_corrected.csv` 只证明该 literal RIC 在样本 universe 的覆盖期，不能证明 AI 业务。
3. `official_issuer_description`：公司官网/IR 业务描述是当前或检索日证据；本 run 不下载或快照这些页面，因此只能作为 `provisional_static_source`，不能回填 2015–2026 的历史成员资格。
4. `etf_constituent_snapshot`：若未来取得带 retrieval date 的 ETF holdings，只能作为当日外部交叉验证；现有 raw 目录只有 AIQ、BOTZ、IGV、SOXX、XLK 的价格，没有 holdings/constituents 表，本 run 不把 ETF 价格当成 constituents 证据。

候选注册表将 `pit_evidence_status=direct_local_pit` 与 `provisional_static_source` 分开。后者在正式 pool 中必须标为 `membership_unknown`，直到取得 formation-date 前可见的历史业务证据。不得把无证据的年份前向填充为成员，也不得把未知当作非 AI。

## 版本冻结和审核门

- 词典、类别、候选名单和来源指针由 `ai_pool_expansion_v1_config.json` 固定；任何新词或新类别都必须升版本并重新审计。
- 空字符串/供应商 NA 保留为 NA；不插值、不零填、不前后填充、不 winsorize，不静默去重。
- 原始 raw CSV、旧 v1 membership、旧日志均不可覆盖。
- 输出至少包含：全量 universe screening、49 行 candidate registry、LSEG 局部证据、来源目录、输入/输出 hashes、gate report 和限制说明。
- 本阶段不读取 `returns`、`targets`、`labels`、`model_outputs` 或 portfolio 文件；没有收益筛选或表现结论。

## 当前设计层候选分布

| group_id | 数量 |
|---|---:|
| `gpu_accelerator` | 5 |
| `ai_semiconductor` | 6 |
| `memory_hbm_storage` | 5 |
| `server_network` | 6 |
| `cloud_software` | 10 |
| `data_center_power_cooling` | 9 |
| `robotics_autonomy` | 8 |
| 合计（unique literal RIC） | 49 |

该分布是供应链覆盖设计，不是投资组合权重，也不意味着每个 RIC 在全部历史期间都属于 AI 产业链。

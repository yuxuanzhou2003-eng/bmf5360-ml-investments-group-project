# AI 行业股票池规范 v1.1

日期：2026-09-10。研究对象是点时定义的 AI 行业股票池，之后的因子交易和机器学习模型只使用池内股票。v1.1 是用户要求的预注册宽口径版本，加入 AI 产业链基础设施和存储。

## 成员资格

- 观察单位为 `RIC + formation_date`。只使用形成日前已公开的历史业务分部名称及原始公告日期（`ReportingState=Orig`）。没有可用公告的期间记为 `membership_unknown`，不能当作非 AI。
- 不用当前公司名称、未来并购、未来收入或未来价格表现回填历史成员资格。
- 退市记录保留到最后可交易日；退市本身不构成排除理由。公司可随业务变化进入或离开池子。

## 预注册 taxonomy

### Core AI（主分析池）

分部名称匹配明确 AI 或计算词组：独立词 `AI`、`artificial intelligence`、`generative AI`、`machine learning`、`deep learning`、`neural`、`AI accelerator`、`GPU`、`graphics processor`、`tensor`、`inference`、`data-center compute`、`high-performance AI computing`。独立 `AI` 必须按词边界匹配，不能把包含字母组合的普通词误算为 AI。

### AI industry broad（主基准扩大池）

以下任一产业链类别满足词组条件即可进入扩大池：

- **计算芯片与加速器**：`GPU`、`AI accelerator`、`accelerated computing`、`tensor`、`inference chip`、`data-center processor`；
- **存储与内存**：`HBM`、`high bandwidth memory`、`memory for AI`、`AI memory`、`data-center memory`、`SSD/NAND for data center`；单独的 `memory`、`storage`、`NAND` 或 `flash` 不足以入池；
- **服务器、网络与数据中心**：`AI server`、`AI networking`、`data center`、`data-center infrastructure`、`high-performance computing`、`accelerated server`；
- **云与软件平台**：`AI software platform`、`machine-learning platform`、`generative AI platform`、`intelligent cloud`、`AI services`；单独的 `software` 或 `cloud` 不足以入池；
- **机器人与自主系统**：`robotics`、`autonomous driving`、`autonomous systems`、`industrial AI`；
- **数据中心设备与配套**：`data-center power`、`data-center cooling`、`liquid cooling`、`AI power systems`、`data-center equipment`。

### AI ecosystem（旧称，稳健性子池）

分部名称匹配数据中心、云、服务器、网络、机器人等直接相关词组。该子池保持旧 v1.0 口径，用于比较宽口径新增的边际影响。

### Ambiguous

无法由预注册规则判断、只含泛化词、或分部对账/日期冲突的记录标记为 `ambiguous`，保留审计表但不进入主池。新增词须重新冻结 taxonomy。

## 输出与研究用途

清洗阶段生成 `ai_pool_membership_pit.csv`、`ai_pool_membership_audit.csv`、`ai_pool_summary.json`。每条记录保留 RIC、形成日、池标签、状态、命中词、原始分部名称、公告日、taxonomy 版本和原因码。检查形成日前 cutoff、RIC-日期唯一性、Core/ecosystem 重叠、unknown/ambiguous 数量、退市覆盖和原始哈希对应关系；本阶段不生成未来收益、标签、预测或组合业绩。

Core AI 是纯度最高的敏感性池；AI industry broad 是主基准扩大池；AI ecosystem 是旧口径稳健性子池。因子排序和 Random Forest 样本先按形成日过滤到相应池内，再加入规模、价值、盈利能力、投资、杠杆、动量、波动、beta、流动性和市场状态控制变量。任何宽泛词单独出现都不能入池；新增词须重新冻结版本。

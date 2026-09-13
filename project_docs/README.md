# BMF5360 项目文档包

NUS BMF5360 Machine Learning in Investments 组队项目的全部文档，截至 2026-09-10。

这是**文档快照**，不含数据和代码。数据在原项目的 `data/`（最大的 `prices.csv` 有 577 MB），代码在原项目根目录。新项目要复用数据时按本文档里的路径去原目录取。

---

## 现在做到哪了

选题已经收敛为 **AI 产业链股票的日频预测与组合策略**。主线不再依赖 event study，也不把“AI 暴露度”作为预测目标；AI taxonomy 只负责定义投资池，模型在池内预测未来 21 个交易日相对 SPY 的超额收益方向。

当前主线已有完整的 49-RIC 日频开发样本、宏观状态、AI 行业状态、Logistic 与 Random Forest 基准。训练期为 2015–2020，验证期为 2021–2022，测试期 2023-01-01 至 2026-06-30 仍严格封存。覆盖一致主版本已经独立核验，当前最均衡的开发期模型是“技术 + AI 行业状态”Logistic：验证 AUC 0.5799、PR-AUC 0.5745、Brier 0.2476。该模型已转换为正式验证期组合，累计净收益 -3.10%、年化 -1.63%、Sharpe -0.119、最大回撤 -13.15%、对 SPY beta 0.064；独立审计 13 PASS、1 WARN、0 FAIL。这是保留的真实负结果，正在做贡献归因，不在验证期调阈值。

| 层 | 产物 | 验证 |
|---|---|---|
| 股票池 | 781 只 point-in-time，含 154 个退市 RIC | corrected spans 已审计，退市不作为删除理由 |
| 行情 | 2,084,177 行日频，含开高低收量与买卖报价 | 31/31 passed |
| 财报事件 | 28,995 个公告 | 31/31 passed |
| 关系图（残差相关） | 107,532 条边 | 31/31 passed，**假设已被否证** |
| 关系图（共同分析师） | 111,128 条合格边 | 图 21/21、model-ready 59/59 passed |
| 共同分析师模型 | Ridge / HGB / two-stage | 最终审计 46/46 passed，**validation 无稳定增量** |
| AI 财务/分部字段探针 | 1,395 行、11 类请求 | 28/28 passed；Orig 与 current-state 有实质差异 |
| AI 正式原料 collector | 357 个独立请求 | 3 批小试已返回，正式全量待恢复机制修复后执行 |
| AI 产业链候选池 | 49 个 RIC、7 个产业环节 | 全部存在于 781-RIC 母池；9 家有本地 PIT 分部证据，40 家为探索性静态归类 |
| 日频模型主表 | 110,829 行；H21 非重叠训练/验证锚点 2,334/992 | 测试期目标、预测和指标为 0 |
| 日频宏观状态 | 18,345 行 FRED 原始观测；20 个可用变量 | raw 75/75、clean 158/158 checks passed；6 个信用利差变量因开发期无覆盖而排除 |
| AI 行业状态 | 2,889 个形成日；主版本 21 个覆盖一致增量变量 | 严格前一交易日；AIQ/BOTZ/IGV 的 18 个高结构性缺失变量只留在稳健性 run |
| 覆盖一致增强模型 | 4 组特征 × Logistic/RF；992 行验证预测 | 独立审计 74/74 passed；主候选为技术 + AI 状态 Logistic |
| 正式验证期组合 | 23 个完整21-session区间；AI多头 + SPY beta hedge | 净收益 -3.10%；独立审计13 PASS/1 WARN/0 FAIL；测试仍封存 |

---

## 文档在哪

### 01_protocol — 选题与研究协议

| 文件 | 内容 |
|---|---|
| `RESEARCH_PROTOCOL.md` | **最重要**。v0.1 到 v0.9 逐版记录每个阶段冻结了什么规则、为什么。所有已审批的数据与建模口径都在这里 |
| `AI_SECTOR_FEASIBILITY.md` | 转向 AI 行业的定位、筛选口径、AI 暴露因子的三类数据源结论 |
| `AI_FACTOR_DATA_SPEC.md` | AI_REV、AI_RD、AI_MKT、控制变量、股票—月 RF、数据 gate 与输出契约 |
| `AI_FACTOR_CLEANING_PLAN.md` | raw 到 clean 的空值、期间、分部、ETF、退市与审计规则 |
| `AI_BASELINE_MODEL_SPEC.md` | Momentum、Logistic、Random Forest 和因子排序基准的共同样本与评估协议 |
| `INFORMATION_DIFFUSION_PROPOSAL.md` / `_PILOT.md` | 最初的信息扩散选题与 30 股试点，已被 v2/v3 取代，保留作沿革 |

### 02_feasibility — 数据可行性

| 文件 | 内容 |
|---|---|
| `LSEG_DATA_FEASIBILITY.md` | 最早的十选题筛选与 LSEG 权限边界 |
| `RELATION_DATA_FEASIBILITY.md` | 关系数据三轮探测：共同分析师与共同持股可用，供应链未能证实，含 broker token 身份验证 |
| `FIELD_SEMANTICS_REVIEW.md` / `EPS_FIELD_CANDIDATES.md` | EPS 字段口径考证 |
| `STOCK_SPLIT_REVIEW.md` | 拆股核查，结论是价格已含公司行动调整，禁止再乘拆股因子 |

### 03_data — 数据状态与规则

| 文件 | 内容 |
|---|---|
| `DATASET_STATUS.md` | 每个阶段的数据集状态与"能用/不能用"判断 |
| `DATA_PROCESSING_LOG.md` | **最详细的一份**（53 KB）。每次运行的时间、输入输出路径、脚本哈希、影响行数、隔离位置、检查结果、限制。包括失败和事故 |
| `DATA_DICTIONARY.md` | 字段定义与固定清洗规则 |
| `eligible_spans_2015_2026_corrected.csv` | 781 只股票池清单，所有取数以它为准 |
| `../../AI_DAILY_STATE_MODEL_RESULTS_v1.md` | 当前日频数据处理、模型设计、完整验证表、限制与下一步 |

### 04_records — 工作规范与 AI 使用记录

| 文件 | 内容 |
|---|---|
| `AGENTS.md` | 数据处理透明度要求。**新会话应先读这个** |
| `AI_USE_LOG.md` | 课程要求的 AI 使用记录，按阶段记录哪个模型做了什么、哪些判断需要人工复核 |
| `CLAUDE_HANDOFF_2026-09-09.md` | 上一次交接文档，含当时的完整状态 |

---

## 几条不能破的规矩

这些是项目一路走下来定的，破了前面的工作就白做：

1. **测试期封存**。2023-01-01 至 2026-06-30 在规格冻结前不得计算任何目标、预测、指标或策略表现。为机械日历读取含后期数据的文件不等于获得评分授权。
2. **任何实质数据处理都要先说明、后报告并写日志**——退市处理、缺失值、去重、隔离、单位换算、异常值、窗口截断。已经授权并记录的常规步骤不需要反复等待批准。
3. **不删行，用标记**。缺失保持 NA 不填补，排除项带 reason code 留在主表里。
4. **区分"不可解析"和"无权限"**。字段名不认 ≠ 这个 license 没有该数据，两者在报告里不能混。
5. **失败要留痕**。error sidecar、被取代的运行、事故经过都保留并记录，不静默清理。
6. **不因为一个子样本好看就采纳它**。训练与验证方向必须一致。

---

## 两件必须先定、否则会毁掉测试期的事

**① 多重比较。** 四个策略若都在同一测试集上评估再挑最好的报告，与"扫九个信号组合挑一个"是同一种错误。测试期打开前必须先选定：预先声明主策略、采用多重比较校正、或明确声明为探索性研究。

**② AI 分部词典与覆盖门槛。** 历史分部字段已取到，但重复 total、对账差异和名称分类都必须先冻结规则。R&D/分部缺失保持 NA；模型若用中位数与 missing indicator，只能在训练折内拟合并同时报告 complete-case。

---

## 已经吃过的亏

写下来是为了不再犯第二次。

**方法论上**

- 池化 IC 会高估事件内排序能力。receiver Ridge 池化 IC 0.096，事件内只有 0.0159——投资动作需要的是后者。
- 只在验证期好看的子组不是发现。"高 surprise + 正相关在 10 天"验证期 IC 0.028、多空差 45.7 bp，但训练期几乎平坦，判为不采纳。
- 训练与验证符号相反是常见死法。lead-lag 图和纯覆盖 Jaccard 都栽在这里。
- 用今天的认知选历史股票就是前视。PLTR 2020 年才上市，NVDA 2015 年是游戏显卡公司。

**LSEG 平台上**

- 取指数变动方向，只有**拼错的** `TR.IndexJLConstituentituentChange` 才返回 Joiner/Leaver 的 Change 列，正确拼写会静默丢列。
- 新闻的时间戳在 DataFrame 的 **index**（`versionCreated`）里，`to_csv(index=False)` 会丢掉。
- 请求 EDate 超过退市日时，LSEG 会把最后一个交易日重复填充到 EDate。
- broker 名显示为 `Permission Denied <数字>` 时，那个数字是稳定的 broker ID，跨公司可比，不要因为看不到名字就丢弃——被遮蔽的恰恰是覆盖面最广的大行。
- 推荐记录的范围请求会按交易日展开出 167 倍冗余，且响应不含标明日期的列，只能用单点 as-of。
- 新闻 API 单次 100 条上限、按 newToOld 排序，且服务端频繁超时，不适合全量取数。

---

## 新会话怎么接手

按顺序读：`04_records/AGENTS.md` → `01_protocol/RESEARCH_PROTOCOL.md`（尤其 v0.8 到 v0.9）→ `01_protocol/AI_FACTOR_DATA_SPEC.md` → `01_protocol/AI_SECTOR_FEASIBILITY.md` → 本文件的“不能破的规矩”和“必须先定的两件事”。

需要某个具体运行的细节时再查 `03_data/DATA_PROCESSING_LOG.md`，那份 53 KB 不必通读。

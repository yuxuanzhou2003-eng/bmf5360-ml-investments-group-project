# AI 因子、Random Forest 选股与择时：数据规格 v0.3

> 状态说明（2026-09-10）：用户已将研究目标改为“先定义 AI 行业股票池，再在池内交易”。本文件中把连续 AI 暴露作为主交易因子的部分由 `RESEARCH_PROTOCOL.md` v1.0 supersede；原始字段仍可作为 AI 池成员资格证据和描述性变量保留。

日期：2026-09-10。状态：**待团队冻结的执行规格；本文件本身不表示已经取数、清洗、建模或完成回测**。

本规格只新增文档，不发起 LSEG 请求，不读取或推导测试期目标，也不改写现有数据和文档。文中将“已证实”“候选”“待证实”和“建议默认值”分开写；任何待证实项目通过前，不得把候选字段升级为正式因子输入。

## 1. 研究对象和最终基金范围

这里的 `RF` 指 Random Forest（随机森林），不是 risk-free rate。

最终研究对象是美国大盘股 AI 产业链交易，研究期为 2015-01-01 至 2026-06-30。股票池先由**历史可得的成员资格**决定，再用随时间变化的 AI 暴露连续分数排序；不得按今天的知识制作静态 AI 名单回填到 2015 年。

基金研究分成两层：

| 层 | 角色 | 预先约束 |
|---|---|---|
| AI 因子主线 | 用 point-in-time AI 暴露分数形成横截面多空因子组合 | 因子只在形成时点已可得的信息上计算；研究组合、收益口径、成本和可交易性独立记录 |
| RF 选股与择时 | 在形成时点预测每只股票下一持有期相对 SPY 收益为正的概率；用横截面概率排序选股，并把概率分布聚合成组合风险开关 | RF 一行是一只股票 × 一个月，只使用截至决策 cutoff 的特征；主标签固定为未来 21 个 SPY session 的个股相对收益方向 |
| event sleeve | 机制检验和卫星仓位，观察财报意外、信息传导或共同分析师覆盖是否解释 AI 产业链反应 | 与 AI 因子和 RF 特征分表；保留现有 event study 的事件级审计；不得用 event sleeve 的结果选择主线规则 |

主线的研究基准组合建议固定为：每月最后一个 SPY 交易日形成，下一 SPY 交易日收盘执行；在形成时点的合格股票中，AI 分数最高 20% 等权做多、最低 20% 等权做空，形成 `AI_factor_spread`。每边至少 10 只股票，否则该形成日不形成因子并保留原因码。RF 同时对每只合格股票预测未来 21 个 SPY session 相对收益为正的概率；该概率用于 AI 高暴露股票内部的选股排序，月度概率分布再形成预先规定的 0.5/1.0 风险倍数。最终基金是长偏、纯多空或只做多，须在建模前由 root 冻结；本规格先把可比较的多空 spread 和 RF 增量定义清楚。

event sleeve 的主问题是“事件信息是否提供 AI 产业链的机制证据”，不是用它替换因子主线。现有 `panel_v3` 的 event 表、五交易日相对 SPY 标签和日终执行资料可以作为工程输入，但关系边的最终版本、是否采用共同分析师覆盖图，以及卫星仓位上限仍要单独冻结。事件标签不能进入 RF 的同日或未来特征。

## 2. 当前输入盘点及版本边界

陌生成员先按下面的路径理解已有层。它们是现有工程输入，不是本规格已经执行的结果。

| 内容 | 当前路径/结构 | 读法和限制 |
|---|---|---|
| 修正后的成员区间 | `data/audit/universe_rebuild/eligible_spans_2015_2026_corrected.csv` | 读入 `ric,name,delisted_ric,member_from,member_to,member_days,fetch_start,fetch_end`；同一 RIC 的区间是 PIT 资格依据 |
| v3 原始计划和响应 | `data/raw/universe_v3/20260909T012417705069Z/` | `plan.json` 保存请求、批次和哈希；actual/estimate 是长表，prices 是按 RIC 展开的宽表；失败尝试保留 `.error.json` |
| v3 清洁价格/流动性 | `data/clean/v3/20260909T012417705069Z/prices.csv` | 长表主键目标为 `Instrument + Date`；有 `TRDPRC_1, OPEN_PRC, HIGH_1, LOW_1, ACVOL_UNS, BID, ASK, TRNOVR_UNS` 以及可用性、quoted spread、dollar volume、来源和复权记录 |
| 现有收益 | `data/clean/v2/returns.csv` | `Instrument,Date,Total Return,return_decimal,in_sp500_that_day`；原始 `Total Return` 按现有清洗约定除以 100 得 `return_decimal`；它与 v3 价格并非自动同版本，使用前要做版本和总回报对账 |
| v2 原始收益 | `data/raw/universe_v2/returns_*.csv`、`benchmark_returns.csv` | 长表样例可见相同键重复；清洁层只保留完全相同记录的一份，冲突键进入隔离表；SPY 原始标识为 `SPY.P` |
| 财报实际值 | `data/clean/v3/.../actuals.csv` | `Instrument,announcement,announcement_day,period_end,calc_date,eps_actual,...,actual_source,timezone_status`；是 event sleeve 输入，不能充当 AI 因子本身 |
| 周频一致预期 | `data/clean/v3/.../estimates_weekly.csv` | 同一 `Instrument + Period End Date` 的历史 snapshot；事件匹配要求严格早于公告日、精确同财季、最大 14 日陈旧度 |
| event panel | `data/panel_v3/20260909T035245922341Z/` | `events.csv`、`features.csv`、`labels.csv`、`execution_inputs.csv`、`diagnostics_ex_post.csv` 物理分离；future label 和事后诊断不得进入主线特征 |
| 现有 model-ready 示例 | `data/model_ready_v1/20260909T064151673764Z/` | `model_features.csv` 是 45 个入场前候选特征，`eligibility.csv` 保存逐行资格和原因；可复用结构，不把它当成 AI 因子已冻结结果 |

现有只读盘点存在一个必须先解决的计数差异：修正区间文件有 781 行，其中 154 个 `delisted_ric=True`、627 个非退市行；旧的 `universe_spans_summary.json`/可行性文档写的是 782 个 RIC、155 个退市 RIC，v3 raw plan 还明确排除了 `EVHC.N^L16`。这可能是旧联合名单和 benchmark/错误区间混在一起造成的，但在新 collector 的 `universe_manifest` 中必须逐一对账，不能凭数字猜测。

## 3. Point-in-time universe

### 3.1 基础成员资格

默认基础股票池来自 `eligible_spans_2015_2026_corrected.csv`，不是当前成分股名单。对每个形成日 `t`，只有满足

```text
member_from <= t <= member_to
```

的 `ric` 才是该日的历史成员；单日区间也按上式保留。`member_from`、`member_to` 的边界含义、同日加入/退出顺序和 781/782 的计数对账要在 collector 前写入 manifest。

成员资格和数据可得性是两件事：

- 成员在形成日合格但缺价格、AI 暴露或流动性时，保留该成员和缺口原因；不能把它从历史公司集合静默删除。
- 退市证券在其 `member_to` 之前仍可入池；不得因为后来退市而从历史样本删除。退市后的价格、结算和目标缺口按 `DELISTED_TERMINAL_UNRESOLVED` 或相应原因保留。
- `member_to` 之后不向前延长持仓资格，也不以前值替代缺失价格。

### 3.2 行业范围和 AI 产业链

AI 可行性文档提出 TRBC Technology 129 只加相关行业的宽池，但现有 `data/raw/trbc_diagnostic_v1/trbc_current.csv` 只有当前快照：711 行，其中当前 Technology 129 行。当前行业不能回填历史，`TRBC Economic Sector Name` 和 `TRBC Industry Group Name` 的历史有效日期尚未证实。

因此按以下顺序执行：

1. 基础 PIT 成员集合先完整生成，保留全部 781 个修正区间对应证券以及单独的 SPY benchmark。
2. 只有取得带有效日期或可复现 as-of 结果的历史 TRBC 分类，才能正式限制到 Technology/相关行业；行业字段必须有 `valid_from/valid_to` 或供应商明确的 as-of 语义。
3. 在 PIT TRBC 通过前，不能用当前 Technology 129 只构造正式 2015–2026 因子。若团队决定暂时使用全 PIT 成员作为研究宽池，必须把它写成“宽池敏感性/默认研究宇宙”并固定，不得称为历史 AI 行业名单。
4. “相关行业”必须列出可重复的 TRBC industry group 名称或代码；不能由公司名称、今天的 AI 印象、新闻关键词或最终收益反推成员。

### 3.3 身份键

每行至少保留四类键：

| 键 | 作用 | 规则 |
|---|---|---|
| `ric` | LSEG 取数和证券级价格键 | 原样保留，包括 `^` 退市/历史后缀；不因 ticker 更名直接覆盖 |
| `security_id` | 研究期内稳定的上市证券键 | 由显式映射表生成；不得用字符串猜测替换 |
| `issuer_id` | 发行人级合并/分组键 | 优先使用带有效日期的 Organization PermID 或经证据支持的映射；当前 master 的 PermID 不能自动回填历史 |
| `identity_map_version` | 复现和审计 | 记录来源、有效区间、映射类型、证据、审核状态和哈希 |

`identity_map.csv` 至少包含 `raw_ric,security_id,issuer_id,valid_from,valid_to,map_type,evidence_path,map_status,reason_code`。更名、重组、并购、ADR/普通股和同一发行人多上市证券要分开判定；没有一对一证据时保留原始行，标 `IDENTITY_UNRESOLVED` 或 `IDENTITY_AMBIGUOUS`，不强行合并。FDX/FDXF、历史后缀 RIC 和 AMCR 等已知异常必须进入映射审计清单。

## 4. 预先固定的 AI 因子、控制变量和使用时点

下面是取数阶段的数据规格和建模前需要冻结的候选设计。取数只收集原始解释变量，不等于已经批准任何因子权重。公式、窗口、最大陈旧度、分位点、标签和回退路径都要写进 `factor_config.json`；建模后不得从 test 表现倒推修改。

### 4.1 形成日和 cutoff

- 形成日：每个自然月最后一个 SPY 交易日 `formation_session`。
- 决策 cutoff：形成日收盘；可用信息必须在该日收盘前已存在。形成日收盘只用于生成下一交易日组合，不把形成日之后的价格写回特征。
- 执行日：形成日之后的第一个 SPY 交易日收盘 `entry_session`。
- 形成日的成员资格、AI 评分、控制变量和组合权重在 `formation_session` 冻结；RF 也在同一 cutoff 输出 `p_up`。
- 因子持有期：从 `entry_session` 收盘之后到其后第 21 个 SPY session 收盘。若采用不同执行或持有时点，形成全新 `factor_config`、标签和 purge 清单。

### 4.2 AI 暴露组件

#### `AI_REV`: 历史业务分部收入暴露

`ReportingState=Orig` 探针已经证实 `TR.F.BUSTotRevBizActiv(Period=FY0)` 可返回历史分部代码、名称、财务期间、日期和收入值。正式清洁时先保留供应商原始行，再由事先登记、与收益无关的词典把 named segment 分成 `core_ai`、`broad_ai_infrastructure`、`non_ai`、`ambiguous`。初版词典必须在读取任何因子未来收益前冻结，并保存版本、每次规则修改和人工复核记录。

```text
ai_segment_share(i,p) = sum(revenue of accepted AI segments) / selected_company_total_revenue
```

只有在公司—期间有可识别 total、分母为正、原始公告日不晚于形成日且 reconciliation 状态合格时才计算。探针的 54 个公司—期间中有 4 组重复 null-total，14 组所选 total 与 `TR.F.TotRevenue` 不完全相等；正式流程不得强制对齐或静默去重，而要分别标记 `SEGMENT_TOTAL_AMBIGUOUS`、`SEGMENT_RECONCILIATION_GAP`。`All Other`、eliminations、blank-name、blank-code named rows均保留，不自动归入 AI。分部名称变化保留为历史事实，不能用今天的名称回填过去。

`AI_REV` 是最直接、也最稀疏的 AI 经营暴露候选。它衡量披露口径下的 AI 相关收入，不等于全公司的真实 AI 经济暴露；词典分类需要进行 core/broad 两套事先登记的敏感性分析。

#### `AI_RD`: R&D 强度

对证券 `i` 在形成日 `t`，取最新一条满足以下条件的同一发行人、同一财务期间记录：

```text
rd_intensity(i,t) = ResearchAndDevelopment(i,p) / Revenue(i,p)
```

条件是：

1. R&D 和 Revenue 的 `period_end` 完全一致，币种、单位和 scale 已通过语义门槛；`Revenue > 0` 且两个值有限。
2. 该观察有供应商明确的 `available_at`/发布日期/有效 vintage，且 `available_at <= formation_session`。只有 `fperiod` 不能证明当时可得，不能把财年结束日当发布日期。
3. 在候选记录中取 cutoff 前最近的可得 vintage，并保留 `period_end, available_at, vintage_id, snapshot_age_days, raw_file, raw_row`。默认最大陈旧度建议为 450 个日历日；该数值需 root 冻结。超过上限标记 `FUNDAMENTAL_STALE`，不前值延长。
4. 缺 R&D、缺 Revenue、Revenue 为零/负数、币种或单位不明、缺 available date/vintage，均为 NA 和原因码；不填零、不用行业均值、不用上一期盲目填补。

在每个 formation cross-section 内，对有效 `rd_intensity` 做百分位排名 `AI_RD_RANK`（0–1，平值使用平均名次）。因子值是经济量，排名只用于组合构造，不能消除原始单位/口径问题。

v3 探针证明 `TR.ResearchAndDevelopment`、`TR.Revenue` 和原始公告日可以分别返回。`ReportingState=Orig` 与默认 current-state 值在样本中存在实质差异，因此正式输入只能使用 Orig run，并以 `TR.ISOriginalAnnouncementDate <= formation_session` 作为可用时点门槛。探针还发现 WMT 的 9 个财年没有 R&D keyed row，另有 1 个无键空值占位行；这类缺失保持 NA，不解释为真实零。字段单位和币种仍需在正式清洁 gate 中验证。

#### `AI_MKT`: 控制市场后的 SOXX 敏感度

正式 collector 会分别保存 `SOXX.OQ`、`XLK.P`、`BOTZ.OQ`、`AIQ.OQ`、`IGV.P`，并可显式拉取 `SPY.P` 对账。主候选使用覆盖较长的 SOXX，并先控制整体市场，避免把普通市场 beta 误称为 AI 暴露。对每个形成日，在 `[t-126 sessions, t-1 session]` 内先估计 SOXX 对 SPY 的回归残差，再估计个股对 SPY 的残差与该 SOXX 残差的斜率：

```text
u_SOXX = residual from r_SOXX = a + b * r_SPY
u_i    = residual from r_i    = c + d * r_SPY
AI_MKT_RESID_BETA_126(i,t) = Cov(u_i, u_SOXX) / Var(u_SOXX)
```

至少 100 个三方共同观测且残差方差为正，否则为 NA；最晚观测日必须严格早于 cutoff。对有效斜率做横截面百分位排名 `AI_MKT_RANK`。该量仍主要代表半导体/AI 基础设施的市场隐含暴露，不能解释为完整 AI 收入占比。

其它 ETF 只作为事先登记的机制分解和稳健性，不通过验证期或测试期挑选：

- `AI_MKT_XLK_RESID_V1`：控制 SPY 后的 XLK 敏感度，用来识别广义科技成分。
- `AI_MKT_THEME_RESID_V1`：BOTZ/AIQ/IGV 各自控制 SPY 后的敏感度，仅在各 ETF 实际上市后的有效窗口计算，不回填上市前历史。

使用价格 beta 定义 AI 暴露，再用它预测价格存在循环性风险。价格 beta 只能作为严格历史窗口内的特征，不能作为标签；报告必须把该风险和 ETF basket 的产业含义一起披露。

#### `AI_COMPOSITE`: 训练期冻结的组合暴露

正式 RF 保留 `AI_REV_RANK`、`AI_RD_RANK` 和 `AI_MKT_RANK` 三个独立特征，让非线性模型在 training-only CV 内学习增量。用于传统 long/short 因子交易的组合分数候选为三组件等权：

```text
AI_COMPOSITE_3(i,t) = (AI_REV_RANK + AI_RD_RANK + AI_MKT_RANK) / 3
```

`AI_COMPOSITE_3` 只在三项都有效时计算，不因缺一项而重新归一化，也不把缺失项设为零。考虑到分部和 R&D 披露可能显著缩小样本，另登记 `AI_COMPOSITE_2 = (AI_RD_RANK + AI_MKT_RANK)/2` 为覆盖稳健性；是否把它作为可交易主分数，只能根据 training 覆盖和 training-only CV 在第一次 validation 前冻结。所有原始值、rank、缺失指标、陈旧度、分部 reconciliation 状态和有效组件数都单独输出。

### 4.3 因子组合和交易标签

在每个形成日的 PIT 合格且有冻结版本 `AI_COMPOSITE` 的成员中：

- 最高 20% 为 long，最低 20% 为 short；精确平值按 `security_id` 升序稳定打破，分组规则写入 config。
- 每边至少 10 只；不足则整日 `FACTOR_SIDE_TOO_SMALL`，保留该日和成员，不形成零收益。
- 每边等权，long 权重和为 `+1`、short 权重和为 `-1`；同一股票不能同时两边。
- 形成日到入场日之间不重新排名；换手、spread、dollar volume 和借券约束在执行层单独计算。

默认 RF 主标签是个股相对收益的二元方向：

```text
R_i,H(t) = product_{d in (entry_session_t, exit_session_t]}(1 + r_i,d) - 1
R_SPY,H(t) = product_{d in (entry_session_t, exit_session_t]}(1 + r_SPY,d) - 1
stock_excess_h21(i,t) = R_i,21(t) - R_SPY,21(t)
RF_LABEL_H21(i,t) = 1{stock_excess_h21(i,t) > 0}
```

`exit_session_t` 是入场后第 21 个 SPY session；区间 `(entry_session, exit_session]` 表示不使用入场收盘前的日内/当日变动。标签为**未扣成本**的个股相对收益方向，成本后的基金收益另表。只有个股和 SPY 的必需收益、入场和退出记录齐全，且退出未越过 split boundary 时标签才有效；缺失不设为零。`RF_LABEL_H5`、`RF_LABEL_H63` 只可作为预先登记的稳健性标签，不能在看过 validation/test 后替代 H21 主标签。

### 4.4 RF 输入和控制变量

RF 的一行是一个 `security_id + formation_session`。这在训练期形成数万条股票—月观察，同时所有交叉验证仍按完整月份向前推进；同月股票不会被随机拆入训练和验证，也不把 event graph edge 当独立样本。默认候选输入只来自 cutoff 之前的数据：

| 组 | 预先固定的变量 |
|---|---|
| AI 暴露 | `AI_REV_RANK`、`AI_RD_RANK`、`AI_MKT_RANK`、冻结版 `AI_COMPOSITE`、组件是否缺失、陈旧度、segment reconciliation 状态；缺失指标与数值分开 |
| 公司控制 | lagged log market cap、book-to-market、gross profitability/assets、operating cash flow/assets、asset growth、debt/assets；只有 Orig + original-announcement gate 通过的字段才进入 |
| 个股行情 | 1/5/20/60/252 session momentum、20/60 session volatility、126 session market beta、idiosyncratic volatility、20 session volume/dollar volume/spread |
| 因子状态 | 截至 cutoff 的 AI spread 1/5/20 session 收益、20/60 session 波动、AI score 横截面中位数/标准差与有效覆盖率 |
| 市场状态 | SPY 的 1/5/20/60 session 动量，20/60 session 年化波动，SPY beta/有效观测数（如需要） |
| 可交易性 | 个股有效价格、双边报价、dollar volume、quoted spread，以及当月横截面覆盖率；缺口单独作为指标，不能改写为零 |
| 行业控制 | 只有 PIT TRBC 通过后才加入历史 sector/industry 哑变量；当前 TRBC 快照不作历史控制变量 |

按现有数据字典计算：momentum 为窗口内 `prod(1+r)-1`；波动为日收益样本标准差乘 `sqrt(252)`；beta 用至少 100 个成对观测的 126-session OLS；idiosyncratic volatility 用同一回归残差标准误乘 `sqrt(252)`；流动性中位数只在最后 20 个 SPY session 的实际非空观测上计算。`r` 的总回报/价格收益口径必须在 `return_basis` 中固定。

当前 v3 清洁层明确有复权价格、volume、bid/ask、turnover 和 `dollar_volume_source`；它没有和 v2 `returns.csv` 自动等价的 v3 总回报表。因此默认先把“v3 adjusted close 的价格收益”和“v2 clean total return”作为两个显式候选口径，root 必须在标签生成前选定一个主口径。不能把 adjusted close 价格收益称为 total return，也不能混用两套版本而不保留来源。

### 4.5 event sleeve 的使用时点

event sleeve 沿用已写入 `RESEARCH_PROTOCOL.md` 的可审计时点：

- 实际 EPS 使用 `ActType=Reported`，`Report Date` 按供应商原样保留；正式入场用公告日历日之后的第一个 SPY 交易日收盘，盘前/盘中/盘后只作诊断。
- 一致预期按相同 Instrument 和精确 Period End Date 匹配，snapshot 严格早于公告日且不超过 14 日；缺失和过旧不填补。
- 主 event 标签是入场收盘后 5 个交易日 receiver 相对 SPY 的累计收益，1/10 日只作已登记敏感性。
- event 的 actual/consensus/dispersion、未来公告和 overlap 诊断不进入 AI 因子或 RF 的同日特征。关系图若改用共同分析师覆盖，必须以自己的 collector/clean/model-ready 版本接入并保留旧图结果。

## 5. LSEG 字段清单和验证门槛

### 5.1 字段状态

| 用途 | 字段/数据项 | 当前状态 | 进入正式输入前必须通过 |
|---|---|---|---|
| R&D 组件 | `TR.ResearchAndDevelopment` + `.date/.fperiod`, `ReportingState=Orig` | **6 股×9 财年探针已返回**；45/54 keyed slots 有值，WMT 9 槽 no-row，另有 1 个无键空占位 | 正式全量覆盖、币种/scale、与 Revenue 同期间配对、original announcement gate |
| Revenue 与控制 | `TR.Revenue`、`TR.F.TotRevenue/TotAssets/ComEqTot/GrossProfIndPropTot/NetCashFlowOp/DebtTot`，`ReportingState=Orig` | **探针各覆盖 54/54 期间槽**；Orig 与默认 current-state 在 369 个可比值中有 52 个差异 | 正式全量覆盖、单位/币种、同期间键、original announcement gate；不得混用 current-state 值 |
| 财务可用日期 | `TR.ISOriginalAnnouncementDate`, `TR.ISStatementLastUpdatedDate`, `TR.ISPeriodEndDate`，`ReportingState=Orig` | **探针 54/54 返回**；原始公告相对期间末 11–38 天，无负滞后；Last Update 仅作诊断 | 以 original announcement 作为可用门槛，保留 last update，不把 period end 或 last update 误作首次可用日 |
| ETF/股票价格 | `TRDPRC_1, OPEN_PRC, HIGH_1, LOW_1, ACVOL_UNS, BID, ASK, TRNOVR_UNS` | **字段和 v3 raw/clean 结构已证实**；collector 用 `get_history`、1D、完整调整列表 | 每个 ETF RIC 的日期覆盖、重复键、正值、调整标志、首个有效日期、与 benchmark 日期对齐；总回报语义仍待证实 |
| ETF 候选 | `SOXX.OQ, XLK.P, BOTZ.OQ, AIQ.OQ, IGV.P` | **候选覆盖已在可行性文档中报告**，未完成本规格所需完整持久化 | 每个组件的实际响应、上市前空白、复权/分红口径、basket 最低组件数和版本 |
| benchmark | `SPY.P` 价格；如使用总回报则对应已验证总回报字段 | 价格结构已存在；总回报版本一致性 **待证实** | SPY session calendar、price/return 对账、`return_basis` 冻结 |
| event actual | `TR.EPSActValue(ActType=Reported)` 及 `.announcedate/.calcdate/.periodenddate/.fperiod/.currency` | **字段响应和 DIB 对 split-standardized actual 的窄语义已证实** | EPS basis、币种/每股单位、历史 vintage、公告时区；不能写成发行人 GAAP diluted EPS |
| event consensus | `TR.EPSMean.calcdate/.periodenddate/.fperiod/.currency`, `TR.EPSMean`, `TR.EPSStdDev`, `TR.EPSNumIncEstimates` | **已有周频清洁表** | snapshot < announcement_day、精确 period end、14 日陈旧度、修订/vintage 解释 |
| PIT 行业 | `TRBC Economic Sector Name`, `TRBC Industry Group Name` 的历史有效/as-of 版本 | 现有 `trbc_current.csv` 是**当前-only；历史不可用** | 历史有效日期或可复现 as-of 查询；通过前不得用当前行业回填 |
| 业务描述 | `TR.BusinessSummary` | **已证实返回，但当前-only** | 不进入任何回溯特征或筛选；只作当前描述 |
| 业务分部收入 | `TR.F.BUSTotRevBizActiv(Period=FY0).segmentCode/.segmentName/.fperiod/.date/value`，`ReportingState=Orig` | **正确字段已证实**：探针 269 行、54 个公司—期间组；58 个 null-total candidate、4 组重复 total、14 组与 `TR.F.TotRevenue` 不完全一致 | 全量覆盖；total 选择/reconciliation 原因码；冻结 AI 词典；不得静默去重或强制调平 |
| 其它主题/专利 | `TR.ThemeExposure`, `TR.PatentCount`, `TR.CompanyDescription` 及旧分部拼写 | **探针字段无法解析**；这是字段名/解析失败，不等同于权限拒绝 | 另行 DIB 字段确认；当前不进入主模型 |
| 新闻 | `ld.news.get_headlines` | **可返回候选新闻，但不作为全量因子**；探针保存的 CSV 仅有 headline/storyId/sourceCode，没有可用时间列 | 若做小样本代理验证，必须保留 versionCreated/availability_utc；不进入正式全量 AI 因子 |

### 5.2 六道 gate

任何 gate 失败都要在 `gate_report.json` 标记 `blocked`，并把受影响行放入审计/隔离表。

1. **解析 gate**：每个字段单独请求和保存 literal field name、响应列名、状态、错误 sidecar；不能因组合请求失败就推断其它字段不存在。
2. **结构 gate**：预期列、类型、日期格式、主键、批次数、行数和 SHA-256 一致；同一主键重复且值冲突要 quarantine，完全相同记录才可保留一份。
3. **语义 gate**：R&D/Revenue 的单位、币种、期间、scale；价格的调整和 quote 定义；收益是价格收益还是总回报；EPS 的 basis 与公司行动；每个结论附 DIB/response 证据路径。
4. **PIT gate**：每个财务 observation 有可复现 available date/vintage 且不晚于 formation cutoff；每个 ETF/价格特征最晚日期 `< cutoff`；当前-only TRBC/BusinessSummary 禁止进入历史输入；event snapshot 严格 `< announcement_day`。
5. **覆盖 gate**：按 RIC、formation month、period、退市状态分别报告 non-null、stale、no-row 和 failed-fetch。缺口保留，不能用成功公司数量掩盖失败公司。
6. **独立复算 gate**：从 clean 表抽样独立重算 R&D ratio、ETF basket/beta、rank、long/short 权重、标签算术和 split/purge；保存 `validation.json`。通过工程检查不等于经济含义或策略有效性通过。

## 6. 缺失、退市、split、身份和 vintage 处理

### 6.1 缺失值

空白、不可解析日期、数值转换失败和非有限值统一为 NA，并保留原始值/原始位置。NA 与真实零严格区分：真实 R&D 为 0、真实收益为 0 和没有返回记录不是一回事。

正式默认规则是：不均值/中位数填补、不前填/后填、不用零替代、不静默 winsorize。模型若经 root 批准使用 training-only 中位数加 missing indicator，必须同时保留 complete-case 结果、训练拟合参数、缺失指示器和每个 split 的覆盖报告；validation/test 只应用 training 参数。

### 6.2 退市和终止价格

退市只改变未来的成员资格和可执行性，不是历史排除理由。对每个形成日记录 `pit_member_at_formation`、`delisted_ric`、`member_to` 和 `terminal_data_status`。若未来收益缺失，标 `LABEL_MISSING`/`DELISTED_TERMINAL_UNRESOLVED`，不赋零收益；若已建仓但没有结算价格，基金核算必须单独披露未解决处理。

### 6.3 Split 和公司行动

v3 clean price 记录 `price_adjustments=exchangeCorrection,manualCorrection,CCH,CRE,RPO,RTS`，并保存公司行动探针原始记录。`TR.EPSActValue` 已由 DIB 证明包含公司行动标准化（明确包括 stock split），因此不得再次对 EPS、预期或 clean price 手工乘 split factor。split 表用于对账、影响清单和异常诊断，不用于重复修正。调整字段缺失或语义不明标 `PRICE_ADJUSTMENT_UNVERIFIED`，不能假定已复权。

### 6.4 身份变更

RIC、ISIN、PermID、公司名称和发行人身份分别存储。更名或 ticker 变化只有在有效区间和证据支持下才连接；合并、分拆、ADR、多个上市证券不能仅按名称合并。映射不确定时，行保留在 raw/clean 审计，模型资格置 false 并写 `IDENTITY_UNRESOLVED`/`IDENTITY_AMBIGUOUS`。

### 6.5 可得日期、时区和 vintage

- `Report Date` 和 `announcement` 按供应商原样落盘；当前 `timezone_status` 是 vendor clock naive，不能把字符串直接称为美东时刻。正式 event 入场只用公告日历日之后第一个 SPY session。
- `Calc Date`、`Period End Date`、`fperiod` 的含义分开保存。财年结束日不是投资者获得报表的日期。
- R&D/Revenue 必须有供应商 available/release/vintage 证据。若只有当前修订值和财年标签，历史因子为 `VINTAGE_NOT_PROVEN`，不使用。
- estimates 取严格公告日前最近的同 Instrument、同 Period End snapshot，并保存 `snapshot_age_days`；不能把后续修订当历史已知信息。
- ETF/股票价格的可得时点是对应交易日收盘后；使用它形成下一交易日组合，不把形成日后价格放进特征。

## 7. 训练、验证、测试和 purge

### 7.1 固定分区

| split | formation/事件日期 | 允许的决定 |
|---|---|---|
| training | 2015-01-01—2020-12-31 | 因子诊断、RF 的 walk-forward 内部 CV、预处理拟合、有限候选配置 |
| validation | 2021-01-01—2022-12-31 | 在首次拟合前登记的候选中做一次模型/参数选择和稳健性报告 |
| test | 2023-01-01—2026-06-30 | 规则冻结后一次正式评价；在冻结前保持封存 |

分区由 formation date（event sleeve 用 announcement_day）决定，不按收益、不按最终覆盖和不按静态 AI 名单调边界。RF 的开发 CV 采用时间向前的 expanding/rolling 训练窗口，不用随机 K-fold。预处理、缺失参数、阈值、feature selection 只在每个训练窗口拟合。

### 7.2 因子标签 purge

主标签 H=21。对于 split 边界 `B`，较早 split 的形成日若 `entry_session >= B` 或 `exit_session >= B`，该形成日的全部股票样本标记 `boundary_purged=True`，不用于该 split 的监督训练，原始组合/成员行保留。形成日是原子单位，不能只保留其中标签完整的股票来挽救月份。

RF 内部 CV 也至少留出覆盖完整 H=21 目标窗口的 gap/embargo；训练窗口不能看到验证窗口的目标或使用跨界收益。event sleeve 按 `(source, announcement_day, period_end)` 分组，同一事件的所有 receiver edge 一起分区和 purge；不能把同一事件拆到不同 split。

在本规格阶段只构造分区计划、特征 cutoff 和 purge 清单，不读取/推导 test 的 `factor_spread`、RF label、预测、指标或基金表现。test target 在规则 hash、gate report 和 root 冻结记录完成后才允许由专门步骤生成或解封；任何提前访问都必须标记违规并停止正式评价。

## 8. 候选尝试登记和多重比较

每个候选在运行前登记一行 `candidate_registry.csv`，至少包括：

```text
candidate_id, family_id, role, parent_spec_hash, data_version,
universe_rule, factor_definition, basket_version, rebalance_rule,
return_basis, label_horizon, label_threshold, missing_rule,
feature_list, model_config, split_usage, pre_registered_at,
status, output_path, reviewer, notes
```

建议首次登记的有限候选：

| candidate_id | role | 固定内容 |
|---|---|---|
| `AI_3COMP_RF_H21_V1` | primary candidate | `AI_REV/AI_RD/AI_MKT` 独立输入、股票—月样本、H21、`stock excess > 0`，按概率排序形成组合；是否成为正式 primary 要在 training 覆盖 gate 后、首次 validation 前冻结 |
| `AI_REV_FACTOR_H21_V1` | economic mechanism | 仅按冻结 core/broad 词典形成的历史 AI segment revenue share 排序；完整披露覆盖与 reconciliation 状态 |
| `AI_RD_RF_H21_V1` | registered robustness | 仅 R&D intensity rank；只有 Orig + announcement + unit gate 通过才可运行 |
| `AI_MKT_RF_H21_V1` | registered robustness | 控制 SPY 后的 SOXX residual beta rank；不得用 validation/test 在 ETF 间择优 |
| `AI_COMPOSITE2_RF_H21_V1` | coverage robustness | R&D rank 与 SOXX residual-beta rank 等权；只按 training 覆盖决定是否可交易 |
| `AI_COMPOSITE_RF_H5/H63_V1` | registered robustness | 同一 score，只改变已登记持有期；不替代 H21 主标签 |

RF 模型也只允许小型、预先登记的候选表。primary 模型使用固定 `RandomForestClassifier`（建议 `n_estimators=500,max_depth=4,min_samples_leaf=100,max_features=sqrt,class_weight=balanced,random_state=5360`）；只在 training 的按月 walk-forward 折比较最多两个深度/叶节点敏感性，并与 Logistic Regression 和简单 momentum 排名比较。validation 只作一次模型确认，test 不搜索参数。模型配置若需变化，建立新 `candidate_id` 和新 data/model-ready 版本。

多重比较规则：

1. `family_id` 记录“因子定义 × basket × horizon × model × portfolio rule”的完整候选数量；报告所有已登记候选的结果和覆盖，而不是只报 validation/test 最好者。
2. 主确认问题只有一个：冻结的主候选在 test 是否有预先定义的方向性/经济效益证据。若团队把多个候选作为正式检验，须在打开 test 前冻结校正方法；默认可用 family 内 Holm-Bonferroni，并报告未校正和校正后的 p-value/置信区间。
3. 未登记的事后尝试、test 后调参、改变分位点/持有期/回退层级的结果均标 `exploratory_after_test`，不能称为独立最终测试，也不能覆盖旧结果。
4. 验证期单个子组、年份或 high-score 现象不能单独升级为发现；同时报告训练、验证、分年份、覆盖率和成本敏感性。

## 9. Collector、clean、model-ready 的输出契约

所有 run 使用新目录和 UTC `run_id`，成功 raw 文件不可覆盖；每个阶段保存输入/输出路径、版本、代码或配置 SHA-256、规则、前后行数/证券数、受影响行、quarantine 路径、checks、限制和状态。实际运行时按项目要求追加 `DATA_PROCESSING_LOG.md` 和 `AI_USE_LOG.md`；本规格撰写阶段没有新增处理记录。

### 9.1 Collector：只保存供应商响应

建议路径：`data/raw/ai_factor_v1/<run_id>/`。

产物：

- `plan.json`：确切 universe、字段 literal name、参数、日期、batch、代码/配置 hash、预期输出和不取 test target 的声明。
- `collector_manifest.json`：每个请求的开始/结束时间、状态、HTTP/LSEG 错误、响应列、行数、SHA-256、credential redaction 状态。
- `rd_<batch>.csv`、`revenue_<batch>.csv`、六类 TR.F 控制、original statement dates、business segments、monthly market cap、`etf_prices_<ric>.csv` 及对应 `.meta.json`。
- 失败尝试保留 `.error.json`；字段未解析、空返回、超时和权限/服务错误分开记录。

Collector 不拼接 ratio、不改单位、不转时区、不去重、不补值、不删除退市证券、不生成 factor score 或 RF label。raw 输入必须能回放到 clean。

### 9.2 Clean：语义和行级审计

建议路径：`data/clean/ai_factor_v1/<run_id>/`，隔离目录为 `data/audit/ai_factor_v1/<run_id>/quarantine/`。

| 表 | 主键 | 必备字段/目的 |
|---|---|---|
| `fundamental_observations.csv` | `security_id + period_end + vintage_id` | R&D、Revenue、六类控制原值、currency、unit/scale、period_end、original announcement、`rd_intensity`、来源和有效性 flags；同期间冲突不任意选最后一条 |
| `segment_observations.csv` | `security_id + period_end + raw_segment_row` | 原始 segment code/name/value、total candidate、词典版本、AI 分类、reconciliation gap 和原因码；重复/空白原样可追溯 |
| `etf_prices.csv` | `Instrument + Date` | 原始价格/流动性字段、adjustment set、has flags、raw file/row、字段缺口 |
| `asset_returns.csv` | `Instrument + Date` | 明确 `return_basis`（price 或 total）、算法/来源、基准、可用性；不把价格收益重命名成总回报 |
| `ai_market_proxies.csv` | `proxy_version + Date` | SOXX/XLK/theme ETF 与 SPY 收益、共同观测、残差计算版本、组件缺口和起始日期 |
| `membership_snapshots.csv` | `formation_session + ric` | PIT membership、行业状态、delisted 状态、成员区间来源、identity map 版本、reason code |
| `identity_map.csv` | `raw_ric + valid_from + valid_to` | security/issuer 映射、类型、证据、审核状态 |
| `split_audit_link.csv` | `Instrument + split_event_id` | split 原始证据、adjustment 对账、是否禁止二次修正 |
| `quarantine.csv` | `source_table + raw_file + raw_row + reason_code` | 原始位置、原值摘要、失败阶段、是否行级不纳入/公司级不纳入、恢复条件 |
| `missingness_summary.csv` | `table + field + split/month` | row/instrument/period 缺失、failed fetch、stale、真实零计数；不把 missing 当 zero |
| `clean_summary.json`、`gate_report.json` | run | 版本、哈希、before/after counts、原因码计数、门槛状态和限制 |

Clean 允许把空白/不可解析值变为 NA、按已批准语义计算 ratio/returns、隔离冲突和结构性宽表 padding；不允许填补、winsorization 或因退市删公司。

### 9.3 Model-ready：形成日级因子和 RF 输入

建议路径：`data/model_ready_ai_factor_v1/<run_id>/`。

| 表 | 主键 | 内容和限制 |
|---|---|---|
| `factor_scores.csv` | `formation_session + security_id` | 成员资格、AI_REV/RD/MKT/composite 原值和 rank、as-of age、segment reconciliation、component flags、score version；只用 cutoff 前信息 |
| `factor_portfolio_members.csv` | `formation_session + security_id` | long/short side、rank、weight、可交易性 flags、组合形成原因；每边权重和必须可复算 |
| `rf_features.csv` | `formation_session + security_id` | AI 暴露、公司控制、个股行情、市场/因子状态、可交易性与缺失指标；不含 future label/entry-day EOD |
| `rf_targets_dev.csv` | `formation_session + security_id` | **仅供 training/validation 开发使用**的个股 H21 excess-return 方向、entry/exit、return basis、target availability 和 purge flags；test 行不解封 |
| `rf_targets_test_sealed/` | `formation_session + security_id` | 规则冻结后才由专门步骤生成/解封；冻结前只保存密封状态和 manifest，不读取目标值 |
| `event_sleeve_features.csv`、`event_sleeve_labels.csv` | `sample_id` | 与现有 event panel 对接；event 未来标签与事后诊断不得 join 到 RF features |
| `eligibility.csv` | `formation_session + security_id` 或 `sample_id` | split、PIT、feature/score/price/side/label/purge flags 及逐行 reason code；flags 不删除 master row |
| `feature_missingness.csv` | `split + feature` | NA 和可用比例；处理参数只从 training 拟合 |
| `split_counts.csv` | `split` | formation day、security、factor side、purge、label、feature 和监督资格计数 |
| `metadata.csv` | 主键 | data/config/code hash、return basis、universe/factor/basket/version、source file/row、vintage、cutoff |
| `candidate_registry.csv` | `candidate_id` | 运行前登记和运行后路径/状态；保留失败和废止候选 |
| `summary.json`、`validation.json` | run | 版本、哈希、before/after counts、审计结果、限制、test seal 状态 |

Model-ready 不做缩放、缩尾或填补，除非在 training 内部拟合并写入处理日志；原始 master row、缺口和不合格组合形成日全部保留。

## 10. Reason code 词典

Reason code 要稳定、可统计、可操作。一个对象可以有多个 code，但要有一个 `primary_reason_code`；reason code 表示资格/状态，不等于物理删行。

| 类别 | 建议代码 | 含义 |
|---|---|---|
| universe | `UNIVERSE_NOT_MEMBER`, `MEMBERSHIP_DATE_CONFLICT`, `MEMBERSHIP_UNKNOWN`, `PIT_CLASS_UNAVAILABLE`, `CURRENT_CLASS_ONLY`, `SECTOR_SCOPE_UNRESOLVED`, `PRE_WARMUP_NO_SNAPSHOT` | 形成日成员/历史分类/图预热资格问题 |
| field/fetch | `FIELD_UNRESOLVED`, `REVENUE_FIELD_UNVERIFIED`, `REQUEST_ERROR`, `TIMEOUT`, `NO_ROWS`, `SCHEMA_MISMATCH`, `REQUIRED_FIELD_MISSING`, `DATE_PARSE_FAIL`, `NONFINITE_VALUE` | 供应商字段或响应结构问题 |
| semantics | `UNIT_UNKNOWN`, `CURRENCY_UNKNOWN`, `PERIOD_MISMATCH`, `REVENUE_NONPOSITIVE`, `VINTAGE_NOT_PROVEN`, `AVAILABLE_DATE_MISSING`, `RETURN_BASIS_UNVERIFIED`, `PRICE_ADJUSTMENT_UNVERIFIED`, `ADJUSTMENT_DOUBLE_APPLICATION_BLOCKED` | 经济口径和时点未通过 |
| stale/missing | `RD_MISSING`, `REVENUE_MISSING`, `FUNDAMENTAL_STALE`, `ETF_COMPONENT_MISSING`, `BASKET_COMPONENTS_INSUFFICIENT`, `HISTORY_INSUFFICIENT`, `ENTRY_PRICE_MISSING`, `QUOTE_MISSING`, `DOLLAR_VOLUME_MISSING` | 缺失/陈旧/历史长度不足；不等于零 |
| identity/delist | `IDENTITY_UNRESOLVED`, `IDENTITY_AMBIGUOUS`, `RIC_CHANGE_UNMAPPED`, `MERGER_UNRESOLVED`, `DELISTED_PIT_ELIGIBLE`, `DELISTED_TERMINAL_UNRESOLVED` | 身份和退市处理状态 |
| factor/portfolio | `FACTOR_COMPONENT_MISSING`, `FACTOR_SCORE_UNAVAILABLE`, `FACTOR_SIDE_TOO_SMALL`, `WEIGHT_FORMATION_FAILED`, `CAPACITY_UNAVAILABLE`, `BORROW_UNVERIFIED` | score、两边组合或投资约束问题 |
| split/label | `LABEL_MISSING`, `LABEL_EXIT_MISSING`, `LABEL_CROSSES_SPLIT`, `BOUNDARY_PURGED`, `FEATURE_CUTOFF_VIOLATION`, `TEST_TARGET_SEALED` | 标签、边界和信息隔离状态 |
| event | `EVENT_ACTUAL_UNAVAILABLE`, `EVENT_SNAPSHOT_MISSING`, `EVENT_SNAPSHOT_STALE`, `EVENT_EDGE_UNVERIFIED`, `EVENT_TARGET_INCOMPLETE` | 卫星 event sleeve 的资格问题 |

`ROW_QUARANTINED` 只说明该行进入隔离表；`COMPANY_EXCLUDED` 只有在预先写明“公司级规则”并给出所有受影响期间时才可使用。不能把一条字段缺口写成公司永久排除。

## 11. 审计和完成判定

每个阶段至少交付以下检查：

1. raw CSV 与 `.meta.json` SHA-256 一致，plan 请求和实际响应一一对应；失败尝试和重试不覆盖。
2. 输入/输出表列名、主键、日期排序、重复/冲突键、instrument 数和行数可复算。
3. 成员区间无未解释重叠/反转；PIT classification 不引用 current-only 字段；781/782/退市计数差异有书面处置。
4. 每个 factor observation 的 `available_at <= formation_session`，ETF/市场窗口最晚 `< cutoff`；随机抽样逐行检查没有未来日期。
5. R&D/Revenue 同期间、单位/币种明确、Revenue 正值；ETF basket 的组件数、起始日期、权重和 beta 最小观测数达标。
6. 价格调整只应用一次；EPS actual 不再乘 split factor；price/return basis、benchmark 和股息处理明示。
7. 缺失、failed fetch、stale、真实零、退市终止和结构性 padding 分开计数；无静默填补、前值延伸、winsorization 或因退市删行。
8. factor rank、top/bottom 分组、side count、权重和、同股唯一性、换手与容量输入可由 clean 独立复算。
9. RF features 不含 label、未来收益、entry-day EOD、事后 overlap、未来成员身份或 test target；test seal 状态在 manifest 中可验证。
10. H21 目标的入场/退出、缺口和 boundary purge 逐行可复算；event 组和 formation 月不能跨 split。
11. `candidate_registry` 覆盖每个运行；报告候选数量、family、所有结果、覆盖、年份稳定性和校正方法。
12. 独立 validator 输出 `validation.json` 和 `status=passed/blocked`。只有字段语义、PIT、覆盖、执行资格、purge 和 test seal 都通过，才可称为 model-ready；工程表存在不等于基金回测完成。

## 12. Root 在取数前必须决定的项目

1. **历史宇宙**：使用 PIT TRBC Technology + 哪些明确 industry group，还是先使用全部 PIT 成员作宽池；如何处理 corrected 781 行与旧 782/155 摘要的差异。
2. **主因子**：采用 `AI_COMPOSITE`，还是将 `AI_MKT` 或 `AI_FUND` 作为正式主因子；R&D vintage 失败时是否允许预先登记的单组件 fallback。
3. **R&D 可得日期规则**：LSEG 若不提供历史 available date/vintage，是暂停 fundamental 组件，还是批准一个固定、保守且不依赖结果的报告滞后；不得默认把 fperiod 当可得日。
4. **ETF basket**：basket-5、SOXX/XLK basket-2 或 thematic basket 的正式选择；最少有效组件数和 beta 起始期。
5. **收益口径**：AI 因子标签用已验证总回报，还是 v3 adjusted-close price return；股息、公司行动、benchmark 和成本分别怎么进组合核算。
6. **组合约束**：long-short、long-only 或长偏；top/bottom 分位点、每边最小数量、容量阈值、借券和 RF 关闭时的现金/基准资产。
7. **RF 标签和模型**：H21 与 `spread > 0` 是否为唯一主标签；RF 的小型参数表、概率阈值和是否只在 `p_up >= 0.5` 持有主线。
8. **event 卫星**：仓位上限、是否使用共同分析师覆盖图、与主线是否只做独立机制报告；不得让 event 结果选主线因子。
9. **多重比较**：主确认 family、候选数量、Holm/FDR 或全探索性报告的校正方案；test 是否只解封一次。
10. **冻结与记录**：root 批准后保存 `factor_config.json`、`candidate_registry.csv`、配置 hash 和 test-seal 状态，之后的变更只能生成新版本并保留旧结果。

在上述决定和 gate 通过前，本项目可以完成 collector 计划、字段确认、clean 审计和不含目标的 feature/split 结构；不能宣称 AI 因子已可交易、RF 已有效、event 机制已成立或 test 已通过。

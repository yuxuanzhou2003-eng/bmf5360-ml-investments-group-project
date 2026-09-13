# 当前数据集状态：工程面板 v2（工程验证版）

## 2026-09-09 EPS 字段探针完成：发现口径差异

AAPL.OQ/NVDA.OQ字段探针已完成：主探针18次隔离请求17次完成、1次组合字段本地列处理错误；拆开后的5次补充请求全部成功。default actual 与 `ActType=Reported` 在36行完全一致；actual/estimate currency样本均为USD；scale suffix全空且`Scale=0/6`未改变值；fperiod有值；effective/activation未提供可用vintage信息。原始路径为 `data/raw/eps_field_probe_v2/20260908T165319452604Z/` 与 `data/raw/eps_alternative_probe/20260908T165548467804Z/`，审计在 `data/audit/eps_field_semantics/20260908T165833960007Z/`。

NVIDIA 2024样本中，`EPSActValue(ActType=Reported)=0.612`对应发行人non-GAAP 6.12的拆股后尺度；`DilutedEPSExclExtra=EPSNormalizeddil=0.59787`约对应GAAP 5.98。说明`Reported`不能直接解释为发行人GAAP；用不同口径会让该事件z-score由2.5796变为1.9100。现有EPS信号仍未批准用于正式模型，数据未改写。

## 2026-09-09 拆股公司行动原始数据与审计完成

LSEG 公司行动已覆盖当前清单的 782 个 RIC：8/8 批成功，原始 807 行中有 113 个有效拆股事件、涉及 88 个 RIC；694 行为空事件占位，保留且不当成零。113 个事件均满足 adjustment factor≈old shares/new shares，且无重复事件。原始运行在 `data/raw/stock_splits_v2/20260908T163414365646Z/`，审计与独立复核在 `data/audit/stock_split_cases/20260908T163803607846Z/`。

Apple 和 NVIDIA 个案显示当前历史 EPS 数值与后续拆股尺度吻合，但 Tesla 公告日及 NVIDIA 生效/交易日期的对账显示 LSEG 公司行动日期字段不能直接互换。没有把拆股因子应用于 EPS、预期、收益或面板。EPS 会计口径、历史 vintage 和公告时区仍未解除，正式模型输入仍待批准。

全样本只读匹配显示：22,523 个有邻居源事件中，1,336 个（5.93%）公告后至少发生过拆股，涉及68个源 RIC 和 6,347 条边（5.90%）。这是潜在历史尺度影响清单，并非错误行清单；详见 `data/audit/stock_split_exposure/20260908T164324372468Z/`。共同缩放保持 standardized surprise 的算术检查通过，但不能证明 actual/mean/std 同口径或当时 vintage。

## 2026-09-08 覆盖与时间拆分审计完成

`data/audit/coverage_split_inventory/20260908T1128053483021Z/` 已完成独立复核：782 个代码中 17 个 actual、14 个 estimate 缺口全部保留，26 个不完整标签涉及 12 个 receiver，缺失原因未知。训练/验证/测试边行分别为 54,421/19,415/33,696，完整标签 54,411/19,412/33,683，含边事件 11,626/3,961/6,936；跨边界事件 0。8/26 行缺少 entry-day return observation，需价格/执行复核，不代表不可交易。无填补、退市删除、新 LSEG 请求或模型分析。

## 2026-09-08 研究协议与字段语义复核（设计/审阅记录）

`RESEARCH_PROTOCOL.md` 为建模前设计，未执行；训练 2015–2020、验证 2021–2022、测试 2023-01-01 至 2026-06-30，按完整事件边界 purge。建议的经核验 ET 日期后首个收盘入场尚未实施。`FIELD_SEMANTICS_REVIEW.md` 基于既有探针/脚本/官方资料，无新 LSEG 请求或输入转换；timezone/DST、EPS basis/units/split adjustment/vintage unresolved，actual/mean/std 未批准为正式输入。coverage split inventory 随后完成，见本文件顶部完成记录。

最新更新：2026-09-08。**v2 工程事件/邻居面板及 overlap diagnostic 修正版已完成限定范围的独立验证：21/21 checks passed，107,532 条 label/edge 行、40 条 OLS 残差相关独立复算；8/8 修正测试通过。该验证仍不构成 full model-ready。**

当前 v2 面板修正版：`data/panel_v2/20260908T111211716224Z/`；修正审计：`data/audit/panel_v2/20260908T111211716224Z/diagnostic_correction.json` / `.md`；独立验证：`data/audit/panel_v2/20260908T111836060858Z/validation.json`。原始运行 `20260908T053814601256Z` 及其 5 个输出哈希保留，events/features/labels 内容不变。面板为 engineering_only：无模型、无回测、无基金业绩结论。

| v2 工程面板计数 | 数量 |
|---|---:|
| 事件总行 | 29,003 |
| matched snapshot | 22,824 |
| graph-ready | 22,732 |
| history 不足 | 92 |
| graph-ready 零邻居 | 209 |
| 至少1条边 | 22,523 |
| 选定边/feature-label 行 | 107,532 |
| 不完整标签行 | 26（12 个唯一 receiver RIC） |
| 完整标签行 | 107,506 |
| 负相关边 | 10,615 |

规则：历史图窗口为公告日前 126 个交易日且至少 100 个有效配对；边要求 `abs(rho) >= 0.3`，最多 5 条，允许负相关；source/receiver 使用公告日精确成员资格；无数值填补、无仅因退市删除、无身份替换、无日期/时区转换。EPS raw 仅作追溯字段；announcement timezone/vintage 仍未解决。事件匹配保留此前的 22,824 matched 口径；面板未宣称 full model-ready。

Overlap 修正：762 行、11 个 receiver RIC 因缺少可用 actuals 从 `False` 改为 `NA`；修正版计数为 True 37,808、False 68,962、Unknown/NA 762。False 只表示未观察到匹配，不证明没有事件；可用 actuals 也不代表完整覆盖。

后续成员区间修正（最新口径）：按同日净变动处理 EVHC.N^L16 同日加入/退出，移除一条错误的历史成员区间，保留全部原始记录。只改777个成员布尔标记和8个事件资格，收益值不变。最新事件报告 data/audit/event_readiness/20260908T020634146977Z/：总事件29,003，候选22,831，匹配22,824，缺财季4、过旧3；以下22,839/22,832是修正前历史数字。

2015-01-01 至 2026-06-30 公告日窗口有 29,003 条事件，22,839 条符合公告当日成员条件。其中 22,832 条匹配到严格公告前、14 天内的同财季预期；4 条缺财季、3 条快照过旧。44 条 dispersion 为零，事件保留但标准化 surprise 不可计算。22,788 条可计算 surprise 已独立复核。事件级状态与报告在 data/audit/event_readiness/20260908T013521791934Z/。

请求清单中 17 个代码没有有效财报，其中 15 个带退市标记；14 个没有有效预期，其中 13 个带退市标记。已取得事件的高匹配率不能证明这些缺失代码的覆盖完整。

| v2 基础表 | 原始行数 | 清洗保留 | 隔离 | 有效证券代码数 |
|---|---:|---:|---:|---:|
| 收益（含 SPY） | 2,256,958 | 2,081,816 | 175,142 | 783 |
| 财报实际值 | 33,206 | 32,415 | 791 | 765 |
| 分析师预期 | 463,871 | 424,101 | 39,770 | 768 |

84 个预期请求批次已全部成功落盘；其中两个失败批次本轮补回 7,719 行。成功响应仍有字段缺失，已单独隔离。收益和财报清洗输出与上一版逐字节相同。没有数值填补、极值删除或仅因退市删除证券。

完整处理规则、样本损失、旧输出备份、补采重复请求错误及运行记录见 DATA_PROCESSING_LOG.md。当前审计见 data/audit/v2/quality_report.json、validation.json；固定运行副本见 data/audit/v2/runs/20260907T172611428132Z/。工程检查通过不代表公告时区、历史成员重建、EPS 原始披露版本或策略有效性已获证实。

以下保留 **2026-09-07 的 v1 历史记录**，不能与上方 v2 数量混用。v1 的采集—清洗—财季匹配—事前统计网络—未来标签流程曾复跑通过；其结果不代表 v2 面板已完成。

## 数据范围与规模

| 数据层 | 当前范围 | 清洗后的记录 | 状态 |
|---|---|---:|---|
| 股票与基准总回报 | 30 只股票 + SPY，2014-01-02 至 2018-01-12 | 31,496 | 31 个标的各 1,016 个观测交易日，对 SPY 日期无缺口 |
| 季度财报实际值 | 2015–2017 年公告；29 家有数据 | 349 | 1 条 FDX.N 空记录隔离 |
| 季度盈利预期 | 2014-12-05 至 2017-12-29，周频 | 4,669 | 1 条 FDX.N 空记录隔离 |
| 公司基本信息及行业 | 获取时点的 30 家公司快照 | 30 | 当前资料，仅用于识别与描述，不用于历史行业网络 |
| Reuters 新闻版本及公司标签 | 2014 年 12 月，围绕 30 股查询 | 1,131 | 1,301 条跨窗口重复记录去重，7 个时间矛盾版本隔离 |
| 新闻共同标记关系候选 | 同一新闻包含样本中两家公司的 RIC 标签 | 465 条事件，101 对不同公司 | 待语义审核；不能称为供应链或已验证业务关系 |
| 财报—统计邻居面板 | 2015–2017 年事件，各取 3 个残差相关邻居 | 1,047 | 清洗后重建有效记录与旧版一致；仍是统计基线 |

股票分布偏科技、金融、工业与能源，且为手工挑选的存续公司。它不代表完整市场，更不是已解决存活偏差的历史股票池。

## 本轮修复和新增

- 建立原始快照、清洗表、隔离表、匹配面板四层目录。原始行可追溯到文件与行号，下载请求和校验值可查。
- 新闻请求虽然设为 1,000，服务端元数据的 pageLimit 实为 100；已按时间递归细分，并确认未饱和子窗口覆盖整个 2014 年 12 月查询区间。
- 7 个新闻版本出现 version_created 早于 first_created，差距达数小时；未猜测修复时间，已隔离。
- 收益没有按缺失填零；FDX.N 的财报和预期缺失显式保留。NVIDIA 2016-11-11 的约 +29.81% 总回报只列为大幅变动复核项，没有自动删除。
- 特征、未来收益标签、事后同期财报诊断分别输出，防止事后信息混入训练输入。
- 独立检查全部 1,047 个未来五日标签，与清洗后的日收益重算一致；原始文件完整性、行数守恒、唯一键、时间顺序等总计 61 项工程检查通过，其中多项是逐文件校验。

## B、C、D 的关系数据有了什么进展？

新增新闻标签能提供可追溯的关系候选。例如该月原始共同标记统计中出现了 XOM–CVX、BAC–JPM、AAPL–MSFT 等公司对。但共同标记也可能来自市场综述或宽泛提及；本轮未把它们自动加入正式图模型，也未声称存在实质业务联系。

清洗后的关系候选带有新闻 ID 和可用时间，后续可以回到原文审核，并按严格早于信号的滚动窗口构造关系。当前新闻只有一个月，不能直接覆盖三年的事件面板。

## 历史股票池的新进展

LSEG 官方文章的指数增减成员写法返回了 2015–2017 年 205 条记录，包含 Joiner/Leaver 方向，说明这条路径值得继续。但历史初始名单仍未核验：带日期链并附历史价格参数返回 503 个当前成员，其中仅 453 个历史价格非空，仍出现不应属于当年的成员。不能删除历史价格为空的公司后就宣称解决了存活偏差。

205 条变动记录不是完整历史股票池。需要可靠的期初或期末锚点、覆盖完整区间的变动数据，以及退市、更名和重组代码映射，再重建并交叉验证成员集合。

## 正式建模前的四道数据关卡

### 2026-09-09 字段语义状态更新

LSEG Workspace Data Item Browser 已直接确认：`TR.EPSActValue` 是按 I/B/E/S 默认币种及公司行动（明确包括 stock splits）标准化的 actual。证据在 `data/audit/lseg_data_item_browser/20260909T012702710Z/`。这解决了 actual 字段是否包含拆股标准化这一窄问题，也意味着不能再对它手工应用 split factor。

数据仍不是正式模型 ready。DIB 对该字段的定义允许 analyst-specific item inclusion/exclusion，`Reported` 也不能自动等同于发行人 GAAP diluted EPS；actual 与 consensus mean/stddev 的共同口径、公告时区和 point-in-time vintage 仍未通过。因此当前面板和 EPS 探针继续属于工程/语义审计输出，不升级为可投资业绩证据。

1. **股票池与身份时间线**：确定历史可投资公司、退市与更名映射，处理 FDX.N 缺失。
2. **事件时点和口径**：确认公告时区、原始披露版本、EPS 拆股与修订方式；将周频预期升级到公告前最近可得快照。
3. **关系样本的完整性与含义**：新闻扩展到覆盖建图预热期及事件期，检查别名、转载、公司实体相关性；历史行业和供应链资料仍需另取。
4. **回测必需数据**：可执行价格、流动性、交易成本、借券成本与最终样本外划分。当前总回报标签不能替代成交价及投资组合核算。

目前最合理的工作重点仍是数据。图模型应在上述关卡明确后进入；无需为了提前展示收益而降低数据要求。

详细字段与规则见 DATA_DICTIONARY.md。机器可读检查结果位于 data/audit/v1/quality_report.json 和 validation.json。本轮没有模型收益或基金业绩结果。

## 接口依据

- 新闻公司代码与原始元数据：https://community.developers.lseg.com/discussion/133280/initial-query-can-this-function-return-ric-ld-news-get-headlines-fr-tsla-o-and-language-len-and-s/p1
- 历史指数成员重建：https://developers.lseg.com/en/article-catalog/article/building-historical-index-constituents
- 历史行业产品路径需另核验：https://www.lseg.com/content/dam/data-analytics/en_us/documents/brochures/lseg-data-for-quant-research-brochure.pdf
## 2026-09-09 v3 核心清洁数据层完成

v3 raw run `20260909T012417705069Z` 的 explicit-Reported actuals 39/39 批和执行价格/流动性 163/163 批已经成功并通过逐批请求、行数和 SHA-256 核验。清洁输出在 `data/clean/v3/20260909T012417705069Z/`，独立验证为 31/31 checks passed。

当前可用核心表为：32,402 个 valid actual events、423,940 条 weekly point-in-time estimates、2,084,177 条 price/liquidity security-date rows。Actual quarantine 790；price structural-padding quarantine 167,134；EVHC estimate universe exclusions 161。所有缺失和排除均有明细，没有均值/中位数/前值/零填补，没有因退市直接删除证券，没有对 I/B/E/S EPS 再做拆股调整。

AMCR.N 当前 LSEG 已无法重拉带日期 actual；29 个有效历史事件从 v2 原始档案带来源标签继承，正式结果必须附剔除 AMCR 的敏感性分析。2015-2026H1 事件中 20 个无严格事前同财季预期、9 个快照超过 14 天、27 个源事件在保守入场日缺 close；均保留状态，不填补。

**状态：核心数据可进入 v3 面板重建；整条项目尚未 full model-ready。** 尚需完成新入场规则下的关系网络/特征/标签面板、receiver 可执行性与成本规则、边界 purge、模型与组合回测。日频 estimates 目前只有 32,290 行单批 pilot，不是完整核心表。

## 2026-09-09 v3 面板已完成并独立复核

当前正式工程面板为 `data/panel_v3/20260909T035245922341Z/`：28,995 个研究期公告全部进入事件审计，22,824 个事件获得严格事前、精确同财季且不超过 14 天的周度一致预期；22,732 个事件具有足够的历史统计网络，最终形成 107,532 条 source—receiver edge。107,499 条五交易日标签完整，33 条不完整标签保留并显式标记。

输出分为 `features.csv`、`labels.csv`、`execution_inputs.csv` 和 `diagnostics_ex_post.csv`，四表均为 107,532 个唯一且一致的 sample_id。入场日固定为供应商公告日历日之后第一个 SPY session；模型流动性字段取前一 session，入场日 EOD 数据与事后信息不在 features 中。独立验证 `data/audit/panel_v3/20260909T041022558174Z/validation.json` 为 **31/31 passed**，并全量复算未来收益与价格键。

面板没有做任何缺失值填补或无记录删行。入场执行数据共有 15 个不同 sample 存在至少一种缺口：9 条无价格行、12 条缺 close、9 条缺 volume、15 条缺 quote、9 条缺 dollar volume，逐行见同一审计目录的 `entry_missing_rows.csv`。此外，7 条 edge 的 receiver 在公告日属于指数、但在下一交易日入场时已不属于指数；master panel 保留它们并在 diagnostics 标记，正式可交易样本必须按预先固定的 entry membership 和 entry close 规则另建清单。

**状态：数据清洁层和 v3 master panel 已 ready；模型/基金回测输入尚需完成特征冻结、时间分区与边界 purge、执行资格清单及成本模型。** 这一区分防止把“工程数据可用”误写成“策略已经可投资”。

## 2026-09-09 model-ready v1 已完成

无泄漏历史特征、event-level 时间分区、边界 purge 和执行/标签/特征资格已经写入 `data/model_ready_v1/20260909T064151673764Z/`。数据保持 107,532 条 master edge，含 45 个只用入场前信息的候选特征；没有拟合任何 preprocessing，也没有物理删行或填补 NA。

训练/验证/测试分别为 54,421 / 19,415 / 33,696 条。3 个年底事件的 14 条 edge 跨越下一分区边界并整组标记 purge；17 条在入场时不满足成员资格/价格行/close 条件；33 条未来标签不完整；213 条至少缺一个候选特征。四类状态分别保存，监督学习资格共 107,270 条。独立验证位于 `data/audit/model_ready_v1/20260909T064816383676Z/`，31/31 passed。

**状态：可以开始只使用 training 的 preprocessing 与基线模型开发。** 测试集仍不得用于特征选择、缺失处理、超参数选择或模型选择；成本模型、同股同日信号聚合、持仓重叠和组合约束仍未完成，因此当前仍没有基金回测结果。

## 2026-09-09 baseline v1 结果

训练/验证基线位于 `data/model_runs/baseline_v1/20260909T070830647279Z/`，独立验证 `data/audit/baseline_v1/20260909T071048556782Z/` 为 13/13 passed。测试期没有预测或指标。

当前证据偏弱：receiver-only Ridge 在验证期有最高 pooled Spearman 0.0960 和唯一略正的 weighted R² 0.0012；network Ridge 的 mean event Spearman 最高但只有 0.0172，且 weighted R² 为 -0.0007。相对 receiver-only 的事件内排序增量约 0.0013，HGB 更差。内部 2019 年表现多为负，表明关系/目标或状态依赖仍需改进。

**状态：数据和开发基线 ready；策略 edge 尚未通过验证门槛。** 下一步应在 validation 范围内检查按 surprise 强度、相关性符号、行业/流动性和市场状态的条件表现，并比较真正的 lead-lag/关系图；测试期继续封存。


## 2026-09-09 基线后三轮诊断：图假设在当前形式下未通过

baseline v1 之后，在 training 与 validation 范围内完成三轮探索性诊断，全部为负向结果：

1. **horizon 与条件诊断**（正式 run `data/analysis/validation_signal_diagnostics_v1/20260909T073556574431Z/`，4/4 checks passed）：raw `network_signal` 在 1/2/3/5/10 session 上接近零或不稳定。唯一看起来强的是 high surprise + positive correlation 在 10 session 的验证子组（mean event IC 约 0.02845、top-minus-bottom 约 45.7 bp），但同一设定在训练期几乎平坦，属于验证期单侧现象，不作为规则采用。
2. **有向 lead-lag 图诊断**（正式 run `data/analysis/lead_lag_diagnostics_v1/20260909T073416575653Z/`，4/4 checks passed）：lead 与 asymmetry 信号在训练期接近零或为负，只在验证期转正，符号不一致。
3. **next-session-open 目标诊断**（正式 run `data/analysis/next_open_target_diagnostics_v1/20260909T075928064059Z/`，独立验证 `data/audit/next_open_targets_v1/20260909T080525524443Z/` **26/26 passed**）：把理论入场从次日收盘提前到次日开盘，raw 网络信号仍在 ±0.007 内且训练/验证符号不一致，network Ridge 事件内 IC 在 0.0121–0.0219，与收盘基线同量级。

前两项的首次运行（`20260909T072158563895Z`、`20260909T072907041017Z`）缺少 10-session 的 horizon 专属边界清除，已被修正版取代，保留但不得引用；修正后 10-session 各清除 77 行。next-open 诊断因工具超时被执行两次，两个 run 目录输出逐文件 SHA-256 相同，正式版为通过独立验证的那个，另一个保留为证据。

**状态：数据工程层与开发流水线保持 ready 且已独立复核；同期残差相关图作为 alpha 假设在当前形式下没有通过开发期门槛。** 是否转向两阶段设定（事件层预测平均反应 + 事件内预测相对排序）或改换研究问题，需团队决策后才能冻结规格。测试期 2023-01-01 至 2026-06-30 仍完全封存，未计算任何测试期目标、预测、指标或策略表现。

## AI 行业交易主线进展（更新至 2026-09-10）

共同分析师图已完成正式 training/validation 模型与 46/46 独立检查。training-only CV nominee `analyst_hgb` 在 validation 的 mean event Spearman 为 -0.00399，而控制模型 `context_ridge` 为 0.01990；配对差 -0.02389，95% block-bootstrap 区间 [-0.05227, 0.00811]。结论是该图没有稳定增量，降级为探索性 event sleeve；测试集没有预测或评分。

AI 原始字段 v3 探针 `data/raw/ai_fundamentals_pit_probe_v3/20260909T175636988821Z/` 已完成，独立审计 `data/audit/ai_fundamentals_pit_probe_v3/20260909T181610854824Z/` 为 28/28 passed。11/11 请求返回、35 个 raw 文件和 1,395 行在验证前后哈希不变。`ReportingState=Orig` 与默认 current-state 在 369 个配对财务值中有 52 个差异；Original Announcement Date 在 11/54 槽不同。正式财务输入因此固定 Orig + original-announcement 可用时点。

业务分部正确字段已证实可用：269 行、54 个公司—财年组；58 个 null-total candidate、4 组重复 total、14 组 selected total 与 `TR.F.TotRevenue` 不完全一致。以上缺失、重复和对账差异全部原样保留，尚未冻结 AI 关键词词典，也未清洗成因子。Combined monthly market cap 为 648 个唯一 `Instrument + Date` 键，每家公司 108 个月。

正式 AI-factor collector 已生成 357 个独立请求计划：781 个原始 RIC（627 live 标识、154 delisted 标识），11 个股票请求族各 32 批，另有 5 个 ETF 请求。小试 run `data/raw/ai_factor_v1/20260909T181731782199Z/` 已返回前三批 R&D：75/75 个计划 RIC 都有响应，共 501 行，R&D 数值非空 418 行、缺失 83 行，16 个退市 RIC 保留；未清洗、填补、去重或删除。其余 354 个请求待独立小试审计通过后恢复执行。较早计划 `20260909T181534358580Z` 因脚本哈希变化在发请求前中止并保留。

当前状态是“数据字段和正式 collector 小试已就绪，完整 AI 原料仍在收集”；还不能称所有数据 ready，也没有 AI 因子、RF、组合或基金回测结果。测试期 2023-01 至 2026-06 的未来收益、标签、预测和业绩继续封存。

# LSEG EPS 字段语义审查（回溯证据版）

## 2026-09-09 字段探针补充

两股受控探针现已验证：default `TR.EPSActValue` 与 `ActType=Reported` 在36行逐值相同；actual与estimate的`.currency`分别返回36/36、18/18个USD；`.scale`均为空，global `Scale=0/6`未改变返回值；`.fperiod`有值；effective/activation未产生可用vintage证据。Restated/GoForward语法可接受但只返回空值。详情在 `data/audit/eps_field_semantics/20260908T165833960007Z/eps_field_probe_report.md`。

NVIDIA 2024构成新的口径反例：IBES `ActType=Reported` 为0.612，拆股前尺度6.12对应发行人non-GAAP；`DilutedEPSExclExtra`及`EPSNormalizeddil`为0.59787，拆股前约5.9787对应发行人GAAP 5.98。故本文此前“不把Reported命名为发行人GAAP”的限制得到更强支持。currency在该样本得到确认，scale、默认basis及point-in-time vintage仍未得到DIB定义，正式门槛不变。

## 2026-09-09 拆股公司行动补充证据

本项目随后通过本地 LSEG Workspace 对 782 个 RIC 请求 `CAEventType=SSP`，取得113个有效拆股事件；原始与审计路径分别为 `data/raw/stock_splits_v2/20260908T163414365646Z/` 和 `data/audit/stock_split_cases/20260908T163803607846Z/`。全部事件的 adjustment factor 在5e-6容差内等于 old shares/new shares，无重复事件。

LSEG 公司行动指南说明 Capital Change factor 可向后用于 per-share earnings。这与 Apple/NVIDIA 历史 actual 的整倍数案例一致，但仍没有定义 `TR.EPSActValue`、mean、stddev 是否自动调整、是否同一 comparable basis，或当前历史值是否保持当时 vintage。因此本补充没有把 EPS 字段升级为正式模型输入，也没有将因子写回任何研究表。

日期字段仍需分别解释：Tesla 2020 的供应商 CA announcement date 与发行人页面相差一天；NVIDIA 2024 的供应商 ex/effective date 是06-07，而发行人写明06-10开始按拆股后价格交易。公司行动日期不能用于替代尚未核验的 actual announcement timestamp。

## 审查结论

本文件记录对既有只读探针、项目脚本和官方公开资料的语义审查，不执行新的 LSEG API 请求，不修改 `data/raw`、`data/clean`、`panel`、覆盖表、共享日志或状态文件。审查时间为 2026-09-08（Asia/Singapore）。

当前可以采用的最窄结论是：

1. `TR.EPSActValue.announcedate` 在本账户返回一个名为 `Report Date` 的带分钟墙上时钟字符串；本样本未返回 offset 或时区，`origtimezone` 候选也未被 API 接受。因此公告时区、夏令时规则以及它是否严格代表发行人可见的公告时刻仍是 **unknown**。
2. `TR.EPSActValue(ActType=Reported)` 的参数语法在本次小样本可调用，且与默认 actual 恰好相同；这只证明该请求在这三条记录上返回相同值，不能把默认字段或 `Reported` 全历史命名为原始未调整披露。
3. LSEG 官方产品说明明确存在 comparable、restated、go-forward actuals，并说明可按共识口径调整 actuals；它没有公开说明本项目所用 `TR.EPSActValue` 默认值、`ActType` 全部允许值、拆股/单位规则或历史版本选择方式。
4. `TR.EPSMean.calcdate` 是带日期的历史共识记录。LSEG 社区示例支持用带 `calcdate` 的日频历史请求构造 point-in-time 共识，但示例是 `RevenueMean(Period=FY1)`，不能直接证明本项目周频 `EPSMean(Period=FQ1)` 的截止时刻、后续修订和当时可见性。
5. Apple 个案中，发行人 2015-01-27 公告的 2015 财年第一季稀释 EPS 为 `$3.06`，本地 LSEG actual 为 `0.765`，二者严格相差四倍。Apple 投资者关系页列出 2014-06-09 的 7-for-1 与 2020-08-31 开始按拆股调整交易的 4-for-1；由于 2015 公告已经发生在 2014 年拆股之后，四倍关系与后来 2020 年 4-for-1 调整相符。这是“当前供应商视图可能按后续拆股调整”的强个案证据，不是全历史调整算法、可比口径或 point-in-time vintage 的证明。

因此，正式回测仍需把 EPS 数值和标准化 surprise 视为未批准输入；先完成字段级定义、时区/日期确认、单位与调整匹配、以及 vintage 可重现性门槛。

## 证据范围和完整性

### 本地输入

| 证据 | 内容 | 完整性边界 |
|---|---|---|
| `data/raw/semantics_probe/20260908T020426798613Z/` | 一次 session、5 次 `get_data`；RIC 为 `AAPL.O`、`MSFT.O`、`INTC.O`，窗口 2015-01-01 至 2015-03-31，`Frq=FQ`、`Period=FQ0`。脚本 `probe_eps_semantics.py` SHA-256 `0cd25ee708209afbd2bfc9981d0fca7c3a5984ddc55d65b76845defd7c96226e`。 | 成功字段只有三个三行响应；不代表全市场或全历史行为。原始报告 SHA-256 `6c0b91c867e2c16d518224f4daf1503de31830d1cddda59582da951bb6ee89e8`。 |
| `data/raw/missing_coverage_probe/20260908T052439003473Z/` | 缺口身份、FDX/FDXF、历史代码和 DAY 的 7 次只读请求；复用此前 actual/estimate 语义结论。脚本 SHA-256 `0f8f1853d30610abfc7703057f49568f79a20b9e25cfd96f8a45a8a40f576c81`。 | 该探针没有修复或补齐数据；空字段仍只是本次服务端响应为空。原始语义报告 SHA-256 `b0211c2bf5f30bd6b251449a5d6ba8967846a5e485a704235f536022b5af4995`。 |
| `data/raw/universe_v2/actuals_live_000.csv`、`estimates_live_000.csv` | AAPL 2014-12-31 季度实际与 2015-01-23 公告前共识的原始行。 | 这些行未经本审查改写；没有乘四、反拆股、填补或重新匹配。 |

现有缺口探针确认 17 个 clean actuals 缺口与 14 个 clean estimates 缺口的记录保留在 quarantine/审计范围；本审查不将缺失当零，也不因退市或分拆代码替换实体。

### 本地原始数值复核

`base_2015q1.csv` 的三行是：

| Instrument | Report Date | Period End Date | Earnings Per Share - Actual |
|---|---|---|---:|
| `AAPL.O` | `2015-01-27 16:30:00` | `2014-12-31` | `0.765` |
| `MSFT.O` | `2015-01-26 16:10:00` | `2014-12-31` | `0.75` |
| `INTC.O` | `2015-01-15 16:00:00` | `2014-12-31` | `0.74` |

`date_outputs_2015q1.csv` 中三条记录均满足 `Date = Report Date`，而 `Calc Date` 只保留同一天的日期。`reported_2015q1.csv` 中 `ActType=Reported` 与默认 actual、Report Date、Period End Date 逐行相等。对应文件 SHA-256 分别为：

- `base_2015q1.csv`: `3163fb623d442cfdddf14124ae1b6dd3e7039f5d825265589bf210cb1518bac7`
- `date_outputs_2015q1.csv`: `ce3899f78cb76b44d6e0812a5aab2684b1ed656a64e27400de5655dcf7dd3dfc`
- `reported_2015q1.csv`: `c49593a3c6d02a3162d763bc4975eaa2525cfe486dd04efe022c0318f9c28600`

两个失败请求的原始错误也保留：

- `TR.EPSActValue.origtimezone` 与 `TR.EPSActValue.announcedate.origtimezone`：`Unable to collect data for all requested fields`。
- 合并候选 `ActType=Comparable/Restated/GoForward`：在 `Comparable` 处报 `COMPARABLE is unrecognized for parameter ACTTYPE`；因此其余候选没有被独立测试，不能把失败解释为“没有 comparable actuals”。

## 官方资料证据

### LSEG 数据集层面的事实

LSEG 的 [IBES Actuals 产品页](https://www.lseg.com/en/data-catalogue/company-data/ibes-estimates/actuals) 说明：

- actuals 数据覆盖 announced values、changes、surprise windows 和 estimate periods；
- actuals 支持 announcement dates，用于 point-in-time earnings analysis；
- 数据集包含 comparable actuals、restated actuals 和 go-forward actuals；
- reported actuals 可以按多数分析师使用的 operating/comparable basis 对齐，在 reported figure 含 unusual、one-time 或 non-comparable items 时按 consensus basis 调整；
- 数据集描述还提到 effective date、activation date、period advance indicators、normalization flags、currency 和 unit scales。

这些是数据集层面的官方说明。它们证明需要检查调整和版本字段，却没有把上述概念映射到本项目请求的具体 `TR.*` 字段，也没有指定 `TR.EPSActValue` 默认的 `ActType`、单位、币种或时区。

### 时区证据的适用边界

LSEG 社区的 [I/B/E/S timestamps in which time zone?](https://community.developers.lseg.com/discussion/78942/i-b-e-s-timestamps-in-which-time-zone) 给出的解决方式是为 `TR.EPSEstDate` 请求 `TR.EPSEstDate.origtimezone`，并明确讨论该估计日期可能来自 EST、GMT 或 IST。该答复只针对 `TR.EPSEstDate`；本地探针对 `TR.EPSActValue` 及其 `announcedate` 后缀的实际请求失败，所以不能把 `EPSEstDate` 的规则外推给 actual announcement date。

### 参数和 point-in-time 证据的适用边界

- [List of parameters for a given TR field](https://community.developers.lseg.com/discussion/23646/list-of-parameters-for-a-given-tr-field) 的 LSEG 社区答复建议在 Workspace 的 Data Item Browser（DIB）或 Formula Builder 查看某个 TR 字段的参数；当时可用的 metadata API 尚未成熟。这使 DIB 的字段截图/导出和精确公式成为本项目下一步的必要证据。
- [Point in Time Estimates](https://community.developers.lseg.com/discussion/121794) 的 LSEG 社区答复给出 `TR.RevenueMean(Period=FY1).calcdate` 与 `TR.RevenueMean(Period=FY1)`、`Frq=D`、历史 `SDate/EDate` 的 point-in-time 请求模式。它支持把 `calcdate` 当作历史共识序列的检索维度来验证，但不是对 `TR.EPSMean` 的字段字典、每日截点、修订标记或原始披露 vintage 的保证。
- [Query about EPSMean and associated parameters](https://community.developers.lseg.com/discussion/48816/query-about-epsmean-and-associated-parameters) 展示了 `TR.EPSMean(Period=FY1)`、`.PeriodEndDate`、`.CalcDate` 与 `.Date` 的输出组合；版主答复只确认 API 取数成功并建议内容问题联系内容支持，不能替代字段定义。
- [I am pointing out a problem with the LSEG-data API](https://community.developers.lseg.com/discussion/132082/i-am-pointing-out-a-problem-with-the-lseg-data-api) 的代码示例包含 `TR.EPSActValue(ActType=Reported)`。这支持本项目使用该语法作为一个可复现 selector，但该讨论解决的是列顺序/空格导致的 API 输出问题，并没有定义 `Reported` 与默认 actual 的内容差异。

### Apple 发行人证据

- [Apple 2015-01-27 official release](https://www.apple.com/uk/newsroom/2015/01/27Apple-Reports-Record-First-Quarter-Results/) 写明：2015 财年第一季截至 2014-12-27，Apple 当日公布季度结果，净利润约 180 亿美元，稀释每股收益 `$3.06`。
- [Apple investor dividend history](https://investor.apple.com/dividend-history/default.aspx) 的表格列出 2014-06-09 开始按拆股调整交易的 7-for-1、2020-08-31 开始按拆股调整交易的 4-for-1，并说明表中股息金额不做拆股调整。该页面为本个案的直接拆股日期证据。
- [Apple 2020 third-quarter release](https://www.apple.com/newsroom/2020/07/apple-reports-third-quarter-results/) 是同一 2020 年公司行动公告的发行人页面；本审查使用投资者关系历史表中的明确交易日期作为拆股日期引用。

## 语义判定矩阵

| 对象 | 已由证据支持 | 尚未支持、不能宣称 | 当前研究用法 |
|---|---|---|---|
| `TR.EPSActValue.announcedate` / `Report Date` | 返回字符串含日期和分钟；三条样本的列名为 `Report Date`。 | 时区、DST、是否发行人首发时刻、是否供应商报告/处理时刻、跨午夜日期语义。 | 原始字符串保留；暂标 `announcement_timezone=unknown`。不能正式生成 ET 盘前/盘中/盘后标签。 |
| `TR.EPSActValue.date` | 三条样本逐行等于 `announcedate`。 | `.date` 是否只是别名、默认日期输出还是不同经济日期。 | 不能用三条相等样本定义字段含义。 |
| `TR.EPSActValue.calcdate` | 在日期输出请求中返回同一日日期。 | 计算/激活/发布日期定义、时间截点及是否可用于 actual vintage。 | 不作为可见时点证明。 |
| `TR.EPSActValue(ActType=Reported)` | 本账户、本窗口、本三条记录可请求；与默认三行相同。官方社区代码也出现该语法。 | 全历史 selector 语义、默认是否 reported、是否 split-adjusted、是否 restated、允许值清单。 | 只可保存为显式 `reported_selector_requested=true` 的追溯字段。 |
| Comparable / Restated / Go-forward | LSEG 产品页确认数据集层面存在这些内容。 | 本 API 的确切参数值、字段名、默认选择和 activation/effective date。`Comparable` 候选在一次合并请求中失败。 | 在 DIB 定义前不能做 comparable surprise。 |
| `TR.EPSMean` / `TR.EPSStdDev` / `TR.EPSNumIncEstimates` | 本地字段有 `Calc Date`、`Period End Date`、mean、stddev、included estimates；社区展示 point-in-time `calcdate` 请求模式。 | 每日/周频截点、修订历史、当时可见性、币种/单位/拆股、是否与 actual 同一 comparable basis。 | 只能作为工程快照；严格 `< announcement day` 是程序时间规则，不是供应商 vintage 证明。 |
| Apple `3.06` vs `0.765` | `3.06 / 4 = 0.765`；发行人 2015 公告、2014 7-for-1 和 2020 4-for-1 日期均有官方来源。 | 全历史是否统一按 2020 拆股、actual/mean/stddev 是否都同因子、是否存在其他 corporate action、当时 API 可见值。 | 不乘四改写数据；只作为 split-scale 敏感性个案。 |

## Apple 个案的严格解释

本地原始行和官方公告可写成：

```text
issuer 2015-01-27 diluted EPS                  = 3.06
vendor TR.EPSActValue                         = 0.765
3.06 / 4                                      = 0.765 exactly
vendor TR.EPSMean (2015-01-23)                = 0.64767
arithmetic x4 of mean                         = 2.59068
vendor TR.EPSStdDev (2015-01-23)              = 0.02905
arithmetic x4 of standard deviation           = 0.11620
```

最后三项乘法是审查中的算术演示，没有写回原始或 clean 文件。`2.59068` 与 `0.11620` 不能被称为发行人共识或已验证的 LSEG 原始单位；本地没有独立来源证明当时 analysts 的均值、分歧和 actual 使用同一个四倍因子。

2015 年发行人公告已经在 2014 年 7-for-1 之后，所以不能把 `3.06` 与 `0.765` 的四倍差解释成 2014 拆股本身。2020 年 4-for-1 是更匹配的解释。这个个案因此提高了“LSEG 当前历史视图对 EPS 做后续拆股调整”的可信度，但仍有四个逻辑不能跨越：

1. 一个 exact factor 只能证明可能的 per-share scale 变化，不能证明 `ActType=Reported` 是发行时原值、默认 actual 没有 later restatement，或所有公司都采用同样规则。
2. 如果 actual、consensus mean 和 standard deviation 都被同一个正因子乘上，`(actual - mean) / stddev` 在数学上不变；这只能说明共同线性缩放下 z-score 具有尺度不变性。它不能证明三者确实同因子、同币种、同 basic/diluted 定义或同 comparable basis。
3. `actual - mean` 不具有这种尺度不变性；若 raw absolute difference 或模型使用 EPS 水平，必须先确认单位和调整口径。
4. 即使三项同尺度，也不能证明 point-in-time vintage：当前服务端可能返回后来修订/调整后的历史值，而 `Calc Date` 可能只是行的日期索引。拆股调整和发行时可见性是两个独立问题。

## 需要在 Data Item Browser 精确回答的问题

下面的问题是可直接复制到 DIB 字段说明、参数面板或内容审核记录中的核对清单；本任务没有向 LSEG 或任何其他人发送消息。

### Actual 字段

1. 对 `TR.EPSActValue`，请记录 DIB 的完整 field description、measure、basis、currency、unit/scale、basic/diluted/normalized 定义，以及是否默认做 split/corporate-action adjustment。
2. `TR.EPSActValue` 的 `ActType` 允许值完整清单是什么？`Reported`、`Comparable`、`Restated`、`GoForward` 是否为合法值，还是对应不同字段/参数名？请保存每个值的定义和可复现公式。
3. 默认 `TR.EPSActValue` 选择哪一种 actual？它是否会随公司/时期/数据版本改变？默认值是否可能在后续 restatement、corporate action 或 comparable adjustment 后回写？
4. 对 `TR.EPSActValue.announcedate`、`.date`、`.calcdate`、`.periodenddate`，分别说明经济事件、数据激活/计算、财季结束和 display date 的含义；`Report Date` 是发行人公告 timestamp、报告日还是供应商处理日？
5. `announcedate` 每个值的原始时区是什么，是否可通过 actual 专用的 `origtimezone` 或其他字段取得？如果没有字段，LSEG 是否提供固定时区/夏令时规则和跨午夜处理规则？请用一个含时区/offset 的 AAPL actual 示例确认。
6. 产品页提到的 effective date、activation date、period advance、normalization flag、currency、unit scale 在 Workspace DIB 中对应哪些确切 `TR.*` 字段？哪些字段能识别 later restatement 与 original release？

### Consensus 字段

7. 对 `TR.EPSMean`、`TR.EPSStdDev`、`TR.EPSNumIncEstimates`，`Calc Date` 是共识快照观察日、最后计算日、数据激活日还是供应商收盘截点？日期不带时间时，日内截止时间和时区是什么？
8. 用 `Period=FQ1`、`Frq=D` 请求的 EPSMean 是否返回每个历史日当时可见的 consensus，还是以今天的数据库状态重建/修订后的序列？是否有 activation/revision/original-vintage 字段？请提供 AAPL `2014-12-31` 财季在 `2015-01-23` 和若干后续日期的可复现结果。
9. `PeriodEndDate` 是否唯一识别同一财季？对财年变更、53 周财年和 restated fiscal periods，需使用哪些 period/fperiod 字段避免把不同财季合并？
10. `EPSMean`、`EPSStdDev` 和 actual 是否保证同一 currency、per-share scale、basic/diluted定义、split factor 和 comparable/GAAP basis？若不保证，DIB 中的 conversion/normalization/scale 字段和规则是什么？
11. 对 Apple 2014-12-31 季度，能否分别返回 Reported、Comparable、Restated、Go-forward actual（或其实际字段名），并给出每一行的 effective/activation date、scale 和 currency，以解释 `$3.06`、`$0.765` 与共识的关系？

### Vintage 和可重现性

12. 能否在指定历史 as-of 时间读取 actual 和 consensus 的原始 vintage，并区分“当时可见值”与“当前回溯值”？如果可以，请记录准确的 field/parameter 组合；如果不能，正式研究应把 EPS surprise 限制为非 point-in-time 描述性变量。
13. 对同一 observation，如果后来发生拆股、会计重述或 comparable reclassification，API 如何标记旧值、激活时间和新值？请提供一个有已知 corporate action 的回溯例子，而不是只给当前值。

DIB 记录至少应保存：字段名、完整参数和值域、公式文本、截图或导出、检索日期、Workspace/API 版本、测试 RIC、SDate/EDate/Frq/Period、返回列名、以及每个文件的 SHA-256。没有这些字段级记录时，产品页只能作为背景说明，不能作为研究假设的通过证据。

## 正式回测的操作门槛

### Gate A：公告日期和时区

- 必须获得 `announcedate` 的字段定义和时区规则，并用包含夏令时、盘前、盘中、盘后及日期边界的代表性记录复核。保留 vendor 原始字符串和一个显式的 `verified_announcement_ts_et`；不覆盖原字段。
- 未确认时，事件只能进入 engineering/sensitivity 表；不得把字符串按 ET 或 UTC 解析，也不得声称盘前/盘中/盘后可交易。
- ET 验证后，协议中的主入场规则可写成：

  > `verifiedETdate` 是经字段定义与原始时区验证后、按 `America/New_York` 夏令时规则转换得到的公告日。`entry_date` 取该日期之后（严格 `>`）的第一个常规美国股票交易日；在已核验的可执行收盘价上入场。缺失、无时区、无法消歧或跨午夜语义未确认的事件标为 `timezone_unverified`，不进入正式主回测。

- 这是一个预先声明且偏保守的规则：盘前和盘后事件都跳过公告日收盘，避免依赖未核实的日内可交易分类。它要求 `verifiedETdate` 先存在；当前面板的临时 ET 标签和建议规则都没有被本审查重算。
- “第一个常规交易日”需要固定交易所日历/假日规则；“收盘价”需要用可执行价格字段验证，不能把 `TR.TotalReturn` 直接当成交价或把收盘拍卖成交假设隐藏在标签中。

### Gate B：实际、共识和分歧的共同口径

- DIB 必须给出 actual、mean、stddev 的 currency、unit/scale、basic/diluted、split/corporate-action 和 comparable/GAAP/normalized 定义。
- 必须证明实际值和共识均值、标准差在事件样本上使用同一尺度；若只能证明共同乘法缩放，允许做预先声明的 z-score scale sensitivity，但不自动改写 raw EPS，也不解锁 `eps_difference` 或 EPS level 特征。
- Apple 四倍个案不能作为全局 `actual *= 4` 或 `consensus *= 4` 规则。应使用供应商字段/因子；无法确认时，EPS raw 只保留 traceability。

### Gate C：point-in-time vintage

- 必须说明 `Calc Date` 的历史可见性、日内截止时间、修订机制和 `PeriodEndDate` 键；最好以日频历史请求复核事件日前和事件后的同一财季，并保留当前值与历史值差异。
- 必须能区分 actual 的发行/激活日期、后来 restatement 和当前回溯值。只看到一条 `Report Date` 或 `Calc Date` 不足以证明“当时知道”。
- 在 vintage 不能重现时，正式模型不能使用 raw actual、mean、stddev 或 standardized surprise；可以报告明确标注的工程数量和缺失审计。

### Gate D：覆盖、身份和收益

- 17 个 actuals 和 14 个 estimates 的公司级缺口继续保留审计和 reason code；不能以裸 RIC、分拆实体、当前代码或退市状态补齐。
- 通过前述三项后，还需验证总回报字段的单位/复权方式、实际可执行 close/open 价格、公司行动、持仓重叠、借券和交易成本。标签可用于工程核对不等于已完成基金回测。
- 任何通过状态都应绑定字段字典版本、DIB 证据哈希、输入/输出快照哈希和新的数据版本；不能把本审查的“resolved/unresolved”直接写回既有 panel。

## 对 `RESEARCH_PROTOCOL.md` 的审阅意见（未修改该文件）

协议把公告时区、EPS 单位/调整、vintage 和可执行价格列为正式实验前门槛，方向正确。建议保持以下边界：

1. 把“经核验的美东公告日期”明确成字段语义、原始时区、DST 和跨午夜处理均有证据后的 `verifiedETdate`；仅凭返回时钟看起来像 ET 不够。
2. `nextETdayclose` 规则本身适合作为主规则：按 ET 日历日期严格向后取第一个常规交易日，统一跳过同日 close，降低盘前/盘中/盘后分类依赖。它是研究设计选择，不是 LSEG 字段语义的证据；当前没有按该规则重算面板。
3. 把“共同尺度”与“可比口径”分开验收。共同四倍缩放可能保持 standardized surprise 数学值不变，但不能证明 actual 已按共识 basis 调整，也不能证明历史值是当时 vintage。
4. `nextETdayclose` 解锁后仍不能自动解锁回测：入场 close 的可执行价格、`TR.TotalReturn` 的定义、缺失/退市结算、成本和跨分区标签边界仍需单独通过。

## 相对既有报告的新增发现

- Apple 2015 公告已发生在 2014 年 7-for-1 之后；因此 `3.06 → 0.765` 的四倍差更具体地指向后来的 2020 年 4-for-1 视图，而不是 2014 拆股重复解释。
- 当前 LSEG IBES Actuals 产品页明确列出 effective/activation/period-advance/normalization/currency/unit-scale 这些应当进入 DIB 核验的问题；此前报告只把它们概括为数据集存在多种 actual 口径。
- LSEG 社区 point-in-time 示例明确用 `calcdate` 加日频历史请求来取得历史 consensus；这增强了“可验证的检索路径”证据，但由于示例是 RevenueMean FY1，仍不能把它当作 EPSMean FQ1 的字段级保证。
- DIB/Formula Builder 是官方社区给出的字段参数发现路径；因此“准确参数值未知”不是可以用猜测解决的工程问题，必须留下 DIB field-level artifact。

## 当前状态

`FIELD_SEMANTICS_REVIEW.md`：**review_completed_with_unresolved_gates**。

已完成的是证据整合、来源核对、Apple split-scale 个案解释和协议门槛审阅；未完成的是 actual announcement timezone、actual 默认/selector basis、共识与 actual 的共同单位/调整、以及历史 point-in-time vintage 的字段级确认。没有模型拟合、正式回测、数据改写或外部消息发送。

## 2026-09-09 Data Item Browser 直接核验

已在本机 LSEG Workspace 1.26.831 中打开 Data Item Browser，以 `AAPL.OQ` 检查 `TR.EPSActValue`。证据保存在 `data/audit/lseg_data_item_browser/20260909T012702710Z/`，四张截图的 SHA-256 见同目录 `manifest.json`。本次只是读取界面；没有改写任何数据表、模型或回测。

DIB 将该字段定义为按 I/B/E/S 默认币种及公司行动标准化的 actual，且明确以 stock splits 为例。这通过了 `TR.EPSActValue` 当前历史视图的拆股标准化子门槛。因此不能把发行人原始 EPS 与当前 LSEG 值的倍数差直接当数据错误，也不能再对 actual 手工乘拆股因子。

DIB 同时说明，该 EPS 是 contributing analyst 认为适合估值的口径，纳入或排除的项目可随 analyst model 而变。默认 `Actual Type=Reported`；选择器的完整可见值为 `Reported`、`Restated`、`GoForward`、`All`、`Latest`，没有显示 `Comparable`。这支持“Reported 是 I/B/E/S actual selector”，但不支持“Reported 必然等于发行人 GAAP diluted EPS”。

可见默认参数为 `Period=FY0`、`Actual Type=Reported`、`Roll Periods=TRUE`、`Align Type=PrelimDate`、`Consolidation=PrimaryBasis`、`Currency=Default`、`Methodology=InterimSum`、`FX Conversion Rate=PeriodEnd`、`Output=value`；底部表达式为 `TR.EPSActValue(Period=FY0)`。AAPL 描述面板另显示 Calc Date、Date、Period End Date 和 Report Date，但这些当前显示值不能证明历史 point-in-time vintage。

因此 Gate B 只部分更新：actual 的 corporate-action normalization 已有字段级证据；actual 与 mean/stddev 的共同 basis、currency/scale 保证仍未通过。Gate A 的公告时区与 Gate C 的历史 vintage 仍未通过。正式模型继续不得把 raw EPS surprise 当作已核验输入。
## 2026-09-09 已接受的操作口径

- `TR.EPSActValue(ActType=Reported)`：明确用于 v3 actual collection。含义仍是 I/B/E/S 标准化 actual/consensus basis，不等同于发行人 GAAP diluted。DIB 已确认包含 stock split 等公司行动标准化，不再手工拆股复权。
- `TR.EPSMean.calcdate`：作为日期级 point-in-time snapshot。正式匹配要求 Calc Date 严格早于 Report Date 日历日并精确匹配 Period End Date。AAPL 探针显示 FQ1 在公告日滚动，支持该双重约束。
- `TRDPRC_1, OPEN_PRC, HIGH_1, LOW_1, ACVOL_UNS, BID, ASK, TRNOVR_UNS`：通过 get_history 获取，并明确指定 exchange/manual corrections 与 CCH/CRE/RPO/RTS。AAPL 2020 拆股个案显示价格/成交量相反方向调整、turnover 不变；这是一项个案交叉验证，不证明所有证券完整无误。
- `BID/ASK`：日末报价代理，只用于成本/流动性特征和压力情景，不作为保证成交价。
- `TRNOVR_UNS`：首选 dollar-volume 字段；缺失时 `TRDPRC_1 × ACVOL_UNS` 是显式、有来源标签的派生值，不是缺失值填补。
- `Report Date` 时钟：仍无全市场 timezone declaration。正式入场只依赖其供应商日历日之后的首个交易日，消除盘中分类造成的前视风险。

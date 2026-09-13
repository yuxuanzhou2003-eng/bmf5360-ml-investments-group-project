# AI 行业交易：选题定位与 AI 暴露因子可行性（更新至 2026-09-10）

## 项目定位变更

选题从纯粹的 Ripple Alpha（信息扩散）转为 **AI 行业交易**，用户 2026-09-09 决定：**Ripple Alpha 作为该策略的一个组成部分**，不是被替代，也不是并行的独立项目。

策略组合：

| 层 | 内容 | 现状 |
|---|---|---|
| 股票池 | 历史时点可得的 S&P 500 成员池，781 个原始 RIC（含 154 个退市标识） | corrected spans 已审计；不按退市状态删除 |
| AI 暴露因子 | 历史业务分部收入、R&D 强度、控制 SPY 后的 SOXX 敏感度 | 字段探针已通过；全量 raw 收集中 |
| 策略 1 | Factor trading | 主线；待完成清洁、因子构造与开发期验证 |
| 策略 2 | Random Forest 横截面分类与组合择时 | 主线；一行是一只股票×一个形成月，待 model-ready |
| 策略 3 | Event study（财报事件） | 卫星/机制模块；事件基础设施已完成 |
| 策略 4 | Ripple Alpha 信息扩散 | 全市场共同分析师版本验证失败，仅保留为探索性卫星模块 |

Ripple Alpha 可以留作机制研究，因为 AI 产业链存在芯片、服务器、数据中心、云和电力之间的经济传导。但现有全 S&P 500 共同分析师图的正式 validation 结果为负：预先由 training CV 选出的 `analyst_hgb` mean event Spearman 为 -0.0040，低于控制模型 `context_ridge` 的 0.0199，配对差为 -0.0239，95% block-bootstrap 区间为 [-0.0523, 0.0081]。因此它不能作为当前主 alpha，也不能用之后的 AI 子样本结果覆盖这项否证。

## 筛选方式：为什么不能用静态 AI 名单

按今天的认知定 AI 名单再回测 2015 年，等于在 2015 年就知道 2024 年谁会赢。具体反例：PLTR 2020-09 才上市；VRT 2020 年通过 SPAC 上市；NVDA 2015 年是游戏显卡公司，数据中心收入占比很小；SMCI 2015 年是普通服务器厂商。

当前方案是：**历史 S&P 500 成员池 + point-in-time AI 暴露连续变量**。现有 TRBC 只有当前快照，供应商静态 TRBC 字段没有历史时间序列，不能把今天的 Technology 129 只回填到 2015 年。只有取得带有效日期的历史行业分类，行业哑变量才可进入正式控制。

## AI 暴露因子：可行性探测结论

探测脚本 `probe_ai_exposure_v1.py`，输出 `data/raw/ai_exposure_probe_v1/`。12 个候选源逐个独立请求。

### 已通过小样本结构与时点探测

| 源 | 字段 | 时点依据 | 探针结果 |
|---|---|---|---|
| 业务分部收入 | `TR.F.BUSTotRevBizActiv(Period=FY0)` 的 code/name/fperiod/date/value，`ReportingState=Orig` | 原始财报公告日不晚于形成日 | 269 行、54 个公司—财年组；存在 4 组重复 total 和 14 组 total reconciliation 差异，均保留待清洁 |
| R&D 强度 | `TR.ResearchAndDevelopment` + `TR.Revenue`，`ReportingState=Orig` | `TR.ISOriginalAnnouncementDate` 不晚于形成日 | R&D 45/54 keyed slots 有值；Revenue 54/54；缺失保持 NA |
| 财务控制 | Revenue、Assets、Equity、Gross Profit、Operating Cash Flow、Debt，`ReportingState=Orig` | 同上 | 六类控制各覆盖 54/54；Orig 与默认 current-state 在 369 个配对值中有 52 个差异 |
| 市场隐含暴露 | SOXX、XLK、BOTZ、AIQ、IGV 与 SPY 的历史价格 | 所有窗口严格截止形成日前一交易日 | 正式 collector 已配置；主候选为控制 SPY 后的 SOXX residual beta，不再默认用五只 ETF 等权 basket |

这些变量都随时间变化，用来描述“公司何时、通过哪种渠道具有 AI 暴露”，而不是给公司贴永久 AI 标签。`ReportingState=Orig` 探针的原始公告相对期间末滞后为 11–38 天；Last Update 晚得多，只作修订诊断，不能当首次可用日。

### 可用但不可 point-in-time

`TR.BusinessSummary` 返回公司业务描述，但只有当前版本、无历史日期。**不得用于任何回溯期的筛选或特征**，只能作为当前时点的参考。

### 工程上不可行

新闻。`ld.news.get_headlines` 确实带时间戳（index `versionCreated`，datetime64，精确到秒），主题查询也有区分度——`R:WMT.N and "artificial intelligence"` 2018 全年返回 0 条，而 NVDA 同类查询有结果。但：

- 单次查询有 100 条上限且按 newToOld 排序，`R:NVDA.OQ` 查 2018 全年只覆盖 11-16 至 12-27，取全年需分段翻页；
- 多次出现 `ReadTimeout` 与 500 Network Error，服务端不稳定；
- 覆盖 129 只 × 46 个季度、每段还需翻页，按实测速度与超时率估计需数十小时且大概率中途失败。超时来自服务端，并行无法解决。

**结论：新闻不做全量，降级为小样本验证**——抽取若干公司—季度，检查 R&D 强度与 beta 较高时 AI 相关报道是否确实更多，用于验证代理变量的合理性。

### 已纠正的旧结论

旧探针中的五种分部拼写无法解析，但这不代表分部数据不存在。使用正确字段 `TR.F.BUSTotRevBizActiv(Period=FY0)` 后，历史分部数据已经成功返回。旧失败记录继续保留，用来说明字段发现过程；当前不得再写“LSEG 没有分部数据”。`TR.ThemeExposure`、`TR.PatentCount` 和 `TR.CompanyDescription` 的已试拼写仍无法解析，暂不进入主模型。

## 必须写进报告的风险

**循环性**：用价格 beta 定义 AI 暴露，再用它预测价格。缓解措施是严格使用历史窗口、先控制 SPY、将它与分部收入和 R&D 分开报告；它仍是市场隐含代理，不能被解释成经营层面的 AI 收入。

**R&D 和分部覆盖缺口**：缺失可能来自公司不披露、字段不适用或供应商覆盖。缺失不填零，不用行业均值；模型只允许 training-fitted median + missing indicator，并同时报告 complete-case 敏感性。分部重复 total 和对账差异不能静默去重或强制调平。

**控制变量**：RF 必须控制 lagged size、book-to-market、gross profitability/assets、operating cash flow/assets、asset growth、debt/assets、动量、波动、market beta、idiosyncratic volatility、成交额、价差和市场状态。行业只有 PIT 分类得到验证后才加入。由此才能检验 AI 暴露是否提供超越常见规模、价值、质量、动量和风险效应的增量。

**多重比较**：四个策略若都在同一测试集上评估、再挑最好的报告，与此前扫九个信号组合挑一个是同一种错误。测试期打开前必须先定：预先声明主策略，或采用多重比较校正，或明确声明为探索性研究。

## 沿用不变的基础设施

point-in-time 股票池（781 只，含 154 个退市 RIC）、清洗后的约 208 万行日行情、28,995 个财报事件、时间分区（training 2015–2020 / validation 2021–2022 / test 2023-01 至 2026-06）、事件级边界 purge、特征 cutoff 与入场规则。测试期保持封存；当前 collector 只拉特征原料，不读取或生成测试期未来收益、标签、预测或业绩。

# AI 因子清洁层执行计划 v0.1

日期：2026-09-10。本文是对 `data/raw/ai_factor_v1/20260909T183801359195Z/` 的处理计划，先记录规则，再执行清洁脚本。计划不改变 raw 文件。

## 目标和输入

将 LSEG 原始返回转换为可审计的公司—财年、业务分部、ETF 日行情和月度市值表，为之后形成日因子与股票—月 Random Forest 提供输入。股票身份沿用 corrected spans 中的 literal RIC；不根据公司名称猜测替代 RIC，不把不同 RIC 自动合并。

## 明确处理规则

1. **空值和类型**：只在 clean 内把空字符串、供应商空值和无法解析的数值/日期转为 NA，并保留 `raw_file`、`raw_row`、原始文本和 `reason_code`。文本 `0` 是观测值，不改成 NA；NA 不改成 0。
2. **财务期间对齐**：R&D、Revenue、Assets、Equity、Gross Profit、Operating Cash Flow、Debt 按 literal `Instrument + parsed period_end` 对齐；不能按返回行位置拼接。`fperiod` 只是期间标签。相同键多行保留，并分别标记 `DUPLICATE_KEY_IDENTICAL` 或 `DUPLICATE_KEY_CONFLICT`。
3. **原始口径与可用时点**：财报值只接受 `ReportingState=Orig` 的 raw family。`TR.ISOriginalAnnouncementDate` 是拟用可用时点；只有 `original_announcement <= formation_session` 才能进入该形成月。`Period End Date` 不能代替公告日，`Last Update` 只作修订诊断。公告日缺失或期间键不一致标记 `PIT_DATE_UNAVAILABLE` / `PERIOD_KEY_MISMATCH`。
4. **单位、币种和比率**：在单位、币种、scale 通过语义 gate 前不计算比率。通过后计算 `rd_intensity = R&D / Revenue`，仅当同期间、Revenue>0 且两值有限；否则保留 NA 和 `RATIO_INVALID` 原因。不会将不同币种静默换算。
5. **业务分部**：保留每个原始 segment row，包括 blank code/name、All Other、eliminations、null-total candidate 和重复行。使用预先登记的 core/broad 关键词词典生成分类，不使用收益表现反推关键词。输出 `segment_classification_version`、命中词、`ai_class`、原始收入和 raw provenance。每个公司—期间选择 total 时，若有多个 null-total 标记 `SEGMENT_TOTAL_AMBIGUOUS`；与总收入不一致标记 `SEGMENT_RECONCILIATION_GAP`，不强制调平。
6. **市值**：按 `Instrument + Date` 显式键读取月度市值；不能按行位置依赖 value/date。市值请求是市场数据，计划中明确不传 `ReportingState=Orig`。单位/币种若未证实，则保留原值并标记 `MARKET_CAP_SEMANTICS_UNVERIFIED`，不计算规模比率。
7. **ETF 和收益**：ETF 价格、OHLC、成交量、bid/ask、turnover 保留供应商调整字段。只在日期可解析、价格为正且前一交易日存在时计算 `price_return = P_t/P_{t-1}-1`，不称为 total return；上市前不回填。SOXX、XLK、BOTZ、AIQ、IGV 与 SPY 独立保留，缺口标记 `ETF_NO_OBSERVATION`。
8. **退市和成员资格**：退市 RIC 仍保留全部可用财务、分部和行情行。形成日成员资格由 corrected spans 计算；`member_to` 后的缺失行情标记终止状态，不删除公司。只有明确的 row-level 结构性 padding 或冲突才进入 quarantine，并保留恢复条件。
9. **不在清洁层做的事**：不填补、不前向填充、不 winsorize、不按当前 TRBC 回填历史、不生成未来收益/标签、不训练模型、不选择组合或参数。

## 计划输出

- `fundamental_observations.csv`：原始财务字段、期间、原始公告日、可用性 gate、R&D ratio、单位/币种与原因码。
- `segment_observations.csv`：原始分部行、词典分类、total candidate、对账差异与 provenance。
- `etf_prices.csv` 与 `asset_returns.csv`：原始 ETF 行及明确的 price-return 口径。
- `market_cap_monthly.csv`：显式键的月度市值及语义状态。
- `ai_market_proxies.csv`：只保存形成日以前的 ETF/SPY 代理输入，不保存标签。
- `membership_snapshots.csv`、`identity_map.csv`、`quarantine.csv`、`missingness_summary.csv`、`clean_summary.json`、`gate_report.json`。

每次 run 使用新 UTC 目录，记录输入/输出哈希、前后行数和证券数、受影响行、reason code 计数、quarantine 路径、检查、限制和执行状态。清洁前不宣称模型数据 ready；完成独立验证后才进入 model-ready。

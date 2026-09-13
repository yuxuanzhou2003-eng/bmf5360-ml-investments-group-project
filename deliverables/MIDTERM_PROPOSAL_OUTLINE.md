# 期中 Proposal 写作提纲（草案，待用户确认）

日期 2026-09-11。课程要求原文为 "Midterm Report & Presentation (10%): idea generation, data preparation, and early-stage analysis"。本提纲把交接包 `BMF5360_Baseline_Package` 里的材料按这三项重新编排，所有数字均来自包内文件，写作时逐条回查。

## 2026-09-11 已定决策与交付状态

### 第四轮（2026-09-12）：单实验版 v4，当前稿

用户决定只写一个实验，选数字更好的股票日频面板（Logistic 0.5799），事件面板降为第 1 节选题过程里的一句话；要求"能解释的解释，不能解释的避开"，图文并茂。当前稿 `Midterm_Report_GroupX_v4.docx/.pdf`，正文恰好 5 页（第 2 到 6 页），参考文献在第 7 页，AI 声明在封面底部。

- 标题 "From Factor Signals to Investable Strategies: A Machine-Learning Stock-Selection Pilot in US Large Caps"。
- 三张图：Fig 1 流程图（单面板版）；Fig 2 主模型预测概率五分位的 21 日超额收益（pooled，Q1 -2.43% 到 Q5 +1.90%，来自 `validation_predictions.csv`）；Fig 3 验证期 NAV vs SPY，阴影标出 4 个跳过调仓、持仓延长的窗口。
- 风险处理：(1) 40/49 无 PIT 证据写成"已有 9 家、其余 40 家待扩展，第 5 节第一步"；(2) 择时 vs 选股写成"信号分两层"的设计说明，给日均 rank IC 0.089 和同日 top-bottom 0.74%；(3) 8 模型选最大值写成"预注册的嵌套因子集阶梯，全部报告"；(4) 组合亏损做了归因：执行调仓后的 21 日窗口复利 +14.6%，4 个跳过调仓的延长窗口 -15.4%（跳过日模型最高概率 0.43–0.59），说明规则忽略了模型自己的信号，v2 规则改为跳过日空仓/仅对冲 + 按同日排名选前 20%；(5) 两年 regime、宏观非 vintage 不提。
- 归因数据来自 `data/backtests/ai_portfolio_v1/20260910T133000000000Z/nav_daily.csv` 与 `portfolio_targets.csv`，计算见会话记录；这是验证期事后归因，v2 规则只能预注册不能重跑验证期。
- 旧脚本保留在 scratchpad：`build_midterm_report_v3_two_pilots.py`。


### 第二轮修订（2026-09-11 晚）：淡化 AI，改为因子框架

用户反馈：AI 不应是重点，开头泛泛写策略，主方法是 factor analysis（含 earnings event analysis 和自建量化因子），用 ML 实现。据此重写为 v2：

- 开头按用户反馈（"太 straight forward"）加了一段动机：横截面预测是核心问题、单信号弱且不稳、传统因子模型线性固定权重、ML 可组合更多信号但易过拟合（引 Gu-Kelly-Xiu 2020）、因此评估协议与模型同等重要。
- 第三轮（用户："data 部分也有点 straightforward，开头不要名词开头；要不要可视化"）：2、2.2、3、4 节各加一句引导句；新增 Figure 1 研究流程图（宇宙→因子库→两个面板→模型→组合规则 + 时间轴，`_artifact_work/figures/fig1_pipeline.png`）和 Figure 2 验证期 NAV vs SPY（数据来自 `data/backtests/ai_portfolio_v1/20260910T133000000000Z/nav_daily.csv`），Figure 2 取代原 Table 4，指标移到图注；删掉 Reproducibility 段并压缩 2.3 与第 5 节以控制页数。文件 `Midterm_Report_GroupX_v3.docx/.pdf`（2026-09-12 凌晨已重转，正文恰好 5 页，底部余量约 5 行）。2.2 节事件面板段按用户"这段是想说啥"的反馈重写成"一行是什么、怎么定义关联、surprise 怎么算、入场与标签、规模"的顺序。为控页数：AI 工具声明移到封面底部，Table 1 去掉 Status 列并入第三列，Table 3 去掉全因子 RF 行（图注注明），段后距 6 到 5 pt、标题前距缩小。绘图脚本 `_artifact_work/build_figures.py`。
- 标题改为 "From Factor Signals to Investable Strategies: A Machine-Learning Equity Research Framework with Two Early-Stage Pilots"。
- 第 1 节先写策略（PIT S&P 500 池、21 日相对 SPY 排序、beta 对冲），再写三个因子族（市场因子 / 财报事件因子 / 基本面与主题暴露因子，Table 1），AI 产业链只作为其中一个试点子池。
- 第 4 节新增事件因子结果（Table 2，来自 `data/model_runs/baseline_v1/20260909T070830647279Z/validation_metrics.csv`）：receiver-only Ridge pooled Spearman 0.096、within-event 0.0159；network Ridge 0.0172；HGB -0.0017；共同分析师图 -0.0040 vs 0.0199。结论是事件传导不能单独成 alpha，surprise 留作股票层候选因子。
- 股票选择结果与验证期组合结果保留（Table 3、Table 4），去掉 49 家分组表和六准则表（准则写成一句）。
- 文件：`Midterm_Report_GroupX_v2.docx/.pdf`。v1（AI 试点版）文件因用户在 Word 中打开未覆盖，保留为沿革；脚本旧版在 scratchpad `build_midterm_report_v1_ai_pilot.py`。


- 定位采用 GPT 交接文档的写法：可复用研究框架，AI 产业链是第一个已完成的试点。标题 "From ML Signals to Investable Strategies: An AI Supply-Chain Pilot Study"。用户原话："我还是倾向于 gpt 的定位，反正只是一个探索"。
- 课程格式要求：正文 3–5 页（封面、目录、参考文献不计），Times New Roman 11 pt，1.2 倍行距，四边 1 英寸，PDF 提交，文件名 `Midterm_Report_Group#`，截止 Week 5 周六 23:59。
- 初稿已生成：`Midterm_Report_GroupX.docx` / `.pdf`（生成脚本 `_artifact_work/build_midterm_report.py`）。正文约 4.6 页，A4，封面上组号与成员为占位符，交前需替换并改文件名。
- 主模型口径为 technical + AI-state Logistic（0.5799），technical + macro（0.5616）作为"择时 vs 选股"对照；验证期组合负结果如实写入 4.3 节。

## 总体定位（原建议，已被用户否决，保留作沿革）

原建议以基金为主体写。用户选择 GPT 的框架定位，理由是期中只是探索阶段，终稿策略尚未选定。

主模型口径统一为 technical + AI-state Logistic（验证期 pooled AUC 0.5799），因为验证期组合就是用这个模型跑的。technical + macro Logistic（0.5616）作为对照出现在"择时 vs 选股"的教训里。

验证期组合的负结果（累计 -3.10%）如实写进第 3 节，作为 early-stage analysis 的主要发现，并直接推出第 4 节的下一步。

## 结构与证据来源

### 0. 摘要（半页）

基金目标、股票池、持有期、ML 的角色、当前进度四句话。

### 1. 投资想法与 ML 的必要性（idea generation）

| 内容 | 证据文件 |
|---|---|
| 选题漏斗：10 个候选选题按 LSEG 数据可得性筛选（2026-09-07）→ 财报信息扩散在全 S&P 500 上检验并否证（analyst_hgb 事件 Spearman -0.0040 vs ridge 0.0199，配对差 95% 区间 [-0.0523, 0.0081]）→ 收敛到 AI 产业链 | `LSEG_DATA_FEASIBILITY.md`、`AI_SECTOR_FEASIBILITY.md` |
| 基金目标：在美股 AI 产业链 7 个环节 49 家公司内，预测未来 21 个交易日相对 SPY 的超额收益方向，做多高分股并用 SPY 对冲 beta，波动目标 10% | `AI_PORTFOLIO_BACKTEST_SPEC_v1.md` |
| 经济机制：GPU → 存储/HBM → 服务器/网络 → 云/软件 → 数据中心电力的产业链传导；AI 资本开支周期下环节间收益离散度高 | `AI_SUPPLY_CHAIN_TAXONOMY_v2.md` |
| 为什么用 ML：技术、流动性、宏观、行业状态共 60 余个弱信号，二分类是自然的监督任务；Logistic 做透明基准，RF 做非线性对照 | `BASELINE_LOGISTIC_MODEL_CARD.md` |
| 为什么不能用今天的 AI 名单回测 2015 年（PLTR 2020-09 上市、VRT 2020 SPAC、NVDA 2015 是游戏显卡公司） | `AI_SECTOR_FEASIBILITY.md` |

### 2. 数据准备（data preparation）

| 内容 | 证据文件 |
|---|---|
| 数据源：LSEG 781 只 PIT S&P 500 RIC（含 154 退市）、约 208 万行清洗日行情、28,995 个财报事件；FRED 7 个宏观序列 18,345 行；SOXX/XLK/AIQ/BOTZ/IGV ETF 价格 | `DATASET_STATUS.md`、`AI_DAILY_STATE_MODEL_RESULTS_v1.md` |
| 股票池：7 组 49 RIC，选择不用收益/模型表现，退市保留（JNPR.N^G25），9 家有本地 PIT 分部证据、40 家为暂定静态证据 | `support/report_content.md` §1–2 |
| 特征：21 技术/流动性/市场 + 20 宏观 + 21 AI 行业状态，全部截止形成日前一交易日；6 个信用利差变量因覆盖从 2023-09-11 起而排除 | `support/report_content.md` §3 |
| 标签：y = 1 若个股 21 日总收益 > SPY 21 日总收益 | 模型卡 |
| 切分：训练 2015–2020（2,334 锚点）/ 验证 2021–2022（992 锚点）/ 测试 2023-01 至 2026-06 封存未开；H21 非重叠锚点；RF 调参用训练期内 2018/2019/2020 purged expanding folds | 模型卡、README |
| 已审批的数据处理规则：末日填充隔离、多 RIC 不合并、盘前/盘后入场时点、极端收益只复核不删；缺失只在模型内做训练期中位数 + 指示变量 | `DATA_PROCESSING_LOG.md`、memory |

### 3. 早期分析（early-stage analysis）

| 内容 | 证据文件 |
|---|---|
| 8 行结果表（4 特征组 × Logistic/RF）；主模型 technical + AI-state Logistic AUC 0.5799，相对技术 Logistic 增量 0.0405，按形成日 bootstrap 95% 区间 [0.0007, 0.0801]；2021 年 0.5906、2022 年 0.6213 | `AI_DAILY_STATE_MODEL_RESULTS_v1.md` |
| 择时 vs 选股分解：41 个变量中 27 个在同一形成日内为常数；宏观相对技术的同日 pair-weighted AUC 增量 0.00624，区间跨零；pooled AUC 不等于选股能力 | `audits/timing_selection_validation/summary.json`、docx 报告 §5 |
| 验证期组合（预注册规则）：累计 -3.10%，年化 -1.63%，Sharpe -0.119，最大回撤 -13.15%，beta 0.064，年化换手 4.54 倍，价差成本 0.24%；同期 SPY +13.11% | `AI_DAILY_STATE_MODEL_RESULTS_v1.md` |
| 结论：排序改善没有在当前组合规则下转成收益，成本不能解释亏损；Logistic 优于 RF，复杂度未带来价值 | 同上 |

### 4. 风险、局限与下一步

| 内容 | 证据文件 |
|---|---|
| 局限：40/49 缺历史 PIT 成员证据、幸存者偏差、宏观非 vintage、样本按形成日聚类、2022 年 ECE 0.1247、测试期未开 | `REPORT_POSITIONING_AND_NEXT_STAGE.md` |
| 下一步：对负收益做归因（多头/对冲/跳过日/环节）；v2 预注册组合规则（每月固定日调仓、前 20% 等权、单边 0/5/10/25 bps 成本敏感性）；用六条准则选定终稿策略；一次性打开测试期 | docx 报告 §6、positioning 文档准则表 |
| 分工与时间表 | 待用户提供 |

### 附录

包内容与哈希（`PACKAGE_CONTENTS.md`）、AI 使用声明（`AI_USE_LOG.md`）。

## 用词边界

可以写 "leakage-controlled, reproducible prototype; possible mixed timing/selection signal"。不写 "proves stable alpha"、"macro improves stock selection"、"ready for investment"。0.5799 是分类排序指标，不是收益或胜率。

## 待用户确认

1. 定位按基金主体（本提纲）还是按 GPT 的框架主体。
2. 期中只写 AI 产业链选股一条主线，信息扩散作为"已检验并否证、保留为卫星模块"一句带过；event study 与 factor trading 不在期中单独成节。
3. 主模型口径用 technical + AI-state Logistic（0.5799），technical + macro（0.5616）降为对照。
4. 验证期组合负结果写进期中。
5. 格式约束：页数、幻灯片页数、截止日期、语言（默认英文）、组员与分工。

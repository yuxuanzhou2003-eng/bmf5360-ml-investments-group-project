# 数据字典与清洗规则（v1）

本数据集是工程试验版。币种、单位、时点和来源须与实际字段定义一并审核；通过工程检查不代表已经满足正式投资回测要求。

| 表 | 主键 | 主要字段及含义 | 时间与使用限制 |
|---|---|---|---|
| returns | Instrument + Date | Total Return：供应商原始数值；return_decimal：原始值除以 100；raw_file/raw_row：原始位置 | 日频总回报；复权与分红方法仍需对账；不能当成成交价 |
| actuals | Instrument + Report Date + Period End Date | Earnings Per Share - Actual：季度实际 EPS | 公告时间无已核验时区，保留原样；不是原始披露版本保证 |
| estimates | Instrument + Calc Date + Period End Date | Mean：同财季预期均值；Standard Deviation：预测分歧；Number of Included Estimates：覆盖数量 | 周频历史快照；仅匹配公告日前、同财季记录；不以年度 FY1 替代季度 FQ1 |
| company_master_current_only | Instrument | RIC、公司名称、Organization PermID、ISIN、当前 TRBC 行业 | 记录实际获取时间，historical_use_allowed=false。不可将当前行业回填历史 |
| news_versions | story_id + version_created | headline、first_created、version_created、rics_json、family_id、availability_utc | UTC；用版本时间作为可用时点；首次发表时间不能代表更新版本已知时间 |
| news_cotag_candidates | left + right + family_id | 两公司在同一新闻的 RIC 标签；availability_utc 为首次观测到该配对的版本时间 | 公司共标签不代表业务关系或全文中实质讨论；所有边待审核 |
| features | sample_id | source、receiver、公告与财季、事前预期、实际 EPS、惊喜、事前残差相关性 | 仅工程样本；EPS 调整口径、时区、股票池尚未通过研究门槛 |
| labels | sample_id | entry_close、exit_close、五日收益、基准收益及二者之差 | 未来标签；不能作为模型输入；不是扣费后的策略收益 |
| diagnostics_ex_post | sample_id | 接收公司同期财报重叠、其事件覆盖是否可用 | 含事后信息，只能做诊断，不能直接用于预测或事前交易过滤 |

## 固定清洗规则

1. 原始数据保留并计算 SHA-256；原始 pilot_v1 文件改变时停止并要求建立新版本，不能覆盖。
2. 空白统一为缺失，日期显式解析，数值转换失败与非有限数值按缺失处理；毫秒和非毫秒时间均显式支持。
3. 必需字段缺失的行进入 quarantine，记录原因和原始位置。不能把缺失收益、缺失事件或缺失预期填成零。
4. 完全相同的记录保留一份；同一业务主键有冲突时隔离冲突记录，不任意选择最后一条。
5. 预期标准差和覆盖数量不能为负；标准差为零时不计算标准化惊喜，不用微小常数制造极端值。
6. 新闻按版本去重；时间先后矛盾的新闻版本隔离。保留标题及公司标签；转载、RPT、内容相似但 story_id 不同的报道仍需语义去重。
7. 行情汇总、列表式报道等只标记为审核候选，不据简单正则判断其无效。BRIEF 也可能含有重要公司消息。
8. 单日绝对收益超过 20% 仅列入复核清单，不自动删除或缩尾。这是诊断阈值，未按收益优化。
9. 网络与预期只用事件前数据。未来收益与事后重叠诊断拆表；使用稳定 sample_id 连接。
10. 新闻窗口达到服务端页上限时细分请求；核对未饱和子窗口能否覆盖目标区间。该检查不能证明供应商之外不存在遗漏新闻。

## 路径和复现

- 原始快照：data/raw/pilot_v1/；补充数据：data/raw/supplement_v1/。
- 清洗表：data/clean/v1/；隔离行、逐公司覆盖、质量检查：data/audit/v1/。
- 匹配面板及拆分的特征/标签：data/diffusion_pilot_clean/。
- 本地复跑：使用 .venv/Scripts/python.exe 运行 run_data_pipeline.py。
- 补充取数再清洗：同上并加 --collect。已缓存且校验一致的补充请求不会重复下载。
- 初始 30 股原始取数由 probe_lseg.py 的 diffusion_pilot 模式生成；如重新取数产生不同结果，应另建新数据版本。
- 网络访问需要本地 Workspace 登录及 .env。代码和文档不含凭证，原始数据目录不进入 Git。

正式训练前需要专门定义 point-in-time 股票池、历史关系、数据发布时区、成本，以及独立最终测试期。当前 2015–2017 年数据已用于开发。

## v3 面板字段补充（2026-09-09）

正式工程面板路径为 `data/panel_v3/20260909T035245922341Z/`，以下定义覆盖本节之前的 v1 面板说明。

| 表 | 主键 | 主要字段及含义 | 使用限制 |
|---|---|---|---|
| events | 每个 source + announcement + period_end 事件 | 全部 28,995 个研究期 actual；status、graph_status、neighbor_count 记录每一步去向 | 是完整事件处理审计；没有生成 edge 的事件也保留 |
| features | sample_id | source、receiver、announcement_day、精确财季 actual/consensus、surprise、历史 residual correlation；`lagged_*` 为 entry 前一 session 的价格/流动性及来源 | 只允许白名单字段进入模型；`actual_source` 用于 AMCR 敏感性筛选，不当作经济预测变量 |
| labels | sample_id | entry_session、exit_session、五 session receiver/SPY total return、差值与 label_complete | entry/exit 是交易日日期；未来收益不能进入特征；缺失标签不填零 |
| execution_inputs | sample_id | entry session 的 adjusted close、volume、quoted spread、dollar volume、来源、复权规则与可用性 flags | 是日终执行/成本诊断，不是同日收盘交易前可知特征；receiver 通过 sample_id 与 features 连接 |
| diagnostics_ex_post | sample_id | receiver 公告日/入场日成员资格、财报覆盖及持有期自身公告重叠 | 含事后或资格诊断，不进入模型；正式可交易清单须另行冻结 |

`entry_price_row_available=False` 表示 clean price 表没有精确 `(receiver, entry_session)` 行；此时 price、volume、spread、dollar-volume 和来源保持 NA。`entry_has_close`、`entry_has_volume`、`entry_has_two_sided_quote` 是供应商字段非空标记；当前没有单独存 `entry_has_dollar_volume`，应按 `entry_dollar_volume.notna()` 派生。不能把 NA 来源改写为供应商值。

`dollar_volume_source=TRNOVR_UNS` 表示使用供应商 turnover；`TRDPRC_1_times_ACVOL_UNS` 表示 clean 层按同一行的 adjusted close × adjusted volume 计算并显式标记。两者不可混写。`price_adjustments` 固定记录 `exchangeCorrection,manualCorrection,CCH,CRE,RPO,RTS`；不得再对 clean price 或 I/B/E/S EPS 手工乘拆股因子。

## model-ready v1 表（2026-09-09）

| 表 | 主键 | 内容 | 限制 |
|---|---|---|---|
| model_features | sample_id | 45 个公告/入场前候选特征：surprise/network、source/receiver return dynamics、market regime、历史流动性与横截面 rank | 不含 entry-day EOD、未来收益、执行资格或事后诊断；尚未缩放、缩尾或填补 |
| metadata | sample_id | source/receiver、事件/财季/快照、actual provenance、原始 EPS 追踪字段、reference/entry/exit session | 用于追踪与分组；原始 actual/consensus/dispersion 不在首轮 ML 白名单 |
| targets | sample_id | 与 panel v3 相同的未来五 session receiver/SPY return、excess 和 label_complete | 只用于训练损失/评价，不用于当时交易资格 |
| eligibility | sample_id | split、boundary purge、entry membership/price/close、capacity/spread、label availability、feature availability、监督样本资格及逐行原因 | flags 不删除 master row；不同模型可有不同 feature availability，但必须另存规则 |
| feature_missingness | split + feature | 每个特征在 training/validation/test/all 的 NA 数与比例 | 是描述性审计；不能用 test 缺失分布拟合处理参数 |
| split_counts | split | event/edge、purge、label、entry、core feature 与 supervised eligibility 数量 | 同一 `(source, announcement, period_end)` 事件不可跨分区 |

收益特征按 total-return 日序列计算：momentum 为窗口内 `prod(1+r)-1`；volatility 为日收益样本标准差乘 `sqrt(252)`；beta 为至少 100 个成对观测的 126-session OLS market beta；idiosyncratic volatility 为同一回归残差标准误乘 `sqrt(252)`。流动性中位数只在最后 20 个 SPY session 内按实际非空观测计算，不跨日期填充。

## next-open 诊断持久化字段（2026-09-09，探索性）

字段位于 `data/analysis/next_open_target_diagnostics_v1/20260909T075928064059Z/open_targets_and_availability.csv`，仅供开发期诊断使用，不属于正式建模表，不得作为特征输入。

- `receiver_entry_open`、`benchmark_entry_open`：receiver 与 SPY 在 entry session 的 `OPEN_PRC`，取自 clean v3 prices 的精确 `(Instrument, Date)` 键，缺失保持 NA。
- `entry_open_available`：上述两个开盘价同时存在且为正。
- `exit_session_{h}`：entry session 之后第 h 个 SPY 交易日（h = 0 时即 entry session 本身）；日历末端不足时为空。
- `target_open_to_close_{h}`：receiver 从 entry session 开盘到 `exit_session_{h}` 收盘的价格收益，减 SPY 同区间价格收益。**价格收益，不是股息全收益**；仅在 `target_available_{h}` 为真时非空。
- `boundary_purged_{h}`：该 horizon 的退出日进入下一分区（training 的 ≥ 2021-01-01，validation 的 ≥ 2023-01-01），目标被屏蔽。
- `target_available_{h}`：退出日存在、未跨分区、入场开盘价与退出收盘价均可用且为正。与 `target_open_to_close_{h}` 的非空完全等价，已由独立验证核对。

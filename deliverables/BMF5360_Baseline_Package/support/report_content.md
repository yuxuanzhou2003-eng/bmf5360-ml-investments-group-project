# 基础模型交付材料：报告内容草稿

> 回溯整理日期：2026-09-10。本文只从项目内冻结规范、候选注册表、已执行的 validation-only 模型 run 及其可读结果说明中提取内容，供基础模型报告编排使用。它不打开 sealed test targets，不生成 test 指标，也不把设计候选自动解释为完整历史 PIT 成员。

## 0. 口径与证据层级

本稿的数值主线是日频 `technical_plus_macro / Logistic` run `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/`。`AI_SUPPLY_CHAIN_TAXONOMY_v2.0` 和 49-RIC candidate registry 是冻结的供应链设计候选层；9 个 `direct_local_pit` 与 40 个 `provisional_static_source` 的证据状态必须与模型结果分开表述。模型结果来自已经执行的训练/验证 run；验证期以外的测试目标仍封存。早期通用协议 `AI_BASELINE_MODEL_SPEC.md` 可作为研究协议参考，但本稿的 0.561559 来自上述日频 run，采用该 run 的每个 SPY session 决策日与 H21 非重叠锚点口径。

## 1. 49 个 RIC 的选择逻辑与七组 taxonomy

候选池从当前 781-RIC universe 中按 literal RIC 选择供应链角色候选。选择不使用历史收益、波动、规模、流动性、模型预测或 ETF 表现；不以公司名称、ticker 猜测替换或合并 RIC。未选的 732 行留在 `universe_screening.csv`，不是物理删除；候选清单记录为 49 行、49 个 unique literal RIC，物理删除为 0、quarantine 为 0，旧 v1 member overlap 为 9。退市状态与可交易覆盖期分开记录，退市本身不是排除条件。

冻结的七组及候选如下。注册表按 primary group 计数；同一公司如有 secondary group，不把它解释为额外的投资权重。

| 供应链组 | 数量 | 49 个 literal RIC |
|---|---:|---|
| GPU / AI 加速器 (`gpu_accelerator`) | 5 | `NVDA.OQ`, `AMD.OQ`, `AVGO.OQ`, `INTC.OQ`, `MRVL.OQ` |
| AI 芯片及配套半导体 (`ai_semiconductor`) | 6 | `QCOM.OQ`, `TXN.OQ`, `ADI.OQ`, `NXPI.OQ`, `ON.OQ`, `MPWR.OQ` |
| 存储 / HBM / 数据中心内存 (`memory_hbm_storage`) | 5 | `MU.OQ`, `WDC.OQ`, `STX.OQ`, `SNDK.OQ`, `NTAP.OQ` |
| 服务器 / 网络 / 光互连 (`server_network`) | 6 | `HPE.N`, `DELL.N`, `ANET.N`, `CSCO.OQ`, `JNPR.N^G25`, `CIEN.N` |
| 云 / AI 软件 / EDA (`cloud_software`) | 10 | `MSFT.OQ`, `AMZN.OQ`, `GOOG.OQ`, `META.OQ`, `ORCL.N`, `IBM.N`, `NOW.N`, `PLTR.OQ`, `SNPS.OQ`, `CDNS.OQ` |
| 数据中心电力 / 散热 (`data_center_power_cooling`) | 9 | `EQIX.OQ`, `DLR.N`, `AMT.N`, `IRM.N`, `VRT.N`, `ETN.N`, `TT.N`, `JCI.N`, `CARR.N` |
| 机器人 / 自主系统 (`robotics_autonomy`) | 8 | `TSLA.OQ`, `TER.OQ`, `ROK.N`, `HON.OQ`, `ISRG.OQ`, `ZBRA.OQ`, `TRMB.OQ`, `APTV.N` |
| **合计** | **49** | **49 个 unique literal RIC** |

taxonomy 的直接证据门槛按业务/产品分部的明确语义设置，泛化词单独出现不足以入组：

| 组 | 可接受的直接证据示例 | 仅有以下泛化词时的处理 |
|---|---|---|
| GPU / AI 加速器 | GPU、AI accelerator、accelerated computing、tensor/inference chip、data-center processor | 单独的 `technology` 或 `chip` 不足以入组 |
| AI 芯片及配套半导体 | 明确服务 AI/数据中心负载的 AI/ML SoC、CPU/ASIC、edge inference、networking/analog/power silicon | 单独的 `semiconductor` 或 `electronics` 标为 ambiguous |
| 存储 / HBM / 数据中心内存 | HBM、high bandwidth memory、AI/data-center memory、enterprise SSD/NAND、data-center storage | 单独的 `memory`、`storage`、`NAND` 或 `flash` 不足以入组 |
| 服务器 / 网络 / 光互连 | AI/accelerated server、HPC、data-center networking、Ethernet switch/adapter、optical interconnect、服务器 OEM | 泛化 IT hardware 或 telecom 不足以入组 |
| 云 / AI 软件 / EDA | AI/ML/generative-AI platform、intelligent cloud、AI services、AI design/simulation/developer platform | 单独的 `software`、`cloud` 或 consulting 标为 ambiguous |
| 数据中心电力 / 散热 | data-center power/cooling、liquid cooling、UPS、电气配电、热管理或明确的数据中心负荷供电 | 泛化 utility、HVAC 或 construction 不足以入组 |
| 机器人 / 自主系统 | robotics、industrial automation/vision、autonomous driving/systems、明确的机器人测试设备 | 普通汽车、工业或航空业务不单独入组 |

## 2. PIT 证据状态与历史成员限制

49/49 候选在本次 LSEG raw run 有分部行和公告行。只有固定历史分部词命中，并按 `Instrument + period_end` 连接 `Orig` announcement 的候选才记为 `direct_local_pit`。9 个 direct local PIT 候选为：

`NVDA.OQ`, `INTC.OQ`, `MU.OQ`, `HPE.N`, `MSFT.OQ`, `EQIX.OQ`, `AMT.N`, `IRM.N`, `TER.OQ`。

其余 40 个为 `provisional_static_source`。它们的注册表保存官方公司/IR 页面指针，但本 run 没有下载正文或保存页面 hash；这些页面只能作为当前/检索日参考，不能回填 2015–2026 的历史 membership。下游 PIT pool 在获得形成日前可追溯的历史业务证据、Orig 公告日和逐期 cutoff 审计前，应把这些候选保持为 `membership_unknown`，不能当作历史上始终属于 AI。

`JNPR.N^G25` 标记 `delisted_ric=True`，`member_to=2025-07-08`，仍保留在候选清单；退市和覆盖期单独记录。项目内 `AIQ.OQ`、`BOTZ.OQ`、`IGV.P`、`SOXX.OQ`、`XLK.P` 只有 ETF 价格，没有 holdings/constituents 表，因此没有从 ETF 价格反推公司成员资格。上述证据层限制意味着模型结果仍有历史成员身份不完整和幸存者偏差风险。

## 3. 41 个输入变量的选择逻辑

`technical_plus_macro` 的 41 个原始输入是冻结配置中的 21 个技术/流动性/市场变量加 20 个宏观变量。所有特征最晚使用决策日前一个 SPY session 的可得信息；主模型的变量集合在比较前固定，没有根据验证期表现逐个挑选。

### 21 个技术、流动性和市场变量

| 目的 | 数量 | 实际变量 |
|---|---:|---|
| 个股动量 | 4 | `momentum_1`, `momentum_5`, `momentum_20`, `momentum_60` |
| 个股风险与市场敏感度 | 5 | `volatility_20_ann`, `volatility_60_ann`, `beta_126`, `idio_vol_126_ann`, `beta_obs_126` |
| 个股交易流动性 | 6 | `volume_median_20_log1p`, `volume_median_60_log1p`, `dollar_volume_median_20_log1p`, `dollar_volume_median_60_log1p`, `spread_median_20_bps`, `spread_median_60_bps` |
| SPY 市场状态 | 6 | `spy_momentum_1`, `spy_momentum_5`, `spy_momentum_20`, `spy_momentum_60`, `spy_volatility_20_ann`, `spy_volatility_60_ann` |
| **合计** | **21** | **技术 + 风险 + 流动性 + 市场状态** |

这组变量覆盖四类预先定义的控制目的：多个短中期窗口捕捉个股近期价格状态；滚动波动率、beta、特质波动率和 beta 有效观察数描述风险与市场暴露；成交量、美元成交额与报价价差描述可执行性；SPY 动量和波动率描述共同市场状态。窗口长度和字段由配置固定，不能把单个系数解释成独立因果效应。

### 20 个宏观变量

| 目的 | 数量 | 实际变量 |
|---|---:|---|
| VIX 风险情绪与位置 | 5 | `vix_level`, `vix_change_1d`, `vix_change_5d`, `vix_change_20d`, `vix_percentile_252d` |
| 美国国债利率状态 | 9 | `dgs3mo_level`, `dgs3mo_change_5d`, `dgs3mo_change_20d`; `dgs2_level`, `dgs2_change_5d`, `dgs2_change_20d`; `dgs10_level`, `dgs10_change_5d`, `dgs10_change_20d` |
| 广义美元状态 | 4 | `broad_dollar_level`, `broad_dollar_change_5d`, `broad_dollar_change_20d`, `broad_dollar_change_60d` |
| 利率曲线斜率 | 2 | `term_spread_10y_2y`, `term_spread_10y_3m` |
| **合计** | **20** | **VIX + 利率 + 美元 + 曲线** |

宏观变量按风险情绪、利率水平及变化、美元趋势和期限结构四个经济状态维度预先登记。FRED 原始序列按日期与 SPY session 对齐，随后整体滞后至少一个 SPY session；周末/假日缺口只在源观测已公布后向后携带，源值、来源日期、携带天数和 reason code 保留。当前版本的 FRED 下载不是完整 vintage 数据库，修订风险仍需披露。

### 被排除的 6 个信用利差变量

排除项完整为：

`high_yield_oas_level`, `high_yield_oas_change_5d`, `high_yield_oas_change_20d`, `investment_grade_oas_level`, `investment_grade_oas_change_5d`, `investment_grade_oas_change_20d`。

原因是本地下载版本的两组 OAS 只从 2023-09-11 开始，训练期 2015–2020 与验证期 2021–2022 没有可用覆盖。原始记录保留在宏观数据层，但六列没有进入本轮模型；没有零填充、回填或用验证期信息补齐。配置和运行检查同时要求这六列不出现在任何入模 feature set 中。

## 4. H21 标签、时间切分与信息隔离

本稿使用日频基础模型口径。每个 SPY trading session 是一个 decision/formation session；所有滚动特征截止形成日前一个 SPY session。形成日收盘作为入场时点，目标收益窗口为之后第 1 至第 21 个 SPY session。标签为：

```text
y = 1, 若 stock forward 21-session total return - SPY forward 21-session total return > 0
y = 0, 否则
```

缺失的未来收益或入场/退出记录不设为 0。主模型只使用预先标记的 H21 非重叠锚点；同一 literal Instrument 在连续 21 个 SPY session 区间内至多保留一个监督锚点。样本没有随机打乱。训练期内部使用按时间扩展的 2018、2019、2020 折；每个折要求训练行的 `exit_session` 严格早于下一折验证起点，留出完整 21-session purge/embargo，避免标签窗口重叠。

| 分区 | 日期 | H21 非重叠锚点 | 覆盖 RIC | 形成日 |
|---|---|---:|---:|---:|
| training | 2015–2020 | 2,334 | 39 | 444 |
| validation | 2021–2022 | 992 | 44 | 89 |
| test | 2023-01-01—2026-06-30 | 未解封 | 未报告 | 未解封 |

验证集正标签比例为 `50.7056%`。测试期目标没有被打开、预测或计算指标；本稿没有访问 sealed test target。

## 5. Logistic Pipeline 与参数

`technical_plus_macro / Logistic` 的固定 sklearn Pipeline 为：

```text
41 个原始输入变量
→ SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
→ StandardScaler()
→ LogisticRegression()
→ p_up
```

Logistic 参数为：

| 参数 | 值 |
|---|---|
| `C` | `1.0` |
| `class_weight` | `balanced` |
| `solver` | `lbfgs` |
| `max_iter` | `2000` |
| `random_state` | `5360` |

中位数和标准化参数在每个训练折内单独拟合；最终模型只用 primary training anchors 拟合，validation 只应用训练得到的参数。该 run 中 41 个原始变量因 `spread_median_60_bps` 产生一个额外的缺失指示变量，形成 42 个模型输入。没有零填充、前向填充、winsorization、原始行删除或源数据修改。模型输出 `p_up` 是 `y=1` 的估计概率，不是未来收益率大小；拟合截距记录为 `0.008767`。

## 6. Validation 结果与对照

以下结果来自同一批验证样本：992 行、44 个 RIC、89 个 formation sessions、正标签比例 `0.5070564516`。精确 ROC-AUC 原值来自 `validation_metrics.csv`；报告显示时可四舍五入到六位小数。

| 特征组 / 模型 | ROC-AUC | PR-AUC | Brier | 0.50 阈值方向准确率 | Pooled rank IC | 日均 rank IC | 21日 top-minus-bottom spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| `technical_only` / Logistic | 0.5394626108 | 0.5420980470 | 0.2510953427 | 51.61% | 0.1403233 | 0.0776001 | 0.6152% |
| `technical_plus_macro` / Logistic | **0.5615590709** | **0.5524040990** | **0.2508244874** | **56.45%** | **0.1739062** | **0.0858063** | 0.5244% |
| `technical_plus_macro` / Random Forest | 0.5361898141 | 0.5454062926 | 0.2495687160 | 52.92% | 0.0955498 | 0.0356310 | -0.1076% |
| `technical_plus_ai_state` / Logistic（后续状态增强对照） | 0.5799395854 | 0.5745134788 | 0.2476321267 | 54.54% | 0.1866749 | 0.0889306 | 0.7386% |
| `full_state` / Logistic（宏观 + AI 状态） | 0.5819927063 | 0.5455707547 | 0.2699851983 | 55.65% | 0.1858652 | 0.0929521 | 0.7032% |

`technical_plus_macro / Logistic` 相对 `technical_only / Logistic` 的 ROC-AUC 增量为 `0.0220964601`，报告口径为 `0.022096`；方向准确率提高约 `4.84` 个百分点，Brier 改善约 `0.000271`。但是诊断性的 top-minus-bottom spread 从 `0.6152%` 降至 `0.5244%`，所以这组结果支持“宏观状态改善验证期方向分类”，不能单独证明可交易收益提升。

同一验证样本中，加入 AI 行业状态的 Logistic AUC 为 `0.579940`，全状态 Logistic 为 `0.581993`；因此 `0.561559` 是宏观增强的基础模型结果，不是当前数值最高的 Logistic。全状态 Logistic 的 Brier 为 `0.269985`，明显高于宏观增强版本，概率可靠性不能只按 AUC 排名。

## 7. 结果限制与可复现位置

- 49 个候选中只有 9 个有本地历史 LSEG 分部 + `Orig` 公告日期的 direct local PIT 证据；40 个 static candidates 仍需历史证据，不能把当前公司描述当作完整 2015–2026 membership。
- 结果是 training/validation-only 的分类比较。AUC、rank IC 和标签期 spread 不是封存测试结果，也不是已经扣除全部交易约束的基金绩效承诺。
- 验证期正式组合另有独立结果与限制；本稿只解释基础 Logistic 的验证分类结果，不把 0.561559 直接改写为组合收益。
- 本次摘录没有进行源数据转换：没有插补、去重、缩尾、单位或时区变换、标识映射、成员删除或标签过滤；原始/clean/model-ready 行保持不变，affected source rows=0，quarantine=none。模型内部训练期中位数处理仅按既有配置说明记录。

可复现和审计位置：

| 内容 | 项目内路径 |
|---|---|
| 49-RIC 清单与 9/40 PIT 限制 | `AI_POOL_EXPANSION_CANDIDATES_v1.md`；`data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv` |
| 七组 taxonomy 与证据门槛 | `AI_SUPPLY_CHAIN_TAXONOMY_v2.md` |
| 日频宏观/AI 状态比较与文字结论 | `AI_DAILY_STATE_MODEL_RESULTS_v1.md` |
| Logistic 模型卡 | `BASELINE_LOGISTIC_MODEL_CARD.md` |
| 固定特征、预处理、参数和信用利差排除 | `ai_pool_daily_state_models_v1_1_config.json` |
| 训练/验证指标（精确值） | `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/validation_metrics.csv` |
| 实际 feature set | `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/feature_sets.json` |
| Logistic artifact | `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/models/technical_plus_macro_logistic.joblib` |
| 独立审计 | `data/audit/ai_pool_daily_state_models_v1_1/20260910T070500000000Z/` |

核心证据文件 SHA-256（用于核对版本）：

| 文件 | SHA-256 |
|---|---|
| `data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv` | `b165060c7d61be0a3ff31f3b337dda1730f0e154efb1a4ac1b61cf693f038cde` |
| `ai_pool_daily_state_models_v1_1_config.json` | `d0b92845f6297e2fb2ef4cfeaf8c0c4881b7e80ab3511f47c812ee0398de9177` |
| `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/validation_metrics.csv` | `b30974e624de29fa0b32222a1c9c37d9fa68beef89bae75e3a6c8ada856919b5` |
| `data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/feature_sets.json` | `1ffbdb60efc6b3fa0e5c2c0e11b08795a1ede4b105e4015738a0e0213c6e80cb` |

## 8. 来源定位

- `AI_POOL_EXPANSION_CANDIDATES_v1.md:3,5-29`：49 个 RIC、七组数量、732 行保留、0 删除、9 direct local PIT、40 provisional static、退市示例。
- `AI_SUPPLY_CHAIN_TAXONOMY_v2.md:3,7-10,12-33,43-56`：选择禁用项、七组证据门槛、PIT 证据优先级与七组分布。
- `BASELINE_LOGISTIC_MODEL_CARD.md:6-28,30-58,60-92,111-117`：0.561559 的含义、样本/切分、Pipeline、参数、41 个变量、六个信用利差排除、结果表与复现路径。
- `ai_pool_daily_state_models_v1_1_config.json:14-38,65-79,96-112`：H21、训练/验证年份、21 个技术变量、20 个宏观变量、6 个排除项、训练期预处理与 Logistic 参数、purged tuning/test policy。
- `AI_DAILY_MACRO_FEATURE_SPEC.md:3-23`：宏观序列和可得时点、AI 状态输入原则、四组模型比较规则。
- `AI_DAILY_STATE_MODEL_RESULTS_v1.md:13-37,39-62,88-110`：9/40 限制、宏观缺失原因、四组特征 × 两类模型结果、测试封存与独立检查。
- `run_ai_pool_daily_state_models_v1.py:110-119,371-380,512-532,584-592`：实际 sklearn Pipeline、feature set 合并、信用利差排除检查、test/预处理检查。


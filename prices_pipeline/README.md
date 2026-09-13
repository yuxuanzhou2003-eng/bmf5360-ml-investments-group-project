# prices.csv 是怎么来的

这个文件夹是**只读快照**，把生成 `data/clean/v3/20260909T012417705069Z/prices.csv` 的代码按执行顺序整理在一起，方便查阅和交接。

> 权威版本仍在项目根目录。改代码请改根目录的那一份，不要改这里的副本，否则两边会不同步。
> 快照时间：2026-09-09。

产物：`data/clean/v3/20260909T012417705069Z/prices.csv`，2,084,177 行 × 21 列，577 MB。

---

## 执行顺序

### 01_universe — 先确定"哪些股票、什么时候在池子里"

`rebuild_universe_v2.py` → `build_universe_spans_v2.py` → `eligible_spans_2015_2026_corrected.csv`（781 只）

从 2026-09-07 的 503 只 S&P 500 成分股锚点，配合 751 条指数变动记录**反向回滚**，重建 2015–2026 每一天的成分股名单，含 155 只退市 RIC（带 `^` 后缀）。价格取数的范围由它决定——**不是用今天的成分股去取历史价格**，那样会有幸存者偏差。

这里有个 LSEG 的坑值得记住：只有拼错的字段名 `TR.IndexJLConstituentituentChange` 才会返回 Joiner/Leaver 的 Change 列，正确拼写的 `TR.IndexJLConstituentChange` 会静默丢掉该列。

### 02_collect — 取原始数据

`collect_universe_v3.py`

- 字段：`TRDPRC_1, OPEN_PRC, HIGH_1, LOW_1, ACVOL_UNS, BID, ASK, TRNOVR_UNS`
- 复权：显式指定 `exchangeCorrection, manualCorrection, CCH, CRE, RPO, RTS`
- 163/163 个价格批次全部成功，宽表日期行 461,217
- 版本化目录、断点续传、请求身份 + 行数 + SHA-256 三重校验后才复用已有文件，失败保留 `.error.json`

采集期间的两个事件都留了痕迹：`prices_live_046` 遇到连接重置（`attempt_01.error.json` 保留，续传成功）；一个分片写共享 summary 时撞上 Windows 文件锁（`summary.json.42304.tmp` 保留，数据文件未损坏）。

### 03_clean — 清洗

`clean_universe_v3.py`

把 2,251,311 个"证券×日期"宽表单元展开成长表，然后：

| 处理 | 影响行数 | 规则 |
|---|---|---|
| 结构性空行隔离 | 167,134 → quarantine | 宽表对齐产生的、所有字段全空的单元，不是真实缺失 |
| close 缺失 | 117 | **保留为 NA，不填补** |
| volume 缺失 | 75 | 同上 |
| 双边报价缺失 | 26 | 同上 |
| dollar_volume 优先用供应商 turnover | 1,945,045 | `TRNOVR_UNS` |
| turnover 缺失时回退 `close × volume` | 139,055 | 打标记 `dollar_volume_source` |
| dollar_volume 仍缺失 | 77 | 保留为 NA |

**没有做的事**（这些是刻意的）：没有均值/中位数/前值/零填补，没有插值，没有缩尾，没有因为退市就删除证券，没有对已经复权的价格再乘拆股因子。

隔离的行全部写进 `prices_quarantine.csv` 并带 `rejection_reason`，不是删掉。

### 04_validate — 独立验证

`validate_clean_v3.py` → `data/audit/validate_clean_v3/20260909T015809930464Z/validation.json`，**31/31 checks passed**。

用与清洗脚本不同的实现重算，不是复读清洗脚本自己的记账。

### 05_probes — 支撑上述决策的探针

- `probe_price_execution_v3.py` / `audit_price_execution_v3.py`：验证复权口径。默认复权与显式指定六项复权的 AAPL 文件**逐字节一致**；拆股前后 unadjusted/adjusted close 与 volume 的中位比均为 4，turnover 差为 0。同时发现显式复权响应会漏掉请求首日，所以全量采集用了更早的预热期，边界缺口不填补。
- `collect_stock_splits_v2.py` / `audit_stock_splits_v2.py` / `audit_panel_split_exposure.py`：确认价格已含公司行动调整，因此**禁止**再乘拆股因子。

---

## 21 个字段

| 字段 | 含义 |
|---|---|
| `Instrument` / `Date` | 主键，唯一 |
| `raw_date` | 供应商原始日期字符串，未解析前的样子 |
| `TRDPRC_1` | 收盘价（已复权） |
| `OPEN_PRC` / `HIGH_1` / `LOW_1` | 开/高/低（已复权） |
| `ACVOL_UNS` | 成交量 |
| `BID` / `ASK` | 日末买卖报价 |
| `TRNOVR_UNS` | 供应商成交额 |
| `raw_wide_row` / `raw_file` | 溯源：来自哪个原始文件的第几行 |
| `has_close` / `has_volume` / `has_two_sided_quote` | 可用性标记 |
| `quoted_spread_bps` | 报价价差（基点） |
| `dollar_volume` | 成交额 |
| `dollar_volume_source` | `vendor_turnover` 或 `close_times_volume` |
| `in_sp500_that_day` | 当日是否在指数内 |
| `price_adjustments` | 复权口径字符串 |

---

## 用之前必须知道的三条限制

1. **这是价格序列，不是股息总收益序列。** `price_adjustments` 只说明做了交易所更正和公司行动调整，**不能**当作分红再投资的证明。需要总收益请用 `data/clean/v2/returns.csv`（模型标签用的就是它）。
2. **日末 bid/ask 只是价差代理**，不代表能按该价成交。
3. **缺失不代表退市。** 不要从"没有远期收益"推断退市，缺失原因未经单独核实前一律记为未知。

## 已知待核实项

`MRNA.OQ` 在 2026-08-19 单日 +177% 且当天在指数内。按规则，|日收益|>20% 只进复核清单，不删不缩尾。

## 完整记录在哪

- 逐次运行的详细记录：根目录 `DATA_PROCESSING_LOG.md`
- 字段定义与固定清洗规则：`docs/DATA_DICTIONARY.md`
- 拆股核查过程：`docs/STOCK_SPLIT_REVIEW.md`
- 机器可读的验证结果：`data/audit/validate_clean_v3/20260909T015809930464Z/validation.json`

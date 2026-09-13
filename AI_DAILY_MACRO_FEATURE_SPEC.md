# AI 日频宏观与行业状态特征规范 v1.0

日期：2026-09-10。本文在查看新增模型结果前冻结第二轮 daily baseline 的新增变量。所有变量最晚使用决策日前一个可用交易日；不使用测试期目标选择变量。

## 宏观风险状态

- `VIXCLS`：VIX level、1/5/20 日变化、252 日滚动 percentile。
- `DGS3MO`、`DGS2`、`DGS10`：利率水平、5/20 日变化、`10Y-2Y` 和 `10Y-3M` 期限利差。
- `BAMLH0A0HYM2`：美国高收益债 OAS level、5/20 日变化。
- `BAMLC0A0CM`：美国投资级公司债 OAS level、5/20 日变化。
- `DTWEXBGS`：广义美元指数的 5/20/60 日变化。

日频宏观序列按观测日期左连接到 SPY 交易日，再整体滞后至少一个 SPY session。周末/假日缺口只允许在源序列已公布以后向后携带；不得向前填充未来值。原始值、首次可用日期、携带天数和 reason code 必须保留。FRED 当前下载不是完整 vintage 数据库，因此结果应披露修订风险。

## AI 行业状态

使用现有 SOXX、XLK、AIQ、BOTZ、IGV 日行情，生成 5/20/60 日动量、20/60 日波动、60 日最大回撤。主行业变量固定为 SOXX 相对 SPY 的 5/20/60 日收益；其他 ETF 作为辅助状态，不按验证期表现选择。

在 49-RIC 扩大池内计算过去一日可得的 breadth 和 dispersion：上涨比例、位于 20/60 日均线上方比例、横截面收益标准差、等权池相对 SPY 收益。计算时保留当日有效标的数，少于 20 家则标记低覆盖。

## 模型比较

依次比较：`technical_only`、`technical_plus_macro`、`technical_plus_ai_state`、`full_state`。Random Forest 超参数只在 training 内的 purged walk-forward 中选择；validation 只用于一次模型族比较，test 继续封存。主统计使用 H21 非重叠锚点；all-daily + boundary purge 是相关样本稳健性分析。

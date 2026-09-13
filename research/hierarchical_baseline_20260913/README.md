# AI 行业—赛道—企业：截图对应基准

本目录对应2026-09-13讨论的五行对照表，不是后续RF调参、准入或风险预测实验。研究股票池为34家美国上市美元证券/ADR；2021–2023训练，2024–2025为反复使用的开发期。当前静态AI成员资格不是历史点时资格，结果不代表已证明的投资优势。

## 策略和数值口径

- 行业固定AI基础设施，行业模型决定AI候选与现金之间的配置。
- 三层模型：行业概率至少0.5时投资；选择排名前三赛道；各赛道买预测排名前一半企业。Logistic C=0.5；RF200树、深度4、最小叶子20、特征比例0.7、随机种子5360。这不是后续深度3收益回归模型。
- 等赛道对照采用所有可用赛道及企业；动量对照按20日动量选前三赛道，并持有这些赛道的全部合格企业。
- 单股最多10%、单赛道最多40%，约束在调仓时执行，剩余资金留现金，期间允许权重漂移。
- 每5个美国交易日生成信号，下一交易日收盘执行；五日训练收益从形成日后第2至第6个交易日计算。训练标签结束日必须在2024年前。
- 核心因子：5/20/60日动量、20/60日波动、60日回撤、相对赛道动量。行业层另加8个已滞后宏观变量，具体名称及公式见主代码和协议。
- 表中成本为每单位绝对成交额10bps；现金使用3月国债年率按实际日数/365代理。净Sharpe是每日净收益减现金收益后年化，年化波动率是每日净收益的样本标准差乘sqrt(252)。SPY首次执行后买入持有，支付10bps买入成本，无期末强制清仓。

**注意：主运行文件的原始 `performance.csv` 中 `sharpe` 使用零无风险利率；截图使用的是 `sharpe_excess_cash`。请使用本次导出器或原审计文件计算截图指标，不能混用两列。**

五行汇总：[comparison.md](summary/comparison.md)，精确值：[comparison.csv](summary/comparison.csv)。这里仅公开聚合策略指标，没有股票日行情、个股预测、完整持仓或模型二进制。

## 代码入口

| 文件 | 用途 |
|---|---|
| `run_hierarchical_fund_v1.py` | 三层特征、分类模型、对照/消融、连续持仓及成本回测；支持路径参数 |
| `export_hierarchical_comparison_v1.py` | 从完成的回测导出截图口径，并计算SPY对照；支持路径参数 |
| `build_hierarchical_input_v1.py` | 本地冻结源到34公司面板的原始构建流程 |
| `validate_hierarchical_input_v1.py` | 原始输入审计 |
| `validate_hierarchical_run_v1.py` | 原始标签抽样、逐日账本、净Sharpe及SPY核查 |
| `diagnose_hierarchical_selection_v1.py` | 原始赛道/公司排名诊断 |
| `HIERARCHICAL_FUND_V1_PROTOCOL_20260913.md` | 数据处理、交易规则和研究限制 |

后三个原始构建/审计入口保留冻结日期路径，部分会在导入时执行。不要把它们当作通用库导入；需在原工作区源数据齐备且输出目录不存在时单独运行。它们不是自包含的数据下载工具。

## 安装与运行

在仓库根目录、单独虚拟环境中安装这一版依赖，避免覆盖其他历史实验环境：

```text
python -m pip install -r requirements-hierarchical.txt
python run_hierarchical_fund_v1.py --panel-dir data/model_ready_hierarchical_v1/20260913T033945Z --output-dir data/model_runs/hierarchical_public_rerun/run01
python export_hierarchical_comparison_v1.py --panel-dir data/model_ready_hierarchical_v1/20260913T033945Z --run-dir data/model_runs/hierarchical_public_rerun/run01 --output-dir data/model_runs/hierarchical_public_rerun/summary01
```

两个输出目录必须尚不存在，以免覆盖结果。主运行训练全部固定模型及消融，不按开发成绩自动选赢家。

### 本地数据前提

需要在有权使用这些数据的本地环境提供冻结面板：

- `stock_daily.csv`：至少Date、Instrument、primary_group、return_decimal、close；一行公司×美国交易日；收益是小数形式的总回报，不能再除100。
- `macro_daily.csv`：Date及主代码MACRO中的8列；宏观在输入层已按前一交易日可得信息映射。
- `spy_daily.csv`：Date、return_decimal，供对照导出器使用。

这里没有随代码重新分发LSEG价格、总回报、源数据、密钥或完整本地交付包。**仅克隆仓库不能从头重跑这一版，需要上述授权本地输入。** 旧README中“可复现”的冻结输入对应较早的行业状态模型，不等于本次三层模型的数据。

## 本次发布检查

导出器在本地冻结运行上执行，五行数值与截图四舍五入后逐项一致。提交脚本进行了语法检查，原始已运行的模型/账本代码未为发布改写；没有声称本次发布重新训练了所有模型。发布范围与AI协助见 `PUBLICATION_LOG.md`。

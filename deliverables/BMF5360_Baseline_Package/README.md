# BMF5360 基础模型可交付代码包

本包只覆盖两个预先登记的特征组：`technical_only` 与 `technical_plus_macro`。每个特征组分别拟合 Logistic Regression 和同表 Random Forest，输出四个模型的 validation-only 对照。代码只使用开发目标 `targets_dev.csv`；`targets_test_sealed.csv` 不在输入清单中，不会被打开、解析、哈希、聚合、评分、排序或预测。

## 环境

建议使用 Python 3.12.x。精确依赖见 [code/requirements.txt](code/requirements.txt)：NumPy 2.5.3、pandas 2.3.3、SciPy 1.18.1、scikit-learn 1.9.0、joblib 1.6.0 和 threadpoolctl 3.6.0。项目已有匹配依赖的 `.venv` 时可直接复用；否则在隔离环境中安装：

```text
python -m pip install -r deliverables/BMF5360_Baseline_Package/code/requirements.txt
```

## 输入版本

运行目录默认是项目根目录，即包含 `data/` 的目录。入口会从配置读取以下已冻结版本：

| 用途 | 路径 |
| --- | --- |
| model-ready 特征 | `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z/model_features.csv` |
| model-ready 资格表 | `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z/eligibility.csv` |
| 开发期目标 | `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z/targets_dev.csv` |
| model-ready 摘要 | `data/model_ready_ai_pool_daily_v1/20260910T052100000000Z/summary.json` |
| 训练/验证宏观特征 | `data/clean/daily_macro_v1/20260910T062700000000Z/macro_features_train_valid.csv` |
| 宏观清洗摘要 | `data/clean/daily_macro_v1/20260910T062700000000Z/clean_summary.json` |

上游 model-ready 仍保存测试期日程特征和封存目标文件，但本入口只把 `targets_dev.csv` 中的 training/validation 样本连接到特征；不会读取封存目标文件。默认配置和代码副本分别是 [code/baseline_package_config.json](code/baseline_package_config.json) 与 [code/run_baseline_package.py](code/run_baseline_package.py)。

## 固定口径

- 标签是未来 21 个交易日股票相对 SPY 的超额收益方向 `y`。
- 训练样本为 `non_overlap_selected` H21 anchors，训练期为 2015–2020，验证期为 2021–2022；不随机打乱。
- 资格条件沿用 model-ready：`label_complete`、未 `boundary_purged`、`entry_trade_eligible`、`feature_core_available`、`non_overlap_selected`，且 `y` 与 `forward_excess_return` 非缺失。
- technical-only 使用 21 个技术/流动性/市场变量；technical-plus-macro 在此基础上加入 20 个宏观变量。六个训练/验证无覆盖的信用利差变量明确排除。
- Logistic 固定 `C=1.0`、`class_weight=balanced`、`solver=lbfgs`、`max_iter=2000`、`random_state=5360`。
- Random Forest 使用配置中的 18 组网格；每个特征组单独在 2018、2019、2020 的训练期 purged expanding folds 上按平均 ROC-AUC 选择，平局依次看平均 Brier 和网格顺序。每个 fold 要求 `exit_session < validation_start`。
- 缺失值处理只在模型内部进行：每个训练 fold 和最终训练样本分别拟合 `SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True)`；Logistic 的 `StandardScaler` 只在最终训练样本拟合。没有零填充、前向填充、插值、winsorization 或源表改写。

已有审计 run 的 validation ROC-AUC 参考值如下；重新运行时应在同一输入、依赖和配置下得到相同口径的四行表：

| 特征组 | Logistic | Random Forest |
| --- | ---: | ---: |
| technical-only | 0.539463 | 0.523103 |
| technical + macro | 0.561559 | 0.536190 |

这些是开发期比较，不是测试期性能结论，也不是投资组合回测结果。

## 运行

从项目根目录执行以下命令；使用固定 `--run-id` 可以得到可定位的输出目录：

```text
python deliverables/BMF5360_Baseline_Package/code/run_baseline_package.py --workspace-root . --run-id baseline_package_validation
```

也可以使用项目虚拟环境的解释器：

```text
.venv\Scripts\python.exe deliverables/BMF5360_Baseline_Package/code/run_baseline_package.py --workspace-root . --run-id baseline_package_validation
```

`--config` 可指定另一份同 schema 配置，`--output-root` 可指定新的输出根目录。重复使用同一 `run-id` 会停止并保留已有结果，避免覆盖旧 run。运行完成后入口会在项目根目录 `DATA_PROCESSING_LOG.md` 和 `AI_USE_LOG.md` 追加 run-scoped provenance；不会覆盖原始数据、model-ready 输入、历史模型 run 或审计结果。

## 输出

默认输出为 `data/model_runs/bmf5360_baseline_package/<run-id>/`，包括：

- `validation_predictions.csv`：992 行验证预测和四个 `p_up_*` 概率列。
- `validation_metrics.csv`：technical-only / technical-plus-macro × Logistic / Random Forest 的四行同表指标。
- `feature_importance.csv`：四个已拟合模型的变换后特征重要性；缺失指示变量单独标记。
- `feature_sets.json`：实际使用的两个特征列表。
- `rf_tuning_evidence.json`：每个特征组的 18 组 RF 候选、三期 purged fold 证据和选定参数。
- `models/*.joblib`：四个训练/验证模型 artifact。
- `summary.json`：输入/输出哈希、样本计数、处理规则、验证指标和检查结果；封存测试目标只记录 `opened=false` 的路径描述。
- `audit.json`：测试目标未打开、测试预测未写出和原始输入未修改的布尔审计摘要。

本包当前只交付代码与配置，不在打包阶段重跑模型或生成新的数据结果；真正运行后应以该 run 自己的 `summary.json`、`audit.json` 和日志为准。

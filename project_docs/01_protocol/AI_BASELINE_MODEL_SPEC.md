# AI 行业交易基准模型协议 v0.1

> 口径修订：所有 AI 因子排序和 Random Forest 模型均在点时 AI 行业股票池内进行；全市场连续 AI 暴露主线已由 `AI_POOL_SPEC.md` v1.0 取代。Event study 不是主线。

日期：2026-09-10。本文规定 clean/model-ready 完成后的第一轮基准，不使用测试期未来收益。

## 共同样本和目标

- 观察单位：`security_id + formation_session`，同一形成月的股票不能随机拆到不同分区。
- 形成日：每月最后一个 SPY session；所有特征最晚到形成日前一 session。
- 标签：形成日后第一个 SPY session 入场，持有 21 个 SPY session；`y=1` 当个股累计收益减 SPY 累计收益大于 0，否则为 0。缺失标签不设为 0。
- 分区：training 2015–2020，validation 2021–2022，test 2023-01-01 至 2026-06-30。测试目标保持封存。
- 内部验证：training 内按月份 expanding walk-forward；每个验证折与训练折之间留出完整 21-session gap/embargo。预处理只在折内训练部分拟合。

## 模型候选

1. **Momentum baseline**：按 252-session momentum 排序，报告 top-minus-bottom 和方向准确率；不估计参数。
2. **Logistic regression**：AI_REV、AI_RD、AI_MKT 加控制变量；标准化和中位数填补（若采用）只在训练折拟合，并加入缺失指标。
3. **Random Forest**：固定 `n_estimators=500, max_depth=4, min_samples_leaf=100, max_features=sqrt, class_weight=balanced, random_state=5360`；只在 training-only CV 比较最多两个预先登记的叶节点敏感性。
4. **AI-factor long/short**：按冻结的 AI 分数组成等权 top/bottom 20% 多空组合；它是可解释排序基线，不使用标签来定义分组。

## 评估和交易转换

模型指标包括 validation ROC-AUC、PR-AUC、Brier score、校准曲线、分年份方向准确率、rank IC，以及 top-minus-bottom 的未扣成本收益。组合层同时报告 Sharpe、最大回撤、换手、行业/单股暴露和交易成本敏感性。预测概率只用于预先登记的仓位规则；不在 validation 中反复调阈值。

所有模型都要与以下控制组对比：

- 仅控制变量（size、book-to-market、盈利能力、投资、杠杆、动量、波动、beta、流动性和 SPY 状态）；
- 控制变量 + AI 三组件；
- 仅 AI 三组件。

若加入复杂模型，必须保留这三组对照，报告 AI 变量相对控制组的增量。任何模型选择、缺失规则、阈值、分组比例或持有期变化都建立新 candidate_id，并在第一次 validation 前登记。

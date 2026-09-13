# 多层研究底表字段契约 v1

状态：设计已实施为本文档；数据构建须先通过覆盖审计。此契约不设置投资权重。

## 企业与赛道

企业主键使用项目稳定 security_id；literal RIC 原样保留。一个企业多证券类别不可默认为两家独立公司。

保留 legacy_primary_group，不覆盖旧七组。新增 proposed_subsector、secondary_roles、classification_source、classification_available_at、classification_status，允许 unknown，不强行归类。旧 49 家是起始研究目录，截图名单补充另建核验队列。

分别存储 legacy_universe_member_from/to、listing_from/to、role_evidence_available_at。第一个只代表旧样本成员区间，不推断后两者。退市本身不排除历史观察。历史角色未知不改写为明确非成员。

## 股票日表

主键：(security_id, formation_session)，副键 sample_id。保留 reference_session、feature_source_version、feature_available_at/status。时间采用交易所 session；跨市场数据将来需要 UTC 可得时间对齐。

原特征仅经白名单复制：momentum_1/5/20/60、volatility_20/60_ann、volume_median_20/60_log1p、dollar_volume_median_20/60_log1p、spread_median_20/60_bps、beta_126、idio_vol_126_ann。beta_obs_126 作为覆盖诊断，不默认预测因子。

SPY、VIX 等共同状态另存市场日表，按日期 many-to-one 连接，并显式标记 feature_scope=market。不能把共同变量复制后新增的股票行数当新增独立市场观察。

不继承 supervised_model_eligible、non_overlap_selected、label_complete、exit_session 等未来依赖条件作为特征入选规则。特征缺失保留 NaN 和原因；不静默零填或以缺失作为不涨。重复主键应停止构建并输出审计，不自动保留第一行。

## 赛道日表

主键：(subsector, formation_session)。需分别记录候选数、有资格数、各指标有效分母及缺失数。广度=可计算历史收益且为正的公司数/可计算历史收益的公司数；分母为零时缺失。未上市、缺覆盖、未完成预热不能算作下跌。

赛道收益指标必须声明当日收益是描述性均值还是期初已知持仓收益。若用于交易检验，权重在收益实现前确定；成分缺收益不得事后把权重无成本转移给其余成分。

## 事件表及连接

主键 event_id。字段包括 issuer_id、event_type、event_occurred_at、first_public_at、timezone、source_uri、source_version、timing_precision、revision_status。

公告内容与预期分列；预期 snapshot 必须早于公告。已有表中的 actual、consensus、standardized_surprise 需另行核验，不由列名推断已可用于预测。事件日期不明确时标记 unknown。同一股票多事件不直接 many-to-many 连接放大日表，应使用独立关联表和预定时间窗口聚合。

## 目标及切分

目标单独构造，字段必须有 decision_at、entry_at、exit_at、horizon、return_type、benchmark、target_available、reason_code。入场必须晚于特征可得时点。

训练样本的 exit_at 必须早于验证决策起点；只按 formation year 分割不够。调参按整日期 chronological folds，同日股票不随机分入训练与验证。事件相关记录按事件聚类；误差估计按日期/时间块保留共同冲击与重叠依赖。

2021 起为当前研究范围；2024–2025 已被查看，是开发验证，不恢复为未接触样本。2026 H1 目标仍封存。库存审计不读取任何未来收益。

## 最低验收

原样本数、日期数、证券数与分类后计数对账；无重复主键；所有 reference_session 严格早于 formation_session；缺失未转零；日期不越界；所有 49 个起始候选均出现在覆盖审计中。以上通过仅证明结构与覆盖，不证明行情、企业分类、事件时间和模型收益正确。

# AI pool expansion candidate list v1

本清单对应 `AI_SUPPLY_CHAIN_TAXONOMY_v2.0`，是现有 781-RIC universe 内的 49 个设计候选。逐 RIC 的纳入理由、来源 URL、LSEG 覆盖、退市标记和 PIT 限制见 `data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv`。

## 49 个 literal RIC

| 供应链组 | RIC |
|---|---|
| GPU / AI 加速器（5） | `NVDA.OQ`, `AMD.OQ`, `AVGO.OQ`, `INTC.OQ`, `MRVL.OQ` |
| AI 芯片及配套半导体（6） | `QCOM.OQ`, `TXN.OQ`, `ADI.OQ`, `NXPI.OQ`, `ON.OQ`, `MPWR.OQ` |
| 存储 / HBM / 数据中心内存（5） | `MU.OQ`, `WDC.OQ`, `STX.OQ`, `SNDK.OQ`, `NTAP.OQ` |
| 服务器 / 网络 / 光互连（6） | `HPE.N`, `DELL.N`, `ANET.N`, `CSCO.OQ`, `JNPR.N^G25`, `CIEN.N` |
| 云 / AI 软件 / EDA（10） | `MSFT.OQ`, `AMZN.OQ`, `GOOG.OQ`, `META.OQ`, `ORCL.N`, `IBM.N`, `NOW.N`, `PLTR.OQ`, `SNPS.OQ`, `CDNS.OQ` |
| 数据中心电力 / 散热（9） | `EQIX.OQ`, `DLR.N`, `AMT.N`, `IRM.N`, `VRT.N`, `ETN.N`, `TT.N`, `JCI.N`, `CARR.N` |
| 机器人 / 自主系统（8） | `TSLA.OQ`, `TER.OQ`, `ROK.N`, `HON.OQ`, `ISRG.OQ`, `ZBRA.OQ`, `TRMB.OQ`, `APTV.N` |

合计：49 个 unique literal RIC。每组至少一个候选，未按收益、规模、流动性或模型输出筛选。

## 可审计证据状态

- 781/781 universe RIC 唯一；49/49 候选在 universe；旧 v1 member overlap=9；未选 732 行保留在 `universe_screening.csv`；物理删除=0；quarantine=0。
- 49/49 候选在本次 LSEG raw run 有分部行和公告行。固定历史分部词命中且按 `Instrument + period_end` 连接 Orig announcement 的 9 个为 `direct_local_pit`：`NVDA.OQ`, `INTC.OQ`, `MU.OQ`, `HPE.N`, `MSFT.OQ`, `EQIX.OQ`, `AMT.N`, `IRM.N`, `TER.OQ`。
- 其余 40 个为 `provisional_static_source`。来源目录保存官方公司/IR 页面指针，但本 run 不下载正文、不做页面 hash，因此不能用来回填 2015–2026 历史 membership。
- `JNPR.N^G25` 的 `delisted_ric=True` 且 `member_to=2025-07-08`，仍保留在清单；退市和覆盖期单独记录。
- 项目内 AIQ、BOTZ、IGV、SOXX、XLK 只有 ETF 价格，没有 holdings/constituents 表；没有从 ETF 价格推断成员。

## 后续 PIT 使用

若传给 daily builder，先将 `candidate_registry.csv` 的 `ric` 列按 literal 值复制为 `Instrument` 列，并保留该文件 hash。builder 仍要求形成日前最近的 membership 状态和 corrected member_from/member_to 区间；候选列表本身不把 `provisional_static_source` 升级为 member。静态候选在完成历史业务证据、Orig 公告日和逐期 cutoff 审计前应保持 `membership_unknown`。

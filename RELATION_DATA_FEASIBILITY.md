# 关系数据可行性检查（2026-09-09）

同期残差相关图在开发期被否证（见 `RESEARCH_PROTOCOL.md` v0.6）之后，检查 LSEG 在当前 license 下能否提供**有经济机制的**公司间关系边。本轮只做数据可得性探测：不建图、不建模、不取全量、不触碰测试期，也没有读写任何既有 raw、clean、panel 或 model-ready 文件。

三轮探测共 34 个独立请求，样本为 `AAPL.O`、`MSFT.O`、`WMT.N`。每个候选字段族**单独请求**，避免上一次 `Comparable` 合并请求报错导致其他候选可用性未知的问题。原始响应与逐请求 JSON：

- `data/raw/relationship_probe_v1/20260909T081614330938Z/`（17 个候选）
- `data/raw/relationship_probe_v2/20260909T082013151558Z/`（2 个候选，第 2 个超时后按设计退出）
- `data/raw/relationship_probe_v3/20260909T082330407825Z/`（14 个候选，可续跑）

脚本 `probe_relationships_v1.py`、`probe_relationships_v2.py`、`probe_relationships_v3.py`。凭证只从本地 `.env` 读取，输出与错误均做脱敏。

## 结论汇总

| 关系类型 | 字段 | 历史可取 | 时点字段 | 已知缺陷 | 判断 |
|---|---|---|---|---|---|
| 共同分析师覆盖 | `TR.RecEstBrokerName` + `TR.RecEstValue` + `TR.RecEstDate` | 是，须用单点 as-of | Activation Date，真实历史 | 33.1% 行的 broker 名显示为 `Permission Denied <ID>`；范围请求返回按交易日展开的冗余且不含日期列 | **可用，推荐优先** |
| 共同机构持股 | `TR.InvestorFullName` + `TR.PctOfSharesOutHeld` + `TR.HoldingsDate` | 是，仅单点 as-of | Holdings Filing Date **实为持仓期末日** | 45 天披露滞后必须显式处理；范围请求 Gateway Time-out | **可用，但取数工程量大且需滞后规则** |
| TRBC 行业分类 | `TR.TRBCEconomicSector` 等五层 | **未证实** | 无 | as-of 2015 与当前值完全相同，无法区分"分类未变"与"SDate 被忽略" | 仅可作对照组，且与已知混淆变量重合 |
| 供应链客户—供应商 | 13 种拼写全部尝试 | — | — | 全部返回字段无法解析 | **未能证实可得**（非"不存在"） |
| per-broker EPS 预期 | `TR.EPSEstBrokerName` / `TR.EstBrokerName` | — | — | Broker Name 列返回但全空 | 不可用 |

## 逐项证据

### 共同分析师覆盖：可用

`TR.RecEstBrokerName` 在 as-of 与范围两种写法下都返回真实历史。

- as-of `SDate=2015-12-31`：114 行，Broker Name 全非空，Activation Date 从 2004-10-14 到 2015-12-18，104 个唯一日。
- 范围 `SDate=2015-01-01, EDate=2015-12-31`：29,653 行、72 家唯一 broker、156 个唯一 activation 日。

只有 20.9% 的行 activation date 落在 2015 年内，其余更早。这不是缺陷：返回的是**截至该窗口仍然有效的推荐记录**，一条 2011 年发起、2015 年仍未撤销的推荐会带着 2011 年的 activation date 出现。这正是构造"某时点上哪些 broker 正在覆盖这家公司"所需要的口径。

**范围请求不可用于取数，必须走单点 as-of。** 那 29,653 行里 29,476 行是完全重复，唯一 `(Instrument, Broker Name, Activation Date)` 组合只有 177 个。每个 RIC+broker 组合出现约 252 次，正好等于 2015 年的交易日数——LSEG 把每条仍然有效的推荐按交易日展开了，但**响应中没有任何列标明该行属于哪一天**，只有 activation date。因此范围请求既产生 167 倍冗余，又无法从中恢复"某条推荐在哪些日期有效"。正确做法是像 ownership 一样逐季单点 as-of 请求，每次返回该时点有效的覆盖关系。

取数量级（as-of 单点，去冗余后）：as-of 2015-12-31 对 3 个 RIC 返回 114 行，约 38 行/RIC/季。781 只股票 × 约 47 个季度约 139 万行，比同口径的 ownership（约 3,287 行/RIC/季）轻约 86 倍。该估算基于 3 个大盘股，覆盖度较低的小盘股会更少。

已知缺陷：33.1% 的行 broker 名显示为 `Permission Denied <数字>`，17/72 家 broker 受限。**该数字是稳定的 broker ID，跨公司可比**，因此"同一 broker 是否同时覆盖 A 和 B"这一判断不受影响——共同分析师边只需要 broker 身份的同一性，不需要名称。正式使用前仍需验证该 ID 在全 universe 上的稳定性。

### 共同机构持股：可用，但有两个硬约束

历史确实可取，这是本轮最关键的正面发现：

- `SDate=2015-12-31` → 9,860 行，filing date 全部 ≤ 2015-12-31；
- `SDate=2020-06-30` → 13,963 行，filing date 2018-09-30 至 2020-06-30；
- `SDate=-10Y` → 9,961 行，filing date 2014-12-01 至 2016-09-04，相对日期正确回溯。

**约束一：`Holdings Filing Date` 是持仓期末日，不是披露日。** as-of 2015-12-31 的返回中 99.3% 是月末日，7,785/9,860（79%）正好等于 2015-12-31；as-of 2020-06-30 中 99.6% 是月末日，11,811/13,963（85%）正好等于 2020-06-30。13F 的实际披露在期末后 45 天，因此直接以期末日建图会引入约 45 天前视。可行的修正是建图时只使用**期末日 + 60 天 ≤ 公告日**的最近快照，并把该滞后规则写进配置与处理日志，与现有的快照 age 门槛同一性质。此规则未经审批前不得实施。

**约束二：只能逐季单点取数。** `SDate`+`EDate`+`Frq=FQ` 的范围写法两次均失败，第一次 90 秒超时，重试返回明确的 `UDF Core request failed. Gateway Time-out`。因此历史持仓只能按 as-of 单点逐季请求。3 个 RIC 的单次 as-of 耗时 27 秒；扩到 781 只股票 × 约 47 个季度需要分批与断点续传，量级与 v3 价格采集相当。

### TRBC 行业分类：point-in-time 未证实

`SDate=2015-12-31` 返回 3 行全非空，但三家公司的三个层级与当前值完全相同。这**无法区分**"2015 至 2026 分类确实未变"与"SDate 被忽略、返回的是当前分类"。判定需要找一个已知发生过重分类的公司做对照，本轮未做。

即使证实可取，用行业分类建边的研究价值也有限：v1 阶段的诊断已指出当前信号大概率来自同行业盈利周期共振，用 TRBC 建边等于把已知的混淆变量当作信号。它更适合做对照组。

### 供应链：未能证实可得

13 种字段拼写全部返回 `Unable to resolve all requested fields ... The formula must contain at least one field or function`：`TR.SupplyChainRelationship`、`TR.SCRelationshipPartnerName`、`TR.SupplyChainCustomers`、`TR.SupplyChainSuppliers`、`TR.SCRelationshipType`、`TR.SCRelationshipRevenueShare`、`TR.SupplyChain`、`TR.SupplyChainPartners`、`TR.SCPartnerName`、`TR.CustomerName`、`TR.SupplierName`、`TR.RelationshipType`、`TR.BusinessRelationship`。

**这是字段名无法解析，不是权限拒绝。** 与 VIX 那次明确的 `UserNotPermission` 是两类不同的失败，因此只能记为"这些拼写不可用"，不能推广为"本 license 没有供应链数据"。`ld.discovery.Views` 的 43 个视图中也没有供应链或关系类入口（`ENTITIES`、`INVESTORS`、`ORGANISATIONS`、`PEOPLE` 均为实体检索，不是关系检索）。要证实或证伪需要 Data Item Browser 的字段级确认或 LSEG content support，本轮未做。

### per-broker EPS 预期：不可用

`TR.EPSEstBrokerName`（3 行全空）与 `TR.EstBrokerName`（106 行，Broker Name 0 个非空，EPS 值 97 个非空）都无法给出 broker 身份。broker 级身份只在 recommendation 层面可得，estimate 层面不可得。

## 执行事件记录

第二轮 `ownership_range_2015_2016` 在 90 秒超时后按脚本设计 `os._exit(2)`，导致其后 12 个候选未执行。超时记录保留在 `relationship_probe_v2` 的 summary 中。第三轮脚本改为可续跑：读取 run 目录、跳过已达终态的候选、继续执行下一个，一次超时只损失当前候选。重跑同一请求得到明确的 Gateway Time-out 错误而非再次挂起，因此该组合被判定为不可用而非未知。

## 未验证与安全边界

- 全部结论基于 3 个大盘美股 RIC 与有限窗口，不能推广为 781 只股票池的历史完整性。
- broker token 的身份稳定性已在 30 只股票 × 3 个时点上验证通过（见文末专节）；仍未覆盖退市股票、小盘股，以及 2015 年前与 2022 年后的时点。
- 未验证 ownership 数据的投资方标识符跨公司一致性，而共同持股边完全依赖该一致性。
- 未验证 ownership 在退市股票上的覆盖。
- 未验证 TRBC 分类的 point-in-time 行为。
- 未做任何取全量、建图、建模或回测。测试期 2023-01-01 至 2026-06-30 保持封存。

## 遮蔽 broker token 的身份稳定性验证（2026-09-09）

共同分析师边完全依赖"能否判断同一个 broker 是否覆盖了两家公司"。33%–42% 的 broker 行返回 `Permission Denied <数字>` 而非名称，因此必须先证明该数字是**broker 级稳定标识**而不是行级 token；若是后者，三分之一以上的边无法匹配，整条路径不成立。

脚本 `probe_broker_identity_v1.py`，输出 `data/raw/broker_identity_probe_v1/20260909T084104330997Z/`。样本为 30 只股票，从修正股票池中筛出 2015-01-01 至 2021-12-31 全程在指数内的 365 只，按 RIC 排序等距确定性抽样；三个 as-of 时点 2015-12-31、2018-06-30、2021-06-30。**判据在运行前写死在脚本内**：遮蔽 token 跨公司复现的比例须达到具名 broker 的一半以上且绝对值超过 0.2。

| as-of | 行数 | 遮蔽行占比 | 遮蔽 token 数（跨多公司） | 具名 token 数（跨多公司） | 遮蔽 token 平均覆盖公司数 | 具名平均 |
|---|---:|---:|---:|---:|---:|---:|
| 2015-12-31 | 669 | 41.6% | 21（18） | 78（52） | 13.2 | 5.0 |
| 2018-06-30 | 639 | 41.6% | 20（17） | 73（51） | 13.3 | 5.1 |
| 2021-06-30 | 632 | 39.7% | 17（16） | 81（51） | 14.8 | 4.7 |

跨时点 token 集合重叠（Jaccard）：遮蔽 0.708（2015→2018）与 0.850（2018→2021）；具名 0.736 与 0.621。

**结论：遮蔽 token 是稳定的 broker 标识，且比具名 broker 更稳定。** 跨多公司的 token 占比遮蔽为 88.3%、具名为 66.5%；遮蔽 token 在 30 只样本中平均覆盖 13–15 家公司，具名平均只有 5 家；跨期重叠度不低于具名。

一个值得记入报告的观察：被遮蔽的恰恰是覆盖面最广的大型券商——覆盖越广的机构其数据 license 越贵，因此更可能被 entitlement 屏蔽掉名称。这些正是构造共同覆盖边最有价值的高度数节点，若因无法显示名称而丢弃，会系统性删掉图中连接度最高的部分。因此正式实现直接以 token 字符串作为 broker 标识，不做名称解析。

样本仍限于 30 只全程在池的大中盘股与 3 个时点，未覆盖退市股票与小盘股，也未验证 token 在 2015 年之前或 2022 年之后的稳定性。

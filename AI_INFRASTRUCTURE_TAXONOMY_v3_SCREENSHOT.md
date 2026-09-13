# AI基础设施企业结构v3：以用户截图为主

日期：2026-09-13。用户已指定采用截图结构。本文件替代后续研究的旧粗分组入口，但不改写旧模型或其成员记录。状态：分类框架采用；截图逐项转录与证券匹配另存审计。并非全部企业已取数或具备历史点时成员证据。

## 层级

| 一级环节（截图列） | 细分业务（截图可见） |
|---|---|
| Semi Production／半导体生产 | IC Design、OSAT、Foundry、IDM；底部设备/服务部分被截断 |
| Processor／处理器 | GPU、CPU |
| Server Components／服务器组件 | Power Supply、Passive Component、Thermal Solution、PCB / IC Substrates |
| Server／服务器 | Server Brands、ODM / EMS、DCI (Routing / Optical)、Memory / Storage；底部Cabling名单被截断 |
| Network／网络 | InfiniBand、Ethernet |
| Internal Power / Cooling／内部供电与冷却 | Liquid Cooling、Power Electronics、Uninterruptible Power Supply |
| Power Supply／外部供电 | Grid + Onsite Renew Storage、Generators & Grid Connection、Grid Infrastructure |

保留截图的实际布局：DCI和Memory/Storage在Server列，仍各自拥有独立细分标签，不能再混成无法分辨的“服务器/网络”类别。若以后把DCI归入统一网络分析组，必须另加research_group映射和版本，保留source_group不变。

截图上方仅见Owners / Operators标题，企业名单未显示；这一部分登记为待补，不从标题猜企业。图底部不完整的设备、布线名单同样不猜测。

## 企业与标签规则

1. 同一公司可以出现在多个业务环节。NVIDIA、AMD、Vertiv、Schneider等重复出现保留多角色关系；股票实体在企业表里唯一，不因多标签重复计数或持仓。
2. 保存截图名称、原始证券代码、来源位置、转录置信度；原始Bloomberg式代码不自动拼成LSEG RIC。现有注册表中的精确美股代码匹配只说明本地已有对应候选，不证明当前上市状态、业务事实或全部历史覆盖已核验。
3. 美国、台湾、日本、韩国、欧洲等企业进入同一个候选审计范围。没有本地数据不等于不属于该结构；新增市场须另核验上市地、RIC、币种、时区、交易日历、公司行动和可交易性。
4. 图片标为Private的条目保留为产业研究节点，其私营身份本身仍是图片来源声明，未做外部核验；不会直接当作可交易股票放入模型。
5. 旧49家仍留存在原注册表和旧数据里。图片当前裁切中没出现的公司标为not_visible_in_current_crop，不根据缺席自动判断不相关，更不删除历史数据。
6. 当前AI业务标签不回填成2021年当时已知分类。历史点时成员证据与当前结构分开记录。

## 对策略和测试的影响

基金仍可以分别研究行业总仓位、细分业务轮动和个股差异。采用分类不意味着必须全买，也不意味着必须等权。新基准的主环节/细分权重、多标签分配与集中度限制须在新结果出现前明确，否则企业结构改变本身就会改变收益。

旧rotation_macro_v3与ai_timing_macro_v1的结果保留为旧49家/7组基准。它们没有使用这张截图的新企业全集，也不能声称验证了新结构。下一步先完成公司/证券匹配与覆盖，再生成新版特征和目标。

本次只做结构转录、名单对照和审计文档，不执行删除、重新分配资金、数据填充或新企业收益回测。

## 本次落地结果

审计目录：data/audit/screenshot_taxonomy_v1/20260913025822Z/。

- 7个一级环节、22个可见子类、106条公司×角色关系，按图片证券代码/私营名称暂分87个候选实体键（不是已核验法律实体数量）。
- 19个候选键与旧49家注册表的美股代码精确对应；68个候选键待核验，其中3个带图片Private声明。另有2条公司名称低置信度，不能直接入模。
- 14个候选实体键有多个角色，角色关系保留，不重复计权。
- 旧49条记录完整保留；30条在本裁切范围未见匹配，标注not_visible_in_current_crop，不自动排除或删除。
- 文件：source_image.jpg原图、transcription_draft.json初稿、transcription_reviewed.json复核版、role_memberships.csv关系表、candidate_entities.csv候选表、identifier_review_queue.csv待核队列、old_registry_reconciliation.csv旧名单对照、summary.json计数与哈希。
- 复核更正包括Infineon/STMicro的转录、Sunon的图片Private标签、补回Auras Tech一行、InfiniBand标题、Nan Ya PCB、LS Corp及ABB代码空格；模糊的两家私营名称继续留待原始工作簿核验。原稿保留，未把识别纠正冒充外部企业信息核验。

新价格下载=0、金融数据修改=0、公司删除=0、投资权重修改=0。下一步的首要工作是核验这68个待匹配候选及现有19个标识的覆盖，之后才生成新企业池模型。

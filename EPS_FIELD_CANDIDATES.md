# LSEG EPS field candidates

**Research date:** 2026-09-09 (Asia/Singapore)  
**Scope:** bounded read-only review of public LSEG Developer documentation and LSEG Developer Community pages. No local LSEG session/API call was made in this stage, and no raw, clean, panel, or audit table was changed. Request syntax is recorded separately from field-definition evidence.

## Evidence rules

- **Supported syntax** means an official LSEG page shows the exact field or parameter string in a request or field list.
- **Supported meaning** means the official page states what the returned value represents. A callable field does not establish its unit, currency, accounting basis, split treatment, or historical vintage.
- **Exploratory** means a safe probe candidate only. An accepted request is not proof of the economic definition.
- No public source reviewed here gives a DIB-level definition for the default TR.EPSActValue basis, its complete ActType domain, the basic/diluted/normalized mapping of the three target fields, or the original as-released vintage rule.

## Candidate matrix

| Item | Exact candidate | Evidence status | Confidence | Safe interpretation/use |
|---|---|---|---|---|
| Actual EPS | 'TR.EPSActValue' | **Supported syntax and broad meaning.** An LSEG Community answer gives a historical get_data request and says it retrieves quarterly actual EPS. | High for field/query syntax; medium for detailed basis | Preserve as raw actual. Do not call it GAAP, comparable, basic, diluted, or as-reported without DIB/content-support evidence. |
| Actual fiscal period | 'TR.EPSActValue.fperiod' | **Supported syntax pattern; target-specific definition incomplete.** LSEG examples show .fperiod on EPS estimate fields and fiscal-period outputs on related fields. | Medium | Request beside periodenddate; use both as returned keys. Do not infer that fperiod alone uniquely identifies a quarter. |
| Actual period end | 'TR.EPSActValue.periodenddate' | **Supported syntax in official examples and prior target-field use; detailed definition still requires DIB.** | Medium-high | Retain exactly; match by instrument plus fiscal-period fields, never by row order alone. |
| Actual date | 'TR.EPSActValue.date' | **Supported syntax.** Official examples request it beside actual EPS. A Community question describes the returned actual date as an earnings-announcement date, but the accepted answer warns about cross-category alignment rather than defining this field. | High for suffix syntax; medium-low for economic meaning | Preserve the raw string. Treat as an unverified vendor date until field definition/timezone is confirmed. |
| Actual announcement date | 'TR.EPSActValue.announcedate' | **Target-specific candidate.** Public pages show announcedate for related TR actual fields and the LSEG IBES Actuals product page discusses announcement dates; no public field dictionary mapping for this exact target was found in this bounded search. | Medium for probe syntax; low for definition | Probe separately; retain raw timestamp and do not assign ET/UTC or market-session labels. |
| Actual calculation date | 'TR.EPSActValue.calcdate' | **Suffix candidate; actual-specific meaning unresolved.** .calcdate is explicitly used for historical consensus retrieval, while no page reviewed defines it for actuals. | Medium for syntax; low for meaning | Keep for comparison only. It is not evidence of an as-of or activation timestamp for actuals. |
| Consensus mean | 'TR.EPSMean' | **Supported syntax and broad meaning.** An official LSEG article lists it as the mean EPS estimate and uses it in ld.get_data. | High for field/syntax; medium for exact estimate universe/basis | Raw consensus field; record period and date outputs with it. |
| Consensus standard deviation | 'TR.EPSStdDev' | **Supported syntax.** The official Community answer explicitly suggests TR.EPSStdDev; the official LSEG article lists it as estimate dispersion. | High for field existence; medium for population/units | Raw dispersion only. Do not assume it is in the same scale or basis as actual/mean. |
| Consensus count | 'TR.EPSNumOfEst' | **Supported syntax in official LSEG article.** | High | Candidate companion count. |
| Alternate count spelling | 'TR.EPSNumberofEstimates' | **Supported syntax in an official Community field list; aliases may be view/product dependent.** | Medium | Probe separately from EPSNumOfEst; retain exact column/error. |
| Consensus period end | 'TR.EPSMean.periodenddate' | **Supported syntax in official Community code.** | High for suffix; medium for uniqueness | Use as a period key; validate 53-week years and fiscal-year changes. |
| Consensus fiscal period | 'TR.EPSMean.fperiod' | **Supported syntax in official Community code.** | High for suffix; medium for full definition | Keep with period end and requested Period; do not deduplicate on date alone. |
| Consensus calculation date | 'TR.EPSMean.calcdate' | **Supported syntax and historical retrieval pattern.** LSEG answers show daily historical requests using .calcdate. | High for syntax; medium for PIT retrieval; low for intraday cutoff/vintage guarantee | Strongest public candidate for an engineering as-of index. It does not prove original as-released vintage. |
| Consensus date | 'TR.EPSMean.date' | **Supported syntax.** Official examples request .date; a Community explanation distinguishes regular/released dates from .calcdate. | High for syntax; medium-low for exact event | Keep both date and calcdate; do not substitute one for the other. |
| Consensus value suffix | 'TR.EPSMean.value' | **Supported syntax in official Community code.** | Medium-high | Use only if returned; compare with base field without conversion. |
| Consensus PIT pattern | 'TR.EPSMean(Period=FY1).calcdate' plus value | **Supported request pattern for EPSMean.** The official answer uses daily SDate/EDate and .calcdate for a fixed absolute period. | High for request; medium-low for as-released semantics | Adapt to Period=FQ1 only as a probe. Do not label vintage-correct until independently checked. |
| Diluted EPS alternative | 'TR.DilutedEPSExclExtra' | **Parent-provided candidate; not independently verified in this bounded public search.** No source URL/snippet is attached here. | Low until parent attaches exact official source and DIB definition | Probe separately as an alternative measure. Do not replace TR.EPSActValue or claim comparability. |
| Normalized diluted EPS alternative | 'TR.EPSNormalizeddil' | **Parent-provided candidate; not independently verified in this bounded public search.** No source URL/snippet is attached here. | Low until parent attaches exact official source and DIB definition | Probe separately; the name alone does not establish what normalized excludes or which content set it belongs to. |

## Parameter and selector candidates

| Candidate | Evidence status | Confidence | Safe probe rule |
|---|---|---|---|
| Period=FQ0 | **Supported.** Official Community examples use it with TR.EPSActValue for quarterly actuals. | High for syntax; medium for relative-period interpretation | Use with Frq=FQ; retain returned fperiod/periodenddate. |
| Period=FY0 | **Supported.** Official LSEG article code uses it with TR.EPSActValue. | High for syntax; medium for cross-date interpretation | Do not compare FY0 across companies without returned period end. |
| Period=FY1, FY-1 | **Supported for relative-period syntax.** Official Community guidance describes FY1 as first forecast and FY-1 as previous actual. | High for syntax; medium for content timing | Record requested Period literally; do not rewrite to a calendar year. |
| Frq=FQ, FY, D, M, Y | **Supported.** These frequencies occur in official examples. | High for syntax; medium for observation semantics | Keep original frequency in request metadata. Daily rows may repeat values. |
| SDate / EDate | **Supported.** Official examples use absolute and relative dates. | High | Preserve exact window and returned rows; an empty response is not a genuine zero. |
| Curn=USD | **Supported request syntax.** Official Community code requests actual/estimate fields with Curn='USD'. | High for syntax; low-medium for conversion and target currency | Compare default versus explicit currency on the same RIC/period; never multiply manually. |
| Scale=6 | **Supported request syntax.** Official Workspace article uses Scale='6' with TR.EPSActValue and Curn. | High for syntax; low for numeric meaning of code 6 | Probe default, Scale=3, and Scale=6 separately. Do not call 6 “millions” without DIB metadata. |
| WP | **Supported for the official article's revision-window request.** It passes Period and WP with TR.EPSMean and revision fields. | Medium | Use for estimate-revision probes; do not treat as actual vintage selector. |
| CH / RH | **Supported in Excel formula syntax.** Official Community code shows CH=Fd RH=IN and another example uses RH=calcdate. The accepted answer says mixed categories are not guaranteed to align by date in get_data. | Medium for Excel syntax; low for Python layout semantics | Use explicit suffix fields and key-based matching; do not assume Python reproduces Excel row/header layout. |
| ActType=Reported | **Exploratory selector with official-community syntax evidence in the project's prior review; public pages reviewed here do not define the default or selector meaning.** | Medium for call syntax; low for meaning | Request in an isolated call and compare raw values/period/date columns. Equality on a few rows does not prove original unadjusted reported. |
| ActType=Comparable, Restated, GoForward | **Dataset-level concepts are supported by the official IBES Actuals product page; exact target-field parameter values were not established.** | Low | Issue one candidate per call. An error for one selector does not prove other selectors/concepts are unavailable. |
| Adjusted=0 | **Not supported as a target-field definition.** The corporate-action guide documents adjustment controls for pricing/per-share contexts, while the official broker-EPS Community answer says an unadjusted estimate version was unavailable. | Low | Treat as a negative probe only; rejection or identical output is not proof of adjustment policy. |
| origtimezone on actuals | **Unsupported/unknown for the target in public evidence.** The timezone example concerns TR.EPSEstDate, not TR.EPSActValue. | Low | Probe only if budget allows; retain raw timestamp and do not transfer EPSEstDate timezone rules. |

## Exact official snippets

These snippets are copied from official LSEG pages to preserve reproducible request syntax. They are not field definitions.

### Historical actual EPS

Source: [LSEG Developer Community — history on TR.EPSActValue](https://community.developers.lseg.com/discussion/16838/is-it-possible-to-pull-history-on-non-price-volume-fields-such-as-tr-epsactvalue-get-timeseries/p1)

~~~python
ek.get_data('IBM.N', ['TR.EPSActValue.date', 'TR.EPSActValue'], {'SDate':0, 'EDate':-7, 'Frq':'FQ', 'Period':'FI0'})
~~~

The same page describes this request as quarterly actual EPS for the last eight quarters. Because the example uses FI0, retain it as historical evidence but use the separately documented FQ0 form for a quarterly probe unless DIB says otherwise.

Source: [LSEG Developer Community — historical EPS data](https://community.developers.lseg.com/discussion/73493/get-eps-historical-data-for-stocks/p1)

~~~python
df2,e = ek.get_data('GOOGL.O',['TR.RevenueActValue.date','TR.RevenueActValue','TR.EPSActValue'],parameters = {'SDate':'0','EDate':'-5','Period':'FQ0','Frq':'FQ'})
~~~

### Mean and standard deviation

Source: [LSEG Developer Community — IBES analyst variance](https://community.developers.lseg.com/discussion/88592/ibes-analyst-variance/p1)

~~~text
TR.EPSStdDev
~~~

Source: [LSEG Developer article — finding stocks where analysts and the market disagree](https://developers.lseg.com/en/article-catalog/article/finding-stocks-where-analysts-and-market-disagree)

~~~python
estimate_fields = ["TR.EPSMean", "TR.EPSMean.Date", "TR.EPSNumOfEst", "TR.EPSStdDev"]
estimate_params = {"Period": REVISION_PERIOD, "WP": REVISION_WINDOW}
~~~

### Daily historical consensus calculation date

Source: [LSEG Developer Community — retrieving historical TR.EPSMean](https://community.developers.lseg.com/discussion/134049/retrieving-historical-tr-epsmean-data)

~~~python
fields = ['TR.EPSMean(Period=2026).calcdate','TR.EPSMean(Period=2026)']
parameters = {'SDate':'2024-04-22', 'Frq':'D', 'EDate':'2026-04-22'}
~~~

Source: [LSEG Developer Community — multiple download date and data](https://community.developers.lseg.com/discussion/comment/30858/)

~~~python
fields = ['TR.EPSMean(Period=FY1, Frq=D, SDate=20170101, EDate=20171231).Calcdate', 'TR.EPSMean(Period=FY1, Frq=D, SDate=20170101, EDate=20171231).Value']
~~~

### Currency and scale parameters

Source: [LSEG Developer Community — fundamental quarterly data for stocks](https://community.developers.lseg.com/discussion/28420/fundamental-quarterly-data-for-stocks)

~~~python
df, e = ek.get_data(['MMM.N'], fields, {'Period':'FQ0', 'SDate':0, 'EDate':-79, 'Frq':'FQ', 'Curn':'USD'})
~~~

Source: [LSEG Developer article — Workspace Excel company tearsheet in Python](https://developers.lseg.com/en/article-catalog/article/workspace-excel-company-tearsheet-python-part-1)

~~~python
dfFY0 = ld.get_data(RIC, 'TR.RevenueActValue;TR.EBITDAActValue;TR.EPSActValue', {'Period':'FY0', 'Scale':'6', 'Curn':df.at[0,'Currency']})
~~~

### Excel output headers and date alignment

Source: [LSEG Developer Community — sort get_data by instrument and fiscal period](https://community.developers.lseg.com/discussion/56148/how-to-sort-an-ek-get-data-request-by-instrument-and-fiscal-period-absolute)

~~~text
=TR("GOOGL.OQ";"TR.BSPeriodEndDate.fperiod;TR.BSPeriodEndDate;TR.ISPeriodEndDate;TR.EPSActValue;TR.EPSActValue.date";"Period=FQ0 Frq=FQ SDate=1982-01-01 EDate=2019-12-31 CH=Fd RH=IN";B2)
~~~

That page's accepted answer warns that fields from different categories are not guaranteed to align on the date. Use explicit field suffixes and key-based matching in local probes.

## Minimal safe probe matrix

The matrix is intentionally small and read-only. Each row is a separate request or an explicitly paired comparison. Preserve the exact request, raw response/error, row and instrument counts, returned column names, and hashes. Do not impute, deduplicate, parse timezone, multiply by a scale factor, or write back to research tables during the probe.

| Probe | Request candidates | What it can test | What it cannot establish |
|---|---|---|---|
| A. Actual period/date outputs | TR.EPSActValue, .date, .announcedate, .calcdate, .periodenddate, .fperiod; Period=FQ0, Frq=FQ, fixed SDate/EDate | Which suffixes return, raw date/period relationships, missing/error behavior | Announcement timezone, default actual basis, as-released vintage |
| B. Consensus period/date outputs | TR.EPSMean, .calcdate, .date, .periodenddate, .fperiod, TR.EPSStdDev, TR.EPSNumOfEst; Period=FQ1, Frq=D around a known event | Daily observation pattern, repeated values, period matching, dispersion/count availability | Intraday cutoff, revision policy, same basis/scale as actual |
| C. Currency/scale paired calls | Base actual/mean/std; default, Curn=USD, Scale=3, Scale=6 in separate calls | Parameter acceptance and whether values change | Scale-code meaning, economic conversion, common scale |
| D. Actual selector isolation | One call each: default; ActType=Reported; then Comparable, Restated, GoForward | Selector acceptance and raw value/date differences | Default selector definition; original disclosure value |
| E. Alternative actual measures | TR.DilutedEPSExclExtra; TR.EPSNormalizeddil separately, with returned period/date suffixes if accepted | Availability and raw relationship to target actual | Same measure as IBES TR.EPSActValue; meaning of normalized |
| F. Adjustment negative check | TR.EPSActValue(Adjusted=0) and/or TR.EPSMean(Adjusted=0) only if accepted | Parameter accepted/ignored/rejected | Split policy; identical output is not proof of unadjusted data |
| G. PIT pattern | TR.EPSMean(Period=FQ1).calcdate and value with Frq=D, historical SDate/EDate | Whether documented EPSMean daily/calcdate pattern reproduces | Original vintage, activation time, complete revision history |
| H. Timezone negative check | TR.EPSActValue.origtimezone, optionally .announcedate.origtimezone, isolated | Whether actual-specific timezone suffix is exposed | A failure does not imply ET/UTC; retain raw timestamp |

For every failed exploratory candidate, keep the exact error and mark it “not accepted in this probe,” rather than converting it into a content-definition claim. For every successful candidate, report availability and observed values only until DIB or LSEG content support confirms basis, units, adjustment, and vintage.

## Open evidence gaps

The official public evidence reviewed here does not map these dataset concepts to the three requested TR fields: currency code returned per value, unit/scale meaning, basic versus diluted versus normalized basis, comparable/GAAP/operating basis, default actual selector, split/corporate-action adjustment, effective/activation date, and original versus restated point-in-time vintage. The official IBES Actuals product page confirms these concepts exist at dataset level, but that is not a DIB definition for TR.EPSActValue, TR.EPSMean, or TR.EPSStdDev.

Until field-level definitions are captured, keep EPS fields as traceable engineering observations. Do not unlock raw EPS levels or standardized surprise for a formal model/backtest solely because a request succeeds or because actual, mean, and standard deviation happen to be numerically related in a small sample.


from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, Mm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(r"D:\3 Study 学习资料\2E 金融研二(上)资料\BMF5360 Machine Learning in Investments\Group project_2.0")
OUT = ROOT / "deliverables" / "Midterm_Report_GroupX_v4.docx"
FIG = ROOT / "_artifact_work" / "figures"

FONT = "Times New Roman"
BODY_PT = 11
LINE = 1.2
TABLE_PT = 9
HEAD_SHADE = "E7E6E6"


def set_font(run, size=BODY_PT, bold=False, italic=False):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(k), FONT)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def para(doc, text, size=BODY_PT, bold=False, italic=False, align=None, space_after=5, space_before=0, line=LINE):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing = line
    if align is not None:
        p.alignment = align
    set_font(p.add_run(text), size, bold, italic)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    pf = p.paragraph_format
    pf.space_before = Pt(8 if level == 1 else 4)
    pf.space_after = Pt(3)
    pf.line_spacing = LINE
    pf.keep_with_next = True
    set_font(p.add_run(text), 12 if level == 1 else 11, bold=True)
    return p


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def cell_text(cell, value, bold=False, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    if align is not None:
        p.alignment = align
    set_font(p.add_run(str(value)), TABLE_PT, bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def table(doc, caption, headers, rows, widths, numeric_cols=()):
    cap = para(doc, caption, size=10, bold=True, space_after=2, space_before=4, line=1.0)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        shade(t.rows[0].cells[i], HEAD_SHADE)
        cell_text(t.rows[0].cells[i], h, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cell_text(cells[i], val, align=WD_ALIGN_PARAGRAPH.RIGHT if i in numeric_cols else None)
    for ri, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
            if ri < len(t.rows) - 1:
                for cp in row.cells[i].paragraphs:
                    cp.paragraph_format.keep_with_next = True
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.line_spacing = 1.0
    return t


def figure(doc, path, caption, width=6.0):
    fp = doc.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_after = Pt(2)
    fp.paragraph_format.space_before = Pt(4)
    fp.paragraph_format.keep_with_next = True
    fp.add_run().add_picture(str(path), width=Inches(width))
    return para(doc, caption, size=10, space_after=8, line=1.05)


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Mm(210)
    sec.page_height = Mm(297)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(1))
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.line_spacing = LINE

    # Cover page
    for _ in range(6):
        para(doc, "", space_after=0)
    para(doc, "From Factor Signals to Investable Strategies", size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    para(doc, "A Machine-Learning Stock-Selection Pilot in US Large Caps", size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
    para(doc, "BMF5360 Machine Learning in Investments", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "Midterm Report", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
    para(doc, "Group #", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "[Member name, NUS e-number] · [Member name, NUS e-number] · [Member name, NUS e-number]", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
    para(doc, "Semester 1, AY2026/27 · Week 5", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    for _ in range(10):
        para(doc, "", space_after=0)
    para(doc, "AI tool declaration. We used OpenAI Codex and Claude Code to draft data-collection and modelling scripts, run audits, and improve the clarity of this report. All design decisions, data-processing approvals, and interpretations are our own, and we are responsible for the content and quality of the submitted work.", size=10, space_after=0)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # 1 Project idea
    heading(doc, "1. Project idea")
    heading(doc, "1.1 Motivation and strategy", 2)
    para(doc, "Predicting which stocks will outperform is a central problem in quantitative equity investing. Decades of evidence show that characteristics such as momentum, volatility, liquidity, and the state of a stock's industry carry information about future returns, but each signal is weak on its own, unstable across market regimes, and partly redundant with the others. Traditional factor models combine a handful of these signals with fixed linear weights; machine learning can combine many more of them and capture their interactions, and Gu, Kelly and Xiu (2020) report that tree and neural-network models raise out-of-sample predictive R² well above that of linear benchmarks on US stocks. The same flexibility makes it easy to fit noise, and a model that sees data unavailable at decision time will look excellent in a backtest and fail in trading. The value of ML in investing therefore depends as much on the evaluation protocol as on the model, and that protocol is what this project sets out to build and test.")
    para(doc, "The strategy we are developing is a factor-based US large-cap equity fund. On a point-in-time S&P 500 universe, it ranks stocks by their predicted probability of beating SPY over the next 21 trading sessions and converts the ranking into a beta-hedged portfolio under rules fixed before the backtest runs. The research approach is factor analysis. We construct candidate factors ourselves from daily prices and quotes, macro series, industry-level market data, earnings events, and company fundamentals, evaluate each factor family under one protocol, and let a supervised model combine them into a single prediction (Table 1). The final fund may be long-only with a hedge or long-short; that choice, the universe, and the holding rule will be frozen at the next stage.")
    table(doc, "Table 1. Factor families in the research library",
          ["Family", "Source", "Factors constructed (status)"],
          [["Market", "Daily prices, volume, quotes", "Momentum over 1 to 60 sessions, realised and idiosyncratic volatility, beta, volume, dollar volume, quoted spread, SPY momentum and volatility (built; in the pilot model)"],
           ["Macro state", "FRED daily series", "VIX level, changes and percentile; 3-month, 2-year, 10-year yields and changes; broad dollar; term spreads (built; in the pilot model)"],
           ["Industry state", "Sector ETFs, pilot pool", "SOXX and XLK momentum, volatility, drawdown and return relative to SPY; pool breadth, dispersion, equal-weight return (built; in the pilot model)"],
           ["Event", "28,995 earnings announcements, consensus snapshots", "Standardised surprise against the last pre-announcement consensus, sign and size, announcement timing, transfer to linked companies (built on an event panel; transfer tested; surprise to join the stock-level set)"],
           ["Fundamental", "Original-filing segment revenue, R&D, balance sheet", "R&D intensity, segment-revenue shares, size, value, profitability, investment, leverage (fields probed; cleaning in progress)"]],
          [1.0, 1.55, 3.72])

    heading(doc, "1.2 Why machine learning, and how the pilot was chosen", 2)
    para(doc, "Predicting whether a stock beats SPY over 21 sessions is a supervised classification problem, and a model can weigh dozens of factors jointly under a fixed protocol. Logistic Regression is the transparent benchmark because its coefficients can be reviewed and it tests whether a more flexible model adds anything; Random Forest is the matched non-linear comparison. The ML contribution is measured as a pre-specified improvement over the simpler model with dependence-aware uncertainty, never as the best number found by searching the validation set.")
    para(doc, "We screened ten candidate strategies against LSEG Workspace data access on 7 September 2026, from asset allocation and post-earnings drift to FX carry and the volatility risk premium, and pursued the two that passed the data gate with the fewest open questions. We first built an earnings-event panel on the full S&P 500 to test whether one company's earnings surprise predicts the returns of linked companies; in validation, the receiver's own market factors carried the signal and the transfer factors added 0.001 to the within-event rank correlation, so surprise was returned to the factor library as a candidate and the transfer channel was set aside. We then built the stock-level pipeline that this report presents, and ran it first on a 49-company thematic universe, the US AI supply chain, where a small pool lets every stage be audited before scaling to the full universe. Figure 1 summarises that pipeline.")
    figure(doc, FIG / "fig1_pipeline.png", "Figure 1. Research pipeline. Every stage uses only information dated before the decision session, and the test period stays sealed until the design is frozen.", width=5.4)

    # 2 Data
    heading(doc, "2. Data")
    heading(doc, "2.1 Sources and point-in-time universe", 2)
    para(doc, "Every result in this report depends on whether the data reproduce what an investor could have known on the decision date, so we built the data layer before any model, on two rules. The universe must include companies that later left the index or were delisted, and every field must carry the date on which it became public. Market and company data come from LSEG Workspace. We rebuilt a point-in-time S&P 500 universe of 781 RICs, including 154 delisted identifiers, by rolling the 7 September 2026 constituent list backwards through 751 index-change records, and the cleaned daily panel holds about 2.08 million company-day rows of adjusted prices, volume, and bid/ask quotes from 2015 to 2026. Macro series come from FRED, and SOXX and XLK prices describe the semiconductor and technology industry state.")

    heading(doc, "2.2 Pilot universe", 2)
    para(doc, "The pilot pool holds 49 companies from the parent universe, assigned by economic role to seven supply-chain groups, from GPUs and accelerators, AI semiconductors, memory and storage, and servers and networking to cloud and software, data-centre power and cooling, and robotics and autonomy. Assignment used business-segment disclosures and company descriptions; historical returns, volatility, size, liquidity, and model output played no part, and delisting is not an exclusion (JNPR.N^G25 stays in the registry with its membership ending on 8 July 2025). Each company carries a membership span, an evidence source, and a delisting flag. Direct dated segment evidence is on file for nine companies, and the remaining 40 are documented from current disclosures pending the historical extension described in Section 5. The resulting stock-day panel holds 110,829 company-day observations.")

    heading(doc, "2.3 Factors and information boundary", 2)
    para(doc, "Every feature uses data available through session F-1, one session before the formation date F. The 21 market variables cover stock momentum over 1 to 60 sessions, 20- and 60-session volatility, 126-session beta and idiosyncratic volatility, 20- and 60-session medians of volume, dollar volume, and quoted spread, and SPY momentum and volatility. The 20 macro variables cover VIX, the 3-month, 2-year, and 10-year yields and their changes, the broad dollar index, and two term spreads, each aligned to the SPY calendar and lagged one session. The 21 industry-state variables are SOXX and XLK momentum, volatility, and drawdown; SOXX return relative to SPY; and pool breadth (the share of pool stocks rising or above their moving averages), dispersion, and equal-weight return. Variables with no coverage in the early training years (six credit spreads and 18 variables on ETFs launched after 2015) were excluded beforehand so that their missingness could not act as a calendar label.")

    heading(doc, "2.4 Target, split, and leakage controls", 2)
    para(doc, "The target is y = 1 when the stock's 21-session total return from the close of F exceeds SPY's, and y = 0 otherwise. Training covers 2015-2020 with 2,334 non-overlapping anchors across 39 companies and 444 formation dates; validation covers 2021-2022 with 992 anchors, 44 companies, and 89 formation dates, of which 50.7% are positive. The test period, 1 January 2023 to 30 June 2026, is sealed; its labels have not been opened, predicted, or scored. Samples are never shuffled, and each company keeps at most one anchor per 21-session block so that labels do not overlap within a company. Hyperparameters are chosen only on expanding folds inside the training period, ending in 2018, 2019, and 2020, and every training row must exit before a fold's validation start, which gives a full 21-session purge.")

    heading(doc, "2.5 Processing rules", 2)
    para(doc, "Every action that changes data was listed, approved, and logged before it ran. Empty alignment cells created by the wide-format price download (167,134) and vendor padding rows after a delisting were quarantined with reason codes, not deleted. Pre-market announcements enter at the same-day close and after-market announcements at the next close. Missing values stay missing in every source table; the only imputation is inside the model pipeline, a median fitted on training rows with a missing indicator, and there is no zero fill, forward fill, or winsorization anywhere in the data layer.")

    # 3 Baseline models
    heading(doc, "3. Baseline models")
    para(doc, "The baselines are deliberately simple. Their job is to set the bar that any later model must clear on the same sample. Both models run as fixed scikit-learn pipelines on the same anchors. Logistic Regression uses a training-fitted median imputer with missing indicators, a standard scaler, C = 1.0, balanced class weights, and the lbfgs solver. Random Forest uses 200 trees and an 18-point grid over depth, leaf size, and feature fraction, selected for each factor set by mean ROC-AUC across the three purged training folds, with Brier score as the tie-breaker. Four nested factor sets were registered before any validation score was read, namely market only (21 factors), market plus macro (41), market plus industry state (42), and all three (62), and the validation period was used once, for the final comparison of the eight resulting models. Metrics are ROC-AUC, PR-AUC, Brier score, accuracy at 0.50, and the rank information coefficient (IC) between predicted probability and realised excess return, pooled and by formation date.")

    # 4 Preliminary results
    heading(doc, "4. Preliminary results")
    para(doc, "All results below are validation-period comparisons on frozen samples. They are evidence about the factors and the pipeline, not a performance claim for the fund.")
    heading(doc, "4.1 Classification", 2)
    table(doc, "Table 2. Validation results, 2021-2022, 992 anchors, same sample for every row",
          ["Factor set", "Model", "ROC-AUC", "PR-AUC", "Brier", "Accuracy", "Pooled rank IC", "Mean daily rank IC"],
          [["Market", "Logistic", "0.5395", "0.5421", "0.2511", "51.61%", "0.140", "0.078"],
           ["Market", "Random Forest", "0.5231", "0.5511", "0.2543", "51.92%", "0.074", "0.063"],
           ["Market + macro", "Logistic", "0.5616", "0.5524", "0.2508", "56.45%", "0.174", "0.086"],
           ["Market + macro", "Random Forest", "0.5362", "0.5454", "0.2496", "52.92%", "0.096", "0.036"],
           ["Market + industry state", "Logistic", "0.5799", "0.5745", "0.2476", "54.54%", "0.187", "0.089"],
           ["Market + industry state", "Random Forest", "0.5391", "0.5422", "0.2520", "54.84%", "0.104", "0.087"],
           ["All three", "Logistic", "0.5820", "0.5456", "0.2700", "55.65%", "0.186", "0.093"],
           ["All three", "Random Forest", "0.5349", "0.5481", "0.2506", "52.62%", "0.088", "0.037"]],
          [1.45, 1.0, 0.62, 0.62, 0.58, 0.68, 0.66, 0.66], numeric_cols=(2, 3, 4, 5, 6, 7))
    para(doc, "The market-plus-industry-state Logistic model is the primary model. Its ROC-AUC of 0.5799 is 0.0405 above the market-only Logistic benchmark, and a paired bootstrap clustered by formation date (2,000 draws) gives a 95% interval of [0.0007, 0.0801] for that lift; the AUC is 0.5906 in 2021 and 0.6213 in 2022, so the direction holds in both years. It also leads on PR-AUC, Brier score, and both rank ICs. The all-factor Logistic model has a slightly higher AUC but a worse Brier score (0.2700 against 0.2476), so its probabilities are less reliable. Random Forest is below Logistic in every factor set.")
    figure(doc, FIG / "fig2_quintiles.png", "Figure 2. Realised 21-session excess return over SPY by quintile of the primary model's predicted probability, validation period, pooled across formation dates (992 anchors).", width=5.0)
    para(doc, "Figure 2 shows how the probability translates into returns. Sorting the 992 validation anchors into quintiles of predicted probability gives a mean 21-session excess return of -2.43% in the lowest quintile and +1.90% in the highest, with the middle quintiles between them. The signal has two layers. The industry-state variables take the same value for every stock on a formation date, so they move the date-level probability, that is, whether the pool as a whole is likely to beat SPY over the next month. The ranking of stocks within a date comes from the 21 market variables; there the mean daily rank IC is 0.089 and the average top-minus-bottom quintile spread is 0.74% per 21 sessions. A portfolio rule therefore has to use both layers, sizing exposure by the date-level signal and picking names by the within-date rank.")

    heading(doc, "4.2 First portfolio mapping", 2)
    para(doc, "To test whether the ranking converts into returns, we fixed a portfolio rule before running it. Every 21 sessions the primary model scores all eligible stocks, and those with predicted probability of at least 0.55 are held, subject to a 10% single-stock cap, a 30% group cap, a 10% volatility target, a 50% one-way turnover cap per rebalance, and a SPY beta hedge, with half the quoted spread charged on every trade. A date with fewer than five qualifying stocks is skipped and, as the rule was written, the existing positions are kept until the next rebalance. Figure 3 shows the result over 2021-2022: a cumulative net return of -3.10% with a realised beta of 0.064, annualised volatility of 9.78%, a maximum drawdown of -13.15%, and spread costs of 0.24% of initial NAV, against +13.11% for SPY. An independent daily re-accounting of cash, positions, trades, and costs passed 13 of 14 checks with one non-blocking warning.")
    figure(doc, FIG / "fig3_nav.png", "Figure 3. Pre-registered hedged portfolio versus SPY, 2021-2022, growth of 1 net of spread costs. Shaded bands mark the four windows in which a rebalance was skipped and the previous positions were held for a further 21 sessions.", width=5.2)
    para(doc, "The loss has a single source. In the 21 sessions after each executed rebalance, the horizon the model was trained for, the hedged portfolio compounded to +14.6% net. All of the loss came from the four shaded windows, which compounded to -15.4%. On those dates the model's highest probability had fallen to between 0.43 and 0.59, fewer than five names qualified, and the rule held the previous positions for another 21 sessions; the largest of these windows, January to February 2022, coincided with the start of the technology drawdown. The model's date-level signal was right on those dates, and the rule ignored it. The hedge worked as designed (realised beta 0.064) and costs are immaterial. Because this attribution was made on the validation period, the corrected rule (move to cash or hedge-only when a date is skipped, and select by within-date rank rather than a fixed probability threshold) is pre-registered for the next stage and is not re-run on the same data.")

    # 5 Limitations and next stage
    heading(doc, "5. Limitations and next stage")
    para(doc, "Three limitations govern how the numbers should be read. The validation sample has 89 formation dates, and stocks on the same date share market shocks, so every interval above uses date or block resampling. The probabilities are not calibrated (expected calibration error 0.057 overall and 0.125 in 2022), which is why the portfolio rule uses ranks and a fixed threshold rather than probability levels. And the pilot pool is small by design; dated historical membership evidence is complete for nine of the 49 companies and the remaining 40 are documented from current disclosures, so scaling to the full point-in-time universe is the first step below.")
    para(doc, "The next stage has five steps. First, extend dated membership evidence to the remaining pilot companies and run the same pipeline on the full 781-RIC point-in-time universe. Second, complete the fundamental family and the standard controls, and add earnings surprise to the stock-level factor set, so that all factor families are evaluated on one sample by rank IC, quantile spreads, and turnover. Third, pre-register portfolio rule v2 (cash or hedge-only on skipped dates, top 20% by within-date rank at equal weight, monthly rebalance, one-way costs of 0 to 25 basis points) with settings chosen on training data only. Fourth, choose the final fund strategy on six written criteria (economic mechanism, point-in-time feasibility, sample adequacy, pre-specified ML contribution, an implementable rule with costs, and one untouched test period). Fifth, freeze the design and open the sealed test period once.")

    # References
    heading(doc, "References")
    refs = [
        "Gu, S., Kelly, B., and Xiu, D. (2020). Empirical asset pricing via machine learning. Review of Financial Studies, 33(5), 2223-2273.",
        "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. (Purged cross-validation and embargo.)",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Federal Reserve Bank of St. Louis. FRED economic data: VIXCLS, DGS3MO, DGS2, DGS10, DTWEXBGS.",
        "LSEG Workspace and LSEG Data Library for Python. Daily prices, quotes, index constituents, earnings events, consensus estimates, and segment fundamentals.",
    ]
    for r in refs:
        p = para(doc, r, size=10, space_after=3, line=1.1)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)

    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    build()

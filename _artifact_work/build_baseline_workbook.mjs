import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workDir = path.resolve(".");
const projectRoot = path.resolve(workDir, "..");
const outputDir = path.join(projectRoot, "deliverables", "BMF5360_Baseline_Package");
const previewDir = path.join(workDir, "xlsx_previews");
const data = JSON.parse(await fs.readFile(path.join(workDir, "delivery_data.json"), "utf8"));
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const wb = Workbook.create();
const sheets = {};
for (const name of ["Summary", "Company Universe", "Selection Logic", "Factor Dictionary", "Model Results", "Logistic Coefficients", "Processing Audit", "Sources"]) {
  sheets[name] = wb.worksheets.add(name);
  sheets[name].showGridLines = false;
}

const NAVY = "#17365D";
const BLUE = "#2F75B5";
const PALE_BLUE = "#D9EAF7";
const PALE_GREEN = "#E2F0D9";
const PALE_AMBER = "#FFF2CC";
const PALE_RED = "#FCE4D6";
const LIGHT = "#F3F6F9";
const TEXT = "#1F2937";
const MUTED = "#5B6573";
const FONT = "Arial";

function colName(n) {
  let s = "";
  while (n > 0) { n--; s = String.fromCharCode(65 + (n % 26)) + s; n = Math.floor(n / 26); }
  return s;
}

function title(sheet, text, endCol) {
  sheet.getRange(`A2:${endCol}2`).format.borders = { bottom: { style: "thin", color: BLUE } };
  sheet.getRange("A2").values = [[text]];
  sheet.getRange("A2").format.font = { name: FONT, size: 15, bold: true, color: NAVY };
  sheet.getRange("A3").values = [["BMF5360 | validation-only baseline package | test period remains sealed"]];
  sheet.getRange("A3").format.font = { name: FONT, size: 9, italic: true, color: MUTED };
}

function header(sheet, address) {
  const r = sheet.getRange(address);
  r.format.fill = NAVY;
  r.format.font = { name: FONT, size: 10, bold: true, color: "#FFFFFF" };
  r.format.horizontalAlignment = "center";
  r.format.verticalAlignment = "center";
  r.format.wrapText = true;
  r.format.borders = { insideVertical: { style: "thin", color: "#FFFFFF" }, bottom: { style: "thin", color: "#FFFFFF" } };
}

function body(sheet, address) {
  const r = sheet.getRange(address);
  r.format.font = { name: FONT, size: 10, color: TEXT };
  r.format.verticalAlignment = "center";
  r.format.borders = { insideHorizontal: { style: "thin", color: "#E5E7EB" }, bottom: { style: "thin", color: "#D1D5DB" } };
}

function section(sheet, address, text) {
  const range = sheet.getRange(address);
  range.format.fill = PALE_BLUE;
  range.format.font = { name: FONT, size: 11, bold: true, color: NAVY };
  sheet.getRange(address.split(":")[0]).values = [[text]];
}

// Summary
{
  const s = sheets["Summary"];
  title(s, "AI Stock Daily Logistic Baseline - Submission Workbook", "N");
  section(s, "A5:F5", "Exploratory pooled classification result");
  s.getRange("A6:B13").values = [
    ["Model", "Technical + macro Logistic"],
    ["Prediction target", "Future 21-session stock excess return vs SPY > 0"],
    ["Validation ROC-AUC", null],
    ["Pooled AUC lift vs technical-only", null],
    ["Validation accuracy at 0.50", null],
    ["Validation observations", null],
    ["Raw / transformed features", null],
    ["Test status", "SEALED - no predictions or metrics"],
  ];
  s.getRange("B8").formulas = [["='Model Results'!H7"]];
  s.getRange("B9").formulas = [["='Model Results'!H7-'Model Results'!H5"]];
  s.getRange("B10").formulas = [["='Model Results'!K7"]];
  s.getRange("B11").formulas = [["='Model Results'!D7"]];
  s.getRange("B12").formulas = [["='Logistic Coefficients'!K5&\" / \"&'Logistic Coefficients'!K6"]];
  s.getRange("A6:A13").format.fill = LIGHT;
  s.getRange("A6:A13").format.font = { name: FONT, size: 10, bold: true, color: TEXT };
  s.getRange("A6:B13").format.borders = { insideHorizontal: { style: "thin", color: "#D7DEE7" }, bottom: { style: "thin", color: "#B8C4D1" } };
  s.getRange("B8:B10").format.numberFormat = "0.0000";
  s.getRange("B10").format.numberFormat = "0.00%";
  s.getRange("B11").format.numberFormat = "#,##0";
  s.getRange("A15:B15").values = [["Logistic specification", "Value"]];
  header(s, "A15:B15");
  s.getRange("A16:B21").values = [
    ["Pipeline", data.model.pipeline],
    ["C", data.model.params.C],
    ["Class weight", data.model.params.class_weight],
    ["Solver / iterations", `${data.model.params.solver} / ${data.model.params.max_iter}`],
    ["Training period", data.sample.train_years],
    ["Validation period", data.sample.validation_years],
  ];
  body(s, "A16:B21");
  s.getRange("H5:I5").values = [["Logistic model", "ROC-AUC"]];
  header(s, "H5:I5");
  s.getRange("H6:H9").values = [["Technical only"], ["Technical + macro"], ["Technical + AI state"], ["Full state"]];
  s.getRange("I6:I9").formulas = [["='Model Results'!H5"], ["='Model Results'!H7"], ["='Model Results'!H9"], ["='Model Results'!H11"]];
  s.getRange("I6:I9").format.numberFormat = "0.000";
  body(s, "H6:I9");
  const chart = s.charts.add("bar", s.getRange("H5:I9"));
  chart.title = "Validation ROC-AUC by Logistic feature set";
  chart.titleTextStyle.typeface = FONT;
  chart.titleTextStyle.fontSize = 12;
  chart.hasLegend = false;
  chart.xAxis = { axisType: "textAxis", textStyle: { typeface: FONT, fontSize: 9 } };
  chart.yAxis = { numberFormatCode: "0.00", numberFormatSourceLinked: false, textStyle: { typeface: FONT, fontSize: 9 } };
  chart.setPosition("H11", "N25");
  s.getRange("A23:F23").values = [["Interpretation after timing-versus-selection audit", null, null, null, null, null]];
  s.getRange("A24:B27").values = [
    ["Meaning", "0.5616 is pooled classification AUC: it mixes common date-level state/timing and stock selection; it is not a portfolio return."],
    ["Block inference", "The macro-minus-technical pooled AUC lift is 0.0221; its 21-session block-bootstrap 95% CI is -0.0394 to +0.0704."],
    ["Stock selection", "Same-date pair-weighted AUC lift is 0.0062 with CI -0.0066 to +0.0218: no verified cross-sectional improvement."],
    ["Test protection", "Development evidence only; 2023-2026 test labels remain unopened."],
  ];
  s.getRange("A23:F23").format.fill = PALE_AMBER;
  s.getRange("A23").format.font = { name: FONT, bold: true, color: "#7F6000" };
  s.getRange("A24:A27").format.fill = LIGHT;
  s.getRange("A24:A27").format.font = { name: FONT, size: 10, bold: true, color: TEXT };
  s.getRange("B24:B27").format.font = { name: FONT, size: 10, color: TEXT };
  s.getRange("B24:B27").format.wrapText = true;
  s.getRange("A1:N30").format.font.name = FONT;
  s.getRange("A:A").format.columnWidth = 29;
  s.getRange("B:B").format.columnWidth = 72;
  s.getRange("C:G").format.columnWidth = 3;
  s.getRange("H:H").format.columnWidth = 24;
  s.getRange("I:I").format.columnWidth = 13;
  s.getRange("J:N").format.columnWidth = 11;
  s.tabColor = NAVY;
}

// Company universe
{
  const s = sheets["Company Universe"];
  title(s, "Registered AI Supply-Chain Company Universe", "O");
  const headers = ["RIC", "Company", "Primary group", "Inclusion reason", "Member from", "Member to", "Delisted RIC", "PIT evidence", "PIT eligible", "Local evidence rows", "Source type", "Source title", "Source URL", "Reason code", "PIT limitation"];
  s.getRange(`A5:${colName(headers.length)}5`).values = [headers];
  header(s, `A5:${colName(headers.length)}5`);
  const rows = data.registry.map(x => [x.ric, x.canonical_name, x.primary_group, x.inclusion_reason, x.member_from, x.member_to, x.delisted_ric, x.pit_evidence_status, x.pit_eligible_now, x.local_keyword_evidence_rows, x.source_type, x.source_title, x.source_url, x.reason_code, x.pit_limitation]);
  s.getRange(`A6:${colName(headers.length)}${5 + rows.length}`).values = rows;
  body(s, `A6:${colName(headers.length)}${5 + rows.length}`);
  s.getRange(`D6:D${5 + rows.length}`).format.wrapText = true;
  s.getRange(`O6:O${5 + rows.length}`).format.wrapText = true;
  s.getRange(`A6:A${5 + rows.length}`).format.font = { name: FONT, size: 10, bold: true, color: NAVY };
  s.getRange("A:O").format.columnWidth = 14;
  s.getRange("B:B").format.columnWidth = 29;
  s.getRange("C:C").format.columnWidth = 27;
  s.getRange("D:D").format.columnWidth = 48;
  s.getRange("H:H").format.columnWidth = 23;
  s.getRange("K:L").format.columnWidth = 26;
  s.getRange("M:M").format.columnWidth = 38;
  s.getRange("N:N").format.columnWidth = 44;
  s.getRange("O:O").format.columnWidth = 55;
  s.freezePanes.freezeRows(5);
  s.freezePanes.freezeColumns(2);
}

// Selection logic
{
  const s = sheets["Selection Logic"];
  title(s, "Company Selection Logic and Evidence Quality", "F");
  section(s, "A5:F5", "Selection process");
  const headers = ["Step", "Rule", "Implementation", "Financial rationale", "Bias control"];
  s.getRange("A7:E7").values = [headers]; header(s, "A7:E7");
  const rows = data.selection_steps.map(x => [x.step, x.rule, x.implementation, x.rationale, x.bias_control]);
  s.getRange(`A8:E${7 + rows.length}`).values = rows; body(s, `A8:E${7 + rows.length}`);
  s.getRange(`B8:E${7 + rows.length}`).format.wrapText = true;
  const start = 10 + rows.length;
  section(s, `A${start}:F${start}`, "Seven supply-chain groups");
  s.getRange(`A${start + 2}:E${start + 2}`).values = [["Primary group", "Description", "Companies", "Direct local PIT", "Provisional static"]];
  header(s, `A${start + 2}:E${start + 2}`);
  const grows = data.group_summary.map(x => [x.primary_group, x.description, x.companies, x.direct_local_pit, x.provisional_static]);
  s.getRange(`A${start + 3}:E${start + 2 + grows.length}`).values = grows;
  body(s, `A${start + 3}:E${start + 2 + grows.length}`);
  const totalRow = start + 3 + grows.length;
  s.getRange(`A${totalRow}:E${totalRow}`).values = [["Total", "", null, null, null]];
  s.getRange(`C${totalRow}`).formulas = [[`=SUM(C${start + 3}:C${totalRow - 1})`]];
  s.getRange(`D${totalRow}`).formulas = [[`=SUM(D${start + 3}:D${totalRow - 1})`]];
  s.getRange(`E${totalRow}`).formulas = [[`=SUM(E${start + 3}:E${totalRow - 1})`]];
  s.getRange(`A${totalRow}:E${totalRow}`).format.fill = PALE_BLUE;
  s.getRange(`A${totalRow}:E${totalRow}`).format.font = { name: FONT, bold: true, color: NAVY };
  s.getRange("A:A").format.columnWidth = 28;
  s.getRange("B:B").format.columnWidth = 40;
  s.getRange("C:E").format.columnWidth = 32;
  s.freezePanes.freezeRows(7);
}

// Factor dictionary
{
  const s = sheets["Factor Dictionary"];
  title(s, "Technical and Macro Factor Dictionary", "J");
  const headers = ["Order", "Feature", "Included", "Category", "Source", "Lookback / status", "Unit", "Calculation", "Why included", "Timing / exclusion reason"];
  s.getRange("A5:J5").values = [headers]; header(s, "A5:J5");
  const rows = data.features.map(x => [x.order, x.feature, x.included, x.category, x.source, x.window, x.unit, x.operation, x.logic, x.timing]);
  s.getRange(`A6:J${5 + rows.length}`).values = rows; body(s, `A6:J${5 + rows.length}`);
  s.getRange(`B6:B${5 + rows.length}`).format.font = { name: FONT, size: 10, bold: true, color: NAVY };
  s.getRange(`D6:J${5 + rows.length}`).format.wrapText = true;
  s.getRange(`C6:C${5 + rows.length}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "FALSE", format: { fill: PALE_RED, font: { color: "#9C0006", bold: true } } });
  s.getRange("A:A").format.columnWidth = 8;
  s.getRange("B:B").format.columnWidth = 34;
  s.getRange("C:C").format.columnWidth = 12;
  s.getRange("D:D").format.columnWidth = 22;
  s.getRange("E:E").format.columnWidth = 33;
  s.getRange("F:G").format.columnWidth = 20;
  s.getRange("H:J").format.columnWidth = 43;
  s.freezePanes.freezeRows(5);
  s.freezePanes.freezeColumns(2);
}

// Model results
{
  const s = sheets["Model Results"];
  title(s, "Validation Metrics - All Frozen Baseline Comparisons", "O");
  const headers = ["Model key", "Feature set", "Model", "Rows", "Instruments", "Formation dates", "Positive rate", "ROC-AUC", "PR-AUC", "Brier", "Accuracy @0.50", "Pooled rank IC", "Mean daily rank IC", "Top-bottom spread", "AUC change vs technical Logistic"];
  s.getRange("A4:O4").values = [headers]; header(s, "A4:O4");
  const rows = data.metrics.map(x => [x.model_key, x.feature_set, x.model, x.rows, x.instruments, x.formation_sessions, x.positive_label_rate, x.roc_auc, x.pr_auc, x.brier, x.directional_accuracy_at_0_5, x.pooled_rank_ic, x.mean_daily_rank_ic, x.mean_daily_top_minus_bottom_spread, null]);
  s.getRange(`A5:O${4 + rows.length}`).values = rows;
  s.getRange("O5").formulas = [["=H5-$H$5"]];
  s.getRange(`O5:O${4 + rows.length}`).fillDown();
  body(s, `A5:O${4 + rows.length}`);
  s.getRange(`G5:O${4 + rows.length}`).format.numberFormat = "0.0000";
  s.getRange(`G5:G${4 + rows.length}`).format.numberFormat = "0.00%";
  s.getRange(`K5:K${4 + rows.length}`).format.numberFormat = "0.00%";
  s.getRange(`N5:N${4 + rows.length}`).format.numberFormat = "0.00%";
  s.getRange("A7:O7").format.fill = PALE_GREEN;
  s.getRange("A7:O7").format.font = { name: FONT, size: 10, bold: true, color: "#375623" };
  s.getRange("A:A").format.columnWidth = 36;
  s.getRange("B:B").format.columnWidth = 27;
  s.getRange("C:C").format.columnWidth = 18;
  s.getRange("D:O").format.columnWidth = 17;
  s.freezePanes.freezeRows(4);
}

// Coefficients
{
  const s = sheets["Logistic Coefficients"];
  title(s, "Technical + Macro Logistic Coefficients", "K");
  const headers = ["Absolute rank", "Transformed feature", "Base feature", "Missing indicator", "Standardized coefficient", "Absolute coefficient", "Odds multiplier per +1 SD", "Direction"];
  s.getRange("A5:H5").values = [headers]; header(s, "A5:H5");
  s.getRange("J5:K6").values = [["Raw features", data.model.raw_feature_count], ["Transformed features", data.model.transformed_feature_count]];
  s.getRange("J5:J6").format.fill = LIGHT;
  s.getRange("J5:J6").format.font = { name: FONT, size: 10, bold: true, color: TEXT };
  s.getRange("K5:K6").format.numberFormat = "0";
  const rows = data.coefficients.map(x => [x.absolute_rank, x.transformed_feature, x.base_feature, x.missing_indicator, x.coefficient_standardized, x.absolute_coefficient, null, null]);
  s.getRange(`A6:H${5 + rows.length}`).values = rows;
  s.getRange("G6").formulas = [["=EXP(E6)"]]; s.getRange(`G6:G${5 + rows.length}`).fillDown();
  s.getRange("H6").formulas = [["=IF(E6>0,\"Positive\",IF(E6<0,\"Negative\",\"Zero\"))"]]; s.getRange(`H6:H${5 + rows.length}`).fillDown();
  body(s, `A6:H${5 + rows.length}`);
  s.getRange(`E6:G${5 + rows.length}`).format.numberFormat = "0.0000";
  s.getRange(`B6:C${5 + rows.length}`).format.font = { name: FONT, size: 10, color: NAVY };
  s.getRange("A:A").format.columnWidth = 13;
  s.getRange("B:C").format.columnWidth = 42;
  s.getRange("D:H").format.columnWidth = 22;
  s.getRange("I:I").format.columnWidth = 3;
  s.getRange("J:J").format.columnWidth = 24;
  s.getRange("K:K").format.columnWidth = 12;
  s.freezePanes.freezeRows(5);
}

// Processing audit
{
  const s = sheets["Processing Audit"];
  title(s, "Material Data Processing and Audit Trail", "G");
  const headers = ["Stage", "Input", "Rule", "Before", "After", "Missing / excluded treatment", "Status"];
  s.getRange("A5:G5").values = [headers]; header(s, "A5:G5");
  const rows = data.processing.map(x => [x.stage, x.input, x.rule, x.before, x.after, x.missing_or_excluded, x.status]);
  s.getRange(`A6:G${5 + rows.length}`).values = rows; body(s, `A6:G${5 + rows.length}`);
  s.getRange(`A6:A${5 + rows.length}`).format.font = { name: FONT, size: 10, bold: true, color: NAVY };
  s.getRange(`B6:G${5 + rows.length}`).format.wrapText = true;
  s.getRange(`G6:G${5 + rows.length}`).conditionalFormats.add("containsText", { text: "passed", format: { fill: PALE_GREEN, font: { color: "#375623", bold: true } } });
  s.getRange("A:A").format.columnWidth = 23;
  s.getRange("B:B").format.columnWidth = 33;
  s.getRange("C:C").format.columnWidth = 58;
  s.getRange("D:E").format.columnWidth = 28;
  s.getRange("F:F").format.columnWidth = 48;
  s.getRange("G:G").format.columnWidth = 26;
  s.freezePanes.freezeRows(5);
}

// Sources
{
  const s = sheets["Sources"];
  title(s, "Sources, Reproduction Notes and Submission Caveats", "D");
  s.getRange("A5:B5").values = [["Artifact", "Project-relative path"]]; header(s, "A5:B5");
  const rows = data.sources.map(x => [x.artifact, x.path]);
  s.getRange(`A6:B${5 + rows.length}`).values = rows; body(s, `A6:B${5 + rows.length}`);
  const r0 = 8 + rows.length;
  section(s, `A${r0}:D${r0}`, "Submission notes");
  s.getRange(`A${r0 + 2}:B${r0 + 7}`).values = [
    ["Main result", "Technical + macro Logistic validation ROC-AUC = 0.561559"],
    ["Interpretation", "AUC measures ranking/classification quality; it is not return, Sharpe or R-squared."],
    ["Evidence limit", "Only 9/49 companies have direct local point-in-time role evidence; 40/49 use provisional static descriptions."],
    ["Missing data", "Source NA values remain NA. Median imputation and missing indicators are fitted inside the training pipeline only."],
    ["Corporate actions", "Adjusted prices already include supplier corporate-action treatment; no second stock-split adjustment is applied."],
    ["Test protection", "2023-01-01 to 2026-06-30 target data remain sealed."],
  ];
  s.getRange(`A${r0 + 2}:A${r0 + 7}`).format.fill = LIGHT;
  s.getRange(`A${r0 + 2}:A${r0 + 7}`).format.font = { name: FONT, bold: true, color: TEXT };
  s.getRange(`A${r0 + 2}:B${r0 + 7}`).format.wrapText = true;
  s.getRange("A:A").format.columnWidth = 28;
  s.getRange("B:B").format.columnWidth = 105;
}

// Apply restrained base typography and row heights.
for (const name of Object.keys(sheets)) {
  const s = sheets[name];
  const used = s.getUsedRange();
  used.format.font.name = FONT;
  used.format.verticalAlignment = "center";
  used.format.autofitRows();
}

wb.recalculate();

const checks = await wb.inspect({
  kind: "formula",
  maxChars: 8000,
  options: { maxResults: 200 },
});
await fs.writeFile(path.join(workDir, "workbook_formula_inspect.ndjson"), checks.ndjson ?? String(checks), "utf8");
const errors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  maxChars: 4000,
  options: { maxResults: 100 },
});
await fs.writeFile(path.join(workDir, "workbook_error_scan.ndjson"), errors.ndjson ?? String(errors), "utf8");

for (const name of Object.keys(sheets)) {
  const preview = await wb.render({ sheetName: name, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${name.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const out = await SpreadsheetFile.exportXlsx(wb);
const outputPath = path.join(outputDir, "BMF5360_AI_Baseline_Model_Calculations.xlsx");
await out.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: Object.keys(sheets), formulaInspect: path.join(workDir, "workbook_formula_inspect.ndjson"), previewDir }));

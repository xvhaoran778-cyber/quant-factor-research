#!/usr/bin/env python3
"""Build the multi-factor PDF report from saved JSON results."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "multifactor_combination_results.json"
OUT = ROOT / "output" / "pdf" / "multifactor_combination_report.pdf"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def weight_text(weights: dict[str, float]) -> str:
    labels = {
        "correlation_breakdown": "相关性突变",
        "low_volatility_20": "低波动",
        "liquidity_strength_20": "流动性强度",
        "downside_volatility_20": "下行波动",
    }
    return " / ".join(f"{labels[k]} {v:.0%}" for k, v in weights.items() if v)


def metrics_row(idx: int, row: dict) -> list[str]:
    m = row["metrics"]
    return [
        str(idx),
        weight_text(row["weights"]),
        pct(m["total_return"]),
        pct(m["annual_return"]),
        f"{m['sharpe']:.2f}",
        pct(m["max_drawdown"]),
        f"{m.get('win_rate', 0) * 100:.1f}%",
        f"{m.get('profit_loss_ratio', 0):.2f}",
        str(m.get("max_drawdown_duration_weeks", 0)),
        pct(m.get("benchmark_total_return", 0)),
        pct(m.get("excess_return", 0)),
        str(m["closed_trades"]),
    ]


def yearly_table(row: dict) -> list[list[str]]:
    yearly = row["metrics"].get("yearly_returns", {})
    return [["年份", *map(str, yearly.keys())], ["收益", *[pct(v) for v in yearly.values()]]]


def p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text), style)


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    pdfmetrics.registerFont(TTFont("CNFont", font_path))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNTitle", parent=styles["Title"], fontName="CNFont", fontSize=20, leading=26))
    styles.add(ParagraphStyle(name="CNH2", parent=styles["Heading2"], fontName="CNFont", fontSize=14, leading=20, spaceBefore=10))
    styles.add(ParagraphStyle(name="CN", parent=styles["BodyText"], fontName="CNFont", fontSize=10.5, leading=16))

    recent = data["recent_backtests"]
    long = data["long_validation"]
    best_recent = recent[0]
    robust = max(long, key=lambda row: row["metrics"]["total_return"])

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm)
    story = [
        Paragraph("多因子组合挖掘报告", styles["CNTitle"]),
        Paragraph(f"生成时间：{data['generated_at'].replace('T', ' ')}；数据目录：{data['market_dir']}", styles["CN"]),
        Spacer(1, 8),
        Paragraph("结论摘要", styles["CNH2"]),
        Paragraph(
            "本次用 0.1 步长枚举 282 组多因子权重，先在 2024-2026 做快速预筛，再对前 30 组做真实周频 next-open 回测，并将近期前 8 组放到 2019-2023 验证。近期最高收益组合并不稳健；更值得继续研究的是长期验证收益最高的折中组合。",
            styles["CN"],
        ),
        Paragraph(f"近期最高收益：{weight_text(best_recent['weights'])}，2024-2026 总收益 {pct(best_recent['metrics']['total_return'])}，最大回撤 {pct(best_recent['metrics']['max_drawdown'])}。", styles["CN"]),
        Paragraph(f"稳健折中候选：{weight_text(robust['weights'])}，2019-2023 总收益 {pct(robust['metrics']['total_return'])}，最大回撤 {pct(robust['metrics']['max_drawdown'])}；2024-2026 对应收益见近期排名表。", styles["CN"]),
        Paragraph("因子公式与原理", styles["CNH2"]),
    ]

    cell = ParagraphStyle(name="Cell", parent=styles["CN"], fontSize=8.2, leading=11)
    factor_rows = [[p(value, cell) for value in row] for row in [
        ["因子", "方向", "公式", "原理"],
        ["相关性突变", "高好", "corr20(个股收益, 市场收益) - corr60(个股收益, 市场收益)", "寻找和市场同步性下降、可能出现独立行情的股票。"],
        ["低波动", "低好", "std20(日收益)", "控制波动，避免组合集中在短期剧烈波动股票。"],
        ["流动性强度", "高好", "mean20(收盘价 * 成交量)", "偏向更活跃、更容易成交的股票，降低小样本和成交风险。"],
        ["下行波动", "低好", "std20(min(日收益, 0))", "专门惩罚下跌波动，作为回撤控制信号。"],
    ]]
    story.append(Table(factor_rows, colWidths=[2.4 * cm, 1.5 * cm, 4.8 * cm, 7.3 * cm], style=[
        ("FONTNAME", (0, 0), (-1, -1), "CNFont"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))
    story += [
        Spacer(1, 8),
        Paragraph("组合打分公式", styles["CNH2"]),
        Paragraph("Score = Σ 权重_i × 横截面排名_i。正向因子按值越大排名越高；负向因子先反向排名。每周五收盘后产生信号，下一交易日开盘成交；持仓 5 只，使用市场过滤、手续费和滑点。", styles["CN"]),
        Paragraph("2024-2026 近期真实回测 Top 10", styles["CNH2"]),
    ]

    top_recent_rows = [["#", "权重", "总收益", "年化", "夏普", "最大回撤", "胜率", "盈亏比", "回撤周", "基准", "超额", "交易数"]]
    top_recent_rows += [metrics_row(i, row) for i, row in enumerate(recent[:10], 1)]
    story.append(Table(top_recent_rows, colWidths=[0.6 * cm, 4.7 * cm, 1.4 * cm, 1.3 * cm, 1.0 * cm, 1.4 * cm, 1.1 * cm, 1.1 * cm, 1.0 * cm, 1.2 * cm, 1.2 * cm, 1.0 * cm], style=[
        ("FONTNAME", (0, 0), (-1, -1), "CNFont"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story += [PageBreak(), Paragraph("2019-2023 长期验证", styles["CNH2"])]
    long_rows = [["#", "权重", "总收益", "年化", "夏普", "最大回撤", "胜率", "盈亏比", "回撤周", "基准", "超额", "交易数"]]
    long_rows += [metrics_row(i, row) for i, row in enumerate(long, 1)]
    story.append(Table(long_rows, colWidths=[0.6 * cm, 4.7 * cm, 1.4 * cm, 1.3 * cm, 1.0 * cm, 1.4 * cm, 1.1 * cm, 1.1 * cm, 1.0 * cm, 1.2 * cm, 1.2 * cm, 1.0 * cm], style=[
        ("FONTNAME", (0, 0), (-1, -1), "CNFont"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story += [
        Spacer(1, 8),
        Paragraph("稳健折中候选逐年收益", styles["CNH2"]),
        Table(yearly_table(robust), style=[
            ("FONTNAME", (0, 0), (-1, -1), "CNFont"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]),
        Spacer(1, 8),
        Paragraph("下一步建议", styles["CNH2"]),
        Paragraph("不要把近期最高收益组合直接加入当前候选。优先把稳健折中候选作为研究候选继续跑 2012-2018、2024-2026 和全周期，并加入换手、行业暴露、单票贡献检查。当前结果仍属于研究结果，不构成实盘稳定盈利结论。", styles["CN"]),
    ]
    doc.build(story)


if __name__ == "__main__":
    main()

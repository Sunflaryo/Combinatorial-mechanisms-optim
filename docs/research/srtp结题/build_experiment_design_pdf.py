from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon


BASE = Path(__file__).resolve().parent
OUT = BASE / "初步实验设计方案_HRL_AMA.pdf"


def register_fonts():
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    bold_candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    ]
    font = next((p for p in candidates if p.exists()), None)
    bold = next((p for p in bold_candidates if p.exists()), font)
    if not font:
        return "Helvetica", "Helvetica-Bold"
    pdfmetrics.registerFont(TTFont("CN", str(font)))
    pdfmetrics.registerFont(TTFont("CN-Bold", str(bold)))
    return "CN", "CN-Bold"


FONT, FONT_BOLD = register_fonts()


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "CNTitle",
            fontName=FONT_BOLD,
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "CNSubtitle",
            fontName=FONT,
            fontSize=10.5,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5E6A75"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            "H1CN",
            fontName=FONT_BOLD,
            fontSize=14.5,
            leading=19,
            textColor=colors.HexColor("#1F4E79"),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "H2CN",
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#2F5D62"),
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyCN",
            fontName=FONT,
            fontSize=9.8,
            leading=15,
            textColor=colors.HexColor("#27313A"),
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallCN",
            fontName=FONT,
            fontSize=8.2,
            leading=11.5,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            "CalloutCN",
            fontName=FONT,
            fontSize=9.4,
            leading=14,
            leftIndent=8,
            rightIndent=8,
            borderColor=colors.HexColor("#BFD7EA"),
            borderWidth=0.8,
            borderPadding=7,
            backColor=colors.HexColor("#F3F8FC"),
            textColor=colors.HexColor("#243B53"),
            spaceBefore=5,
            spaceAfter=8,
        )
    )
    return styles


S = make_styles()


def p(text, style="BodyCN"):
    return Paragraph(text, S[style])


def bullet(text):
    return Paragraph("• " + text, S["BodyCN"])


def cell(text, style="SmallCN"):
    return Paragraph(text, S[style])


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(1.7 * cm, 1.1 * cm, "HRL-AMA 初步实验设计方案")
    canvas.drawRightString(A4[0] - 1.7 * cm, 1.1 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def table(data, col_widths, header_rows=1, font_size=8.0):
    t = Table(data, colWidths=col_widths, repeatRows=header_rows, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT),
                ("FONTNAME", (0, 0), (-1, header_rows - 1), FONT_BOLD),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 3),
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
                ("BACKGROUND", (0, header_rows), (-1, -1), colors.HexColor("#F8FAFC")),
                ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8E2EA")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def architecture_diagram():
    d = Drawing(500, 155)
    box_fill = colors.HexColor("#EAF4F4")
    edge = colors.HexColor("#2F5D62")
    accent = colors.HexColor("#1F4E79")

    def box(x, y, w, h, title, sub):
        d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=box_fill, strokeColor=edge, strokeWidth=1))
        d.add(String(x + w / 2, y + h - 16, title, fontName=FONT_BOLD, fontSize=9, textAnchor="middle", fillColor=accent))
        d.add(String(x + w / 2, y + 13, sub, fontName=FONT, fontSize=7.3, textAnchor="middle", fillColor=colors.HexColor("#374151")))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#6B7280"), strokeWidth=1.2))
        d.add(Polygon([x2, y2, x2 - 6, y2 + 3, x2 - 6, y2 - 3], fillColor=colors.HexColor("#6B7280"), strokeColor=colors.HexColor("#6B7280")))

    box(10, 96, 105, 42, "历史市场状态", "预算 / bids / revenue")
    box(145, 96, 105, 42, "高层 Controller", "阶段性目标 g")
    box(280, 96, 105, 42, "低层 Worker", "AMA 参数 w, φ")
    box(145, 28, 105, 42, "AMA Solver", "清算分配与支付")
    box(280, 28, 105, 42, "结果反馈", "revenue / budget")
    box(410, 62, 78, 42, "下一轮状态", "state update")

    arrow(115, 117, 145, 117)
    arrow(250, 117, 280, 117)
    arrow(332, 96, 198, 70)
    arrow(250, 49, 280, 49)
    arrow(385, 49, 410, 78)
    d.add(String(448, 112, "下一轮回到历史状态", fontName=FONT, fontSize=7, textAnchor="middle", fillColor=colors.HexColor("#6B7280")))

    d.add(String(250, 146, "图 1  实验中的 HRL-AMA 闭环", fontName=FONT_BOLD, fontSize=10, textAnchor="middle", fillColor=colors.HexColor("#17324D")))
    return d


def revenue_chart():
    # Preliminary numbers copied from current ourwork PDF; used only to motivate design.
    groups = ["Uniform", "Lognormal", "LLG", "Non-stationary"]
    vcg = [27.68, 28.15, 17.31, 28.84]
    static = [27.29, 32.38, 19.99, 32.80]
    flat = [30.80, 32.61, 20.34, 32.59]
    series = [("VCG", vcg, "#8BA6A9"), ("Static AMA", static, "#8EC07C"), ("Flat DRL", flat, "#3A7CA5")]
    d = Drawing(500, 220)
    d.add(String(250, 204, "图 2  现有初步结果可作为 baseline 设计依据", fontName=FONT_BOLD, fontSize=10, textAnchor="middle", fillColor=colors.HexColor("#17324D")))
    left, bottom, width, height = 45, 38, 400, 135
    maxv = 36
    d.add(Line(left, bottom, left, bottom + height, strokeColor=colors.HexColor("#9CA3AF")))
    d.add(Line(left, bottom, left + width, bottom, strokeColor=colors.HexColor("#9CA3AF")))
    for tick in [0, 10, 20, 30]:
        y = bottom + height * tick / maxv
        d.add(Line(left - 3, y, left + width, y, strokeColor=colors.HexColor("#E5E7EB"), strokeWidth=0.5))
        d.add(String(left - 8, y - 3, str(tick), fontName=FONT, fontSize=7, textAnchor="end", fillColor=colors.HexColor("#6B7280")))
    group_w = width / len(groups)
    bar_w = 9
    for gi, g in enumerate(groups):
        gx = left + gi * group_w + 18
        for si, (_, vals, color) in enumerate(series):
            val = vals[gi]
            h = height * val / maxv
            x = gx + si * (bar_w + 3)
            d.add(Rect(x, bottom, bar_w, h, fillColor=colors.HexColor(color), strokeColor=None))
        d.add(String(gx + 16, bottom - 12, g, fontName=FONT, fontSize=7.1, textAnchor="middle", fillColor=colors.HexColor("#374151")))
    for i, (name, _, color) in enumerate(series):
        x = 125 + i * 95
        d.add(Rect(x, 12, 9, 9, fillColor=colors.HexColor(color), strokeColor=None))
        d.add(String(x + 14, 13, name, fontName=FONT, fontSize=7.5, fillColor=colors.HexColor("#374151")))
    d.add(String(250, 188, "说明：这里不是最终 HRL 结论，只说明 VCG / Static AMA / Flat DRL 应作为第一批 baseline。", fontName=FONT, fontSize=7.5, textAnchor="middle", fillColor=colors.HexColor("#6B7280")))
    return d


def timeline_chart():
    d = Drawing(500, 150)
    d.add(String(250, 136, "图 3  初步实验推进顺序", fontName=FONT_BOLD, fontSize=10, textAnchor="middle", fillColor=colors.HexColor("#17324D")))
    left, top = 42, 112
    phases = [
        ("1. 复现实验", 0, 85, "#B7D7E8"),
        ("2. HRL 实现", 85, 105, "#CDEAC0"),
        ("3. 消融与压力测试", 190, 120, "#FFE5B4"),
        ("4. 整理成稿", 310, 105, "#E5D4EF"),
    ]
    d.add(Line(left, 72, left + 420, 72, strokeColor=colors.HexColor("#94A3B8"), strokeWidth=1))
    for name, start, length, color in phases:
        x = left + start
        d.add(Rect(x, 62, length, 28, rx=4, ry=4, fillColor=colors.HexColor(color), strokeColor=colors.HexColor("#64748B"), strokeWidth=0.6))
        d.add(String(x + length / 2, 73, name, fontName=FONT_BOLD, fontSize=7.7, textAnchor="middle", fillColor=colors.HexColor("#1F2937")))
    for i, label in enumerate(["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]):
        x = left + i * 105
        d.add(Line(x, 58, x, 96, strokeColor=colors.HexColor("#CBD5E1"), strokeWidth=0.5))
        d.add(String(x, 43, label, fontName=FONT, fontSize=7, textAnchor="middle", fillColor=colors.HexColor("#64748B")))
    d.add(String(250, 22, "先把“能复现、能比较、能解释”做稳，再补更复杂的 strategic / scalability 压力测试。", fontName=FONT, fontSize=8, textAnchor="middle", fillColor=colors.HexColor("#374151")))
    return d


def metric_radar_like():
    d = Drawing(500, 160)
    d.add(String(250, 145, "图 4  评价指标不只看 revenue", fontName=FONT_BOLD, fontSize=10, textAnchor="middle", fillColor=colors.HexColor("#17324D")))
    labels = [
        ("Revenue", 250, 112),
        ("IC regret", 354, 78),
        ("IR violation", 314, 32),
        ("Runtime", 186, 32),
        ("Adaptation", 146, 78),
    ]
    points = [(250, 108), (340, 78), (308, 40), (192, 40), (160, 78)]
    flat_points = [coord for point in points for coord in point]
    d.add(Polygon(flat_points, fillColor=colors.HexColor("#EAF4F4"), strokeColor=colors.HexColor("#3A7CA5"), strokeWidth=1.2))
    center = (250, 74)
    for x, y in points:
        d.add(Line(center[0], center[1], x, y, strokeColor=colors.HexColor("#CBD5E1"), strokeWidth=0.6))
    for text, x, y in labels:
        d.add(String(x, y, text, fontName=FONT_BOLD, fontSize=8, textAnchor="middle", fillColor=colors.HexColor("#374151")))
    d.add(String(250, 8, "收入提高只有在 IC/IR、运行时间和非平稳适应能力都说得过去时，才算是机制设计上的进展。", fontName=FONT, fontSize=8, textAnchor="middle", fillColor=colors.HexColor("#4B5563")))
    return d


def build():
    frame = Frame(1.65 * cm, 1.7 * cm, A4[0] - 3.3 * cm, A4[1] - 3.4 * cm, id="normal")
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=1.65 * cm,
        rightMargin=1.65 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.6 * cm,
        title="HRL-AMA 初步实验设计方案",
        author="Codex",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=header_footer)])

    story = []
    story.append(p("HRL-AMA 初步实验设计方案", "CNTitle"))
    story.append(p("围绕 repeated combinatorial auction 中的 AMA 参数动态调节", "CNSubtitle"))
    story.append(
        p(
            "这份方案是一个可以继续和老师讨论的初版。它的目标不是一次性把所有实验都做满，而是先把问题拆清楚：我们到底要验证什么、和谁比较、用哪些指标判断结果是否有意义。",
            "CalloutCN",
        )
    )

    story.append(p("1. 实验要回答的核心问题", "H1CN"))
    story.append(
        p(
            "当前 ourwork 的核心不是让 bidder 学会复杂报价，而是让 auctioneer 在 Affine Maximizer Auction (AMA) 这个机制结构内学习参数。AMA 负责每轮的 dominant-strategy incentive compatibility (DSIC) 和 individual rationality (IR)，强化学习负责在重复拍卖中根据历史状态调节参数、提高长期收入。",
        )
    )
    story.append(bullet("问题一：动态调节 AMA 参数，是否真的比静态 VCG / Static AMA 更能提高 cumulative revenue？"))
    story.append(bullet("问题二：这种收入提升是否仍然保持每轮 exact DSIC 和 IR，而不是靠牺牲激励性质换来的？"))
    story.append(bullet("问题三：当 valuation distribution 发生变化、budget 被逐步消耗时，HRL 是否比 Flat DRL 更会适应？"))
    story.append(bullet("问题四：学到的参数是否能解释，还是只是另一个黑箱策略？"))

    story.append(Spacer(1, 5))
    story.append(architecture_diagram())

    story.append(p("2. 实验环境设置", "H1CN"))
    story.append(
        p(
            "建议第一阶段先保持小规模、可复现、可穷举。不要一开始追求大规模，因为 AMA solver 的 allocation enumeration 会随 item 数量指数增长。先把 n=3 bidders、m=2 或 3 items、T=30 rounds 做扎实，再谈扩展。",
        )
    )
    env_data = [
        [cell("模块"), cell("初步设置"), cell("为什么这样设")],
        [cell("Auction horizon"), cell("T = 30 rounds"), cell("足够观察预算消耗和非平稳变化，又不会让训练成本太高。")],
        [cell("Bidders / items"), cell("n = 3, m ∈ {2, 3}"), cell("可以 exact enumerate allocations，便于核对 DSIC/IR 和 solver 正确性。")],
        [cell("Valuation distributions"), cell("Uniform, Lognormal, LLG, Non-stationary"), cell("覆盖平稳、重尾、局部-全局竞争和分布漂移四类基本压力。")],
        [cell("State features"), cell("time, budget ratio, bid mean/std, recent revenue"), cell("不直接使用当轮 bid，避免破坏 per-round DSIC 的承诺顺序。")],
        [cell("Action"), cell("AMA bidder weights w 与 item/allocation weights φ"), cell("RL 只调机制参数，清算仍由 deterministic AMA solver 完成。")],
    ]
    story.append(table(env_data, [3.2 * cm, 5.0 * cm, 7.0 * cm]))

    story.append(p("3. Baseline 与对照组", "H1CN"))
    story.append(
        p(
            "baseline 需要分层安排。先用最容易解释的机制建立底线，再加入学习型机制。这样即使 HRL 暂时没有明显胜出，也能知道问题出在动态学习、参数表达，还是环境本身。",
        )
    )
    base_data = [
        [cell("组别"), cell("机制"), cell("作用")],
        [cell("B0"), cell("Random allocation / random feasible AMA parameters"), cell(" sanity check，只用于确认环境没有明显错误。")],
        [cell("B1"), cell("VCG"), cell("exact DSIC/IR 的经典静态基准。")],
        [cell("B2"), cell("Static AMA"), cell("检验“只优化静态 AMA 参数”能做到什么程度。")],
        [cell("B3"), cell("Flat DRL + AMA"), cell("单层 RL 调参，作为 HRL 的直接对照。")],
        [cell("B4"), cell("HRL-AMA"), cell("高层目标 + 低层 AMA 参数，重点看非平稳场景是否更稳。")],
        [cell("可选"), cell("RegretNet / Lottery AMA"), cell("如果时间够，用作文献近邻对照；若实现成本高，可先只做文字比较。")],
    ]
    story.append(table(base_data, [2.0 * cm, 5.0 * cm, 8.2 * cm]))
    story.append(Spacer(1, 5))
    story.append(revenue_chart())

    story.append(PageBreak())
    story.append(p("4. 实验任务分组", "H1CN"))
    exp_data = [
        [cell("实验"), cell("要验证什么"), cell("具体做法"), cell("预期读法")],
        [
            cell("E1: Solver 与激励性质核对"),
            cell("AMA solver 是否正确，DSIC/IR 是否真的为结构性质。"),
            cell("随机生成 valuation/bid profile；对每个 bidder 做 misreport search；记录 IC regret 和 IR violation。"),
            cell("IC regret 应接近 0；若不为 0，优先查 payment rule 或参数提交顺序。"),
        ],
        [
            cell("E2: 静态 vs 动态调参"),
            cell("Flat DRL/HRL 是否比 VCG、Static AMA 提高 cumulative revenue。"),
            cell("四种 valuation distributions 下跑 200 episodes，报告 mean ± std。"),
            cell("如果只在某些分布上胜出，也要解释为什么，比如 LLG 或 non-stationary 更需要动态性。"),
        ],
        [
            cell("E3: HRL 消融实验"),
            cell("高层 controller 是否真的有用。"),
            cell("比较 Flat DRL、HRL K=5、HRL without budget features、HRL without recent revenue。"),
            cell("如果 HRL 没胜出，至少看 adaptation lag 和参数变化是否更合理。"),
        ],
        [
            cell("E4: 非平稳压力测试"),
            cell("估值分布突变时，机制是否能恢复。"),
            cell("T/2 时 valuation scale 从 U(0,1) 切到 U(0,3)，观察 revenue 和 AMA 参数轨迹。"),
            cell("重点不是单点最高收入，而是 shift 后几轮内能否调整回来。"),
        ],
        [
            cell("E5: 策略性跨轮风险"),
            cell("per-round DSIC 不等于 full dynamic IC。"),
            cell("构造 bid shading agent，前几轮压低 bid，观察是否影响后续参数和总 utility。"),
            cell("这不是要完全解决 dynamic IC，而是诚实报告 limitation 的严重程度。"),
        ],
    ]
    story.append(table(exp_data, [2.4 * cm, 3.7 * cm, 5.1 * cm, 4.0 * cm], font_size=7.5))

    story.append(p("5. 评价指标", "H1CN"))
    story.append(metric_radar_like())
    metric_data = [
        [cell("指标"), cell("怎么计算"), cell("为什么重要")],
        [cell("Cumulative revenue"), cell("每个 episode 内 sum_t payments_t"), cell("主目标，但不能单独看。")],
        [cell("Per-round IC regret"), cell("max misreport utility - truthful utility"), cell("验证每轮 DSIC 是否真的没有被 RL 破坏。")],
        [cell("IR violation"), cell("truthful utility < 0 的幅度和频率"), cell("避免机制靠负效用榨取收入。")],
        [cell("Adaptation lag"), cell("分布突变后 revenue/参数恢复所需轮数"), cell("HRL 的核心卖点之一。")],
        [cell("Budget exhaustion pattern"), cell("bidder budget 随时间下降曲线"), cell("防止早期过度榨取导致后期市场崩掉。")],
        [cell("Runtime"), cell("每轮清算时间、训练时间"), cell("说明 exact solver 的可扩展边界。")],
        [cell("Parameter interpretability"), cell("w 和 φ 的轨迹、和 bidder/item 特征的关系"), cell("让结果不只是黑箱收益表。")],
    ]
    story.append(table(metric_data, [4.0 * cm, 5.2 * cm, 6.0 * cm]))

    story.append(p("6. 推荐输出图表", "H1CN"))
    story.append(bullet("主结果表：不同 distribution 下，VCG / Static AMA / Flat DRL / HRL-AMA 的 cumulative revenue mean ± std。"))
    story.append(bullet("Revenue 曲线：每轮平均 revenue，尤其标出 non-stationary shift 的时间点。"))
    story.append(bullet("IC/IR 检查图：各机制的 IC regret 和 IR violation，证明 revenue 提升不是靠破坏激励性质。"))
    story.append(bullet("参数轨迹图：bidder weights w_t 与 item weights φ_t 随时间变化，看策略是否有解释性。"))
    story.append(bullet("消融表：去掉高层、去掉 budget state、去掉 recent revenue 后性能如何变化。"))
    story.append(Spacer(1, 4))
    story.append(timeline_chart())

    story.append(p("7. 风险与备选解释", "H1CN"))
    risk_data = [
        [cell("风险"), cell("可能原因"), cell("处理方式")],
        [cell("HRL 不如 Flat DRL"), cell("环境太短或高层目标没有学到稳定语义。"), cell("先报告为 preliminary；增加 non-stationary 强度，或调整 K、goal dimension。")],
        [cell("Static AMA 已经很强"), cell("分布平稳，动态调参空间不大。"), cell("把重点转到非平稳、预算消耗和 adaptation lag。")],
        [cell("IC regret 非零"), cell("参数提交顺序错了，或 payment rule 实现有 bug。"), cell("先修 solver 和 timeline，不要急着调 RL。")],
        [cell("训练不稳定"), cell("reward 尺度、budget clipping、action range 设计不稳。"), cell("先固定 Flat DRL，再逐步加 HRL；记录 seed variance。")],
        [cell("算得太慢"), cell("allocation enumeration 指数增长。"), cell("小规模先完成；大规模写成 limitation 或 MIP/DP future work。")],
    ]
    story.append(table(risk_data, [3.5 * cm, 5.3 * cm, 6.4 * cm]))

    story.append(p("8. 最小可交付版本", "H1CN"))
    story.append(
        p(
            "如果时间紧，建议先完成一个最小可交付版本：E1 + E2 + E3 的小规模实验。也就是说，先证明 AMA solver 和 IC/IR 没问题，再比较 VCG、Static AMA、Flat DRL 和一个初版 HRL-AMA。只要这三件事讲清楚，后续再扩展 non-stationary stress test 和 strategic cross-round test 就会顺很多。",
            "CalloutCN",
        )
    )
    story.append(bullet("第一优先级：复现当前 VCG / Static AMA / Flat DRL 结果，并保存随机种子。"))
    story.append(bullet("第二优先级：补上 HRL-AMA 结果，哪怕先只在 Uniform 和 Non-stationary 两个环境跑。"))
    story.append(bullet("第三优先级：把每个结果都配 IC regret / IR violation，不让老师觉得我们只是在追 revenue。"))
    story.append(bullet("第四优先级：画参数轨迹，解释 RL 到底学到了什么。"))

    story.append(p("9. 一句话总结", "H1CN"))
    story.append(
        p(
            "实验设计的重点不是证明“RL 一定比所有机制都强”，而是更耐心地回答：在 exact-DSIC 的 AMA 结构里，动态调参什么时候有用、为什么有用、代价是什么、还有哪些 incentive 和 scalability 问题没有解决。",
            "CalloutCN",
        )
    )

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()

#!/usr/bin/env python3
"""Markdown -> PDF for the flight reports. Headings, tables, lists, bold, code.

    python3 md2pdf.py in.md out.pdf

Deliberately small: it handles the subset the reports are written in rather
than being a general converter, so a report cannot silently lose a table to
a parser it was not tested against.
"""
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

INK = colors.HexColor("#1a202c")
MUTE = colors.HexColor("#4a5568")
RULE = colors.HexColor("#cbd5e0")
HEAD = colors.HexColor("#edf2f7")
ACC = colors.HexColor("#2b6cb0")

ss = getSampleStyleSheet()


def st(name, size, leading, **kw):
    kw.setdefault("fontName", "Helvetica")
    kw.setdefault("textColor", INK)
    return ParagraphStyle(name, parent=ss["BodyText"], fontSize=size,
                          leading=leading, alignment=TA_LEFT, **kw)


S = {
    "h1": st("h1", 19, 23, fontName="Helvetica-Bold", spaceBefore=2,
             spaceAfter=8, textColor=INK),
    "h2": st("h2", 13.5, 17, fontName="Helvetica-Bold", spaceBefore=14,
             spaceAfter=5, textColor=ACC),
    "h3": st("h3", 11.5, 15, fontName="Helvetica-Bold", spaceBefore=11,
             spaceAfter=4),
    "p": st("p", 9.3, 13.2, spaceAfter=5),
    "li": st("li", 9.3, 13.2, leftIndent=11, bulletIndent=2, spaceAfter=2.5),
    "cell": st("cell", 8.0, 10.2, spaceAfter=0),
    "cellb": st("cellb", 8.0, 10.2, fontName="Helvetica-Bold", spaceAfter=0),
    "code": st("code", 8.2, 10.8, fontName="Courier", textColor=MUTE,
               leftIndent=6, spaceAfter=6),
    "small": st("small", 8.0, 11, textColor=MUTE, spaceBefore=6),
}


def inline(t):
    """Markdown emphasis and code to reportlab markup, ampersands escaped."""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Courier" size="8.4">\1</font>', t)
    t = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", t)
    return t


def table(rows):
    head, body = rows[0], rows[1:]
    ncol = max(len(r) for r in rows)
    data = []
    for i, r in enumerate(rows):
        r = list(r) + [""] * (ncol - len(r))
        sty = S["cellb"] if i == 0 else S["cell"]
        data.append([Paragraph(inline(c), sty) for c in r])
    avail = A4[0] - 32 * mm
    # First column wider: these tables name things in it and measure in the
    # rest, so an even split wraps the names and wastes the numbers' space.
    if ncol > 2:
        first = min(avail * 0.34, avail / ncol * 1.9)
        w = [first] + [(avail - first) / (ncol - 1)] * (ncol - 1)
    else:
        w = [avail / ncol] * ncol
    t = Table(data, colWidths=w, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def parse(md):
    out = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            for b in buf:
                out.append(Paragraph(
                    inline(b).replace(" ", "&nbsp;") or "&nbsp;", S["code"]))
            out.append(Spacer(1, 3))
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                out.append(Spacer(1, 2))
                out.append(table(rows))
                out.append(Spacer(1, 7))
            continue
        if ln.startswith("### "):
            out.append(Paragraph(inline(ln[4:]), S["h3"]))
        elif ln.startswith("## "):
            out.append(Paragraph(inline(ln[3:]), S["h2"]))
        elif ln.startswith("# "):
            out.append(Paragraph(inline(ln[2:]), S["h1"]))
        elif re.match(r"^\s*[-*] ", ln):
            out.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", ln)),
                                 S["li"], bulletText="•"))
        elif re.match(r"^\s*\d+\. ", ln):
            n = re.match(r"^\s*(\d+)\. ", ln).group(1)
            out.append(Paragraph(inline(re.sub(r"^\s*\d+\. ", "", ln)),
                                 S["li"], bulletText=f"{n}."))
        elif ln.strip() == "---":
            out.append(Spacer(1, 4))
        elif ln.strip().startswith("*") and ln.strip().endswith("*") \
                and len(ln.strip()) > 2 and "**" not in ln:
            out.append(Paragraph(inline(ln.strip().strip("*")), S["small"]))
        elif ln.strip():
            # A paragraph runs until a blank line.
            buf = [ln]
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not lines[i + 1].startswith(("|", "#", "```"))
                   and not re.match(r"^\s*([-*]|\d+\.) ", lines[i + 1])):
                i += 1
                buf.append(lines[i])
            out.append(Paragraph(inline(" ".join(buf)), S["p"]))
        i += 1
    return out


def footer(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 7.5)
    canv.setFillColor(MUTE)
    canv.drawString(16 * mm, 10 * mm, "DroneVPS - map matching report")
    canv.drawRightString(A4[0] - 16 * mm, 10 * mm, f"page {doc.page}")
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.3)
    canv.line(16 * mm, 13 * mm, A4[0] - 16 * mm, 13 * mm)
    canv.restoreState()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    doc = SimpleDocTemplate(
        dst, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=17 * mm,
        title="DroneVPS map matching report", author="DroneVPS")
    doc.build(parse(open(src).read()), onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()

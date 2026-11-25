"""Build the practice-report DOCX from the provided Word template and Markdown draft."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "practice_report" / "ОтчетПример.docx"
SOURCE = ROOT / "practice_report" / "01_initial_report_draft.md"
OUTPUT = ROOT / "practice_report" / "Отчет_по_федеративному_обучению.docx"


def clear_document_body(document: Document) -> None:
    """Remove body content but preserve the template's sections and styles."""
    body = document._element.body
    section_properties = body.sectPr
    for child in list(body):
        if child is not section_properties:
            body.remove(child)


def set_page_field(paragraph) -> None:
    paragraph.clear()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run = paragraph.add_run()
    run._r.append(field_begin)
    run._r.append(instruction)
    run._r.append(field_end)


def enable_field_updates(document: Document) -> None:
    """Ask Word to refresh PAGE fields when the document is opened."""
    settings = document.settings.element
    update_fields = OxmlElement("w:updateFields")
    update_fields.set(qn("w:val"), "true")
    settings.append(update_fields)


def set_keep_with_next(paragraph) -> None:
    paragraph.paragraph_format.keep_with_next = True


def add_text(paragraph, text: str) -> None:
    """Add Markdown bold and italic spans as Word runs."""
    token = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")
    for part in token.split(text):
        if not part:
            continue
        run = paragraph.add_run()
        if part.startswith("**") and part.endswith("**"):
            run.bold = True
            run.text = part[2:-2]
        elif part.startswith("*") and part.endswith("*"):
            run.italic = True
            run.text = part[1:-1]
        else:
            run.text = part


def add_body_paragraph(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="Times14_РИО2")
    add_text(paragraph, text)


def add_heading(document: Document, text: str, level: int) -> None:
    style_name = "Заголовок1" if level == 1 else "Заголовок2"
    paragraph = document.add_paragraph(style=style_name)
    paragraph.add_run(text.upper() if level == 1 else text)
    set_keep_with_next(paragraph)


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="Times14_РИО2")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.italic = True
    paragraph.paragraph_format.space_after = Pt(10)


def add_figure(document: Document, relative_path: str, caption: str) -> None:
    path = ROOT / "practice_report" / relative_path
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    run.add_picture(str(path), width=Cm(15.5))
    add_caption(document, caption)


def add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = True
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:color"), "000000")
        borders.append(element)
    table._tbl.tblPr.append(borders)
    for cell, text in zip(table.rows[0].cells, headers, strict=True):
        cell.text = text
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "E7E6E6")
        cell._tc.get_or_add_tcPr().append(shading)
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(11)
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row, strict=True):
            cell.text = text
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10.5)
    document.add_paragraph(style="Times14_РИО2")


def parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    cells = lambda line: [part.strip() for part in line.strip().strip("|").split("|")]
    headers = cells(lines[0])
    rows = [cells(line) for line in lines[2:] if line.strip()]
    return headers, rows


def add_markdown_content(document: Document, markdown: str) -> None:
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            index += 1
            continue
        if line.startswith("## "):
            add_heading(document, line[3:], 1)
            index += 1
            continue
        if line.startswith("### "):
            add_heading(document, line[4:], 2)
            index += 1
            continue
        if line.startswith("!["):
            match = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
            caption = ""
            if index + 1 < len(lines) and lines[index + 1].strip().startswith("*"):
                caption = lines[index + 1].strip().strip("*")
                index += 1
            if match:
                add_figure(document, match.group(1), caption)
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            headers, rows = parse_table(table_lines)
            add_table(document, headers, rows)
            continue
        if re.match(r"^[-*] ", line):
            paragraph = document.add_paragraph(style="Times14_РИО2")
            paragraph.paragraph_format.left_indent = Cm(0.75)
            paragraph.paragraph_format.first_line_indent = Cm(-0.5)
            paragraph.add_run("• ")
            add_text(paragraph, line[2:])
            index += 1
            continue
        if re.match(r"^\d+\. ", line):
            paragraph = document.add_paragraph(style="Times14_РИО2")
            paragraph.paragraph_format.left_indent = Cm(0.75)
            paragraph.paragraph_format.first_line_indent = Cm(-0.5)
            marker = re.match(r"^(\d+\.) ", line).group(1)
            paragraph.add_run(f"{marker} ")
            add_text(paragraph, re.sub(r"^\d+\. ", "", line))
            index += 1
            continue
        if line.startswith("`"):
            index += 1
            continue
        add_body_paragraph(document, line)
        index += 1


def add_cover(document: Document) -> None:
    lines = [
        "МИНОБРНАУКИ РОССИИ",
        "САНКТ-ПЕТЕРБУРГСКИЙ ГОСУДАРСТВЕННЫЙ",
        "ЭЛЕКТРОТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ",
        "«ЛЭТИ» ИМ. В.И. УЛЬЯНОВА (ЛЕНИНА)",
        "Кафедра Вычислительной техники",
    ]
    for text in lines:
        paragraph = document.add_paragraph(style="Normal")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(14)
    for _ in range(6):
        document.add_paragraph()
    for text, size in [("ОТЧЕТ", 18), ("по производственной практике", 16)]:
        paragraph = document.add_paragraph(style="Normal")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(18)
    run = paragraph.add_run("Тема: Исследование влияния отравления локальных данных на федеративное обучение при Non-IID-распределении данных")
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)
    for _ in range(9):
        document.add_paragraph()
    for text in ["Санкт-Петербург", "2026"]:
        paragraph = document.add_paragraph(style="Normal")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(14)
    document.add_page_break()


def add_annotation(document: Document) -> None:
    add_heading(document, "Аннотация", 1)
    add_body_paragraph(
        document,
        "В отчёте описан начальный этап разработки программного стенда для исследования федеративного обучения при неоднородном распределении данных. Стенд реализован на Python с применением Flower и PyTorch и предназначен для воспроизводимого сравнения сценариев data poisoning. Представлены реализованные атаки изменения меток и добавления гауссова шума к признакам, методика парных запусков, результаты серии label flipping по трём seed и демонстрационные feature-noise сценарии на MNIST.",
    )
    add_heading(document, "Summary", 1)
    add_body_paragraph(
        document,
        "The report describes the initial development stage of a federated-learning benchmark for non-IID data. The Python-based Flower and PyTorch implementation supports reproducible data-poisoning experiments, including label flipping and Gaussian feature noise. The report presents the paired-run methodology, a three-seed label-flipping series, and demonstrative feature-noise experiments on MNIST.",
    )
    document.add_page_break()


def add_contents(document: Document) -> None:
    add_heading(document, "Содержание", 1)
    for text in [
        "ВВЕДЕНИЕ",
        "Реализованная база стенда",
        "Модель угрозы первого этапа",
        "Программная реализация атаки",
        "Выполненная серия экспериментов",
        "Статус результатов",
        "Дополнительные сценарии отравления признаков",
        "Ограничения начального этапа",
        "Дальнейшее развитие",
    ]:
        paragraph = document.add_paragraph(style="toc 1")
        paragraph.add_run(text)
    document.add_page_break()


def main() -> None:
    shutil.copy2(TEMPLATE, OUTPUT)
    document = Document(OUTPUT)
    clear_document_body(document)
    enable_field_updates(document)
    set_page_field(document.sections[0].footer.paragraphs[0])
    add_cover(document)
    add_annotation(document)
    add_contents(document)
    markdown = SOURCE.read_text(encoding="utf-8")
    add_heading(document, "Введение", 1)
    add_body_paragraph(
        document,
        "Федеративное обучение позволяет обучать общую модель без централизованной передачи локальных данных. Вместе с тем оно затрудняет контроль корректности данных отдельных клиентов. Настоящий отчёт посвящён созданию стенда для анализа отравления локальных данных, а не атак на параметры модели или серверный агрегатор.",
    )
    add_markdown_content(document, markdown)
    document.save(OUTPUT)
    print("Practice report DOCX created")


if __name__ == "__main__":
    main()

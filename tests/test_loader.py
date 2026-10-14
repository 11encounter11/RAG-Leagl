# -*- coding: utf-8 -*-
"""文档加载器的多格式测试(需求 F1.1: 每种格式都有对应解析器, 且都有测试用例)。

覆盖 .txt / .md / .csv / .xlsx / .docx / .pdf 六种格式各一条, 外加一条"不支持的格式"。
只验证"能加载出预期文本 + 打上来源元数据", 不涉及切分与向量化。
"""
import pytest

from app.core.rag.loader import DocumentLoader


@pytest.fixture()
def loader():
    return DocumentLoader()


def _norm(docs):
    """把文档文本拼起来并去掉所有空白, 便于跨格式做稳定断言。"""
    return "".join("".join(d.text.split()) for d in docs).lower()


def test_load_txt(tmp_path, loader):
    p = tmp_path / "劳动法.txt"
    p.write_text("第一条 为了保护劳动者的合法权益。", encoding="utf-8")
    docs = loader.load_file(str(p))
    assert docs and "劳动者的合法权益" in _norm(docs)


def test_load_md(tmp_path, loader):
    p = tmp_path / "note.md"
    p.write_text("# 标题\n\n第一条 测试内容。", encoding="utf-8")
    docs = loader.load_file(str(p))
    assert docs and "测试内容" in _norm(docs)


def test_load_csv_one_doc_per_row(tmp_path, loader):
    p = tmp_path / "faq.csv"
    p.write_text("question,answer\n试用期多久,不超过六个月\n", encoding="utf-8")
    docs = loader.load_file(str(p))
    assert len(docs) == 1
    assert "试用期多久" in _norm(docs)
    assert docs[0].metadata["row"] == 2


def test_load_xlsx(tmp_path, loader):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "table.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["项目", "标准"])
    ws.append(["试用期", "不超过六个月"])
    wb.save(p)

    docs = loader.load_file(str(p))
    assert docs and "不超过六个月" in _norm(docs)
    assert docs[0].metadata.get("sheet")


def test_load_docx(tmp_path, loader):
    docx = pytest.importorskip("docx")
    p = tmp_path / "policy.docx"
    document = docx.Document()
    document.add_paragraph("第一条 为了规范用工管理。")
    document.save(p)

    docs = loader.load_file(str(p))
    assert docs and "规范用工管理" in _norm(docs)


def test_load_pdf(tmp_path, loader):
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "law.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Article 1 test content")
    pdf.save(p)
    pdf.close()

    docs = loader.load_file(str(p))
    # 用 ASCII 文本, 避免默认字体渲染不出中文导致的假失败
    assert docs and "testcontent" in _norm(docs)


def test_metadata_carries_source(tmp_path, loader):
    p = tmp_path / "x.txt"
    p.write_text("内容", encoding="utf-8")
    docs = loader.load_file(str(p))
    assert docs[0].metadata["source_file"] == str(p)
    assert docs[0].metadata["filename"] == "x.txt"


def test_unsupported_extension_raises(tmp_path, loader):
    p = tmp_path / "data.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        loader.load_file(str(p))

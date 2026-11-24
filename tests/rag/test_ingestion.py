"""Document discovery, metadata security, parsing, and chunking tests."""

from dataclasses import replace
from datetime import date
from hashlib import sha256
from pathlib import Path

import pytest
from docx import Document

from ai_support.rag.chunking import chunk_documents, fixed_chunks, heading_chunks, recursive_chunks
from ai_support.rag.config import load_rag_config
from ai_support.rag.loaders import (
    content_hash,
    discover_documents,
    load_descriptors,
    parse_document,
    validate_metadata,
)
from ai_support.rag.sample_data import generate_sample_corpus
from ai_support.rag.schema import DocumentMetadata, SourceDocument


def test_sample_corpus_and_metadata_are_reproducible(tmp_path: Path) -> None:
    config = load_rag_config(Path("configs/rag/student.toml"), tmp_path)
    first_paths, first_cases = generate_sample_corpus(config)
    first_hashes = [content_hash(path) for path in first_paths]
    second_paths, second_cases = generate_sample_corpus(config)

    assert first_paths == second_paths
    assert first_cases == second_cases
    assert first_hashes == [content_hash(path) for path in second_paths]
    assert len(load_descriptors(config)) == 5
    assert {item.metadata.audience for item in load_descriptors(config)} == {"student"}


def test_metadata_policy_rejects_cross_audience_unapproved_expired_and_changed(
    tmp_path: Path,
) -> None:
    config = load_rag_config(Path("configs/rag/student.toml"), tmp_path)
    paths, _ = generate_sample_corpus(config)
    path = paths[0]
    metadata = load_descriptors(config)[0].metadata

    with pytest.raises(ValueError, match="audience isolation"):
        validate_metadata(path, metadata.model_copy(update={"audience": "staff"}), config)
    with pytest.raises(ValueError, match="not approved"):
        validate_metadata(path, metadata.model_copy(update={"approval_status": "draft"}), config)
    with pytest.raises(ValueError, match="expired"):
        expired = metadata.model_copy(update={"review_date": date(2020, 1, 1)})
        validate_metadata(path, expired, config)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_metadata(path, metadata, config)


def test_missing_source_and_sidecar_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source directory"):
        discover_documents(tmp_path / "missing")
    source = tmp_path / "source"
    source.mkdir()
    (source / "article.md").write_text("hello", encoding="utf-8")
    config = replace(load_rag_config(Path("configs/rag/student.toml"), tmp_path), source_dir=source)
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_descriptors(config)


def test_txt_html_docx_and_pdf_dispatch(monkeypatch, tmp_path: Path) -> None:
    metadata = DocumentMetadata(
        document_id="student-test-001",
        audience="student",
        department="IT",
        source="synthetic://test",
        source_version="1",
        content_hash=sha256(b"unused").hexdigest(),
        approval_status="approved",
        review_date=date(2027, 1, 1),
        effective_date=date(2026, 1, 1),
        access_level="student",
    )
    text = tmp_path / "a.txt"
    text.write_text("Text   content", encoding="utf-8")
    html = tmp_path / "a.html"
    html.write_text("<html><nav>skip</nav><p>HTML content</p></html>", encoding="utf-8")
    docx = tmp_path / "a.docx"
    document = Document()
    document.add_paragraph("DOCX content")
    document.save(docx)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"placeholder")
    monkeypatch.setattr("ai_support.rag.loaders._parse_pdf", lambda path: "PDF content")

    assert parse_document(SourceDocument(metadata=metadata, path=text)).text == "Text content"
    assert parse_document(SourceDocument(metadata=metadata, path=html)).text == "HTML content"
    assert parse_document(SourceDocument(metadata=metadata, path=docx)).text == "DOCX content"
    assert parse_document(SourceDocument(metadata=metadata, path=pdf)).text == "PDF content"


def test_all_chunking_strategies_are_configurable(tmp_path: Path) -> None:
    text = "# Heading\n\n" + " ".join(f"word{index}" for index in range(80))
    assert len(fixed_chunks(text, 30, 5)) >= 3
    assert recursive_chunks(text, 30, 5)
    assert heading_chunks(text, 30, 5)[0].startswith("# Heading")

    config = load_rag_config(Path("configs/rag/student.toml"), tmp_path)
    path = tmp_path / "document.md"
    path.write_text(text, encoding="utf-8")
    metadata = DocumentMetadata(
        document_id="student-it-001",
        audience="student",
        department="IT",
        source="synthetic://it",
        source_version="1",
        content_hash=content_hash(path),
        approval_status="approved",
        review_date=date(2027, 1, 1),
        effective_date=date(2026, 1, 1),
        access_level="student",
    )
    chunks = chunk_documents([SourceDocument(metadata=metadata, path=path, text=text)], config)
    assert chunks
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

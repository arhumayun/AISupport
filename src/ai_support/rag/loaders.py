"""Metadata discovery, validation, and multi-format document parsing."""

import json
import re
from hashlib import sha256
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

from ai_support.rag.config import RAGConfig
from ai_support.rag.schema import DocumentMetadata, SourceDocument

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt", ".html", ".htm"}


def content_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def discover_documents(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"RAG source directory not found: {source_dir}")
    paths = sorted(
        path for path in source_dir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no supported documents found in {source_dir}")
    return paths


def metadata_path(document_path: Path) -> Path:
    return document_path.with_name(f"{document_path.name}.metadata.json")


def load_metadata(path: Path) -> DocumentMetadata:
    sidecar = metadata_path(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"document metadata not found: {sidecar}")
    return DocumentMetadata.model_validate_json(sidecar.read_text(encoding="utf-8"))


def validate_metadata(path: Path, metadata: DocumentMetadata, config: RAGConfig) -> None:
    if metadata.audience != config.audience or metadata.access_level != config.audience:
        raise ValueError(f"{metadata.document_id} violates {config.audience} audience isolation")
    if metadata.approval_status != "approved":
        raise ValueError(f"{metadata.document_id} is not approved")
    if metadata.effective_date > config.as_of_date:
        raise ValueError(f"{metadata.document_id} is not yet effective")
    if metadata.review_date < config.as_of_date:
        raise ValueError(f"{metadata.document_id} review date has expired")
    if metadata.content_hash != content_hash(path):
        raise ValueError(f"{metadata.document_id} content hash mismatch")


def load_unvalidated_descriptors(config: RAGConfig) -> list[SourceDocument]:
    descriptors: list[SourceDocument] = []
    identifiers: set[str] = set()
    for path in discover_documents(config.source_dir):
        metadata = load_metadata(path)
        if metadata.document_id in identifiers:
            raise ValueError(f"duplicate document_id: {metadata.document_id}")
        identifiers.add(metadata.document_id)
        descriptors.append(SourceDocument(metadata=metadata, path=path))
    return descriptors


def validate_descriptors(
    documents: list[SourceDocument], config: RAGConfig
) -> list[SourceDocument]:
    for document in documents:
        validate_metadata(document.path, document.metadata, config)
    return documents


def load_descriptors(config: RAGConfig) -> list[SourceDocument]:
    return validate_descriptors(load_unvalidated_descriptors(config), config)


def _parse_pdf(path: Path) -> str:
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _parse_docx(path: Path) -> str:
    document = Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _parse_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    for node in soup(["script", "style", "nav"]):
        node.decompose()
    return soup.get_text("\n")


def normalize_content(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_document(document: SourceDocument) -> SourceDocument:
    suffix = document.path.suffix.lower()
    if suffix == ".pdf":
        text = _parse_pdf(document.path)
    elif suffix == ".docx":
        text = _parse_docx(document.path)
    elif suffix in {".html", ".htm"}:
        text = _parse_html(document.path)
    else:
        text = document.path.read_text(encoding="utf-8")
    normalized = normalize_content(text)
    if not normalized:
        raise ValueError(f"document contains no extractable text: {document.path}")
    return document.model_copy(update={"text": normalized})


def parse_documents(documents: list[SourceDocument]) -> list[SourceDocument]:
    return [parse_document(document) for document in documents]


def write_metadata(path: Path, metadata: DocumentMetadata) -> None:
    metadata_path(path).write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

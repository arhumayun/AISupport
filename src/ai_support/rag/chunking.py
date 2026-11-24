"""Configurable fixed, recursive, and heading-aware chunking."""

import re
from hashlib import sha256

from ai_support.rag.config import RAGConfig
from ai_support.rag.schema import Chunk, SourceDocument

_HEADING = re.compile(r"(?m)^(#{1,6}\s+.+)$")


def _windows(words: list[str], size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    return [chunk for chunk in _windows(text.split(), size, overlap) if chunk]


def recursive_chunks(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}|(?<=[.!?])\s+", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > size:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(_windows(words, size, overlap))
        elif len(current) + len(words) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] + words if overlap else words
        else:
            current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def heading_chunks(text: str, size: int, overlap: int) -> list[str]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return recursive_chunks(text, size, overlap)
    sections: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.extend(recursive_chunks(text[match.start() : end], size, overlap))
    prefix = text[: matches[0].start()].strip()
    return ([prefix] if prefix else []) + sections


def chunk_documents(documents: list[SourceDocument], config: RAGConfig) -> list[Chunk]:
    methods = {
        "fixed": fixed_chunks,
        "recursive": recursive_chunks,
        "heading": heading_chunks,
    }
    chunker = methods[config.chunk_method]
    chunks: list[Chunk] = []
    for document in documents:
        parts = chunker(document.text, config.chunk_size, config.chunk_overlap)
        for ordinal, content in enumerate(parts):
            digest = sha256(content.encode("utf-8")).hexdigest()
            chunk_id = sha256(
                f"{document.metadata.document_id}:{config.chunk_method}:{ordinal}:{digest}".encode()
            ).hexdigest()[:32]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.metadata.document_id,
                    audience=document.metadata.audience,
                    department=document.metadata.department,
                    source=document.metadata.source,
                    source_version=document.metadata.source_version,
                    access_level=document.metadata.access_level,
                    ordinal=ordinal,
                    content=content,
                    content_hash=digest,
                )
            )
    return chunks

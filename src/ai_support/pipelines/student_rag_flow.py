"""Metaflow orchestration for the student RAG knowledge release."""

import os
from pathlib import Path

from metaflow import FlowSpec, Parameter, current, step

from ai_support.rag.chunking import chunk_documents
from ai_support.rag.config import load_rag_config
from ai_support.rag.embeddings import attach_embeddings
from ai_support.rag.evaluation import (
    apply_rag_gate,
    evaluate_retrieval,
    load_benchmark,
    write_evaluation,
)
from ai_support.rag.loaders import (
    load_unvalidated_descriptors,
    parse_documents,
    validate_descriptors,
)
from ai_support.rag.pipeline import make_embedder, make_repository, make_reranker
from ai_support.rag.retrieval import HybridRetriever
from ai_support.rag.tracking import base_manifest, build_release_id, finalize_and_track


class StudentRAGFlow(FlowSpec):
    config_path = Parameter("config", default="configs/rag/student.toml")

    @step
    def start(self):
        self.project_root = str(Path.cwd().resolve())
        self.metaflow_run_id = str(current.run_id)
        self.config = load_rag_config(Path(self.config_path), Path(self.project_root))
        if self.config.audience != "student":
            raise ValueError("StudentRAGFlow requires a student configuration")
        self.next(self.load_documents)

    @step
    def load_documents(self):
        self.documents = load_unvalidated_descriptors(self.config)
        self.next(self.validate_metadata)

    @step
    def validate_metadata(self):
        self.documents = validate_descriptors(self.documents, self.config)
        self.next(self.parse_normalize)

    @step
    def parse_normalize(self):
        self.documents = parse_documents(self.documents)
        self.next(self.chunk)

    @step
    def chunk(self):
        self.chunks = chunk_documents(self.documents, self.config)
        self.next(self.embed)

    @step
    def embed(self):
        self.chunks = attach_embeddings(self.chunks, make_embedder(self.config))
        self.release_id = build_release_id(self.config, self.documents, self.chunks)
        self.next(self.build_index)

    @step
    def build_index(self):
        self.repository = make_repository(os.getenv("AI_SUPPORT_DATABASE_URL"))
        self.manifest = base_manifest(
            self.config,
            self.release_id,
            self.documents,
            self.chunks,
            Path(self.project_root),
            self.metaflow_run_id,
            type(self.repository).__name__,
        )
        self.repository.index_release(
            self.release_id,
            self.config.audience,
            self.documents,
            self.chunks,
            self.manifest,
        )
        self.next(self.evaluate)

    @step
    def evaluate(self):
        retriever = HybridRetriever(
            self.repository,
            make_embedder(self.config),
            self.config,
            self.release_id,
            make_reranker(self.config),
        )
        self.metrics, self.case_results = evaluate_retrieval(
            retriever, load_benchmark(self.config.benchmark_path), self.config
        )
        self.gate = apply_rag_gate(self.metrics, self.config)
        self.evaluation_path = str(self.config.release_dir.parent / "evaluation.json")
        write_evaluation(Path(self.evaluation_path), self.metrics, self.case_results)
        self.next(self.register_release)

    @step
    def register_release(self):
        self.final_manifest = finalize_and_track(
            self.config,
            self.manifest,
            self.metrics,
            self.gate,
            Path(self.evaluation_path),
        )
        self.repository.update_manifest(self.release_id, self.final_manifest)
        self.next(self.end)

    @step
    def end(self):
        print(self.final_manifest)


if __name__ == "__main__":
    StudentRAGFlow()

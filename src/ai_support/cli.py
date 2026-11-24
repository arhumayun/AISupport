"""Reproducible command-line interface for router and RAG workflows."""

import argparse
from pathlib import Path
from typing import cast

from metaflow import Runner

from ai_support.rag.config import load_rag_config
from ai_support.rag.pipeline import config_path_for_audience as rag_config_path
from ai_support.rag.sample_data import generate_sample_corpus
from ai_support.router.config import load_router_config
from ai_support.router.pipeline import config_path_for_audience
from ai_support.router.schema import Audience
from ai_support.router.service import RouterService
from ai_support.router.synthetic import generate_ticket_file
from ai_support.router.tracking import approve_model


def _audiences(value: str) -> tuple[Audience, ...]:
    if value == "all":
        return ("student", "staff")
    return (cast(Audience, value),)


def generate_data(project_root: Path, audience: str, rows: int) -> None:
    for selected in _audiences(audience):
        config = load_router_config(config_path_for_audience(project_root, selected), project_root)
        records = generate_ticket_file(config.data.dataset_path, selected, rows, config.random_seed)
        print(f"generated {len(records)} {selected} tickets at {config.data.dataset_path}")


def train_router(project_root: Path, audience: Audience) -> None:
    flow_file = (Path(__file__).parent / "pipelines" / f"{audience}_router_flow.py").resolve()
    config_path = config_path_for_audience(project_root, audience).resolve()
    with Runner(str(flow_file), cwd=str(project_root), show_output=True).run(
        config=str(config_path)
    ) as running:
        if running.status != "successful":
            raise RuntimeError(f"{audience} router Metaflow run failed")
        manifest = running.run.data.manifest
    print(
        f"{audience}: accuracy={manifest['metrics']['accuracy']:.4f}, "
        f"macro_f1={manifest['metrics']['macro_f1']:.4f}, "
        f"gate_passed={manifest['promotion_gate_passed']}, "
        f"model_version={manifest['registered_model_version']}, "
        f"metaflow_run_id={manifest['metaflow_run_id']}"
    )


def generate_rag_data(project_root: Path, audience: str) -> None:
    for selected in _audiences(audience):
        config = load_rag_config(rag_config_path(project_root, selected), project_root)
        paths, cases = generate_sample_corpus(config)
        print(f"generated {len(paths)} {selected} documents and {len(cases)} benchmark cases")


def build_rag(project_root: Path, audience: Audience) -> None:
    flow_file = (Path(__file__).parent / "pipelines" / f"{audience}_rag_flow.py").resolve()
    config_path = rag_config_path(project_root, audience).resolve()
    with Runner(str(flow_file), cwd=str(project_root), show_output=True).run(
        config=str(config_path)
    ) as running:
        if running.status != "successful":
            raise RuntimeError(f"{audience} RAG Metaflow run failed")
        manifest = running.run.data.final_manifest
    metrics = manifest["metrics"]
    print(
        f"{audience}: release={manifest['release_id']}, "
        f"recall@k={metrics['recall_at_k']:.4f}, mrr={metrics['mrr']:.4f}, "
        f"status={manifest['status']}, metaflow_run_id={manifest['metaflow_run_id']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-support")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Synthetic router dataset operations")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    generate = data_commands.add_parser("generate")
    generate.add_argument("--audience", choices=("student", "staff", "all"), default="all")
    generate.add_argument("--rows", type=int, default=600)

    router = commands.add_parser("router", help="Single-audience router operations")
    router_commands = router.add_subparsers(dest="router_command", required=True)
    train_command = router_commands.add_parser("train")
    train_command.add_argument("--audience", choices=("student", "staff"), required=True)
    approve = router_commands.add_parser("approve")
    approve.add_argument("--audience", choices=("student", "staff"), required=True)
    approve.add_argument("--model-reference", default="candidate")
    predict = router_commands.add_parser("predict")
    predict.add_argument("--audience", choices=("student", "staff"), required=True)
    predict.add_argument("--subject", required=True)
    predict.add_argument("--description", required=True)
    predict.add_argument("--model-reference", default="champion")

    all_routers = commands.add_parser("routers", help="Run both independent router pipelines")
    all_commands = all_routers.add_subparsers(dest="routers_command", required=True)
    run_all = all_commands.add_parser("run-all")
    run_all.add_argument("--generate", action="store_true")
    run_all.add_argument("--rows", type=int, default=600)

    rag = commands.add_parser("rag", help="Single-audience RAG operations")
    rag_commands = rag.add_subparsers(dest="rag_command", required=True)
    rag_generate = rag_commands.add_parser("generate-sample-data")
    rag_generate.add_argument("--audience", choices=("student", "staff", "all"), default="all")
    rag_build = rag_commands.add_parser("build")
    rag_build.add_argument("--audience", choices=("student", "staff"), required=True)

    all_rags = commands.add_parser("rags", help="Run both independent RAG pipelines")
    rags_commands = all_rags.add_subparsers(dest="rags_command", required=True)
    rag_run_all = rags_commands.add_parser("run-all")
    rag_run_all.add_argument("--generate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    if args.command == "data":
        generate_data(project_root, args.audience, args.rows)
    elif args.command == "router" and args.router_command == "train":
        train_router(project_root, cast(Audience, args.audience))
    elif args.command == "router" and args.router_command == "approve":
        audience = cast(Audience, args.audience)
        config = load_router_config(config_path_for_audience(project_root, audience), project_root)
        version = approve_model(config, args.model_reference)
        print(f"{audience} model version {version} is now champion")
    elif args.command == "router" and args.router_command == "predict":
        audience = cast(Audience, args.audience)
        result = RouterService(project_root, args.model_reference).predict(
            audience, args.subject, args.description
        )
        print(result)
    elif args.command == "routers":
        if args.generate:
            generate_data(project_root, "all", args.rows)
        for audience in ("student", "staff"):
            train_router(project_root, audience)
    elif args.command == "rag" and args.rag_command == "generate-sample-data":
        generate_rag_data(project_root, args.audience)
    elif args.command == "rag" and args.rag_command == "build":
        build_rag(project_root, cast(Audience, args.audience))
    elif args.command == "rags":
        if args.generate:
            generate_rag_data(project_root, "all")
        for audience in ("student", "staff"):
            build_rag(project_root, audience)

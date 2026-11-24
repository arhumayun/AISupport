"""Metaflow orchestration for the student department router."""

from pathlib import Path

from metaflow import FlowSpec, Parameter, current, step

from ai_support.router.config import load_router_config
from ai_support.router.pipeline import evaluate, ingest, preprocess, register, train, validate


class StudentRouterFlow(FlowSpec):
    """Five-stage student model workflow with tracked outputs."""

    config_path = Parameter("config", default="configs/router/student.toml")

    @step
    def start(self):
        self.project_root = str(Path.cwd().resolve())
        self.metaflow_run_id = str(current.run_id)
        self.config = load_router_config(Path(self.config_path), Path(self.project_root))
        if self.config.audience != "student":
            raise ValueError("StudentRouterFlow requires a student configuration")
        self.next(self.ingestion)

    @step
    def ingestion(self):
        self.records = ingest(self.config)
        self.next(self.preprocessing)

    @step
    def preprocessing(self):
        self.split = preprocess(self.config, self.records)
        self.next(self.training)

    @step
    def training(self):
        self.model = train(self.config, self.split)
        self.next(self.evaluation)

    @step
    def evaluation(self):
        self.evaluation_result = evaluate(self.config, self.model, self.split)
        self.gate = validate(self.config, self.evaluation_result)
        self.next(self.registration)

    @step
    def registration(self):
        self.manifest = register(
            self.config,
            self.model,
            self.split,
            self.evaluation_result,
            self.gate,
            self.metaflow_run_id,
            Path(self.project_root),
        )
        self.next(self.end)

    @step
    def end(self):
        print(self.manifest)


if __name__ == "__main__":
    StudentRouterFlow()

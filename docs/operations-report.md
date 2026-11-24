# UOWD AI Support Assistant: Deployment and Operations Report

## 1. Architecture as built

The implemented Assessment 3 vertical slice is a single chatbot containing authorization-isolated
student and staff systems. A signed backend session, never an LLM-generated role, selects the router,
knowledge release, profile record, and allowed tools.

```text
Browser -> Streamlit -> FastAPI authentication and audience policy
                         |-> student/staff TF-IDF router -> MLflow champion
                         |-> PostgreSQL FTS + pgvector -> RRF -> optional reranker
                         |-> Ollama grounded answer -> citations
                         |-> session-bound profile/course tools
                         |-> confirmed synthetic email -> Mailpit
                         `-> privacy-safe gap + human handoff

Offline: seeded synthetic data -> DVC -> four Metaflow flows -> MLflow/MinIO
Runtime: FastAPI metrics -> Prometheus rules/alerts -> provisioned Grafana dashboard
Storage: PostgreSQL/pgvector, MLflow PostgreSQL metadata, MinIO artifacts, named volumes
```

### Assessment 2 design compared with the build

| Assessment 2 decision | Assessment 3 build | Reason for difference |
|---|---|---|
| One assistant for visitors, students, and staff | Authenticated student and staff vertical slice | Visitor content and identity behavior would expand evaluation beyond the assessment schedule. |
| Department router in the application image | Two supervised, independently registered student/staff routers | Historical-ticket routing and audience isolation became explicit requirements. Registry loading also enables rollback without rebuilding the image. |
| Audience-filtered shared retrieval design | Separate student/staff sources, benchmarks, manifests, and database filters | Defence in depth reduces the impact of a metadata or query mistake. |
| RAG with limited agent behavior | Deterministic LangGraph workflow with allow-listed tools | Profile lookup, clarification drafts, confirmation, and handoff required controlled orchestration; the LLM never chooses authorization or side effects. |
| NGINX, simulated roles, and possible SSO | Loopback-bound Compose and signed synthetic sessions | This is a local prepared-for-deployment assessment target; TLS termination and institutional SSO need university infrastructure and approval. |
| ServiceNow handoff integration | Portal link plus privacy-safe gap record | The prototype must not create or modify real tickets. |
| Canary release and automatic infrastructure recovery | Manual model/RAG selection and documented restart | Automatic rollback without approval is outside the vertical slice and creates governance risk. |

## 2. Target environment and setup

The system is **prepared for deployment**, not claimed as externally deployed. The named target is a
single macOS or Linux host with Docker Compose v2, UV 0.11.23, Git, sufficient disk space, and access
to download the explicitly selected Ollama and Sentence Transformers models. Administrative ports
bind only to loopback; PostgreSQL is not published; the backend network is internal.

From a clean clone:

1. Copy `.env.example` to `.env` and replace every password, the synthetic login code, and the
   session secret. Never commit `.env`.
2. Run `./scripts/bootstrap_demo.sh --pull-model`. The flag is explicit authorization for the large
   Ollama download; omit it on later runs when the named volume already contains the model.
3. Open Streamlit at `http://127.0.0.1:8501`, MLflow at `http://127.0.0.1:5050`, Grafana at
   `http://127.0.0.1:3000`, and Mailpit at `http://127.0.0.1:8025`.

Every published loopback port is configurable through the `AI_SUPPORT_*_PORT` entries in
`.env.example`; internal Compose addresses remain fixed. This permits an isolated release rehearsal
without disrupting an existing demonstration stack.

The bootstrap builds the application and pinned MinIO source image, starts infrastructure, generates
both deterministic router datasets, trains and registers both routers, manually promotes the passing
candidates to `champion`, generates both RAG corpora, builds both PostgreSQL indexes and immutable
manifests, starts the remaining services, and runs an authenticated acceptance probe. The probe must
observe a student champion routing the timetable question to Registry, cited student-only retrieval,
and a cited knowledge answer. Any failed gate or request stops the command.

## 3. Release and verification

1. Require a clean reviewed commit and green GitHub Actions checks.
2. Regenerate `docs/evidence` and confirm the DVC, Git, Metaflow, MLflow, and evaluation references.
3. Build the exact commit with `docker compose build api minio`; do not rebuild between verification
   and release.
4. Run the four workflows. Passing router models become candidates; a person explicitly changes the
   champion alias. Passing RAG runs write immutable candidate manifests.
5. Run the acceptance verifier and inspect Prometheus targets, Grafana, MLflow aliases, Mailpit, and
   the active model/knowledge gauges.
6. Record the commit, image ID/digest, model versions, RAG release IDs, approver, and verification
   result in the release record.

GitHub Actions installs the locked UV environment, lints, formats, and tests the project. A separate
job validates Compose, performs a real lockfile-based application image build, asserts UID/GID 10001,
starts the image, and calls `/health` before removing the test container.

## 4. Measured performance and scalability

The deterministic router holdouts measured 1.000 accuracy, macro-F1, and top-3 accuracy; calibration
error was approximately 0.0308 for students and 0.0297 for staff. This meets the 0.90 routing target
on deliberately separable synthetic data but is not evidence of production accuracy.

Each five-question RAG benchmark measured 1.000 Recall@5, MRR, nDCG, citation correctness, and the
evidence-term groundedness proxy, with 0.000 unauthorized cross-audience retrieval. Average local
retrieval latency was about 792.49 ms for students and 366.05 ms for staff. The small benchmark meets
the 95% citation target but does not represent real departmental diversity.

A 100-request sequential container probe of `/health` recorded zero errors, 0.678 ms p50, 1.315 ms
p95, and 16.349 ms maximum latency on the arm64 development host. It proves container responsiveness,
not the Assessment 2 six-second end-to-end chat target. A real Ollama integration test passed once
against the locally installed model, but repeatable end-to-end load and throughput measurements remain
required before production capacity claims.

The clearest measured packaging bottleneck is the 5.34 GB unified image, caused by combining training,
RAG, and serving dependencies. The first scaling step is to split trainer/indexer and runtime images.
For runtime pressure, cache public navigation answers, bound concurrency, measure Ollama queue time,
and add stateless API replicas before selecting a larger model. PostgreSQL indexes and connections
must be measured before replication. Proposed capacity investigation triggers are p95 above six
seconds, errors above 1%, sustained CPU above 70%, or an Ollama queue that breaches the response
budget.

## 5. Monitoring and alerts

Prometheus collects API request latency/errors, audience/department predictions, confidence,
retrieval latency, LLM latency/outcome, citation coverage, knowledge gaps, agent failures, email
failures, and active model and knowledge versions. Grafana provisions the version-controlled UOWD
overview dashboard. Prometheus evaluates these initial pilot alerts:

| Signal | Threshold | Severity | Initial response |
|---|---:|---|---|
| API unavailable | `up == 0` for 2 minutes | Critical | Deployment owner checks container and dependencies; enable handoff-only communication. |
| API errors | More than 1% for 15 minutes | Warning | Inspect route/status metrics and recent release; roll back if release-related. |
| API p95 | More than 6 seconds for 15 minutes | Warning | Check Ollama and retrieval latency, CPU, memory, and request queue. |
| Citation coverage | Average below 95% for 15 minutes | Warning | Knowledge owner reviews the affected release and prevents promotion. |
| Tool/email failure | Any in 15 minutes | Warning | Application owner checks the privacy-safe audit and Mailpit/service health. |

Prometheus exposes firing alerts locally; a real university notification receiver is deliberately not
configured with assessment credentials. During the pilot, the deployment owner checks Prometheus and
Grafana during demonstrations and release verification. Before shared deployment, UOWD must select an
approved email, Teams, or on-call receiver and test delivery.

### Alert response

Capture the alert labels, active versions, first/last occurrence, and privacy-safe logs. Stabilize the
service using handoff behavior or a previously approved version, preserve evidence, then diagnose.
Never paste prompts, profile records, credentials, or raw personal data into an incident record. Close
the alert only after the acceptance probe and relevant regression tests pass.

## 6. Maintenance and ownership

| Task | Trigger/frequency | Accountable role | Procedure |
|---|---|---|---|
| Router retraining | About 200 reviewed labels, macro-F1 drop of 0.05, drift, or approved schedule | ML owner with department reviewers | Version data, run audience flow, review gate, approve candidate manually. |
| Knowledge release | Approved content change, expiry, or reviewed gap | Department knowledge owner | Update metadata/source, run affected RAG flow only, review leakage/citation gate. |
| Dependency update | Monthly or critical advisory | Application owner | Update with `uv add`, regenerate `uv.lock`, run tests/image build/acceptance, review licences. |
| Backup/restore | Daily in shared deployment; drill quarterly | Deployment owner | Back up PostgreSQL and MinIO together; restore into an isolated stack and run acceptance. |
| Alert review | Weekly pilot review and after every release | Deployment owner | Review latency, errors, confidence, gaps, failures, and active versions. |
| Access and secret review | Quarterly or personnel/incident change | Security owner | Rotate secrets, review exposure and recipients, invalidate old sessions. |

### Rollback

Router rollback reassigns `champion` to an already approved explicit MLflow version. RAG rollback uses
`ai-support rag select-release` only for a gate-passing immutable release and its matching indexed
configuration. Application rollback starts the previous immutable image. Automatic production
rollback is not enabled. After any rollback, restart API workers, confirm the version gauges, and run
the authenticated acceptance probe.

## 7. Security, privacy, and cost

Only synthetic identities and content are included. The backend signs sessions and derives profile
identity and audience itself. Database queries also require the audience/access level. Retrieved text
is untrusted evidence and cannot grant tools. Side effects require an expiring, session-owned draft and
explicit confirmation. Logs and evidence exports exclude prompts, personal profiles, secrets, and
model binaries. Mailpit accepts only synthetic `example.test` recipients.

`.env.example` contains conspicuous placeholders; the real `.env` is ignored. Services run non-root
where controlled, use `no-new-privileges`, publish administrative ports only on loopback, and keep
PostgreSQL private. The assessment environment does not provide TLS, institutional SSO, a secrets
vault, image signing/scanning, durable session revocation, or tested disaster recovery; these are
required before university deployment.

The local software licence cost is zero and no paid API is required. Actual cost is the host, storage,
backups, electricity, model download/storage, and staff time. A university estimate must be based on
measured chats, Ollama hardware utilization, retention, backup size, and availability requirements;
the project does not invent a cloud price without those inputs.

## 8. Known limitations

- Synthetic tickets and tiny RAG benchmarks make the perfect scores optimistic.
- The groundedness metric is an evidence-term proxy, not expert factual review.
- The stack is prepared locally rather than reachable beyond one developer machine.
- End-to-end chat load, training duration, recovery time, and backup restoration are not yet measured.
- The second embedding profile and CrossEncoder are configurable but lack a recorded comparison run.
- The full image is large, Ollama/model startup is hardware-dependent, and the first bootstrap is slow.
- Alert notification delivery needs an institution-approved receiver.
- Team contribution evidence must match real Git authors before submission.

## 9. Reflection and lessons learned

The build confirmed the central Assessment 2 idea: safe failure and traceability matter more than an
impressive standalone model. The original design treated the assistant as a modular monolith with an
audience-filtered knowledge system, a simple department router, citations, and human handoff. The
implementation kept those principles but made isolation more explicit. Student and staff users now
have separate supervised models, source collections, benchmarks, release manifests, and policy checks.
This increased the number of experiments and operational objects, but it made authorization failures
easier to test and explain. Visitor support was removed to keep a demonstrable vertical slice, while
profile lookup and confirmed email were added because they tested whether agentic behavior could be
bounded by backend policy.

Performance results must be interpreted cautiously. Both synthetic router holdouts exceeded the 90%
target with perfect accuracy and macro-F1, and both five-case RAG benchmarks exceeded the 95%
citation target. Those results show that the pipelines, gates, and metrics work; they do not show that
real tickets or policies will be equally easy. The recorded health latency is far below the six-second
target but excludes retrieval and generation. The honest conclusion is therefore that routing and
retrieval functionality meet the assessment benchmarks while the end-to-end latency target remains
unproven under load. This distinction is more valuable than presenting an unrelated fast health check
as chatbot performance.

Three trade-offs shaped the result. First, TF-IDF and logistic regression were chosen over a transformer
router. This gives fast, explainable training and reliable CPU deployment but will handle ambiguous or
multilingual tickets less effectively. Second, the local Compose stack maximizes reproducibility and
privacy but does not provide TLS, high availability, or institutional identity. Third, training and
serving share one image. That simplified the bootstrap and guaranteed compatible libraries, but created
a 5.34 GB artifact. With more time, the team would retain the explainable baseline until real data
justifies a more complex router, deploy behind university SSO/TLS, and split offline and runtime images.

The system is ready to demonstrate deterministic data generation, four orchestrated pipelines, model
registration, audience-filtered retrieval, controlled tools, rollback, and observable API behavior. It
is not ready to process university data. The first three production fixes would be institutional SSO and
managed secrets; a representative, department-reviewed dataset and knowledge corpus; and tested
end-to-end capacity, backup, alert delivery, and recovery. These address identity, evidence quality, and
operational continuity before adding features.

Within six months, dependencies and container bases will age, articles will approach their review dates,
ticket language will drift, and local model revisions may disappear upstream. Dependency automation
and CI should expose incompatible upgrades; department knowledge owners should notice expiry and gap
alerts; the ML owner should review confidence and labelled performance; and the deployment owner
should verify backups and alerts. Version-pinned source models and immutable manifests reduce the risk
but do not replace ownership.

One process problem is already visible: the repository history and README contribution section do not
yet provide consistent evidence for every group member. That is a teamwork and submission-control
failure, not something that should be hidden or repaired with fabricated attribution. The corrective
action is for each member to review a real area, contribute identifiable final work, and update the
contribution statement to match genuine commits. The same principle applies to all reported results:
the group should demonstrate them, preserve evidence, and remove claims it cannot collectively explain.

The incremental delivery approach was also useful. Building authentication and model provenance before
the agent workflow exposed boundary errors while they were still small, and mocked LLM tests kept CI
deterministic. The remaining lesson is that deployment rehearsal must begin earlier: image size, model
warm-up, persistent-volume initialization, and clean-clone behavior are system qualities that unit tests
cannot reveal. Future iterations should run the isolated acceptance rehearsal at every release candidate,
retain its privacy-safe result, and treat an unexplained operational failure as a failed release.

## 10. References and disclosure

Primary project references are the supplied Assessment 3 brief and Assessment 2 architecture report.
Implementation choices use official documentation for [UV](https://docs.astral.sh/uv/),
[DVC](https://dvc.org/doc), [Metaflow](https://docs.metaflow.org/),
[MLflow](https://mlflow.org/docs/latest/), [scikit-learn](https://scikit-learn.org/stable/),
[Sentence Transformers](https://sbert.net/), [pgvector](https://github.com/pgvector/pgvector),
[PostgreSQL](https://www.postgresql.org/docs/), [Ollama](https://docs.ollama.com/),
[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview),
[FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://docs.streamlit.io/),
[Prometheus](https://prometheus.io/docs/), [Grafana](https://grafana.com/docs/),
[Docker](https://docs.docker.com/), and
[GitHub Actions](https://docs.github.com/actions). The final submitted report should format these URLs
in the institution's required citation style and include access dates if required.

OpenAI Codex assisted with implementation, testing, and documentation. The project group remains
responsible for reviewing every change, validating every measurement, following the lecturer's
generative-AI policy, and being able to explain the submitted system.

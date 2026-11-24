# Implementation Log

This log records material differences between the Assessment 2 proposal and the system built for Assessment 3. Measurements and completion claims will be added only after they are verified.

## 2026-11-24 - Phase 1 foundation

- **Router scope changed:** the Assessment 2 department classifier becomes two independently trained supervised models, one for students and one for staff. Implementation is deferred to Phase 2.
- **RAG isolation strengthened:** the earlier shared-index proposal becomes separate student and staff sources, releases, evaluation datasets, and query policies. Implementation is deferred to Phase 3.
- **Controlled agents added:** profile lookup, course progress, department clarification email, knowledge-gap recording, and human handoff will use allow-listed tools and explicit confirmation for side effects. Implementation is deferred to Phase 4.
- **Visitor scope reduced:** the Assessment 3 vertical slice focuses on authenticated synthetic student and staff identities. Prospective visitors remain a future enhancement.
- **ServiceNow integration reduced:** the prototype will provide a handoff link and privacy-safe gap record, not create or modify real tickets.
- **Foundation implemented:** the default PyCharm script was replaced by a UV-managed Python 3.12 `src/` package, a typed FastAPI health endpoint, JSON logging, automated tests, and GitHub Actions checks.

## Known Phase 1 limitations

- No training, model, data, RAG, agent, user interface, external service, container, or monitoring implementation exists yet.
- CI is defined locally but cannot be claimed as executed until the repository is hosted and the workflow runs.
- Team contribution details and the public demo link remain submission-time inputs.

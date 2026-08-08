# raidar — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.


- [ ] **Follow-up: raidar CI runners are CPU-only so `test_stream_train_smoke` still costs ~8 min there** @todo — GitHub `ubuntu-latest` + the Jenkins agents have no GPU, so the MPS/CUDA speedup only helps local dev; CI still runs the smoke test on CPU every push. Options (user steered toward GPU-when-available for local, CI strategy left open): deselect it in the routine run (`pytest tests -m 'not slow'`) + a separate nightly or parallel `-m slow` job to preserve the dep-bump-drift coverage the test exists for, or move test CI to a GPU runner.

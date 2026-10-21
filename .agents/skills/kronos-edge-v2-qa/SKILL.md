---
name: kronos-edge-v2-qa
description: Run and diagnose the Kronos Edge V2 QA workflow, including local model checks, Gaia worker validation, and Kernelgen result inspection.
---

# Kronos Edge V2 QA

Use this skill when Reza asks to run the repository's QA job, verify a real
model check, rerun Kernelgen, or diagnose a failed or incomplete Gaia run.
Read the repository `AGENTS.md` and `README.md` first. Work only in this
repository and keep credentials out of commands, logs, artifacts, and reports.

## QA modes

There are two different QA targets:

1. **Repository QA** validates the implementation without contacting Gaia.
2. **Live model QA** runs the configured model, TensorRT benchmark, reference
   export, worker inventory check, Kernelgen compile, and report inspection.

Do not call a local test pass a live model QA result. A live result is complete
only when the artifact directory contains both `kernel.py` and
`kernelgen_report.json`, and `eval inspect` renders the report.

## CLI transcripts

Every QA run must preserve the exact command and combined stdout/stderr in a
transcript so the run is reproducible from its artifact directory. Never put
API keys, registry keys, or other secret values in the transcript. Redact
secret-bearing environment assignments while keeping the command structure.

For a live run, capture the output to a temporary file while the workflow is
running, retain its exit status, then write the command, exit status, and
captured output to `<run_path>/qa_cli.txt` after the workflow creates its run
directory. If no run directory was created, use `.artifact/qa_cli-<UTC>.txt`.
The transcript must be written for both successful and failed runs. A shell
pattern is:

```bash
qa_log=$(mktemp)
set +e
<exact QA command> 2>&1 | tee "$qa_log"
qa_status=${PIPESTATUS[0]}
set -e
{
  printf 'Command:\n%s\n\nExit status: %s\n\nOutput:\n' '<redacted exact command>' "$qa_status"
  cat "$qa_log"
} > "<run_path>/qa_cli.txt"
rm -f "$qa_log"
exit "$qa_status"
```

For repository QA, write the same transcript format to
`.artifact/qa_cli-<UTC>.txt`, since repository tests do not create a run
directory.

## Repository QA

From the repository root, run:

```bash
.venv/bin/pytest
```

Report the exit status and the test output exactly. Do not use the live Yasp
API key for this mode.

## Live model QA

Confirm the config, model, and input paths before starting. The documented
entrypoint is:

```bash
PYTHONPATH=src .venv/bin/python -m kronos_edge.workflow \
  path/to/config.yaml path/to/model.py path/to/input.npz
```

Use `--artifacts path/to/dir` only when the workflow's CLI supports it and a
non-default artifact directory is intentional. Let the workflow own the
sequence: preflight, model check, reference export, worker inventory,
Kernelgen compile, and report inspection. Do not upload `--source` or run user
model code on the service side.

Before a live run, verify the persistent worker with the same routing key as
the config:

```bash
yasp-toolkit eval workers --tag routing=<eval_routing>
```

Require a row whose GPU Name equals `config.yaml`'s `gpu` exactly and whose
Routing equals `eval_routing` exactly. Registration is not proof that the
worker can use the GPU.

After the run, check the managed files:

```bash
ls -l .artifact/model.pt2 .artifact/model.engine \
  .artifact/kernel.py .artifact/kernelgen_report.json
yasp-toolkit eval inspect --file .artifact/kernelgen_report.json
```

The successful result should include the local eager/TensorRT benchmark and a
Kernelgen report showing correctness passed. Do not parse or independently
judge fields in the Yasp report; this repository deliberately displays the
toolkit's rendering.

## Kernelgen task tracking

Always preserve the task ID printed by `kernelgen compile`. Kernelgen can take
many minutes, and losing the local polling process does not tell us whether
the remote task succeeded. Query a known task directly:

```bash
yasp-toolkit kernelgen get <task-id> --wait
```

The task UUID is a Kernelgen task ID, not a worker routing tag. If the task
finishes remotely but artifacts are absent locally, report the remote result
and the missing download separately. Do not claim the local QA job completed
unless the expected files and inspection output are present.

## GPU worker diagnosis

If Kernelgen fails with `No CUDA GPUs are available`, inspect the worker
container before retrying:

```bash
docker exec <worker-container> nvidia-smi
docker logs <worker-container> --since <UTC-start-time> | grep -A40 <task-id>
```

The worker may still advertise the correct GPU and routing tag while its
container has no GPU device. In the observed failure, the worker log showed
`gpu_id: null`, and `nvidia-smi` failed with `Unknown Error`. The root cause
was container launch configuration, not model correctness, TensorRT, routing,
or the Kernelgen reference.

The worker image was able to see the RTX 5090 when launched with explicit
Docker GPU access. If the worker must be recreated, first inspect and preserve
its image, command, mounts, environment, network, restart policy, and routing
tags. Prefer renaming the old container for rollback over deleting it. Recreate
with the equivalent of:

```bash
docker run --gpus all ...
```

Then verify all three layers before submitting another job:

1. `docker exec <worker-container> nvidia-smi` reports the target GPU.
2. `yasp-toolkit eval workers --tag routing=<eval_routing>` reports the exact
   GPU and routing tag.
3. A worker-side evaluation reaches `done` with a non-null GPU ID and no
   failure before waiting on Kernelgen generation.

Do not start, stop, remove, or recreate a worker as part of the ordinary
repository workflow. Those are deployment actions and require an explicit
request. If explicitly asked, preserve the existing worker configuration and
verify the replacement before rerunning QA.

## What this run taught us

- A healthy worker registration is only control-plane evidence. Always verify
  device access inside the worker container.
- The first failed job was rejected before correctness evaluation because the
  worker had no CUDA device, even though it advertised an RTX 5090.
- The worker's parent process launched evaluation sandboxes with
  `YASP_LAUNCH_DOCKER_GPUS=all`; that setting does not grant the parent worker
  container a GPU. Check the actual container boundary that runs the job.
- After explicit GPU access, the worker-side evaluation passed and the remote
  Kernelgen task completed successfully, producing an `AgentKernel` with
  correctness passed and a 3.64x median speedup in that run.
- A long-running or lost local poller is an incomplete observation, not a
  failure. Preserve the task ID and query it with `kernelgen get --wait`.
- The documented `eval workers`, `kernelgen export`, `kernelgen compile`,
  `kernelgen get`, and `eval inspect` commands are the supported interface.
  Do not invent a parser for Yasp report fields or rely on undocumented service
  endpoints.

## Handoff

Report:

- the exact command used and repository commit;
- the artifact transcript path and its exit status;
- local test or model-check result;
- worker GPU and routing verification;
- Kernelgen task ID and terminal state;
- whether `kernel.py` and `kernelgen_report.json` exist locally;
- the rendered inspection result; and
- any deployment change, retained rollback container, or unresolved blocker.

Do not include API keys, registry keys, full environment dumps, or unrelated
container secrets in the handoff.

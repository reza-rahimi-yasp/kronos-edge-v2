"""Evaluator check: make sure an evaluator for the config's GPU and routing key is registered before kernelgen.

Runs on the host, between the container and kernelgen. A kernelgen job routed
by ``eval_routing`` to an evaluator that is not registered sits in the queue
indefinitely, so the host checks first and starts one when it has to. Two
nodes:

- ``ensure_evaluator``: lists the evaluators registered under the config's
  routing key with ``yasp-toolkit eval workers`` and looks for one advertising
  exactly the config's ``gpu``. When there is none, it starts a Gaia NVIDIA
  worker container on this machine, following the Gaia Docker deployment
  guide (https://compile.yasp.ai/help/gaia/workers/docker/): scoped registry
  and registration keys are minted with ``yasp-toolkit registry key`` and
  ``yasp-toolkit eval key``, the newest release of the worker image is pulled
  behind a temporary Docker login, and a uniquely named container runs
  ``yasp-eval register --tag routing=<key>`` with the GPU, the Docker socket,
  and the sibling-task settings the guide lists. The worker detects its GPU,
  so its name is not overridden. The node polls the listing until the
  evaluator appears, for a bounded time. The container is recorded in state
  as ``evaluator``; an evaluator that was already registered is not.
- ``release_evaluator``: stops the container in state, if any, then removes
  it. As created here, the worker's launcher uses its default backend and
  runs each task as a subprocess inside the worker container, so the task
  ends with the worker. Configured for its Docker backend (which neither
  this check nor the image does), it runs each task instead in a sibling
  container it creates through the Docker socket under its fixed slot name,
  shared by every worker on the machine, and tears it down when the task is
  cancelled; so the worker is first asked to stop (``docker stop``, bounded:
  SIGTERM, then the daemon kills it) and removed by force only after that.
  Once the worker is gone, the slot is inspected: the worker is started with
  ``KRONOS_EVALUATOR=<container name>`` forwarded into its task containers,
  so a slot whose environment names this worker is its leftover and is
  removed by the id that same inspect returned, never by the shared name,
  which another worker's task may hold by then; one without that marker
  belongs to another worker and is never touched. The graph runs the node
  after every outcome of the kernelgen nodes, so a worker this check started
  never outlives the check; an evaluator it did not start is never touched.
  A failure to remove is reported only when nothing failed before it.

``yasp-eval`` names the executable inside the official worker image; the host
installs no evaluator package and never runs it itself. The host needs only
``yasp-toolkit`` (from ``yasp-agent[gaia]``) and Docker. The user's main API
key stays on the host: only the registration key, scoped to registering,
enters the worker, through the ``docker create`` client's environment, and
the registry key goes to ``docker login`` through stdin. Neither is ever a
command argument, logged, or part of an error. The worker's output is
captured to a temporary file: it must not reach stdout, which carries the
check's result, and it is the failure detail when the worker exits before
registering.
"""

import base64
import json
import os
import re
import signal
import subprocess
import tempfile
import time
import urllib.request
import uuid

from kronos_edge import kernelgen_check
from kronos_edge.kernelgen_check import MISSING_TOKEN, TOOLKIT, UNAUTHORIZED, MissingCredentials, ToolkitFailed, run_logged
from kronos_edge.log import get_logger
from kronos_edge.messages import EVALUATOR as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

logger = get_logger(__name__)

# The evaluator command inside the worker image.
EVALUATOR = "yasp-eval"
WORKER_REGISTRY = "docker.yasp.ai"
WORKER_REPOSITORY = f"{WORKER_REGISTRY}/yasp-eval/nvidia"
WORKER_TAGS_URL = f"https://{WORKER_REGISTRY}/v2/yasp-eval/nvidia/tags/list"
DOCKER_SOCKET = "/var/run/docker.sock"
# The fixed name the worker's launcher gives a task's sibling container when its Docker backend is configured (its
# documented default for YASP_LAUNCH_DOCKER_SLOT_NAME), shared by every worker on the machine so their tasks take turns on the GPU.
TASK_SLOT = "yasp-launcher-slot"
# Set on the worker to its container name and forwarded into its task containers, where it names their owner.
OWNER_VARIABLE = "KRONOS_EVALUATOR"
# One `docker inspect` answer carrying both the container's immutable id and the environment that names its owner.
INSPECT_FORMAT = '{"id":{{json .Id}},"env":{{json .Config.Env}}}'
# How long a started evaluator may take to appear in the listing, and how often the listing is asked.
READY_TIMEOUT_S = 120.0
POLL_INTERVAL_S = 5.0
# A single inventory request must not consume the whole readiness window.
LISTING_TIMEOUT_S = 30.0
# How long `docker stop` gives the worker to cancel its task and tear down the task container before the daemon kills it.
STOP_GRACE_S = 30.0
# How long the attached Docker client may take to exit once its container is removed before it is killed.
STOP_TIMEOUT_S = 10.0
# Bounds for the Docker commands: a first pull of the worker image is large and is not charged to readiness.
DOCKER_TIMEOUT_S = 60.0
PULL_TIMEOUT_S = 1800.0
# The toolkit's key commands print one JWT; a release tag of the worker image is bare semver.
KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
RELEASE_PATTERN = re.compile(r"\d+\.\d+\.\d+")
# What `eval workers` prints instead of a table when nothing matches the filter.
NO_EVALUATORS = "No evaluators"
# The header of the table's first column; a listing without it is not a listing.
TABLE_HEADER = "GPU Name"
# What `docker rm`, `stop`, and `inspect` say about a container that is already gone, which is the state removal is after.
NO_SUCH_CONTAINER = "No such container"

_active_evaluator = None


class DockerFailed(RuntimeError):
    """A Docker command exited with an error; the message is the command (never a key) and the daemon's output."""


class Evaluator:
    """The Gaia worker container this check started, by name, and the ``docker start --attach`` client whose output is its log."""

    def __init__(self, name: str, process: subprocess.Popen, log):
        self.name, self.process, self.log = name, process, log

    def exit_code(self) -> int | None:
        return self.process.poll()

    def output(self) -> str:
        self.log.seek(0)
        return self.log.read()


def run_docker(args: list[str], *, secret: str | None = None, timeout: float = DOCKER_TIMEOUT_S, **kwargs) -> str:
    """Run ``docker`` with ``args`` and return its stdout; a missing command raises FileNotFoundError.

    ``args`` never carry a key: a key reaches Docker only through ``input`` or ``env``, and is passed as
    ``secret`` so the daemon's output can be blanked of it before it becomes the error.
    """
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, **kwargs)
    if result.returncode:
        detail = result.stderr.strip()
        if secret:
            detail = detail.replace(secret, "<key>")
        raise DockerFailed(f"docker {' '.join(args)} exited {result.returncode}\n{detail}")
    return result.stdout.strip()


def worker_key(args: list[str]) -> str:
    """Mint a scoped key with a toolkit key command; its output is a key, so on failure none of it is kept."""
    result = kernelgen_check.run_toolkit(args, timeout=LISTING_TIMEOUT_S)
    key = result.stdout.strip()
    if result.returncode == 0 and KEY_PATTERN.fullmatch(key):
        return key
    command = f"{TOOLKIT} {' '.join(args)}"
    if MISSING_TOKEN in result.stderr or MISSING_TOKEN in result.stdout or UNAUTHORIZED in result.stderr or UNAUTHORIZED in result.stdout:
        raise MissingCredentials(f"{command} found no valid Yasp API key")
    raise ToolkitFailed(f"{command} exited {result.returncode} without printing a key (its output is withheld: it may hold one)")


def worker_tags(registry_key: str) -> list[str]:
    """The tags of the worker image, from the registry's tag list, read with the registry key as the Basic password."""
    auth = base64.b64encode(f"token:{registry_key}".encode()).decode()
    request = urllib.request.Request(WORKER_TAGS_URL, headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(request, timeout=LISTING_TIMEOUT_S) as response:
        return json.load(response)["tags"]


def docker_socket_gid() -> str:
    """The group of the Docker socket, which the worker joins to launch its per-task containers."""
    return str(os.stat(DOCKER_SOCKET).st_gid)


def attach_worker(name: str, log) -> subprocess.Popen:
    """Start the created container and stay attached: the client's output is the worker's, and it exits when the worker does."""
    return subprocess.Popen(
        ["docker", "start", "--attach", name], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def remove_worker(name: str) -> None:
    """Remove the container by name, running or not; one that is already gone is what removal is after."""
    try:
        run_docker(["rm", "--force", name])
    except DockerFailed as exc:
        if NO_SUCH_CONTAINER not in str(exc):
            raise


def stop_worker(name: str) -> None:
    """Ask the worker to stop and wait for it, bounded: on SIGTERM it cancels its task and tears down the task container.

    The daemon kills the worker when the grace runs out. A stop the daemon refuses or that hangs is logged:
    the removal that follows is the contract, and the task container is checked afterwards either way.
    """
    try:
        run_docker(["stop", "--time", f"{STOP_GRACE_S:.0f}", name], timeout=STOP_GRACE_S + DOCKER_TIMEOUT_S)
    except (DockerFailed, subprocess.TimeoutExpired) as exc:
        if NO_SUCH_CONTAINER in str(exc):
            return
        logger.warning("%s: %s; removing %s by force", type(exc).__name__, exc, name, extra={"stage": "evaluator_release"})


def owned_task(name: str) -> str | None:
    """The id of the task container holding the slot if the marker it was forwarded names the worker ``name``; None otherwise.

    The id and the environment come from one inspect: the id is what the removal targets, since the shared
    name may address another worker's task by then.
    """
    try:
        output = run_docker(["inspect", "--type", "container", "--format", INSPECT_FORMAT, TASK_SLOT])
    except DockerFailed as exc:
        if NO_SUCH_CONTAINER in str(exc):
            return None
        raise
    container = json.loads(output)
    return container["id"] if f"{OWNER_VARIABLE}={name}" in (container["env"] or []) else None


def remove_owned_task(name: str) -> None:
    """Remove the task container the worker ``name`` left in the slot, if it is there, by the id the inspect proved
    is its own, never by the shared name; another worker's is never touched, and one gone meanwhile is removed."""
    task = owned_task(name)
    if task is not None:
        logger.warning("the task container %s (%s) of %s is still running; removing it", TASK_SLOT, task[:12], name, extra={"stage": "evaluator_release"})
        remove_worker(task)


def prepare_worker() -> tuple[str, str]:
    """Mint the keys, resolve the newest release of the worker image, and pull it; returns the image and the registration key."""
    logger.info("minting the registry and registration keys for the Gaia worker with %s", TOOLKIT, extra={"stage": "evaluator_start"})
    registry_key = worker_key(["registry", "key"])
    registration_key = worker_key(["eval", "key"])
    releases = [tag for tag in worker_tags(registry_key) or [] if RELEASE_PATTERN.fullmatch(tag)]  # tags is null when empty
    if not releases:
        raise DockerFailed(f"{WORKER_REPOSITORY} has no release tag to pull")
    image = f"{WORKER_REPOSITORY}:{max(releases, key=lambda tag: tuple(map(int, tag.split('.'))))}"
    logger.info("pulling %s", image, extra={"stage": "evaluator_start"})
    # A Docker config of its own, so the login neither replaces the user's nor outlives the pull.
    with tempfile.TemporaryDirectory(prefix="kronos-eval-login-") as config:
        run_docker(
            ["--config", config, "login", WORKER_REGISTRY, "-u", "token", "--password-stdin"], input=registry_key, secret=registry_key,
        )
        run_docker(["--config", config, "pull", image], timeout=PULL_TIMEOUT_S)
    return image, registration_key


def start_evaluator(args: list[str]) -> Evaluator:
    """Start this check's Gaia worker container running ``yasp-eval`` with ``args``, its output captured.

    The container is the Gaia Docker guide's ``docker run`` without the parts of a persistent deployment
    (restart policy, autoheal label, health check): it lives for this check. ``docker create`` takes the
    registration key from its own environment (``-e YASP_API_KEY`` with no value), so the key is never an
    argument and the main key in the host environment is not what the worker gets. From the moment create
    is asked, the container may exist, so any failure after that removes it by name.
    """
    image, registration_key = prepare_worker()
    name = f"kronos-eval-{uuid.uuid4().hex}"
    create = [
        "create", "--name", name, "--runtime", "nvidia",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "--group-add", docker_socket_gid(), "-v", f"{DOCKER_SOCKET}:{DOCKER_SOCKET}",
        "-e", "YASP_API_KEY",
        "-e", f"YASP_LAUNCH_WORKER_IMAGE={image}",
        "-e", "YASP_LAUNCH_DOCKER_GPUS=all",
        "-e", "YASP_LAUNCH_DOCKER_SHM_SIZE=8g",
        "-e", "YASP_LAUNCH_DOCKER_VOLUMES=yasp-dataset-cache:/mnt/dataset-cache",
        "-e", "YASP_DATASET_CACHE_DIR=/mnt/dataset-cache",
        "-e", f"{OWNER_VARIABLE}={name}",
        "-e", f"YASP_LAUNCH_DOCKER_FORWARD_ENV=YASP_DATASET_CACHE_DIR,{OWNER_VARIABLE}",
        image, EVALUATOR, *args,
    ]
    logger.info("starting the Gaia worker container %s", name, extra={"stage": "evaluator_start"})
    log = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
    try:
        run_docker(create, env={**os.environ, "YASP_API_KEY": registration_key}, secret=registration_key)
        process = attach_worker(name, log)
    except BaseException:
        # Includes an interrupt: the daemon may have created the container even if the client did not say so.
        try:
            remove_worker(name)
        except Exception as exc:
            logger.error("%s: %s", type(exc).__name__, exc, extra={"stage": "evaluator_release"})
        log.close()
        raise
    return Evaluator(name, process, log)


def stop_evaluator(evaluator: Evaluator) -> None:
    """Stop and remove this check's container, reap the attached client (killed if it does not exit), drop the log, then
    remove the task container only if this worker's is still in the slot."""
    process = evaluator.process
    try:
        try:
            stop_worker(evaluator.name)
        finally:
            remove_worker(evaluator.name)
    finally:
        try:
            try:
                process.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # exited since the wait gave up
                process.wait(timeout=STOP_TIMEOUT_S)
        finally:
            evaluator.log.close()
    remove_owned_task(evaluator.name)


def listed(output: str, gpu: str, routing: str) -> bool:
    """Whether the toolkit's ``eval workers`` output has a row whose GPU Name and Routing cells are exactly these.

    The table's rows are ``│``-separated cells; a row whose Tags cell is wider than the terminal
    continues on lines whose leading cells are empty, so those are folded into the row above.
    An output that is neither the no-match line nor a table is a toolkit flow that stopped early.
    """
    if NO_EVALUATORS in output:
        return False
    if TABLE_HEADER not in output:
        raise ToolkitFailed(f"{TOOLKIT} exited 0 without listing evaluators\n{output}")
    rows: list[list[str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("│"):
            continue
        cells = [cell.strip() for cell in line.strip("│").split("│")]
        if rows and not cells[0]:
            rows[-1] = [a + b for a, b in zip(rows[-1], cells)]
        else:
            rows.append(cells)
    return any(row[:2] == [gpu, routing] for row in rows)


def _registered(stage: str, gpu: str, routing: str, timeout: float | None = None) -> bool:
    """Ask the toolkit for the evaluators under ``routing`` and look for one advertising ``gpu``."""
    _, output = run_logged(stage, ["eval", "workers", "--tag", f"routing={routing}"], timeout=timeout)
    return listed(output, gpu, routing)


def ensure_evaluator(state: State) -> State:
    stage = "evaluator_inventory"
    global _active_evaluator
    evaluator = None
    try:
        config = state["config"]
        gpu, routing = config.gpu, config.eval_routing
        try:
            found = _registered(stage, gpu, routing, timeout=LISTING_TIMEOUT_S)
        except FileNotFoundError:
            stage = "kernelgen_toolkit"
            raise
        except MissingCredentials:
            stage = "kernelgen_credentials"
            raise
        if found:
            logger.info("an evaluator for %s with routing %s is registered", gpu, routing, extra={"stage": stage})
            return {}

        stage = "evaluator_start"
        args = ["register", "--tag", f"routing={routing}"]
        logger.info("no evaluator for %s with routing %s; starting a Gaia worker on this machine", gpu, routing, extra={"stage": stage})
        try:
            evaluator = start_evaluator(args)
            _active_evaluator = evaluator
        except FileNotFoundError as exc:
            # The toolkit (minting the keys) or Docker (the command, or its socket) was not found.
            stage = "kernelgen_toolkit" if exc.filename == TOOLKIT else "evaluator_command"
            raise
        except MissingCredentials:
            stage = "kernelgen_credentials"
            raise

        stage = "evaluator_ready"
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            code = evaluator.exit_code()
            if code is not None:
                output = evaluator.output()
                stage = "kernelgen_credentials" if MISSING_TOKEN in output or UNAUTHORIZED in output else "evaluator_start"
                raise RuntimeError(
                    f"docker start --attach {evaluator.name} exited {code} before the worker registered\n{output.strip()}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no evaluator advertising {gpu!r} with routing {routing!r} was listed within {READY_TIMEOUT_S:.0f} s "
                    f"of starting the Gaia worker container {evaluator.name}"
                )
            if _registered(stage, gpu, routing, timeout=min(LISTING_TIMEOUT_S, remaining)):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no evaluator advertising {gpu!r} with routing {routing!r} was listed within {READY_TIMEOUT_S:.0f} s "
                    f"of starting the Gaia worker container {evaluator.name}"
                )
            time.sleep(POLL_INTERVAL_S)
        logger.info("the Gaia worker %s is registered", evaluator.name, extra={"stage": stage})
    except Exception as exc:
        update = fail(logger, stage, exc)
        if evaluator is not None:  # started, so release_evaluator must stop it
            update["evaluator"] = evaluator
        return update
    return {"evaluator": evaluator}


def release_evaluator(state: State) -> State:
    global _active_evaluator
    stage = "evaluator_release"
    evaluator = state.get("evaluator")
    if evaluator is None:
        return {}
    try:
        logger.info("removing the Gaia worker container %s", evaluator.name, extra={"stage": stage})
        stop_evaluator(evaluator)
        _active_evaluator = None
    except Exception as exc:
        _active_evaluator = None
        if state.get("error"):
            # The failure that brought the check here is the one to report; the leak is for the operator.
            logger.error("%s: %s", type(exc).__name__, exc, extra={"stage": stage})
            return {"evaluator": None}
        return {**fail(logger, stage, exc), "evaluator": None}
    return {"evaluator": None}


def release_active_evaluator() -> None:
    """Remove the worker outside the graph, on interruption or a signal, preserving the original cause."""
    global _active_evaluator
    evaluator = _active_evaluator
    if evaluator is None:
        return
    try:
        stop_evaluator(evaluator)
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc, extra={"stage": "evaluator_release"})
    finally:
        _active_evaluator = None

"""Evaluator check: confirm a Gaia worker for the config's GPU and routing key is registered before kernelgen.

Runs on the host, between the export and kernelgen. The worker is the user's
persistent deployment, set up once as the Gaia Docker guide documents
(https://compile.yasp.ai/help/gaia/workers/docker/) with a
``--tag routing=<eval_routing>`` added at registration
(https://compile.yasp.ai/help/gaia/workers/#tags-routing); it stays up and serves
every generation. The check never starts, stops, or removes a container and
mints no key: it lists the evaluators registered under the config's routing key
with ``yasp-toolkit eval workers --tag routing=<key>``
(https://compile.yasp.ai/help/reference/eval/#workers) and requires one
advertising exactly the config's ``gpu``. A kernelgen job routed to a pool with
no worker waits forever, so a missing worker stops the check before anything is
submitted.
"""

from kronos_edge.kernelgen_check import TOOLKIT, ToolkitFailed, run_logged
from kronos_edge.log import get_logger
from kronos_edge.messages import EVALUATOR as CUSTOMER_MESSAGES
from kronos_edge.state import State, fail

logger = get_logger(__name__)

# A single listing must not hang the check.
LISTING_TIMEOUT_S = 30.0
# What `eval workers` prints instead of a table when nothing matches the filter.
NO_EVALUATORS = "No evaluators"
# The header of the table's first column (the GPU Name the docs pass to --gpu-name); a listing without it is not a listing.
TABLE_HEADER = "GPU Name"


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


def verify_evaluator(state: State) -> State:
    stage = "evaluator_inventory"
    try:
        config = state["config"]
        gpu, routing = config.gpu, config.eval_routing
        args = ["eval", "workers", "--tag", f"routing={routing}"]
        try:
            _, output = run_logged(stage, args, timeout=LISTING_TIMEOUT_S)
        except FileNotFoundError:
            stage = "kernelgen_toolkit"
            raise
        if not listed(output, gpu, routing):
            stage = "evaluator_missing"
            raise LookupError(
                f"no evaluator advertising {gpu!r} is registered with routing={routing} "
                f"({TOOLKIT} {' '.join(args)}); deploy the Gaia worker on this machine as README.md describes, "
                f"registered with --tag routing={routing}"
            )
        logger.info("an evaluator for %s with routing %s is registered", gpu, routing, extra={"stage": stage})
    except Exception as exc:
        return fail(logger, stage, exc)
    return {}

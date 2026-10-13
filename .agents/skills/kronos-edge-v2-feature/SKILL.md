---
name: kronos-edge-v2-feature
description: Add a feature to the Kronos Edge V2 repository using an approval-gated worktree, Claude implementation, Codex PR review, and Claude follow-up before merge handoff.
---

# Kronos Edge V2 Feature

Use this skill when Reza asks to implement a new feature or substantial change in
`kronos-edge-v2` and wants the established Claude/Codex development loop.

## Step 0: inspect, isolate, and get approval

1. Locate the repository and read its `AGENTS.md`, `README.md`, and any
   architecture document relevant to the requested feature. Treat repository
   instructions as authoritative for structure, testing, dependencies, and
   generated artifacts.
2. Inspect the current branch, worktrees, remote state, and dirty files. Do not
   overwrite existing work or reuse a worktree that contains another task.
3. Create a dedicated worktree from the current `main` (or the explicitly
   requested base) with a feature branch. Keep the existing checkout and other
   agent sessions untouched.
4. Present a concrete implementation plan before delegating or editing feature
   code. Include the intended files/areas, test strategy, external effects,
   credential needs, and the final review/merge handoff. Stop and wait for
   Reza's approval. Creating the isolated worktree is allowed in this step;
   implementation and delegation are not.

Do not infer approval from the feature request itself. If a missing decision
would materially change the design, ask before proceeding.

## Step 1: delegate implementation to Claude Code

After approval, start a dedicated Claude Code session with its working directory
set to the new worktree. Explicitly tell it to read and follow that worktree's
`AGENTS.md`; do not delegate from a different checkout. Give it the approved
plan and require it to:

- implement the smallest complete change;
- add or update tests first where practical;
- update user-facing documentation when the contract or data flow changes;
- run the repository's full relevant test suite and report exact results;
- rebuild required images when repository instructions require it;
- commit the implementation and open a PR.

Use Reza's authenticated Claude Code subscription path. Never pass
`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, or
`OPENROUTER_API_KEY` to Claude Code, and never substitute a paid API-key path.
If delegation or authentication is unavailable, report the blocker rather than
falling back silently.

Credentials are a separate handoff. Do not ask for a secret until the worker
identifies the exact variable and the task genuinely needs it. Keep secrets
local, uncommitted, and out of logs, prompts, PRs, and test fixtures. Tell Reza
precisely where to place the value and resume only after the needed handoff.

Do not merge the PR during this step.

## Step 2: independent Codex PR review

Once Claude reports a PR, verify the branch, commit, PR number, and worktree.
Start a dedicated Codex review session in the same repository/worktree context,
with `AGENTS.md` explicitly in scope. Ask Codex to:

- inspect the PR diff and the surrounding implementation;
- run the relevant tests, plus the full suite when practical;
- look for correctness, regressions, missing failure handling, security,
  compatibility, and inadequate tests;
- post only actionable findings as PR review comments;
- avoid modifying the checkout or implementing fixes.

Report the review result and link each actionable comment. If review is blocked
by authentication or sandboxing, fix the execution setup or report the blocker;
do not claim a review occurred.

## Step 3: Claude response and stabilization

Send the actual review comments back to a fresh or still-valid Claude Code
session rooted in the feature worktree. Ask Claude to inspect every comment,
apply warranted changes, reply to comments that need explanation, and rerun
tests before pushing the PR update.

If `main` advanced while the PR was in flight, have Claude fetch and rebase the
feature branch onto the latest `origin/main`, resolve conflicts carefully, run
the relevant and full tests, then update the PR with `--force-with-lease` when
needed. Never resolve conflicts by discarding unrelated work. Verify the final
worktree is clean and the remote branch points at the tested commit.

## Final handoff

Before reporting completion, verify:

- the PR is open, points at the intended branch, and contains the expected
  implementation;
- review comments are addressed or explicitly answered;
- tests and required build/evaluation steps passed, with exact results;
- no secret, temporary artifact, or unrelated change is committed;
- required CI checks and mergeability are known.

Tell Reza the PR number/link, final commit, test results, unresolved caveats,
and any required local validation. Stop before merging and ask for explicit
permission. A later explicit merge request is a separate action: recheck
mergeability and checks immediately before using the repository's established
merge convention.

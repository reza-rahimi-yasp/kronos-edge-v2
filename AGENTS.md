# AGENTS.md

Instructions for AI coding agents working in this repository.

## Scope

- Work only inside this repository. Do not read, reference, or borrow code from
  any other project on this machine.
- Do not assume anything about the project beyond what is in this repo. If the
  codebase does not show it, treat it as unknown.

## Implementation rules

- Implement the minimum change that satisfies the user's request. No extra
  features, refactors, abstractions, or "while I'm here" cleanups.
- Do not add dependencies, tooling, config, or scaffolding unless the request
  requires it.
- Keep changes small and reviewable. Prefer editing existing files over
  creating new ones.
- Match the existing style and structure of the code you touch.

## When unsure

- If the request is ambiguous, or you are not sure which feature or behavior
  the user wants, ask a concise clarifying question before writing code.
- Do not guess at requirements. A wrong assumption costs more than a question.
- State any small, low-risk assumption you do make explicitly in your reply.

## Before finishing

- Verify the change does what was asked and nothing more.
- Report exactly what was changed, what was not done, and anything that still
  needs the user's input.

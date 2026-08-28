# Development and Git Audit Rules

## Branch model

`main` always represents an audited, runnable milestone. Development happens on short-lived module branches created from the latest `main`:

```text
chore/p0-project-scaffold
feat/p0-runtime-skeleton
feat/p0-coding-loop
feat/p0-observability
feat/p0-textual-tui
feat/p1-capability-routing
experiment/p2-capability-profiling
experiment/p2-routing-evaluation
docs/final-delivery
```

Do not maintain a permanent `develop` branch. Do not commit directly to `main` after repository initialization.

## Commit policy

- Iterate locally without creating a commit for every small edit.
- Commit only after the current module satisfies its acceptance criteria.
- A module normally produces one cohesive implementation commit.
- Review corrections may use separate `fix:` commits; do not rewrite published history.
- Never commit WIP, generated virtual environments, secrets, raw traces, or temporary workspaces.

Commit examples:

```text
chore: complete p0 project scaffold
feat: complete p0 runtime skeleton
feat: complete validated coding loop
feat: add step-level observability
```

## Pre-commit audit

Run all checks before requesting permission to commit:

```text
ruff check src tests
ruff format --check src tests
mypy src
pytest
uv build
git diff --check
```

Also inspect the full diff, verify module dependency direction, and scan for API keys, Authorization headers, private endpoints, `.env` files, raw traces, and benchmark workspaces.

## Pull request and merge policy

1. Push the completed module branch only after explicit authorization.
2. Open a PR against `main` with acceptance evidence.
3. Require CI to pass and review conversations to be resolved.
4. Merge with a merge commit to preserve branch topology.
5. Do not squash or force-push an audited branch.
6. Delete the remote branch after merge.

Recommended `main` rules: require PR, require status checks, block force pushes, restrict deletion, and require conversation resolution. Do not require linear history if merge commits are used.

## Experiment evidence

- Profiling and evaluation must reuse the production runtime.
- Profiling tasks and holdout tasks must be separated by repository.
- Commit benchmark definitions, validation commands, aggregate results, and reproducibility metadata.
- Do not commit credentials, private source repositories, or unredacted raw model output.
- Record model version, strategy, temperature, pricing snapshot date, limits, and random/repetition policy.

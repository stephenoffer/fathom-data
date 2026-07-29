## What this changes

<!-- One concern per pull request. If it closes or narrows an entry in
     docs/problems.md, name it by number. -->

## Does it change what gets invalidated?

- [ ] No — names, messages, docs, or tests only
- [ ] Yes, plans get **wider** (costs compute, always safe)
- [ ] Yes, plans get **narrower**

<!-- If narrower: say why the smaller answer is provably still complete. This is the
     one direction that can serve stale data, and no measured speedup buys it. -->

## Checks

- [ ] `pytest -q`
- [ ] `ruff check src tests && ruff format --check src tests`
- [ ] `mypy`
- [ ] New behaviour has a test that fails without it
- [ ] New errors name the offending value and the next action
- [ ] New `Example:` blocks run (they are doctests)

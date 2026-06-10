# Spec: Migrate Catalax to ArviZ v1

## Objective
Update Catalax so it works against the ArviZ v1 line (`arviz>=1.0`, currently `1.1.0`). The current pin (`arviz>=0.22.0,<0.30.0`) is on the v0 line, which emits `FutureWarning` about the upcoming refactor. Goal: bump the pin to `>=1.1.0,<2.0`, fix every API break that v1 introduces, and verify that all unit + integration tests and example notebooks still run end-to-end.

User-facing Catalax API (`results.plot_posterior(...)`, `results.plot_trace(...)`, `hdi_prob=...`) should **stay the same**. The internal calls into `arviz` change, but Catalax users do not need to migrate code or argument names.

## Tech Stack
- Python 3.12–3.14 (`requires-python` bumped from `>=3.11` to `>=3.12`; forced by `arviz>=1.1.0` whose own `Requires-Python: >=3.12`)
- `arviz>=1.1.0,<2.0` (was `>=0.22.0,<0.30.0`)
- `numpyro>=0.19.0,<0.20.0` (unchanged — supplies `from_numpyro` consumer + `numpyro.diagnostics.hpdi`)
- `matplotlib`, `xarray`, `corner` (transitively used) — no change

## Commands
- Install: `uv sync` (after pyproject change)
- Build sanity: `uv pip check`
- Unit tests: `uv run pytest tests/unit -q`
- Integration tests: `uv run pytest tests/integration -q`
- All tests: `uv run pytest -q`
- Lint: `uv run ruff check catalax tests`
- Notebooks (one-by-one, headless): `uv run jupyter execute examples/HMC.ipynb` (and the same for the other 8 notebooks)
- Sanity import: `uv run python -c "import catalax; print(catalax.__version__)"`

## Project Structure (relevant subset)
```
catalax/__init__.py              → top-level imports + az.style.use(...)
catalax/dataset/__init__.py      → az.style.use(...)
catalax/mcmc/__init__.py         → az.style.use(...)
catalax/mcmc/plotting.py         → az.from_numpyro / plot_posterior / plot_trace / plot_forest / plot_mcse / plot_ess / summary
catalax/mcmc/results.py          → wraps mcmc.plotting + from_numpyro / to_netcdf / from_arviz
catalax/model/__init__.py        → az.style.use(...)
catalax/model/model.py           → az.hdi / az.InferenceData / from_arviz
catalax/neural/__init__.py       → az.style.use(...)
pyproject.toml                   → version constraint
tests/unit/, tests/integration/  → must still pass
examples/*.ipynb                 → 9 notebooks; HMC.ipynb is the only one that touches MCMC/ArviZ directly
docs/hmc/mcmc-basic.mdx          → user-facing docs referencing hdi_prob/plot_posterior on Catalax API (do NOT need to rename)
```

## ArviZ v0 → v1 API delta (empirically verified against arviz==1.1.0 in a clean venv)

| Catalax call | v1 status | Required change |
|---|---|---|
| `az.style.use("arviz-doc")` | ❌ Style removed | Switch to a v1 style. Available: `arviz-cetrino`, `arviz-tenui`, `arviz-tumma`, `arviz-variat`, `arviz-vibrant`. Pick **`arviz-vibrant`** as the closest default. |
| `az.from_numpyro(mcmc)` | ✅ Still in `arviz` namespace via `arviz-base` | No code change. Returns `xarray.DataTree` instead of `InferenceData` (alias for the same object now). |
| `az.plot_posterior(...)` | ❌ Removed | Rename to `az.plot_dist(...)`. Default is KDE; pass `kind="kde"\|"hist"\|"ecdf"` if needed. |
| `az.plot_trace(...)` | ✅ Kept; signature is now keyword-only | Pass arguments by keyword; drop unsupported v0 kwargs (e.g. `figsize` is not in the v1 signature — manage figure size via matplotlib `plt.figure(figsize=...)` *before* the call or via the returned `PlotCollection`). |
| `az.plot_forest(...)` | ✅ Kept; signature changed | Pass `var_names=` by kwarg; drop v0-only kwargs. |
| `az.plot_mcse(rug=True, extra_methods=True, ...)` | ✅ Kept | `rug` + `extra_methods` are still valid kwargs in v1 — keep as-is. |
| `az.plot_ess(kind="evolution", color=..., extra_kwargs=...)` | ⚠️ Kept but signature changed | v1 `plot_ess` has `kind="local"\|"quantile"` — **no `"evolution"`**. Use `az.plot_ess_evolution(...)` instead for the existing behavior. `color`/`extra_kwargs` no longer exist; styling now flows through `visuals=` dict. Simplest fix: switch to `plot_ess_evolution` and drop the v0 color kwargs (leave styling to the v1 default). |
| `az.summary(inf, hdi_prob=hdi_prob)` | ✅ Kept; param renamed | `hdi_prob` → `ci_prob`; also pass `ci_kind="hdi"` to keep HDI semantics (v1 default `ci_kind="eti"`). |
| `az.hdi(samples, hdi_prob=..., skipna=True)` | ✅ Kept; param renamed | `hdi_prob` → `prob`. `skipna` is still accepted. Catalax keeps its **public** `hdi_prob` kwarg; only the inner call changes. |
| `az.InferenceData` type hint | ✅ Importable but emits `MigrationWarning` | Replace type annotation with `xarray.DataTree`. Functionally identical. |
| `.to_netcdf(path)` on the InferenceData | ✅ Method on `DataTree` | Pass `engine="h5netcdf"` to be explicit (default may not pick a working engine without `netcdf4` installed). Add `h5netcdf` to deps. |
| Inside `from_arviz`: `samples.median().posterior` | ⚠️ Needs verification | `DataTree.median()` returns a `DataTree`; `.posterior` accessor still works. Will be verified during implementation. |

## Code Style
Keep existing style. Real example of the kind of change going in:

```python
# v0
inf_data = az.from_numpyro(mcmc)
return az.summary(inf_data, hdi_prob=hdi_prob)

# v1
inf_data = az.from_numpyro(mcmc)
return az.summary(inf_data, ci_prob=hdi_prob, ci_kind="hdi")
```

```python
# v0 — public API unchanged, only the inner call rewritten
hdi = az.hdi(samples, hdi_prob=hdi_prob, skipna=True)
# v1
hdi = az.hdi(samples, prob=hdi_prob, skipna=True)
```

## Testing Strategy
- **Unit tests** (`tests/unit/`) — run as the primary regression gate. The MCMC + model paths have the highest exposure.
- **Integration tests** (`tests/integration/`) — `test_ensemble.py` runs end-to-end neural training; we run it to confirm nothing broke transitively (it does not import arviz).
- **Manual smoke** — run `examples/HMC.ipynb` end-to-end (this is the only example notebook that exercises arviz). Other notebooks must at minimum *import* clean (no `az.style.use` crash on import).
- **No new tests required** — the migration is API-substitution, not new behavior. If any current test relied on a v0-only signature, we update the test rather than mask the change.

## Boundaries
- **Always do:**
  - Keep the **public** Catalax API (`results.plot_posterior`, `hdi_prob=…` etc.) unchanged.
  - Re-run the full pytest suite before declaring done.
  - Bump only the `arviz` constraint; do not touch other version pins unless an actual conflict arises.
- **Ask first:**
  - Adding `arviz-plots` / `arviz-base` / `arviz-stats` as **explicit** deps (they install transitively with `arviz`, so likely unnecessary).
  - Adding `h5netcdf` as a new dep (needed if we want a guaranteed netCDF backend for `to_netcdf`).
  - Renaming any user-visible Catalax method (e.g., calling the wrapper `plot_dist` instead of `plot_posterior`).
  - Choosing a default ArviZ style other than `arviz-vibrant`.
- **Never do:**
  - Pin to a single ArviZ version.
  - Suppress the `FutureWarning` / `MigrationWarning` instead of fixing the root cause.
  - Delete or skip failing tests to make CI green.

## Success Criteria
1. `pyproject.toml` requires `arviz>=1.1.0,<2.0` and `uv sync` produces a consistent lock.
2. `uv run pytest -q` passes locally with **0 failures and 0 errors**.
3. `import catalax` produces no `MigrationWarning` and no `arviz` `FutureWarning`.
4. `uv run jupyter execute examples/HMC.ipynb` runs to completion without raising.
5. The other 8 example notebooks import Catalax without crashing (lightweight check: first code cell executes).
6. Public Catalax API surface is unchanged — existing user code that calls `results.plot_posterior(...)`, `results.plot_trace(...)`, `Model.from_samples(hdi_prob=…)` still works without source changes.

## Decisions (resolved 2026-06-10)
1. **Style**: Drop `az.style.use(...)` entirely. Remove the four `az.style.use("arviz-doc")` lines and the four `import arviz as az` lines that exist solely to call it. Users can apply their own style.
2. **Public API**: Keep current Catalax method names (`results.plot_posterior`, `results.plot_ess`, etc.). Only the internal `az.*` call changes.
3. **`h5netcdf`**: Add as a dependency so `results.save(path)` works out-of-the-box on a fresh install.
4. **Notebook scope**: Run `examples/HMC.ipynb` end-to-end. Smoke-check the other 8 notebooks (import-only / first-cell execution).

## Assumptions
1. We migrate to **arviz 1.1.0+** (latest v1). Not a transition release that supports both v0 and v1.
2. The CI/dev env runs Python 3.13 (matches what is currently installed locally).
3. `numpyro` stays at `>=0.19.0,<0.20.0`. We do not bump it as part of this change.
4. JR-1991/master is the upstream truth; we will eventually PR this back. The branch `update-arviz` lives off the current synced master.

## Plan (Phase 2)

### Components and dependency order
1. `pyproject.toml` — version constraints
2. Package `__init__.py` cleanup — remove `az.style.use(...)`
3. `catalax/model/model.py` — `az.hdi` param rename + `az.InferenceData` type hint
4. `catalax/mcmc/plotting.py` — `plot_posterior`→`plot_dist`, `plot_ess`→`plot_ess_evolution`, `summary` param rename
5. `catalax/mcmc/results.py` — `.to_netcdf(path)` add `engine="h5netcdf"`; docstring touch-ups
6. Tests pass → notebooks run

Each step is independently verifiable: `uv sync`, `import catalax` clean, `pytest tests/unit`, then full pytest, then HMC.ipynb.

### Risks and mitigations
- **`corner.corner` may not accept a `DataTree`** as ArviZ data. **Mitigation:** if corner errors, extract samples to a `np.ndarray` / dict-of-arrays via `mcmc.get_samples()` and pass that instead (corner's native input).
- **v1 plotting returns `PlotCollection`, not a `matplotlib.Figure`.** Existing wrappers do `plt.gcf()` after the call, which works because the matplotlib backend still renders to the current figure. **Mitigation:** if `plt.gcf()` returns an empty figure, switch to using `PlotCollection.fig` (the public attribute) instead.
- **`samples.median().posterior` inside `from_arviz`** — v1 `DataTree.median()` returns a `DataTree`; group access via `.posterior` is preserved (verified empirically below).
- **`figsize`** in v1 plot signatures is gone (absorbed into `pc_kwargs`). Catalax already creates the figure via `plt.figure(figsize=figsize)` *before* the call, which still drives the matplotlib backend's figure size. **Mitigation:** none needed unless tests prove otherwise.
- **`hdi(samples, prob=...)`** — `samples` in `Model.from_samples` is a plain `Dict[str, Array]`. v1 `az.hdi` accepts xarray/DataTree natively; for raw dicts, may need to wrap in `az.convert_to_datatree(samples)`. **Mitigation:** detect at implementation; if dict-input is rejected, wrap with `convert_to_datatree` once.

### Parallel vs sequential
All edits target a small set of files; do them sequentially to keep diffs reviewable. Tests are the throttle — run after each step.

## Tasks (Phase 3)

- [ ] **T1 — Bump pyproject deps + sync env**
  - Acceptance: `pyproject.toml` has `arviz>=1.1.0,<2.0` and `h5netcdf>=1.6.4,<2`. `uv sync` exits 0. `uv run python -c "import arviz, h5netcdf; print(arviz.__version__)"` prints `1.1.0` (or higher).
  - Verify: `uv pip check`
  - Files: `pyproject.toml`, `uv.lock`

- [ ] **T2 — Remove `az.style.use("arviz-doc")` and dead `arviz` imports**
  - Acceptance: No `az.style.use(...)` calls anywhere in `catalax/`. Module-level `import arviz as az` is kept only where the symbol is actually used elsewhere in the file (currently: `catalax/mcmc/results.py`, `catalax/mcmc/plotting.py`, `catalax/model/model.py`). The other four `__init__.py` files no longer import arviz.
  - Verify: `uv run python -W error::FutureWarning -W error::DeprecationWarning -c "import catalax"` exits 0 with no warnings about arviz.
  - Files: `catalax/__init__.py`, `catalax/dataset/__init__.py`, `catalax/mcmc/__init__.py`, `catalax/model/__init__.py`, `catalax/neural/__init__.py`

- [ ] **T3 — `az.hdi` param rename + type-hint cleanup in `model.py`**
  - Acceptance: Two call sites in `from_samples` and `from_arviz` use `prob=hdi_prob` instead of `hdi_prob=hdi_prob`. Type annotation `samples: az.InferenceData` becomes `samples: "xarray.DataTree"` (or `DataTree` if already imported). Public Catalax kwarg `hdi_prob` is preserved.
  - Verify: `uv run pytest tests/unit/model -q` (or full unit run).
  - Files: `catalax/model/model.py`

- [ ] **T4 — `plot_posterior` internal: switch to `plot_dist`**
  - Acceptance: Inside `catalax/mcmc/plotting.py`, the function `plot_posterior(...)` calls `az.plot_dist(inf_data, **kwargs)` instead of `az.plot_posterior`. Public catalax name unchanged.
  - Verify: `uv run pytest tests/unit -q -k posterior or mcmc` (or full).
  - Files: `catalax/mcmc/plotting.py`

- [ ] **T5 — `plot_ess` internal: switch to `plot_ess_evolution`**
  - Acceptance: Catalax's `plot_ess(...)` calls `az.plot_ess_evolution(inf_data, backend=backend)` (no `kind`, no `color`, no `extra_kwargs`). Catalax public method name unchanged.
  - Verify: pytest unit suite, plus eyeball HMC.ipynb output.
  - Files: `catalax/mcmc/plotting.py`

- [ ] **T6 — `summary` param rename**
  - Acceptance: Inside `catalax/mcmc/plotting.py`, `az.summary(inf_data, ci_prob=hdi_prob, ci_kind="hdi")` replaces `az.summary(inf_data, hdi_prob=hdi_prob)`. Catalax's public `summary(mcmc, hdi_prob=...)` signature unchanged.
  - Verify: pytest unit run.
  - Files: `catalax/mcmc/plotting.py`

- [ ] **T7 — `to_netcdf` engine + docstring sync**
  - Acceptance: `HMCResults.to_netcdf(path)` calls `az.from_numpyro(self.mcmc).to_netcdf(path, engine="h5netcdf")`. Docstrings that still reference `arviz.InferenceData` updated to `xarray.DataTree`.
  - Verify: `uv run pytest tests/unit -q` and a quick round-trip: `python -c "..."` writing to a tmp file.
  - Files: `catalax/mcmc/results.py`

- [ ] **T8 — Full pytest + HMC.ipynb run + smoke other notebooks**
  - Acceptance: `uv run pytest -q` is green (0 failed, 0 errored). `uv run jupyter execute examples/HMC.ipynb` runs to completion. Each of the other 8 notebooks: first code cell (typically `import catalax`) executes without error (use `jupyter nbconvert --execute --to notebook --stdout --ExecutePreprocessor.timeout=60` on a subset).
  - Verify: paste pytest summary; show notebook exit codes.
  - Files: none (validation only)

- [ ] **T9 — Lint, final import sanity, commit**
  - Acceptance: `uv run ruff check catalax tests` clean (no new findings introduced by this change). `git status` shows only intended files. Commit the work in one or two logical chunks (T1+T2 = "ArviZ v1 deps and style cleanup"; T3-T7 = "Adapt internal calls to ArviZ v1 API"; T8-T9 verification artifacts if any).
  - Verify: `git diff --stat` shows the expected set of files.
  - Files: `git`

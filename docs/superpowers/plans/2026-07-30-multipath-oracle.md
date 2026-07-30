# Multi-Component Path Transport Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a target-visible `L=1/2/3` Beam-Delay transport ceiling that proves or rejects the representation capacity of multi-component transport before any model training.

**Architecture:** Add a pure NumPy module for deterministic soft source decomposition, coordinate-descent shift/gain fitting, and metric aggregation. Add a thin CLI that reuses the authenticated fold loader and atomic JSON conventions without changing the existing O2 Oracle or O4.1 runtime.

**Tech Stack:** Python 3, NumPy, existing `AntennaLayout` transforms and `MetricAccumulator`, `argparse`, `tqdm`, `unittest`.

## Global Constraints

- Work only on `task/task-018-multipath-oracle` from the approved O4.1 baseline commit.
- Do not change O4.1 train/evaluate/infer behavior or submission generation.
- The Oracle is target-visible and must always report `deployable_prediction=false`.
- `L=1` must reproduce the existing O2 reliability stage.
- Tests run with `python`, never `conda run`.
- Do not add torch dependencies to environment YAML; PyTorch remains a separate CUDA 11.8 wheel install.
- Do not commit datasets, caches, checkpoints, NPY/NPZ/PT/PTH artifacts, submissions, or server JSON outputs.

---

### Task 1: Deterministic soft Beam-Delay components

**Files:**
- Create: `solution/radio_map/learning/multi_component_transport_oracle.py`
- Create: `solution/tests/test_multi_component_transport_oracle.py`

**Interfaces:**
- Produces: `build_soft_components(source, component_count, temperature) -> np.ndarray`
- Consumes: complex NumPy `(H,V,D)` source arrays.

- [ ] **Step 1: Write failing reconstruction and validation tests**

```python
def test_soft_components_are_deterministic_and_reconstruct_source(self):
    source = np.zeros((4, 3, 8), dtype=np.complex64)
    source[0, 1, 2] = 2 + 1j
    source[3, 2, 6] = -1 + 0.5j
    first = build_soft_components(source, 2, 0.35)
    second = build_soft_components(source, 2, 0.35)
    self.assertEqual(first.shape, (2, 4, 3, 8))
    self.assertEqual(first.dtype, source.dtype)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=0), source, atol=1e-6)
    self.assertTrue(np.isfinite(first).all())

def test_one_component_is_exact_source_and_invalid_options_fail(self):
    source = np.ones((2, 2, 4), dtype=np.complex64)
    np.testing.assert_array_equal(
        build_soft_components(source, 1, 0.35)[0], source
    )
    with self.assertRaises(ValueError):
        build_soft_components(source, 0, 0.35)
    with self.assertRaises(ValueError):
        build_soft_components(source, 2, 0.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest solution.tests.test_multi_component_transport_oracle -v
```

Expected: import failure because `multi_component_transport_oracle.py` does not exist.

- [ ] **Step 3: Implement stable peak selection and periodic soft partition**

Implement:

```python
def build_soft_components(source, component_count, temperature):
    values = np.asarray(source)
    _validate_group(values)
    if not isinstance(component_count, int) or isinstance(component_count, bool) or component_count < 1:
        raise ValueError("component_count must be a positive integer")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    power = np.abs(values).reshape(-1) ** 2
    order = np.argsort(-power, kind="stable")
    centers = np.asarray(
        [np.unravel_index(int(index), values.shape) for index in order[:component_count]],
        dtype=np.float64,
    )
    if len(centers) < component_count:
        raise ValueError("component_count exceeds Beam-Delay bins")
    grids = np.indices(values.shape, dtype=np.float64)
    distance = np.zeros((component_count,) + values.shape, dtype=np.float64)
    for axis, size in enumerate(values.shape):
        delta = np.abs(grids[axis][None, ...] - centers[:, axis, None, None, None])
        delta = np.minimum(delta, size - delta) / max(1.0, size / 4.0)
        distance += delta**2
    logits = -distance / float(temperature)
    logits -= logits.max(axis=0, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=0, keepdims=True)
    return (weights * values[None, ...]).astype(values.dtype, copy=False)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```powershell
python -m unittest solution.tests.test_multi_component_transport_oracle -v
```

Expected: both component tests pass.

- [ ] **Step 5: Commit**

```powershell
git add solution/radio_map/learning/multi_component_transport_oracle.py solution/tests/test_multi_component_transport_oracle.py
git commit -m "feat: add soft Beam-Delay components"
```

### Task 2: Coordinate-descent multi-component fit

**Files:**
- Modify: `solution/radio_map/learning/multi_component_transport_oracle.py`
- Modify: `solution/tests/test_multi_component_transport_oracle.py`

**Interfaces:**
- Consumes: `build_soft_components`.
- Produces: `fit_multi_component_group(...) -> tuple[np.ndarray, dict[str, object]]`.
- Reuses: `_best_beam_delay_shift` and `_complex_fit` from `path_transport_oracle_cli.py`.

- [ ] **Step 1: Write failing `L=1` compatibility and two-path capacity tests**

Create a synthetic source with two separated peaks. Construct the target by moving the first peak by
`(+1,0,+1)` with gain `1.3j`, and the second by `(-1,+1,-2)` with gain `0.7-0.2j`.

```python
single, _ = fit_multi_component_group(
    source, target, 1, 1, 1, 2, 0.20, 3
)
multi, details = fit_multi_component_group(
    source, target, 2, 1, 1, 2, 0.20, 3
)
self.assertLess(np.mean(np.abs(multi - target) ** 2), 0.25 * np.mean(np.abs(single - target) ** 2))
self.assertEqual(details["component_count"], 2)
self.assertEqual(details["sweeps"], 3)
```

For compatibility, compare `L=1` against `_oracle_stages(...)[0]["reliability"]` on the same group.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest solution.tests.test_multi_component_transport_oracle -v
```

Expected: failure because `fit_multi_component_group` is missing.

- [ ] **Step 3: Implement coordinate descent and global reliability**

Use exactly this update order:

```python
components = build_soft_components(source, component_count, temperature)
moved = components.copy()
fitted = []
for _ in range(sweeps):
    fitted = []
    total = moved.sum(axis=0)
    for index, component in enumerate(components):
        residual = target - (total - moved[index])
        shift = _best_beam_delay_shift(
            component, residual, max_h_shift, max_v_shift, max_delay_shift
        )
        shifted = np.roll(component, shift=shift, axis=(0, 1, 2))
        gain = _complex_fit(shifted, residual)
        updated = shifted * gain
        total += updated - moved[index]
        moved[index] = updated
        fitted.append({"shift": list(shift), "gain_real": gain.real, "gain_imag": gain.imag})
candidate = moved.sum(axis=0)
delta = candidate - source
denominator = float(np.vdot(delta, delta).real)
reliability = 0.0 if denominator <= 1e-12 else float(
    np.clip(np.vdot(delta, target - source).real / denominator, 0.0, 1.0)
)
output = source + reliability * delta
```

Validate integer limits, positive finite temperature, positive sweeps, matching finite complex shapes, and output dtype.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m unittest solution.tests.test_multi_component_transport_oracle -v
```

Expected: all component and fitting tests pass.

- [ ] **Step 5: Commit**

```powershell
git add solution/radio_map/learning/multi_component_transport_oracle.py solution/tests/test_multi_component_transport_oracle.py
git commit -m "feat: fit multi-component transport oracle"
```

### Task 3: Batch metric audit and promotion report

**Files:**
- Modify: `solution/radio_map/learning/multi_component_transport_oracle.py`
- Modify: `solution/tests/test_multi_component_transport_oracle.py`

**Interfaces:**
- Produces: `audit_multi_component_ceiling(reference_channels, target_channels, layout, ...) -> dict[str, object]`.
- Reuses: `beam_delay`, `inverse_beam_delay`, `MetricAccumulator`.

- [ ] **Step 1: Write failing report test**

```python
report = audit_multi_component_ceiling(
    reference_channels,
    target_channels,
    layout,
    component_counts=(1, 2),
    max_h_shift=1,
    max_v_shift=1,
    max_delay_shift=2,
    temperature=0.20,
    sweeps=3,
    batch_size=1,
    minimum_gain=0.01,
)
self.assertEqual(report["kind"], "path_transport_multi_component_target_visible_ceiling")
self.assertTrue(report["target_visible"])
self.assertFalse(report["deployable_prediction"])
self.assertEqual(set(report["components"]), {"1", "2"})
self.assertGreater(report["components"]["2"]["score"], report["components"]["1"]["score"])
self.assertEqual(report["best_component_count"], 2)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest solution.tests.test_multi_component_transport_oracle -v
```

Expected: failure because `audit_multi_component_ceiling` is missing.

- [ ] **Step 3: Implement batched audit**

For each batch:

1. Convert reference and target channels with `beam_delay`.
2. Loop over B/P/N groups and component counts.
3. Call `fit_multi_component_group`.
4. Convert candidate Beam-Delay tensors back with `inverse_beam_delay`.
5. Update one `MetricAccumulator` per component count.

Build `gain_vs_l1`, choose the best score with deterministic lower-count tie-breaking, and set:

```python
promoted = best_component_count > 1 and best_score - l1_score >= minimum_gain
```

Wrap the batch loop in `tqdm(desc="multi-component transport ceiling", unit="batch")`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m unittest solution.tests.test_multi_component_transport_oracle -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add solution/radio_map/learning/multi_component_transport_oracle.py solution/tests/test_multi_component_transport_oracle.py
git commit -m "feat: audit multi-component transport ceiling"
```

### Task 4: Authenticated fold CLI and atomic JSON

**Files:**
- Create: `solution/radio_map/learning/multi_component_transport_oracle_cli.py`
- Create: `solution/tests/test_multi_component_transport_oracle_cli.py`

**Interfaces:**
- Consumes: `audit_multi_component_ceiling`.
- Reuses: `_load_fold` from `learning.cli`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing parser and synthetic CLI tests**

Test that parser defaults are component counts `(1,2,3)`, temperature `0.35`, sweeps `2`, minimum gain
`0.01`, and that a temporary synthetic fold/coarse file produces a JSON report without mutating inputs.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest solution.tests.test_multi_component_transport_oracle_cli -v
```

Expected: import failure because the CLI module does not exist.

- [ ] **Step 3: Implement the CLI**

Arguments:

```text
--data-dir
--cache-dir
--coarse-validation
--output
--component-counts 1 2 3
--max-h-shift 2
--max-v-shift 2
--max-delay-shift 8
--temperature 0.35
--sweeps 2
--batch-size 2
--minimum-gain 0.01
--limit-samples
```

Validate coarse shape against persisted validation indices, invoke the audit, add manifest fingerprint,
adapter SHA-256 and resolved coarse path, then atomically replace the destination JSON using a temporary file
in the destination directory.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run:

```powershell
python -m unittest solution.tests.test_multi_component_transport_oracle_cli -v
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit**

```powershell
git add solution/radio_map/learning/multi_component_transport_oracle_cli.py solution/tests/test_multi_component_transport_oracle_cli.py
git commit -m "feat: add multi-component oracle CLI"
```

### Task 5: Regression, server handoff, and execution report

**Files:**
- Create: `tasks/task-018-multipath-oracle/执行报告.md`
- Modify: `项目记录.md`

**Interfaces:**
- Produces: local verification evidence, exact SSH command, report path, final commit.

- [ ] **Step 1: Run focused tests**

```powershell
python -m unittest solution.tests.test_multi_component_transport_oracle solution.tests.test_multi_component_transport_oracle_cli solution.tests.test_path_transport_oracle_cli -v
```

Expected: all focused tests pass with no errors.

- [ ] **Step 2: Run full regression**

```powershell
python -m unittest discover -s solution/tests
```

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 3: Prepare the server command**

Use `mamba activate huaweibei-train`, never `conda run`. The command must use the server’s authenticated
fold and coarse validation paths:

```bash
python -m solution.radio_map.learning.multi_component_transport_oracle_cli \
  --data-dir dataset/Round1_Map \
  --cache-dir solution/artifacts/day7/cache_seed42_stable \
  --coarse-validation solution/artifacts/day14/o3_supervision_seed42/coarse_validation.npy \
  --component-counts 1 2 3 \
  --temperature 0.35 \
  --sweeps 2 \
  --batch-size 2 \
  --minimum-gain 0.01 \
  --output solution/artifacts/day19/multipath_oracle/report.json
```

Before delivery, verify the actual coarse filename in the supervision manifest and adjust only that path.

- [ ] **Step 4: Write execution report and concise project record**

Record task identity, commits, tests, server command, pending/actual external result, and recommendation. Do not
copy logs into either document.

- [ ] **Step 5: Commit and push**

```powershell
git add tasks/task-018-multipath-oracle/执行报告.md 项目记录.md
git commit -m "docs: report multipath oracle task"
git push -u origin task/task-018-multipath-oracle
```

Do not merge. Return the branch, final SHA, report path, one-sentence result, and `ACCEPT/REVISE/DISCARD`.

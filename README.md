# ml_split

Outgroup-based subtree merge for divide-and-conquer maximum-likelihood tree building.

`ml_split` joins two phylogenetic subtrees — each carrying one or more outgroups — at a
fresh connector edge. It chooses the best attachment by 5-branch-optimized likelihood over
a window of candidate edge pairs, then restores the inherited outgroups in place. The result
is a single merged tree over the union of the two subtrees' real taxa, with the inherited
outgroups grafted back exactly as they were.

It is a thin Python extension (pybind11) over a C++ likelihood engine shared with the
`mltree` search code.

---

## Installation

Pre-built wheels are distributed separately (by email). A wheel is tied to a specific Python
version — **install the one whose tag matches your interpreter**:

| Python | wheel tag | use it on |
|--------|-----------|-----------|
| 3.11   | `cp311`   | the cluster |
| 3.14   | `cp314`   | a 3.14 machine |

```bash
pip install ml_split-0.1.0-cp311-...-linux_x86_64.whl     # match your Python
```

A `cp311` wheel will not import on 3.14, and vice versa — the import simply fails with
"not a supported wheel on this platform." Check your interpreter with `python --version`.

The wheel needs an AVX2-capable CPU (essentially anything from 2013 on); AVX-512 kernels are
used automatically at runtime where the CPU supports them, with an AVX2 fallback otherwise.

---

## Building from source

Requirements: a C++17 compiler and Python >= 3.8. The build uses PEP 517 isolation, so
pybind11 is fetched automatically — you do not need to install it yourself.

```bash
pip install build           # one-time, if you don't have it
python -m build --wheel     # produces dist/ml_split-...-cpXY-...whl
pip install dist/*.whl
```

Or, to build and install in one step:

```bash
pip install .
```

Building for Python 3.14 requires pybind11 >= 3.0 (the first release with 3.14 support);
build isolation pulls a recent enough version automatically.

---

## Usage

```python
import ml_split

r = ml_split.merge(
    newick_a, newick_b,            # the two subtree topologies (newick strings)
    interest_a, interest_b,        # the outgroup-of-interest leaf on each side
    msa_a=seqs_a, msa_b=seqs_b,    # alignments (see below)
    inherited_a=[["x1", "x2"], ...],   # inherited-outgroup clades on side A
    inherited_b=[...],                  # inherited-outgroup clades on side B
    model="JC",
)

print(r.newick)   # merged tree
print(r.loglik)   # log-likelihood (see "loglik" note under full_blo)
```

**Alignments.** Pass either a single `msa=` (a FASTA path string, or a `{name: sequence}`
dict) covering all taxa, or both `msa_a=` and `msa_b=` as per-side dicts (merged by name; a
taxon present in both must carry the same sequence). The alignment must contain rows for
every real taxon and every inherited-outgroup taxon.

**Models.** `model="JC"` (nucleotide JC69, 4 states) or `model="JTT"` (amino-acid, 20
states). The state count is inferred from the alignment's sequence type.

---

## Options

All of the following are optional; the defaults reproduce the standard "loose" merge —
only the branches around the connector are optimized, the rest of each subtree is left as
estimated.

| argument | default | meaning |
|----------|---------|---------|
| `window` | `14` | size of the attachment-edge window searched on each side |
| `connector_init` | `0.1` | initial connector branch length |
| `full_blo` | `"off"` | full-tree branch-length polishing: `"off"`, `"fast"`, `"thorough"` |
| `eps_5blo` | `1000.0` | convergence threshold of the connector (5-branch) optimization (loose by default; lower = tighter) |
| `eps_fulltree` | `-1.0` | full-tree polishing threshold; `-1` = use the `full_blo` default |

### `full_blo` — full-tree branch-length polishing

By default the merge optimizes only the five branches around the connector. With `full_blo`
set, **all** branch lengths of the merged tree are re-optimized after the connector is
chosen:

```python
ml_split.merge(..., full_blo="fast")       # lighter polish
ml_split.merge(..., full_blo="thorough")   # tighter, slower
```

- The topology never changes — only branch lengths — so the merged tree's structure is
  identical to `full_blo="off"`.
- With `full_blo` on, `r.loglik` is the optimized likelihood of the **full output tree**
  (and matches what e.g. raxml-ng would compute on that tree). With `"off"`, `r.loglik` is
  the connector-search score over the real taxa only.
- `"thorough"` is noticeably slower on large trees; `"fast"` is the practical choice.

### Convergence thresholds

- `eps_5blo` controls the connector optimization. The default (`1000`) is deliberately
  loose — the connector is optimized cheaply per merge, and `full_blo` is there if you want
  to polish the whole tree afterward. Lower `eps_5blo` = tighter connector fit (more sweeps,
  slower).
- `eps_fulltree` overrides the full-tree polishing threshold when `full_blo` is on. Left at
  `-1`, it follows the mode default: `10` for `"fast"`, `0.1` for `"thorough"`.

You can leave all of these alone unless you specifically want to tune the optimization.

---

## Scoring a fixed tree

`ml_split.score` evaluates a tree's log-likelihood at its own branch lengths, with no
optimization — useful for cross-checking against another engine:

```python
ll = ml_split.score(newick, msa, model="JC")
```

---

## Tests

```bash
make test                                   # C++ unit suites
python tests/test_smoke_wheel.py            # end-to-end check of the installed wheel
python python/fuzz_merge.py --cases 1000    # property-based fuzzer (merge contract)
```

The fuzzer generates random trees, partitions, and alignments and asserts, on every case,
that the merge is taxa-complete, faithful to both input subtrees (the merged tree restricted
to each side reproduces that subtree exactly), likelihood-consistent, deterministic, and
invariant to input ordering. Add `--full-blo fast` (or `thorough`) to exercise the
full-tree path.

---

## Layout

```
src/        C++ engine and bindings (umbrella TU: likelihood_scorer_unit.cpp)
python/     Python drivers, validation, and the fuzzer
tests/      C++ unit tests and the wheel smoke test
setup.py    pybind11 extension build
```

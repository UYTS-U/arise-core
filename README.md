# ARISE

**Adaptive Recommendation from Interests and Shared Evidence**, accompanying
“From Initial Interests to Behavioral Evidence: Replay-Free Progressive
Recommendation under Continual User Arrival”.

Anonymous core source release. It contains the method, its training/evaluation
protocol, fixed configurations, input fingerprints, and ML-1M component removals.
Baselines, direct-tag controls, update-volume studies, measured results, data,
checkpoints, server tools, and internal project history are outside this package.
This is a core implementation release, not the complete experimental artifact.

## Structure

```text
src/arise/
  models/main.py   content/ID representation, tag prototypes, TL, CWM module
  tags.py          fixed coefficients for ordered/unordered declaration groups
  cwm.py           short/long scores and strictly lagged gain evidence
  memory.py        CT sufficient statistics, kernel, and reads
  data.py          input loading, temporal split, actual-history packing, masks
  training.py      grouped loss, block freezing, selection, and refit
  runner.py        D0 -> D1 -> D2 -> D3 lifecycle
  cache.py         frozen-backbone cache
  config.py        method configuration and seed registry
  metrics.py       unique-target metrics and user aggregation
  util.py          checkpoints, RNG state, and progress records
configs/          fixed values, seeds, dataset semantics, input hashes
scripts/          input validation, execution, and package verification
tests/            source contracts and optional tensor protocol checks
docs/             prepared-input contract and protocol limits
NOTICE            rights and provenance statement
CITATION.cff      anonymous citation metadata
MANIFEST.sha256   distributed-file checksums
```

TL is Tag Launch, CWM is Coupled Window Mixer, and CT is Cohort Trace. The model
class is `ARISE`; checkpoint field names such as `adapters` remain compatible with
the source implementation. Internal run keys `full`, `no_cold`, `no_adaptive`,
`no_cooc`, and `no_adaptive_cooc` are retained for provenance. The public command
uses the method names below. The CT checkpoint format omits baseline-only state;
start a fresh output directory rather than mixing earlier experiment caches.

## Check and install

Source checks require only Python's standard library and never start training:

```bash
python scripts/verify_release.py
python -B -m unittest discover -s tests -p 'test_release.py' -v
python scripts/run.py --list
```

For execution, use Linux with CUDA and Python 3.8–3.11. The documented source
environment used Python 3.8, PyTorch 2.3.1, NumPy 1.24.3, and SciPy 1.10.1.
The package pins these numerical libraries; pandas is constrained separately.

```bash
python -m pip install -e .
PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_protocol.py' -v
```

The optional tensor checks use synthetic inputs; they do not train models.
Release preparation checked source contracts and preserved expressions, not
training equivalence or the reported experimental scores.

## Inputs and execution

Provide authorized prepared inputs in `data/<dataset>/`, following
[docs/INPUTS.md](docs/INPUTS.md). Input fingerprints are in
`configs/data_manifest.json`. This release neither downloads nor reconstructs
the original preprocessing assets. Raw dataset downloads alone are insufficient
for exact replication.

```bash
python scripts/check_inputs.py --dataset ml1m --verify-hashes
python scripts/run.py --dataset ml1m --method arise --seed 48509147
python scripts/run.py --dataset ml1m --method without-tl --seed 48509147
```

Run ARISE for the same dataset/seed before a removal: removals inherit its D0
coefficient, short window, and scale, while retraining the remaining blocks.
Available removals are `without-tl`, `without-cwm`, `without-ct`, and
`without-cwm-ct`, restricted to ML-1M. Main runs support `ml1m`, `ml10m`, `zhihu`,
`bluesky_a`, and `yelp`.

The three training seeds were randomly drawn before the aligned suite and then
fixed across neural main/removal runs. Their exact values remain in
`configs/seeds.json`; preprocessing, tags, and tie ordering use 2040.
Runtime checkpoints and metrics are written under `results/`. Reuse completed
outputs only for the same source and inputs; use a fresh extraction to rerun.

See [docs/PROTOCOL.md](docs/PROTOCOL.md) for service boundaries and study limits.
Bluesky A is retrospective hidden-group recovery at group-end visibility.
Zhihu Topic-clustering and declaration-snapshot chronology remain unverified.

## Attribution and rights

Keep the method citation and file provenance when discussing this implementation.
Author names and repository links are withheld for review. No public reuse
license is granted by this archive; see `NOTICE`. Checksums detect changes
relative to a trusted copy; they do not prevent copying or establish authorship.

# Validation and release procedure

Automated coverage includes both Agent modes against a local fake Chat Completions server, two operation frames and tool naming schemes, current-observation binding, correction and round budgets, cancellation/late replies, execution and capture failures, frozen measurement provenance, calibration invalidation, visual completion ordering, and the migrated S1 camera/service/executor regressions.

```bash
python -m pip install -e '.[test,build]'
python -m pytest -q
python -m build
python -m twine check dist/*
```

Install the wheel into a fresh virtual environment and run `elevator-harness simulate` outside the checkout. Inspect both archives: no SDK, credentials, recordings, measured calibration, old package, cache, compiled native helper or real images should appear. The wheel contains only the core; the sdist additionally contains source examples, tests and documentation.

The S1 hardware acceptance procedure is in the example README. Do not interpret mocked execution or re-deciding on recorded images as hardware validation. Release preparation does not upload to a package index or alter Git history.

Perception regression coverage additionally checks coarse/fine coordinate units, source-to-crop-to-source mapping, image-boundary crops, refinement failure/partial visibility, canonical-box consistency between Agent and gate, cancellation during refinement, current-frame crop hashes, pose/calibration/geometry invalidation, reopening, deduplicated repeated-motion reviews, and cross-attempt correction memory.

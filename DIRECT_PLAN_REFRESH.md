# Remediation Plan to Align with Market-Fused Bivariate Poisson Specification

1. Implement Overdispersion Toggle
   - Extend `models/bivariate_poisson.py` to support optional Poisson-lognormal components per §4.5, exposing config flags and parameters (lognormal variances tied to features) and updating training, calibration, and prediction flows to honor the toggle.
   - Provide unit tests covering deterministic Poisson and overdispersed cases.

2. Availability Baselines & Historical Backfill
   - Add baseline ingestion utilities (seasonal cache, rolling averages) in `features/availability.py` or a companion module; support top-2 baseline minutes, projected minutes defaults, and player availability aggregates using past games when projections missing.
   - Update schema/loader to surface required baseline data; adjust feature builder to consume cached baselines and add tests.

3. Enhanced Evaluation & Reporting
   - Expand `evaluate.py` and metrics to compute backtests by season, spread, and total buckets; generate calibration plots (PIT, reliability), margin density visuals, and save outputs to `reports/`.
   - Add CLI options/config entries for report generation and persist summary tables.

4. Calibration Refinements
   - Revisit temperature scaling to calibrate marginal team score PMFs rather than flattened joint matrices, aligning with prompt intent. Document assumptions and ensure calibrator outputs remain compatible with inference and evaluation.

5. Documentation & Diagnostics Deliverables
   - Produce comprehensive README updates detailing math, feature tables, IO schemas, and end-to-end usage examples.
   - Create `notebooks/diagnostics.ipynb` with feature importances, calibration plots, and bucketed backtest visuals. Wire instructions into docs.

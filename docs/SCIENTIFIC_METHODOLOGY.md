# Scientific methodology

## Model traceability

Every scientific model should have four traceable elements: a primary reference, an explicit equation or algorithm, an input-domain statement, and a test. [REFERENCES.md](REFERENCES.md) is the bibliography; model documents connect those sources to code behaviour. A source citation without a domain statement is insufficient for an empirical model.

## Verification strategy

RadarSim uses several complementary test types:

- dimensional and limiting-case tests, such as \(R^{-4}\) received-power scaling and zero-rain loss;
- analytic identities, including Doppler/velocity, ambiguity limits, CFAR multipliers, and covariance-intersection consistency;
- deterministic signal tests for delay-bin placement, matched-filter peak, Doppler bin, and non-wrapping delays;
- seeded Monte Carlo tests for Swerling moments, noise statistics, false-alarm behaviour, and estimator errors;
- numerical invariants, including positive semidefinite covariance, Joseph-form update behaviour, finite values, and monotonic pull-off limits;
- integration tests for scenario round trips, simulation steps, track lifecycle, fusion latency, recording, and export; and
- offscreen UI smoke tests for construction, close behaviour, scientific metric wiring, and SAR display normalization.

Passing a test verifies the coded requirement, not every use of the model. Empirical model validation against an independent measured dataset remains a separate activity.

## Reproducible experiments

A result intended for comparison should record:

- RadarSim version and Git commit;
- complete scenario file and any runtime overrides;
- Python, NumPy, and SciPy versions;
- random seed or saved generator state;
- number of trials and discarded warm-up samples;
- estimator initialization and process/measurement covariances;
- detection threshold, pulse count, window, CFAR type, guard/reference geometry, and rank;
- propagation and clutter model choice with environmental inputs; and
- metric definition and confidence interval.

Do not compare a single stochastic run with an expected probability. For a Bernoulli detection rate, report the trial count and a binomial confidence interval. For RCS or clutter moments, report convergence and uncertainty. Preserve the raw configuration rather than only a screenshot.

## Adding or changing a model

Start from a public primary source or standard. Translate its notation into SI units, document all conversions, and identify assumptions that RadarSim adds. Implement input validation at the public boundary. Create tests from an analytic special case and, when feasible, a published numerical example. Test outside-domain behaviour separately from nominal accuracy.

If the literature does not justify an algorithm, expose it as unavailable instead of mapping its name to another method. The SAR omega-k and chirp-scaling entry points follow this rule.

## Numerical practice

Use linear units for sums and ratios, then convert to dB for reporting. Use stable linear solves instead of explicit matrix inverses. Preserve Hermitian/symmetric structure in covariance and complex-signal operations. Avoid circular array operations for physical delays unless periodicity is part of the model. State every clipping floor and whether it changes a physical quantity or only prevents a singular numerical operation.

## Review checklist

A scientific change is ready when its source is identifiable, units and signs are unambiguous, assumptions and invalid regimes are written down, tests fail under a plausible implementation error, all tests pass with fixed dependencies, and user-facing documentation matches the actual API.

Find data leakage, invalid train/evaluation splits, incorrect losses or metrics, numerical
instability, non-reproducibility, statistically unsupported conclusions, and train/serve skew.
Trace the issue through the actual data, training, inference, or evaluation flow. Identify the
unit of statistical independence and verify that splitting happens before any learned
preprocessing, feature selection, normalization, or augmentation that could leak information.

Compare the train, evaluation, and serving transformations and assumptions. Check that the
metric's aggregation matches the claim and decision threshold, and that conclusions are
supported by appropriate baselines, uncertainty or variance, and repeated seeds when stochastic
variation could change the result. Require a concrete path from the methodological flaw to a
misleading measurement, unstable result, or production behavior.

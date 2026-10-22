# Batch-size benchmark

Adam, lr=1e-4, weight decay=0, 256x256, seed=0, random initialization, 5 epochs.

| Batch | Best val Dice | Best epoch | Time (min) | Peak allocated GPU MiB |
|---|---|---|---|---|
| 2 | 0.892869 | 5 | 9.62 | 1086.4 |
| 1 | 0.846363 | 4 | 10.58 | 725.6 |

New run minus baseline Dice: -4.651 percentage points.

The baseline is reused from the previous learning-rate experiment. Configuration, software/GPU and train/validation split were checked. Model/data source changes are comments, formatting and class order; their class/function ASTs match the baseline.

Batch 1 performs 2944 optimizer updates per epoch versus 1472 for batch 2; BatchNorm behavior also changes. This compares fixed-epoch configurations, not equal-update budgets. Timings were measured in separate runs and can vary with system load. Only one seed and five epochs were used; no statistical or convergence claim is made. The test set was not evaluated in this experiment.

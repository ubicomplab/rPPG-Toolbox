# rPPG-toolbox

### Run the code:

* Install the requirements
* Additional testdata and .mat files are needed

### Progress:

| Method_POS       | BVP(RMSE) | PR   | SNR     |
| ---------------- | --------- | ---- | ------- |
| POS_SameRGB      | e-16      | 66.5 | -7.2221 |
| POS_pythonRGB    | e-2       | 66.5 | -7.37   |
| Matlab(standard) | 0         | 66.5 | -7.2221 |

| Method_CHROME    | BVP(RMSE) | PR   | SNR      |                             |
| ---------------- | --------- | ---- | -------- | --------------------------- |
| CHROME_SameRGB   | e-4       | 60   | -3.5549  | Different filtering results |
| CHROME_pythonRGB | e-4       | 66.5 | -3.52421 |                             |
| Matlab(standard) | 0         | 60   | -3.3935  |                             |

| Method_ICA       | BVP(RMSE) | PR   | SNR    |                             |
| ---------------- | --------- | ---- | ------ | --------------------------- |
| ICA_SameRGB      | e-3       | 66.5 | 3.6132 | Different filtering results |
| ICA_pythonRGB    | 1.55      | 66.5 | 3.7305 |                             |
| Matlab(standard) | 0         | 66.5 | 3.6130 |                             |

GrandTruth:67.8

### TODO:


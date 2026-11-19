# Cross-Stock and Cross-Sector Generalisation of Machine Learning Models for Next-Day Stock Direction

Code for the master's thesis of Anass Boulghalgh, MSc Data Science & Society, Tilburg University (2026).

The thesis examines whether machine learning models that predict the next-day direction of a stock (up or down) still work for stocks and sectors that were not part of the training data. Logistic Regression, LightGBM, CatBoost and TabPFN are compared in three settings that share the same test years (2020–2024):

| Setting | Trained on | Tested on |
|---|---|---|
| Within-stock | all 11 stocks | the same 11 stocks, in a later year |
| Cross-stock | 4 technology stocks | the 5th technology stock |
| Cross-sector | 2 sectors | the 3rd sector |

## Research questions

1. **RQ1.** To what extent do Logistic Regression, LightGBM, CatBoost and TabPFN differ in their ability to predict next-day stock return direction?
2. **RQ2.** To what extent do the models generalise to unseen stocks within the same sector?
3. **RQ3.** To what extent do the models generalise to a sector that was not represented during training?
4. **RQ4.** To what extent does the inclusion of technical indicators affect predictive performance and generalisation?

## Data

Daily prices of 11 large U.S. companies from three sectors, January 2015 to December 2024, from Yahoo Finance via the `yfinance` package:

* **Technology:** Apple, Microsoft, Amazon, Alphabet, Tesla
* **Energy:** ExxonMobil, Chevron, ConocoPhillips
* **Food & Beverage:** PepsiCo, Coca-Cola, Mondelez

`Dataset_Thesis.csv` is the exact dataset used in the thesis. `Dataset.py` shows how it was created. Downloading the data again (`make data`) can give slightly different adjusted prices, because Yahoo Finance recalculates them when new dividends are paid.

## Repository structure

| File | What it does |
|---|---|
| `Dataset.py` | Downloads the prices and builds `Dataset_Thesis.csv` |
| `1_Preprocessing.ipynb` | Builds the scale-free features (set A and set B) and saves `model_data.csv` |
| `2_Baseline_LogisticRegression.ipynb` | Majority-class baseline and Logistic Regression |
| `3_LightGBM.ipynb` | LightGBM (row subsampling, five seeds) |
| `4_CatBoost.ipynb` | CatBoost (five seeds) |
| `5_TabPFN.ipynb` | TabPFN v2.5 (optional, needs a GPU) |
| `6_Results.ipynb` | Validation table, main results table, Friedman and Wilcoxon tests, matched comparison of the settings |
| `7_Error_analysis.ipynb` | Confusion matrices, results per held-out stock, sector and year, feature importance, calibration |
| `8_Checks.ipynb` | Eight data and leakage checks |
| `Makefile` | Runs the whole pipeline |

The notebooks save their tables in `results/` and their figures in `figures/`.

## How to run

Python 3.11 with conda:

```bash
conda create -n thesis python=3.11
conda activate thesis
conda install -c conda-forge llvm-openmp   # needed for LightGBM on a Mac
pip install -r requirements.txt
```

Run the pipeline without TabPFN:

```bash
make all       # notebooks 1-4 and 6-8, about 6 minutes on a laptop
```

Add TabPFN (needs a GPU), after `make all`:

```bash
make tabpfn            # runs notebook 5
make results errors    # reruns notebooks 6 and 7, now including TabPFN
```

Do not run `make all` again after TabPFN: it starts with `make clean`, which deletes all results, including those of TabPFN.

Other commands:

```bash
make help       # lists the commands
make get-data   # downloads the data again (adjusted prices may differ slightly)
make clean      # removes all generated files
```

Each notebook can also be run by hand, in the order of the numbers.

## Method in short

* **Features:** returns of the last five days, intraday range, intraday return and volume change (set A), plus distance to the 10- and 50-day moving averages, RSI and MACD (set B). All features are scale-free, so they mean the same for every stock.
* **Evaluation:** expanding-window walk-forward design with one test year at a time (2020–2024) and a one-day gap between training and test data.
* **Tuning:** hyperparameters are chosen on the last training year, using only the training stocks. LightGBM and CatBoost use early stopping and are retrained with five seeds.
* **Metrics:** ROC-AUC (main), Matthews correlation coefficient, and accuracy next to the majority-class baseline.
* **Tests:** Friedman test to compare the models; Wilcoxon signed-rank test for comparisons between two groups, with a Bonferroni correction within each group of tests.

## Citation

Boulghalgh, A. (2026). *Cross-stock and cross-sector generalisation of machine learning models for next-day stock direction* [Master's thesis, Tilburg University].

## License

MIT, see `LICENSE`.

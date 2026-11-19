get-data:
	python Dataset.py

preprocess:
	jupyter nbconvert --to notebook --execute --inplace 1_Preprocessing.ipynb

baseline:
	jupyter nbconvert --to notebook --execute --inplace 2_Baseline_LogisticRegression.ipynb

lightgbm:
	jupyter nbconvert --to notebook --execute --inplace 3_LightGBM.ipynb

catboost:
	jupyter nbconvert --to notebook --execute --inplace 4_CatBoost.ipynb

tabpfn:
	jupyter nbconvert --to notebook --execute --inplace 5_TabPFN.ipynb

results:
	jupyter nbconvert --to notebook --execute --inplace 6_Results.ipynb

errors:
	jupyter nbconvert --to notebook --execute --inplace 7_Error_analysis.ipynb

checks:
	jupyter nbconvert --to notebook --execute --inplace 8_Checks.ipynb

clean:
	rm -rf results figures model_data.csv

all: clean preprocess baseline lightgbm catboost results errors checks

help:
	@echo "make all       - Run the full pipeline (without TabPFN)"
	@echo "make tabpfn    - Run TabPFN (needs a GPU), then: make results errors"
	@echo "make get-data  - Download the data again (prices may differ slightly)"
	@echo "make clean     - Remove all generated files"

.PHONY: all clean get-data preprocess baseline lightgbm catboost tabpfn results errors checks help
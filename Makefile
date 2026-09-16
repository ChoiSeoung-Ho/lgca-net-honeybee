# Reproduce every table and data figure of the article from the released
# per-image predictions and image statistics (CPU only, no images needed).
.PHONY: reproduce-tables verify-env clean
reproduce-tables:
	bash reproduce_analysis.sh ./reproduced
verify-env:
	python -c "import numpy, scipy, pandas, sklearn, matplotlib, PIL; print('analysis environment OK')"
clean:
	rm -rf ./reproduced

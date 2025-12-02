MODULE_NAME := analysis/all_data

PLOT_ALL_MOSMODEL_POINTS := $(MODULE_NAME)/plotAllPoints.py
MOSMODEL_CHART_FILE := $(MODULE_NAME)/table_walks_scatter.pdf

$(MODULE_NAME): $(MOSMODEL_CHART_FILE)

$(MOSMODEL_CHART_FILE): $(MOSMODEL_TEST_MEAN_CSV_FILE) $(MOSMODEL_TRAIN_MEAN_CSV_FILE)
	$(PLOT_ALL_MOSMODEL_POINTS) --test_mean_file=$(MOSMODEL_TEST_MEAN_CSV_FILE) --train_mean_file=$(MOSMODEL_TRAIN_MEAN_CSV_FILE) --output=$(dir $@)

$(MODULE_NAME)/clean:
	rm -f $(MOSMODEL_CHART_FILE)
	cd $(dir $@)
	rm -f *.csv
	rm -f *.pdf


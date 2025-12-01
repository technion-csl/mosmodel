MODULE_NAME := analysis
SUBMODULES := \
	single_page_size \
	growing_window_2m \
	random_window_2m \
	sliding_window \
	linear_models_coeffs \
	pebs_tlb_miss_trace \
	fixed_selector \
	genetic_selector \
	moselect \
	bayesian_optimization \
	mosrange \
	mosmodel \
	all_data \
	manual_layouts \
	vanilla

SUBMODULES := $(addprefix $(MODULE_NAME)/,$(SUBMODULES))

ARRANGE_DATA_TO_PLOT := $(MODULE_NAME)/arrangeDataToPlot.py
SCATTER_PLOT := $(MODULE_NAME)/plotScatter.gp
WHISKER_PLOT := $(MODULE_NAME)/plotWhisker.gp
POLY_PLOT := $(MODULE_NAME)/assessPolynomialModels.py
BUILD_OVERHEAD := $(MODULE_NAME)/buildOverheadSummary.py

COMMON_ANALYSIS_MAKEFILE := $(MODULE_NAME)/common.mk

include $(ROOT_DIR)/common.mk


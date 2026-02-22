MODULE_NAME := analysis/mosmodel


#************* scripts *************
CALCULATE_R_SQUARES := $(ROOT_DIR)/$(MODULE_NAME)/calculateRSquares.py
PLOT_STRATEGY_COMPARE := $(MODULE_NAME)/plot_strategy_compare.py

MOSMODEL_TEMPLATE_MAKEFILE  := $(MODULE_NAME)/template.mk
MOSMODEL_STRATEGY_MAKEFILE  := $(MODULE_NAME)/strategy_eval.mk
MOSMODEL_AUTOGEN_MAKEFILE   := $(MODULE_NAME)/autogen.mk

include $(MODULE_NAME)/strategies.mk

MOSMODEL_COMPARE := analysis/mosmodel/max_errors.pdf

# -------------------------------
# Auto-generate per-strategy module.mk files
# -------------------------------

# Files we will generate
SUBMODULES := $(MOSMODEL_STRATEGIES)
SUBMODULES  := $(foreach s,$(SUBMODULES),$(MODULE_NAME)/$(s))

MOSMODEL_STRATEGIES_DIRS := $(foreach s,$(MOSMODEL_STRATEGIES),$(MODULE_NAME)/$(s))
MOSMODEL_STRATEGY_LAYOUT_MKS := $(addsuffix /layout_generators.mk,$(MOSMODEL_STRATEGIES_DIRS))
MOSMODEL_STRATEGY_TRAIN_MKS  := $(addsuffix /train/module.mk,$(MOSMODEL_STRATEGIES_DIRS))
MOSMODEL_STRATEGY_TEST_MKS   := $(addsuffix /test/module.mk,$(MOSMODEL_STRATEGIES_DIRS))
MOSMODEL_STRATEGY_MKS        := $(addsuffix /module.mk,$(MOSMODEL_STRATEGIES_DIRS))

MOSMODEL_AUTOGEN_MKS := $(MOSMODEL_STRATEGY_LAYOUT_MKS) \
	$(MOSMODEL_STRATEGY_TRAIN_MKS) $(MOSMODEL_STRATEGY_TEST_MKS) $(MOSMODEL_STRATEGY_MKS)

# Bring in the rule generators (define blocks + foreach/eval calls)
include $(MOSMODEL_AUTOGEN_MAKEFILE)

.PHONY: $(MODULE_NAME)
$(SUBMODULES) : $(MODULE_NAME)/bootstrap
$(MODULE_NAME)/bootstrap: $(MOSMODEL_AUTOGEN_MKS)
	
$(MODULE_NAME): $(MOSMODEL_COMPARE)

# -------- clean --------
$(MODULE_NAME)/clean:
	# Delete per-strategy train/test dirs, but keep the per-strategy layout_generators.mk files.
	rm -rf $(sort $(dir $(MOSMODEL_STRATEGY_TRAIN_MKS) $(MOSMODEL_STRATEGY_TEST_MKS)))
	rm -f  $(foreach s,$(MOSMODEL_STRATEGIES_DIRS),$(s)/*.csv) $(foreach s,$(MOSMODEL_STRATEGIES_DIRS),$(s)/*.pdf)
	rm -f $(MOSMODEL_STRATEGY_LAYOUT_MKS) $(MOSMODEL_STRATEGY_MKS)
	rm -f $(MOSMODEL_COMPARE)
	
$(MODULE_NAME)/clean_strategies:
	rm -rf  $(MOSMODEL_STRATEGIES_DIRS)

#************* OUTPUTS *************
GENRATORS_TEST_ERRORS := $(foreach s,$(MOSMODEL_STRATEGIES),$(MODULE_NAME)/$(s)/test_errors.csv)


$(MOSMODEL_COMPARE): $(GENRATORS_TEST_ERRORS)
	python3 $(PLOT_STRATEGY_COMPARE) \
	--output $@ \
	--inputs $(foreach s,$(MOSMODEL_STRATEGIES),$(s)=analysis/mosmodel/$(s)/test_errors.csv)

-include $(ROOT_DIR)/common.mk

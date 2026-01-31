MODULE_NAME := analysis/mosmodel


#************* scripts *************
CALCULATE_R_SQUARES := $(ROOT_DIR)/$(MODULE_NAME)/calculateRSquares.py
PLOT_STRATEGY_COMPARE := $(MODULE_NAME)/plot_strategy_compare.py
# Better names than hardcoding "analysis/mosmodel/template.mk" everywhere
MOSMODEL_TEMPLATE_MAKEFILE  := $(MODULE_NAME)/template.mk
MOSMODEL_STRATEGY_MAKEFILE  := $(MODULE_NAME)/strategy_eval.mk

include $(MODULE_NAME)/strategies.mk

# -------------------------------
# Auto-generate per-strategy module.mk files
# -------------------------------

MOSMODEL_COMPARE := analysis/mosmodel/max_errors.pdf
TEST_EXPERIMENTS :=  random_window_2m growing_window_2m \
		sliding_window/window_20 sliding_window/window_40 sliding_window/window_60 sliding_window/window_80 \
		moselect
# Files we will generate
SUBMODULES := $(MOSMODEL_STRATEGIES) test
SUBMODULES  := $(foreach s,$(SUBMODULES),$(MODULE_NAME)/$(s))
MOSMODEL_STRATEGY_TRAIN_MKS := $(foreach s,$(MOSMODEL_STRATEGIES),$(MODULE_NAME)/$(s)/train/module.mk)
MOSMODEL_STRATEGY_MKS       := $(foreach s,$(MOSMODEL_STRATEGIES),$(MODULE_NAME)/$(s)/module.mk)
MOSMODEL_AUTOGEN_MKS        := $(MOSMODEL_STRATEGY_TRAIN_MKS) $(MOSMODEL_STRATEGY_MKS)

.PHONY: $(MODULE_NAME) $(MODULE_NAME)/bootstrap
$(SUBMODULES) : $(MODULE_NAME)/bootstrap
$(MODULE_NAME)/bootstrap: $(MOSMODEL_AUTOGEN_MKS)
$(MODULE_NAME)/bootstrap/clean:
	rm -rf $(dir $(MOSMODEL_AUTOGEN_MKS))	
$(MODULE_NAME): $(MOSMODEL_COMPARE)
$(MODULE_NAME)/clean: $(MODULE_NAME)/bootstrap/clean
	rm -f $(MOSMODEL_COMPARE)
	

#************* OUTPUTS *************
GENRATORS_TEST_ERRORS := $(foreach s,$(MOSMODEL_STRATEGIES),$(MODULE_NAME)/$(s)/test_errors.csv)

# --- rule generator: train/module.mk per strategy ---
define GEN_TRAIN_MK
$(MODULE_NAME)/$(1)/train/module.mk: $(MODULE_NAME)/strategies.mk $(MOSMODEL_TEMPLATE_MAKEFILE)
	mkdir -p $$(dir $$@)
	@{ \
		echo 'MODULE_NAME := $(MODULE_NAME)/$(1)/train'; \
		echo 'SUBMODULES :='; \
		echo ''; \
		echo 'MODEL_EXPERIMENTS := $(TRAIN_EXPERIMENTS_$(1))'; \
		echo 'include $(MOSMODEL_TEMPLATE_MAKEFILE)'; \
	} > $$@
endef
$(foreach s,$(MOSMODEL_STRATEGIES),$(eval $(call GEN_TRAIN_MK,$(s))))

# --- rule generator: strategy/module.mk per strategy (train only for now) ---
define GEN_STRATEGY_MK
$(MODULE_NAME)/$(1)/module.mk: $(MODULE_NAME)/strategies.mk $(MOSMODEL_STRATEGY_MAKEFILE)
	mkdir -p $$(dir $$@)
	@{ \
		echo 'MODULE_NAME := $(MODULE_NAME)/$(1)'; \
		echo 'SUBMODULES := train'; \
		echo 'SUBMODULES := $$$$(addprefix $$$$(MODULE_NAME)/,$$$$(SUBMODULES))'; \
		echo ''; \
		echo 'include $(MOSMODEL_STRATEGY_MAKEFILE)'; \
	} > $$@
endef
$(foreach s,$(MOSMODEL_STRATEGIES),$(eval $(call GEN_STRATEGY_MK,$(s))))

$(MOSMODEL_COMPARE): $(PLOT_STRATEGY_COMPARE) $(GENRATORS_TEST_ERRORS)
	python3 $(PLOT_STRATEGY_COMPARE) \
  	--output  $(MOSMODEL_COMPARE) \
  	--inputs $(foreach s,$(MOSMODEL_STRATEGIES),$(s)=analysis/mosmodel/$(s)/test_errors.csv) \
	--test_generators  $(TEST_EXPERIMENTS) \
	--percent

include $(ROOT_DIR)/common.mk

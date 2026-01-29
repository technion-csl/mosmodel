MODULE_NAME := analysis/mosmodel


#************* scripts *************
CALCULATE_R_SQUARES := $(ROOT_DIR)/$(MODULE_NAME)/calculateRSquares.py

# Better names than hardcoding "analysis/mosmodel/template.mk" everywhere
MOSMODEL_TEMPLATE_MAKEFILE  := $(MODULE_NAME)/template.mk
MOSMODEL_STRATEGY_MAKEFILE  := $(MODULE_NAME)/strategy_eval.mk

include $(MODULE_NAME)/strategies.mk

# -------------------------------
# Auto-generate per-strategy module.mk files
# -------------------------------

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

include $(ROOT_DIR)/common.mk

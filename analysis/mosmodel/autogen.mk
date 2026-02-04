# analysis/mosmodel/autogen.mk
#
# This file holds the rule generators (define blocks) and their instantiations.
# It assumes the including makefile already defined:
#   MODULE_NAME, MOSMODEL_STRATEGIES, MOSMODEL_TEMPLATE_MAKEFILE, MOSMODEL_STRATEGY_MAKEFILE

# --- rule generator: per-strategy layout-generators file (created once; user-editable) ---
# Default behavior:
#   TRAIN_LAYOUT_GENERATORS := <strategy>
#   TEST_LAYOUT_GENERATORS  := random_window_2m
define GEN_LAYOUT_GENS_MK
$(MODULE_NAME)/$(1)/layout_generators.mk:
	mkdir -p $$(dir $$@)
	@{ \
		echo '# Auto-generated defaults. Edit freely.'; \
		echo '# Train layout generators (space-separated):'; \
		echo 'TRAIN_LAYOUT_GENERATORS := $(1)'; \
		echo ''; \
		echo '# Test layout generators (space-separated):'; \
		echo 'TEST_LAYOUT_GENERATORS := random_window_2m'; \
	} > $$@
endef
$(foreach s,$(MOSMODEL_STRATEGIES),$(eval $(call GEN_LAYOUT_GENS_MK,$(s))))

# --- rule generator: train/module.mk per strategy ---
define GEN_TRAIN_MK
$(MODULE_NAME)/$(1)/train/module.mk: $(MODULE_NAME)/strategies.mk $(MOSMODEL_TEMPLATE_MAKEFILE) $(MODULE_NAME)/$(1)/layout_generators.mk
	mkdir -p $$(dir $$@)
	@{ \
		echo 'MODULE_NAME := $(MODULE_NAME)/$(1)/train'; \
		echo 'SUBMODULES :='; \
		echo ''; \
		echo 'include $(MODULE_NAME)/$(1)/layout_generators.mk'; \
		echo 'MODEL_EXPERIMENTS := $$$$(TRAIN_LAYOUT_GENERATORS)'; \
		echo 'include $(MOSMODEL_TEMPLATE_MAKEFILE)'; \
	} > $$@
endef
$(foreach s,$(MOSMODEL_STRATEGIES),$(eval $(call GEN_TRAIN_MK,$(s))))

# --- rule generator: test/module.mk per strategy (same structure as train) ---
define GEN_TEST_MK
$(MODULE_NAME)/$(1)/test/module.mk: $(MODULE_NAME)/strategies.mk $(MOSMODEL_TEMPLATE_MAKEFILE) $(MODULE_NAME)/$(1)/layout_generators.mk
	mkdir -p $$(dir $$@)
	@{ \
		echo 'MODULE_NAME := $(MODULE_NAME)/$(1)/test'; \
		echo 'SUBMODULES :='; \
		echo ''; \
		echo 'include $(MODULE_NAME)/$(1)/layout_generators.mk'; \
		echo 'MODEL_EXPERIMENTS := $$$$(TEST_LAYOUT_GENERATORS)'; \
		echo 'include $(MOSMODEL_TEMPLATE_MAKEFILE)'; \
	} > $$@
endef
$(foreach s,$(MOSMODEL_STRATEGIES),$(eval $(call GEN_TEST_MK,$(s))))

# --- rule generator: strategy/module.mk per strategy (train + test) ---
define GEN_STRATEGY_MK
$(MODULE_NAME)/$(1)/module.mk: $(MODULE_NAME)/strategies.mk $(MOSMODEL_STRATEGY_MAKEFILE)
	mkdir -p $$(dir $$@)
	@{ \
		echo 'MODULE_NAME := $(MODULE_NAME)/$(1)'; \
		echo 'SUBMODULES := train test'; \
		echo 'SUBMODULES := $$$$(addprefix $$$$(MODULE_NAME)/,$$$$(SUBMODULES))'; \
		echo ''; \
		echo 'include $(MOSMODEL_STRATEGY_MAKEFILE)'; \
	} > $$@
endef
$(foreach s,$(MOSMODEL_STRATEGIES),$(eval $(call GEN_STRATEGY_MK,$(s))))


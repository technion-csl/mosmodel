# analysis/mosmodel/autogen.mk
#
# This file holds the rule generators (define blocks) and their instantiations.
# It assumes the including makefile already defined:
#   MODULE_NAME, MOSMODEL_STRATEGIES, MOSMODEL_TEMPLATE_MAKEFILE, MOSMODEL_STRATEGY_MAKEFILE
#
# Override model from a parent repo by either:
#   1) setting variables before including analysis/mosmodel/module.mk, or
#   2) passing MOSMODEL_USER_CONFIG_MK=/abs/path/to/config.mk
#
# Supported per-strategy overrides (in config.mk / parent makefile / CLI):
#   MOSMODEL_STRATEGIES
#   MOSMODEL_DEFAULT_TEST_LAYOUT_GENERATORS
#   MOSMODEL_TRAIN_LAYOUT_GENERATORS_<strategy>
#   MOSMODEL_TEST_LAYOUT_GENERATORS_<strategy>

# --- rule generator: per-strategy layout-generators file (created once; user-editable) ---
# This file is safe to keep under version control or edit locally.
# It uses ?= so parent/CLI overrides win.
#
# Defaults:
#   train = <strategy>
#   test  = $(MOSMODEL_DEFAULT_TEST_LAYOUT_GENERATORS)
#
# It also defines convenience aliases TRAIN_LAYOUT_GENERATORS / TEST_LAYOUT_GENERATORS
# for older templates that expect them.
define GEN_LAYOUT_GENS_MK
$(MODULE_NAME)/$(1)/layout_generators.mk:
	mkdir -p $$(dir $$@)
	@{ \
		echo '# Auto-generated defaults. Edit freely or override from parent/CLI.'; \
		echo '# Strategy: $(1)'; \
		echo ''; \
		echo 'MOSMODEL_TRAIN_LAYOUT_GENERATORS_$(1) ?= $(1)'; \
		echo 'MOSMODEL_TEST_LAYOUT_GENERATORS_$(1)  ?= $$$$(MOSMODEL_DEFAULT_TEST_LAYOUT_GENERATORS)'; \
		echo ''; \
		echo '# Convenience aliases (used by some templates):'; \
		echo 'TRAIN_LAYOUT_GENERATORS := $$$$(MOSMODEL_TRAIN_LAYOUT_GENERATORS_$(1))'; \
		echo 'TEST_LAYOUT_GENERATORS  := $$$$(MOSMODEL_TEST_LAYOUT_GENERATORS_$(1))'; \
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

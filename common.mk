.PHONY: $(MODULE_NAME)
$(MODULE_NAME): $(SUBMODULES)

CLEAN_TARGET := $(MODULE_NAME)/clean
.PHONY: $(CLEAN_TARGET)
CLEAN_DEPS := $(addsuffix /clean,$(SUBMODULES))
$(CLEAN_TARGET): $(CLEAN_DEPS)

-include $(patsubst %,%/module.mk,$(SUBMODULES))


MODULE_NAME := experiments/vanilla
VANILLA_LAYOUTS ?= layout_all4kb
LAYOUTS := $(VANILLA_LAYOUTS)

VANILLA_EXPERIMENT := $(MODULE_NAME)

NUM_OF_REPEATS := $(DEFAULT_NUM_OF_REPEATS)

VANILLA_RUN = 1
include $(EXPERIMENTS_TEMPLATE)
undefine VANILLA_RUN

$(MODULE_NAME)/clean:
	rm -rf experiments/vanilla/layout_all4kb


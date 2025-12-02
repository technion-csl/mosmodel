MODULE_NAME := experiments/memory_footprint
LAYOUTS := layout4kb

EXTRA_ARGS_FOR_MOSALLOC := --analyze
NUM_OF_REPEATS := 1

include $(EXPERIMENTS_TEMPLATE)

CREATE_MEMORY_FOOTPRINT_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES):
	ram_size_kb=$(shell grep MemTotal /proc/meminfo | cut -d ':' -f 2 | sed 's, ,,g' | sed 's,kB,,g')
	$(CREATE_MEMORY_FOOTPRINT_LAYOUTS) --mem_max_size_kb=$$ram_size_kb \
		--output=$(dir $@)/..

# undefine LAYOUTS to allow next makefiles to use the defaults LAYOUTS
undefine EXTRA_ARGS_FOR_MOSALLOC
undefine LAYOUTS
undefine NUM_OF_REPEATS

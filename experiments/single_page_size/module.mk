MODULE_NAME := experiments/single_page_size
SINGLE_PAGE_SIZE_LAYOUTS ?= layout4kb layout2mb layout1gb  
LAYOUTS := $(SINGLE_PAGE_SIZE_LAYOUTS)

SINGLE_PAGE_SIZE_EXPERIMENT := $(MODULE_NAME)

EXTRA_ARGS_FOR_MOSALLOC := --analyze


include $(EXPERIMENTS_TEMPLATE)

CREATE_SINGLE_PAGE_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE)
	$(CREATE_SINGLE_PAGE_LAYOUTS) --memory_footprint=$< \
		--output=$(dir $@)/..


$(MODULE_NAME)/clean:
	rm -rf experiments/single_page_size/layouts

# undefine LAYOUTS to allow next makefiles to use the defaults LAYOUTS
undefine EXTRA_ARGS_FOR_MOSALLOC
undefine LAYOUTS

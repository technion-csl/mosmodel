MODULE_NAME := experiments/single_page_size
SINGLE_PAGE_SIZE_LAYOUTS ?= layout2mb layout4kb
LAYOUTS := $(SINGLE_PAGE_SIZE_LAYOUTS)

SINGLE_PAGE_SIZE_EXPERIMENT := $(MODULE_NAME)

EXTRA_ARGS_FOR_MOSALLOC := --analyze
CRIU_RUN ?= 0

include $(EXPERIMENTS_ROOT)/criu_single_page_size/module.mk

measurement_run_single_args =
ifeq ($(CRIU_RUN),1)
measurement_run_single_args = --checkpoint-dir "$(call criu_sps_checkpoint_dir,$(1))" --checkpoint-archive-dir "$(CRIU_CHECKPOINT_ARCHIVE_ROOT)/$(CRIU_BENCHMARK_ID)/$(1)"
endif

include $(EXPERIMENTS_TEMPLATE)

ifeq ($(CRIU_RUN),1)
define SINGLE_PAGE_SIZE_CRIU_CHECKPOINT_dependency
$(foreach repeat,$(REPEATS),$(EXPERIMENT_DIR)/$(1)/$(repeat)/perf.out): $(call criu_sps_checkpoint_done,$(1))
endef
$(foreach layout,$(LAYOUTS),$(eval $(call SINGLE_PAGE_SIZE_CRIU_CHECKPOINT_dependency,$(layout))))
endif

CREATE_SINGLE_PAGE_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE)
	$(CREATE_SINGLE_PAGE_LAYOUTS) --memory_footprint=$< \
		--output=$(dir $@)/..


$(MODULE_NAME)/clean:
	rm -rf experiments/single_page_size/layouts

# undefine local variables to allow next makefiles to use their defaults
undefine measurement_run_single_args
undefine EXTRA_ARGS_FOR_MOSALLOC
undefine LAYOUTS

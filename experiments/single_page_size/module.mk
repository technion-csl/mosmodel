MODULE_NAME := experiments/single_page_size
SINGLE_PAGE_SIZE_LAYOUTS ?= layout2mb layout4kb
LAYOUTS := $(SINGLE_PAGE_SIZE_LAYOUTS)

SINGLE_PAGE_SIZE_EXPERIMENT := $(MODULE_NAME)

EXTRA_ARGS_FOR_MOSALLOC := --analyze
CRIU_RUN ?= 0

# Checkpoints are generic; this experiment supplies its layout directory.
CHECKPOINT_LAYOUT_DIR := $(MODULE_NAME)/layouts
include $(EXPERIMENTS_ROOT)/checkpoints/module.mk

measurement_run_prerequisites =
measurement_run_single_args =
measurement_run_pair_args =

ifeq ($(CRIU_RUN),1)
ifneq ($(strip $(I_START1)),0)
measurement_run_prerequisites = $(call checkpoint_archive_done,$(CHECKPOINT_BENCHMARK1_ID),$(1))
measurement_run_single_args = --checkpoint-dir "$(call checkpoint_workspace_dir,$(CHECKPOINT_BENCHMARK1_ID),$(1))" --checkpoint-archive-dir "$(call checkpoint_archive_dir,$(CHECKPOINT_BENCHMARK1_ID),$(1))"
endif

ifeq ($(RUN_MODE),smt)
measurement_run_pair_args = --criu-run
ifneq ($(strip $(I_START1)),0)
measurement_run_pair_args += --checkpoint-dir1 "$(call checkpoint_workspace_dir,$(CHECKPOINT_BENCHMARK1_ID),$(1))" --checkpoint-archive-dir1 "$(call checkpoint_archive_dir,$(CHECKPOINT_BENCHMARK1_ID),$(1))"
endif
ifneq ($(strip $(I_START2)),0)
measurement_run_prerequisites += $(CHECKPOINT_SIDE2_ARCHIVE_DONE)
measurement_run_pair_args += --checkpoint-dir2 "$(call checkpoint_workspace_dir,$(CHECKPOINT_BENCHMARK2_ID),$(SMT_CORUNNER_CHECKPOINT_LAYOUT))" --checkpoint-archive-dir2 "$(call checkpoint_archive_dir,$(CHECKPOINT_BENCHMARK2_ID),$(SMT_CORUNNER_CHECKPOINT_LAYOUT))"
endif
endif
endif

include $(EXPERIMENTS_TEMPLATE)

CREATE_SINGLE_PAGE_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE)
	$(CREATE_SINGLE_PAGE_LAYOUTS) --memory_footprint=$< \
		--output=$(dir $@)/..


$(MODULE_NAME)/clean:
	rm -rf experiments/single_page_size/layouts

# undefine local variables to allow next makefiles to use their defaults
undefine measurement_run_prerequisites
undefine measurement_run_single_args
undefine measurement_run_pair_args
undefine CHECKPOINT_LAYOUT_DIR
undefine EXTRA_ARGS_FOR_MOSALLOC
undefine LAYOUTS

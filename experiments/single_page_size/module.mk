MODULE_NAME := experiments/single_page_size
SINGLE_PAGE_SIZE_LAYOUTS ?= layout2mb layout4kb
LAYOUTS := $(SINGLE_PAGE_SIZE_LAYOUTS)

SINGLE_PAGE_SIZE_EXPERIMENT := $(MODULE_NAME)

EXTRA_ARGS_FOR_MOSALLOC := --analyze
CRIU_RUN ?= 0

# CRIU restore is selected independently per side. I_START=0 keeps that side
# on the native start-barrier path, even when CRIU_RUN=1.
CRIU_RESTORE_SIDE1 := 0
CRIU_RESTORE_SIDE2 := 0
ifeq ($(CRIU_RUN),1)
ifneq ($(strip $(I_START1)),0)
CRIU_RESTORE_SIDE1 := 1
endif
ifeq ($(RUN_MODE),smt)
ifneq ($(strip $(I_START2)),0)
CRIU_RESTORE_SIDE2 := 1
endif
endif
endif

# Backward-compatible ST name used by existing rules.
CRIU_RESTORE_RUN := $(CRIU_RESTORE_SIDE1)

include $(EXPERIMENTS_ROOT)/criu_single_page_size/module.mk

measurement_run_single_args =
ifeq ($(CRIU_RESTORE_SIDE1),1)
measurement_run_single_args = --checkpoint-dir "$(call criu_sps_checkpoint_dir,$(1))" --checkpoint-archive-dir "$(CRIU_CHECKPOINT_ARCHIVE_ROOT)/$(CRIU_BENCHMARK_ID)/$(1)"
endif

measurement_run_pair_args =
ifeq ($(CRIU_RUN),1)
measurement_run_pair_args = --criu-run
ifeq ($(CRIU_RESTORE_SIDE1),1)
measurement_run_pair_args += --checkpoint-dir1 "$(call criu_sps_benchmark_checkpoint_dir,$(CRIU_BENCHMARK1_ID),$(1))" --checkpoint-archive-dir1 "$(CRIU_CHECKPOINT_ARCHIVE_ROOT)/$(CRIU_BENCHMARK1_ID)/$(1)"
endif
ifeq ($(CRIU_RESTORE_SIDE2),1)
measurement_run_pair_args += --checkpoint-dir2 "$(call criu_sps_benchmark_checkpoint_dir,$(CRIU_BENCHMARK2_ID),$(SMT_CORUNNER_CHECKPOINT_LAYOUT))" --checkpoint-archive-dir2 "$(CRIU_CHECKPOINT_ARCHIVE_ROOT)/$(CRIU_BENCHMARK2_ID)/$(SMT_CORUNNER_CHECKPOINT_LAYOUT)"
endif
endif

include $(EXPERIMENTS_TEMPLATE)

ifeq ($(CRIU_RESTORE_SIDE1),1)
define SINGLE_PAGE_SIZE_CRIU_SIDE1_CHECKPOINT_dependency
$(foreach repeat,$(REPEATS),$(EXPERIMENT_DIR)/$(1)/$(repeat)/perf.out): $(call criu_sps_benchmark_checkpoint_done,$(CRIU_BENCHMARK1_ID),$(1))
endef
$(foreach layout,$(LAYOUTS),$(eval $(call SINGLE_PAGE_SIZE_CRIU_SIDE1_CHECKPOINT_dependency,$(layout))))
endif

ifeq ($(CRIU_RESTORE_SIDE2),1)
define SINGLE_PAGE_SIZE_CRIU_SIDE2_CHECKPOINT_dependency
$(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(EXPERIMENT_DIR)/$(layout)/$(repeat)/perf.out)): $(CRIU_SMT_SIDE2_CHECKPOINT_DONE)
endef
$(eval $(call SINGLE_PAGE_SIZE_CRIU_SIDE2_CHECKPOINT_dependency))
endif

CREATE_SINGLE_PAGE_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE)
	$(CREATE_SINGLE_PAGE_LAYOUTS) --memory_footprint=$< \
		--output=$(dir $@)/..


$(MODULE_NAME)/clean:
	rm -rf experiments/single_page_size/layouts

# undefine local variables to allow next makefiles to use their defaults
undefine measurement_run_single_args
undefine measurement_run_pair_args
undefine CRIU_RESTORE_RUN
undefine CRIU_RESTORE_SIDE1
undefine CRIU_RESTORE_SIDE2
undefine EXTRA_ARGS_FOR_MOSALLOC
undefine LAYOUTS

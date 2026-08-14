##### Generic CRIU checkpoint paths and creation rules

CHECKPOINTS_DIR := experiments/checkpoints
CHECKPOINT_BUILD_DIR := $(CHECKPOINTS_DIR)/.build
CRIU_CHECKPOINT_ARCHIVE_ROOT ?= /scratch/mosmodel-checkpoints/archive
CHECKPOINT_MOSALLOC_EXTRA_ARGS := $(EXTRA_ARGS_FOR_MOSALLOC)
CHECKPOINT_LAYOUT_SOURCE_DIR := $(CHECKPOINT_LAYOUT_DIR)

checkpoint_benchmark_id_from_path = $(if $(filter $(benchmarks_root)/%,$(1)),$(patsubst $(benchmarks_root)/%,%,$(1)),$(notdir $(1)))
CHECKPOINT_BENCHMARK1_ID ?= $(call checkpoint_benchmark_id_from_path,$(BENCHMARK_PATH))
CHECKPOINT_BENCHMARK2_ID ?= $(call checkpoint_benchmark_id_from_path,$(BENCHMARK2))

# Local mutable restore workspace. criu_restore.py recreates work/ here from the
# immutable archive before each restore.
checkpoint_workspace_dir = $(CHECKPOINTS_DIR)/$(1)/$(2)

# Immutable archive: the checkpoint.done marker is the Make prerequisite.
checkpoint_archive_dir = $(CRIU_CHECKPOINT_ARCHIVE_ROOT)/$(1)/$(2)
checkpoint_archive_done = $(call checkpoint_archive_dir,$(1),$(2))/checkpoint.done
checkpoint_build_dir = $(CHECKPOINT_BUILD_DIR)/$(1)/$(2)

# Side 1 uses the experiment layout selected by the measurement target.
ifeq ($(CRIU_RUN),1)
ifneq ($(strip $(I_START1)),0)
$(CRIU_CHECKPOINT_ARCHIVE_ROOT)/$(CHECKPOINT_BENCHMARK1_ID)/%/checkpoint.done: | $(CHECKPOINT_LAYOUT_SOURCE_DIR)/%.csv experiments-prerequisites
	$(PYTHON) -m scripts.mosmodel_controller.create_checkpoint \
		--benchmark "$(BENCHMARK_PATH)" \
		--checkpoint-dir "$(@D)" \
		--run-dir "$(call checkpoint_build_dir,$(CHECKPOINT_BENCHMARK1_ID),$*)" \
		--force \
		--layout "$(abspath $(CHECKPOINT_LAYOUT_SOURCE_DIR)/$*.csv)" \
		--i-start "$(I_START1)" \
		--num-threads "$(NUMBER_OF_THREADS)" \
		--prefix "$(SET_CPU_MEMORY_AFFINITY) $(BOUND_MEMORY_NODE) $(CPU_MEMORY_AFFINITY_ARGS1)" \
		--submit "$(RUN_MOSALLOC_TOOL) --library $(MOSALLOC_TOOL) -cpf $(abspath $(CHECKPOINT_LAYOUT_SOURCE_DIR)/$*.csv) $(CHECKPOINT_MOSALLOC_EXTRA_ARGS) --"
endif
endif

# Side 2 uses the fixed SMT co-runner layout. I_START2=0 is native and therefore
# has neither a checkpoint argument nor a checkpoint prerequisite.
SMT_CORUNNER_CHECKPOINT_LAYOUT := $(if $(strip $(SMT_CORUNNER_LAYOUT_FILE)),$(SMT_CORUNNER_LAYOUT),native)
CHECKPOINT_SIDE2_ARCHIVE_DONE := $(call checkpoint_archive_done,$(CHECKPOINT_BENCHMARK2_ID),$(SMT_CORUNNER_CHECKPOINT_LAYOUT))

ifeq ($(CRIU_RUN),1)
ifeq ($(RUN_MODE),smt)
ifneq ($(strip $(I_START2)),0)
ifeq ($(strip $(SMT_CORUNNER_LAYOUT_FILE)),)
$(CHECKPOINT_SIDE2_ARCHIVE_DONE): | experiments-prerequisites
	$(PYTHON) -m scripts.mosmodel_controller.create_checkpoint \
		--benchmark "$(BENCHMARK2)" \
		--checkpoint-dir "$(@D)" \
		--run-dir "$(call checkpoint_build_dir,$(CHECKPOINT_BENCHMARK2_ID),$(SMT_CORUNNER_CHECKPOINT_LAYOUT))" \
		--force \
		--i-start "$(I_START2)" \
		--num-threads "$(SMT_THREADS2)" \
		--prefix "$(SET_CPU_MEMORY_AFFINITY) $(BOUND_MEMORY_NODE) $(CPU_MEMORY_AFFINITY_ARGS2)"
else
$(CHECKPOINT_SIDE2_ARCHIVE_DONE): | $(SMT_CORUNNER_LAYOUT_FILE) experiments-prerequisites
	$(PYTHON) -m scripts.mosmodel_controller.create_checkpoint \
		--benchmark "$(BENCHMARK2)" \
		--checkpoint-dir "$(@D)" \
		--run-dir "$(call checkpoint_build_dir,$(CHECKPOINT_BENCHMARK2_ID),$(SMT_CORUNNER_CHECKPOINT_LAYOUT))" \
		--force \
		--layout "$(abspath $(SMT_CORUNNER_LAYOUT_FILE))" \
		--i-start "$(I_START2)" \
		--num-threads "$(SMT_THREADS2)" \
		--prefix "$(SET_CPU_MEMORY_AFFINITY) $(BOUND_MEMORY_NODE) $(CPU_MEMORY_AFFINITY_ARGS2)" \
		--submit "$(RUN_MOSALLOC_TOOL) --library $(MOSALLOC_TOOL) -cpf $(abspath $(SMT_CORUNNER_LAYOUT_FILE)) $(CHECKPOINT_MOSALLOC_EXTRA_ARGS) --"
endif
endif
endif
endif

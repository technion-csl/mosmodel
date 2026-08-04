##### Layout-specific CRIU checkpoints for the single-page-size experiment

CRIU_SINGLE_PAGE_SIZE_DIR := experiments/criu_single_page_size
CRIU_SINGLE_PAGE_SIZE_LAYOUTS ?= $(SINGLE_PAGE_SIZE_LAYOUTS)

CRIU_CHECKPOINT_ARCHIVE_ROOT ?= /scratch/mosmodel-checkpoints/archive
CRIU_CHECKPOINT_MOSALLOC_EXTRA_ARGS := $(EXTRA_ARGS_FOR_MOSALLOC)

criu_benchmark_id_from_path = $(if $(filter $(benchmarks_root)/%,$(1)),$(patsubst $(benchmarks_root)/%,%,$(1)),$(notdir $(1)))
CRIU_BENCHMARK1_ID ?= $(call criu_benchmark_id_from_path,$(BENCHMARK_PATH))
CRIU_BENCHMARK_ID ?= $(CRIU_BENCHMARK1_ID)
CRIU_BENCHMARK2_ID ?= $(call criu_benchmark_id_from_path,$(BENCHMARK2))

# Backward-compatible helpers for the ST/side1 checkpoint rules.
criu_sps_checkpoint_dir = $(CRIU_SINGLE_PAGE_SIZE_DIR)/$(CRIU_BENCHMARK_ID)/$(1)
criu_sps_checkpoint_done = $(call criu_sps_checkpoint_dir,$(1))/checkpoint.done

# Benchmark-explicit helpers used by SMT side-specific restore arguments.
criu_sps_benchmark_checkpoint_dir = $(CRIU_SINGLE_PAGE_SIZE_DIR)/$(1)/$(2)
criu_sps_benchmark_checkpoint_done = $(call criu_sps_benchmark_checkpoint_dir,$(1),$(2))/checkpoint.done

CRIU_SPS_CHECKPOINTS := $(foreach layout,$(CRIU_SINGLE_PAGE_SIZE_LAYOUTS),$(call criu_sps_checkpoint_done,$(layout)))

.PHONY: $(CRIU_SINGLE_PAGE_SIZE_DIR)

$(CRIU_SINGLE_PAGE_SIZE_DIR): $(CRIU_SPS_CHECKPOINTS)

# Side1/ST checkpoints keep the existing behavior: create the checkpoint from
# the local single-page-size layout when it is not already in the shared archive.
$(CRIU_SPS_CHECKPOINTS): $(CRIU_SINGLE_PAGE_SIZE_DIR)/$(CRIU_BENCHMARK_ID)/%/checkpoint.done: experiments/single_page_size/layouts/%.csv | experiments-prerequisites
	archive_dir="$(CRIU_CHECKPOINT_ARCHIVE_ROOT)/$(CRIU_BENCHMARK_ID)/$*"; \
	if [ -f "$$archive_dir/checkpoint.done" ]; then \
		mkdir -p "$(@D)"; \
		sudo cp -a "$$archive_dir/." "$(@D)/"; \
	else \
		$(PYTHON) -m scripts.mosmodel_controller.create_layout_checkpoint \
			--benchmark "$(BENCHMARK_PATH)" \
			--checkpoint-dir "$(@D)" \
			--layout "$(ROOT_DIR)/$<" \
			--i-start "$(I_START1)" \
			--num-threads "$(NUMBER_OF_THREADS)" \
			--prefix "$(SET_CPU_MEMORY_AFFINITY) $(BOUND_MEMORY_NODE) $(CPU_MEMORY_AFFINITY_ARGS1)" \
			--submit "$(RUN_MOSALLOC_TOOL) --library $(MOSALLOC_TOOL) -cpf $(ROOT_DIR)/$(@D)/layout.csv $(CRIU_CHECKPOINT_MOSALLOC_EXTRA_ARGS) --"; \
		mkdir -p "$$archive_dir"; \
		sudo cp -a "$(@D)/." "$$archive_dir/"; \
	fi

# The SMT co-runner can either run natively (no layout file) or through a fixed
# Mosalloc layout. When a layout file is supplied, its checkpoint uses the same
# layout name as ST and can reuse the shared archive. A checkpoint created without
# Mosalloc is stored under the layout-independent archive key "native".
SMT_CORUNNER_CHECKPOINT_LAYOUT := $(if $(strip $(SMT_CORUNNER_LAYOUT_FILE)),$(SMT_CORUNNER_LAYOUT),native)
CRIU_SMT_SIDE2_CHECKPOINT_DONE := $(call criu_sps_benchmark_checkpoint_done,$(CRIU_BENCHMARK2_ID),$(SMT_CORUNNER_CHECKPOINT_LAYOUT))

ifeq ($(RUN_MODE),smt)
ifeq ($(CRIU_RUN),1)
ifneq ($(strip $(I_START2)),0)
ifeq ($(filter $(CRIU_SMT_SIDE2_CHECKPOINT_DONE),$(CRIU_SPS_CHECKPOINTS)),)
$(CRIU_SMT_SIDE2_CHECKPOINT_DONE): $(SMT_CORUNNER_LAYOUT_FILE) | experiments-prerequisites
	archive_dir="$(CRIU_CHECKPOINT_ARCHIVE_ROOT)/$(CRIU_BENCHMARK2_ID)/$(SMT_CORUNNER_CHECKPOINT_LAYOUT)"; \
	if [ -f "$$archive_dir/checkpoint.done" ]; then \
		mkdir -p "$(@D)"; \
		sudo cp -a "$$archive_dir/." "$(@D)/"; \
	elif [ -z "$(strip $(SMT_CORUNNER_LAYOUT_FILE))" ]; then \
		$(PYTHON) -m scripts.mosmodel_controller.create_layout_checkpoint \
			--benchmark "$(BENCHMARK2)" \
			--checkpoint-dir "$(@D)" \
			--i-start "$(I_START2)" \
			--num-threads "$(SMT_THREADS2)" \
			--prefix "$(SET_CPU_MEMORY_AFFINITY) $(BOUND_MEMORY_NODE) $(CPU_MEMORY_AFFINITY_ARGS2)"; \
		mkdir -p "$$archive_dir"; \
		sudo cp -a "$(@D)/." "$$archive_dir/"; \
	else \
		$(PYTHON) -m scripts.mosmodel_controller.create_layout_checkpoint \
			--benchmark "$(BENCHMARK2)" \
			--checkpoint-dir "$(@D)" \
			--layout "$(abspath $(SMT_CORUNNER_LAYOUT_FILE))" \
			--i-start "$(I_START2)" \
			--num-threads "$(SMT_THREADS2)" \
			--prefix "$(SET_CPU_MEMORY_AFFINITY) $(BOUND_MEMORY_NODE) $(CPU_MEMORY_AFFINITY_ARGS2)" \
			--submit "$(RUN_MOSALLOC_TOOL) --library $(MOSALLOC_TOOL) -cpf $(ROOT_DIR)/$(@D)/layout.csv $(CRIU_CHECKPOINT_MOSALLOC_EXTRA_ARGS) --"; \
		mkdir -p "$$archive_dir"; \
		sudo cp -a "$(@D)/." "$$archive_dir/"; \
	fi
endif
endif
endif
endif

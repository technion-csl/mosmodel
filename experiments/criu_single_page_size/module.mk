##### Layout-specific CRIU checkpoints for the single-page-size experiment

CRIU_SINGLE_PAGE_SIZE_DIR := experiments/criu_single_page_size
CRIU_SINGLE_PAGE_SIZE_LAYOUTS ?= $(SINGLE_PAGE_SIZE_LAYOUTS)

CRIU_CHECKPOINT_ARCHIVE_ROOT ?= /scratch/mosmodel-checkpoints/archive
CRIU_CHECKPOINT_MOSALLOC_EXTRA_ARGS := $(EXTRA_ARGS_FOR_MOSALLOC)
CRIU_BENCHMARK_ID ?= $(if $(filter $(benchmarks_root)/%,$(BENCHMARK_PATH)),$(patsubst $(benchmarks_root)/%,%,$(BENCHMARK_PATH)),$(notdir $(BENCHMARK_PATH)))

criu_sps_checkpoint_dir = $(CRIU_SINGLE_PAGE_SIZE_DIR)/$(CRIU_BENCHMARK_ID)/$(1)
criu_sps_checkpoint_done = $(call criu_sps_checkpoint_dir,$(1))/checkpoint.done

CRIU_SPS_CHECKPOINTS := $(foreach layout,$(CRIU_SINGLE_PAGE_SIZE_LAYOUTS),$(call criu_sps_checkpoint_done,$(layout)))

.PHONY: $(CRIU_SINGLE_PAGE_SIZE_DIR)

$(CRIU_SINGLE_PAGE_SIZE_DIR): $(CRIU_SPS_CHECKPOINTS)

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

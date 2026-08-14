# --- Sanity checks (top of file) ---
ifeq ($(strip $(BENCHMARK_PATH)),)
$(error "===> RUN_MODE=st but BENCHMARK_PATH is not set! <===")
endif

include $(EXPERIMENTS_VARS_TEMPLATE)

ifdef CPU1
  CPU_MEMORY_AFFINITY_ARGS := --cpu $(CPU1)
else
  CPU_MEMORY_AFFINITY_ARGS := --smt 1
endif

# ---------- Templates ----------

define MEASUREMENTS_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv $(call measurement_run_prerequisites,$(1)) | experiments-prerequisites
	echo ========== [INFO] allocate/reserve hugepages ==========
	MOSALLOC_KEEP_HUGEPAGE_POOL=1 $$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< /bin/date
	echo ========== [INFO] start producing ST run: $$@ ==========
	$$(PYTHON) -m scripts.mosmodel_controller.run_single \
		--benchmark "$$(BENCHMARK_PATH)" \
		--run-dir "$$(EXPERIMENTS_RUN_DIR)/$(1)/$(2)/1" \
		--output-dir "$$(@D)" \
		--output-target "$$@" \
		--num-threads "$$(NUMBER_OF_THREADS)" \
		--loop-until "$$(MEASURE_TIMEOUT1)" \
		--prefix "$$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) $$(CPU_MEMORY_AFFINITY_ARGS)" \
		--submit "env MOSALLOC_KEEP_HUGEPAGE_POOL=1 $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) --" $(call measurement_run_single_args,$(1)) \
		$$(RUN_SINGLE_EXTRA_ARGS)
endef

define VANILLA_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: %/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites 
	echo ========== [INFO] start producing: $$@ ==========
	$$(RUN_BENCHMARK) \
		--prefix="$$(CSET_SHIELD_PREFIX)" \
		--num_threads=$$(NUMBER_OF_THREADS) \
		--num_repeats=$$(NUM_OF_REPEATS) \
		--submit_command "$$(MEASURE_GENERAL_METRICS)" \
			-- $$(BENCHMARK_PATH) $$*
endef

define CSET_SHIELD_EXPS_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: %/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites 
	echo ========== [INFO] reserve hugepages before start running: $$@ ==========
	$$(CSET_SHIELD_PREFIX) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) -- sleep 1
	echo ========== [INFO] start producing: $$@ ==========
	$$(RUN_BENCHMARK) \
		--prefix="$$(CSET_SHIELD_PREFIX)" \
		--num_threads=$$(NUMBER_OF_THREADS) \
		--num_repeats=$$(NUM_OF_REPEATS) \
		--submit_command "$$(MEASURE_GENERAL_METRICS)  \
			$$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) --debug" \
		--benchmark_dir=$$(BENCHMARK_PATH) \
		--output_dir=$$* \
		--run_dir=$$(EXPERIMENTS_RUN_DIR)
endef

define TASKSET_EXPS_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: %/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites 
	echo ========== [INFO] reserve hugepages before start running: $$@ ==========
	$$(TASKSET_PREFIX) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) -- sleep 0.1 > /dev/null 2>&1 \
		|| $$(TASKSET_PREFIX) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) -- sleep 0.1
	echo ========== [INFO] start producing: $$@ ==========
	$$(RUN_BENCHMARK) \
		--prefix="$$(MEASURE_GENERAL_METRICS) $$(TASKSET_PREFIX)" \
		--num_threads=$$(NUMBER_OF_THREADS) \
		--num_repeats=$$(NUM_OF_REPEATS) \
		--submit_command "$$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) --debug" \
		--benchmark_dir=$$(BENCHMARK_PATH) \
		--output_dir=$$* \
		--run_dir=$$(EXPERIMENTS_RUN_DIR)
endef

ifdef VANILLA_RUN
$(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call VANILLA_template,$(layout),$(repeat)))))
else
  ifdef SERIAL_RUN
  $(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call MEASUREMENTS_template,$(layout),$(repeat)))))
  else
    ifdef CSET_SHIELD_RUN
    $(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call CSET_SHIELD_EXPS_template,$(layout),$(repeat)))))
    else
      # Assert that ISOLATED_CPUS is not empty
      ifeq ($(strip $(ISOLATED_CPUS)),)
      $(error "===> ISOLATED_CPUS is not set! <===")
      endif
      ifeq ($(strip $(ISOLATED_MEMORY_NODE)),)
      $(error "===> ISOLATED_MEMORY_NODE is not set! <===")
      endif
      $(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call TASKSET_EXPS_template,$(layout),$(repeat)))))
    endif
  endif
endif

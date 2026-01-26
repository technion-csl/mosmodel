include $(EXPERIMENTS_VARS_TEMPLATE)

# --- Sanity checks (top of file) ---
ifeq ($(strip $(BENCHMARK1)),)
$(error "===> RUN_MODE=smt but BENCHMARK1 is not set! <===")
endif
ifeq ($(strip $(BENCHMARK2)),)
$(error "===> RUN_MODE=smt but BENCHMARK2 is not set! <===")
endif

# ---------- Templates ----------

define MEASUREMENTS_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: %/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites
	echo ========== [INFO] allocate/reserve hugepages ==========
	$$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< /bin/date
	echo ========== [INFO] start producing: $$@ ==========
	
	$$(RUN_BENCHMARK) \
		--num_threads=$$(NUMBER_OF_THREADS) \
		--num_repeats=1 \
		--repeat=$(2) \
		--submit_command "$$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) --smt 1 $$(MEASURE_GENERAL_METRICS) \
		$$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) --" \
		--benchmark_dir=$$(BENCHMARK1) \
		--output_dir=$$* \
		--run_dir=$$(EXPERIMENTS_RUN_DIR)/1 &

	
	$$(RUN_BENCHMARK) \
		--prefix="$$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) --smt 2" \
		--num_threads=$$(NUMBER_OF_THREADS) \
		--num_repeats=1 \
		--repeat=$(2) \
		--benchmark_dir=$$(BENCHMARK2) \
		--output_dir=$(EXPERIMENTS_RUN_DIR)/_smt_bg_out/$(1) \
		--run_dir=$$(EXPERIMENTS_RUN_DIR)/2; \
	wait
	rm -rf "$$(EXPERIMENTS_RUN_DIR)/_smt_bg_out/$(1)/$(2)"
endef


# define CSET_SHIELD_EXPS_template =
# $(EXPERIMENT_DIR)/$(1)/1/$(2)/perf.out: %/1/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites
# 	echo ========== [INFO] reserve hugepages before start running: $$@ ==========
# 	$$(CSET_SHIELD_PREFIX) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) -- sleep 1
# 	echo ========== [INFO] start producing: $$@ ==========
# 	@set -e; \
# 	$$(RUN_BENCHMARK) \
# 		--prefix="$$(CSET_SHIELD_PREFIX)" \
# 		--num_threads=$$(NUMBER_OF_THREADS) \
# 		--num_repeats=$$(NUM_OF_REPEATS) \
# 		--submit_command "$$(MEASURE_GENERAL_METRICS) $$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) --smt 1 \
# 		 $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) --debug" \
# 		--benchmark_dir=$$(BENCHMARK1) \
# 		--output_dir=$$*/1 \
# 		--run_dir=$$(EXPERIMENTS_RUN_DIR)/1 &

# 	@set -e; \
# 	$$(RUN_BENCHMARK) \
# 		--prefix="$$(CSET_SHIELD_PREFIX) $$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) --smt 2" \
# 		--num_threads=$$(NUMBER_OF_THREADS) \
# 		--num_repeats=$$(NUM_OF_REPEATS) \
# 		--benchmark_dir=$$(BENCHMARK2) \
# 		--output_dir=$$*/2 \
# 		--run_dir=$$(EXPERIMENTS_RUN_DIR)/2; \
# 	wait
# endef


# define TASKSET_EXPS_template =
# $(EXPERIMENT_DIR)/$(1)/1/$(2)/perf.out: %/1/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites
# 	echo ========== [INFO] reserve hugepages before start running: $$@ ==========
# 	$$(TASKSET_PREFIX) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) -- sleep 0.1 > /dev/null 2>&1 \
# 		|| $$(TASKSET_PREFIX) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) -- sleep 0.1
# 	echo ========== [INFO] start producing: $$@ ==========
# 	@set -e; \
# 	$$(TASKSET_PREFIX) $$(SET_CPU_MEMORY_AFFINITY) $$(ISOLATED_MEMORY_NODE) --smt 1 \
# 		$$(RUN_BENCHMARK) \
# 			--num_threads=$$(NUMBER_OF_THREADS) \
# 			--num_repeats=$$(NUM_OF_REPEATS) \
# 			--submit_command \
# 			"$$(MEASURE_GENERAL_METRICS) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) --debug" \
# 			--benchmark_dir=$$(BENCHMARK1) \
# 			--output_dir=$$*/1 \
# 			--run_dir=$$(EXPERIMENTS_RUN_DIR)/1 & \
# 	$$(TASKSET_PREFIX) $$(SET_CPU_MEMORY_AFFINITY) $$(ISOLATED_MEMORY_NODE) --smt 2 \
# 		$$(RUN_BENCHMARK) \
# 			--num_threads=$$(NUMBER_OF_THREADS) \
# 			--num_repeats=$$(NUM_OF_REPEATS) \
# 			--benchmark_dir=$$(BENCHMARK2) \
# 			--output_dir=$$*/2 \
# 			--run_dir=$$(EXPERIMENTS_RUN_DIR)/2; \
# 	wait
# endef


# define VANILLA_template =
# $(EXPERIMENT_DIR)/$(1)/1/$(2)/perf.out: %/1/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites
# 	echo ========== [INFO] start producing: $$@ ==========
# 	@set -e; \
# 	$$(RUN_BENCHMARK) \
# 		--prefix="$$(CSET_SHIELD_PREFIX)" \
# 		--num_threads=$$(NUMBER_OF_THREADS) \
# 		--num_repeats=$$(NUM_OF_REPEATS) \
# 		--submit_command "$$(MEASURE_GENERAL_METRICS) $$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) --smt 1" \
# 		--benchmark_dir=$$(BENCHMARK1) \
# 		--output_dir=$$*/1 \
# 		--run_dir=$$(EXPERIMENTS_RUN_DIR)/1 &

# 	@set -e; \
# 	$$(RUN_BENCHMARK) \
# 		--prefix="$$(CSET_SHIELD_PREFIX) $$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) --smt 2" \
# 		--num_threads=$$(NUMBER_OF_THREADS) \
# 		--num_repeats=$$(NUM_OF_REPEATS) \
# 		--benchmark_dir=$$(BENCHMARK2) \
# 		--output_dir=$$*/2 \
# 		--run_dir=$$(EXPERIMENTS_RUN_DIR)/2; \
# 	wait
# endef


# ---------- Selector (same structure style as template.mk) ----------
ifdef VANILLA_RUN
$(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call VANILLA_template,$(layout),$(repeat)))))
else
  ifdef SERIAL_RUN
  $(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call MEASUREMENTS_template,$(layout),$(repeat)))))
  else
    ifdef CSET_SHIELD_RUN
    $(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call CSET_SHIELD_EXPS_template,$(layout),$(repeat)))))
    else
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

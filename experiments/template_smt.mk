# --- Sanity checks (top of file) ---
ifeq ($(strip $(BENCHMARK1)),)
$(error "===> RUN_MODE=smt but BENCHMARK1 is not set! <===")
endif
ifeq ($(strip $(BENCHMARK2)),)
$(error "===> RUN_MODE=smt but BENCHMARK2 is not set! <===")
endif

include $(EXPERIMENTS_VARS_TEMPLATE)

SMT_CORUNNER_PREALLOCATE_COMMAND =
SMT_CORUNNER_SUBMIT_ARGS =
ifneq ($(strip $(SMT_CORUNNER_LAYOUT_FILE)),)
SMT_CORUNNER_PREALLOCATE_COMMAND = MOSALLOC_KEEP_HUGEPAGE_POOL=1 $(SET_CPU_MEMORY_AFFINITY) $(BOUND_MEMORY_NODE) $(RUN_MOSALLOC_TOOL) --library $(MOSALLOC_TOOL) -cpf $(abspath $(SMT_CORUNNER_LAYOUT_FILE)) /bin/date
SMT_CORUNNER_SUBMIT_ARGS = --submit2 "env MOSALLOC_KEEP_HUGEPAGE_POOL=1 $(RUN_MOSALLOC_TOOL) --library $(MOSALLOC_TOOL) -cpf $(abspath $(SMT_CORUNNER_LAYOUT_FILE)) $(EXTRA_ARGS_FOR_MOSALLOC) --"
endif

# ---------- Templates ----------
define MEASUREMENTS_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: %/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv $(SMT_CORUNNER_LAYOUT_FILE) $(call measurement_run_prerequisites,$(1)) | experiments-prerequisites
	echo ========== [INFO] allocate/reserve hugepages ==========
	MOSALLOC_KEEP_HUGEPAGE_POOL=1 $$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< /bin/date
	$$(SMT_CORUNNER_PREALLOCATE_COMMAND)
	echo ========== [INFO] start producing SMT run: $$@ ==========
	$$(PYTHON) -m scripts.mosmodel_controller.run_pair \
		--benchmark1 "$$(BENCHMARK1)" \
		--benchmark2 "$$(BENCHMARK2)" \
		--run-dir "$$(EXPERIMENTS_RUN_DIR)/$(1)/$(2)" \
		--side1-output-dir "$$(@D)" \
		--side2-output-dir "$$(EXPERIMENTS_RUN_DIR)/_smt_bg_out/$(1)/$(2)" \
		--output-target "$$@" \
		--num-threads "$$(NUMBER_OF_THREADS)" \
		--loop-until1 "$$(MEASURE_TIMEOUT1)" \
		--loop-until2 "$$(MEASURE_TIMEOUT2)" \
		--prefix1="$$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) $$(CPU_MEMORY_AFFINITY_ARGS1)" \
		--prefix2="$$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) $$(CPU_MEMORY_AFFINITY_ARGS2)" \
		--submit1 "env MOSALLOC_KEEP_HUGEPAGE_POOL=1 $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC) --" \
		$$(SMT_CORUNNER_SUBMIT_ARGS) \
		$(call measurement_run_pair_args,$(1)) \
		$$(RUN_PAIR_EXTRA_ARGS)
endef

# ---------- Selector (existing run-mode selection) ----------
ifdef VANILLA_RUN
$(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call VANILLA_template,$(layout),$(repeat)))))
else
  ifdef SERIAL_RUN
  $(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS), $(eval $(call MEASUREMENTS_template,$(layout),$(repeat)))))
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

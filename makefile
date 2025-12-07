SHELL := /bin/bash
# run all lines of a recipe in a single invocation of the shell rather than each line being invoked separately
.ONESHELL:
# invoke recipes as if the shell had been passed the -e flag: the first failing command in a recipe will cause the recipe to fail immediately
.POSIX:

MODULE_NAME := all
$(MODULE_NAME):

export ROOT_DIR := $(PWD)
export HOST_NAME := $(shell hostname)

# global auxiliary functions
comma := ,
empty :=
space := $(empty) $(empty)

define array_to_comma_separated
$(subst $(space),$(comma),$(strip $1))
endef

SCRIPTS_ROOT_DIR := $(ROOT_DIR)/scripts

# the following list should preserve a topological ordering, i.e., if module B
# uses variables defined in module A, than module A should come before module B
SUBMODULES := analysis slurm experiments

include benchmark.mk
include $(ROOT_DIR)/common.mk

# a top-level "clean" target, which calls all/clean
.PHONY: clean
clean: all/clean

# a generic pattern rule for deleting files
.PHONY: %/delete
%/delete:
	rm -rf $*

# empty recipes to prevent make from remaking the makefile and include files
# https://www.gnu.org/software/make/manual/html_node/Remaking-Makefiles.html
makefile: ;
$(ROOT_DIR)/common.mk: ;


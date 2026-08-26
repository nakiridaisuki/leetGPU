NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++17 -lineinfo

# NVCCFLAGS += -arch=sm_80

BUILD_DIR   := build
PROFILE_DIR := profile
SRC_DIRS    := easy medium hard
SRCS        := $(wildcard $(addsuffix /*.cu, $(SRC_DIRS)))
TARGETS     := $(notdir $(basename $(SRCS)))

vpath %.cu $(SRC_DIRS)

.PHONY: all clean help $(TARGETS)

all: help

help:
	@echo "LeetGPU practice compile tool"
	@echo "Usage:"
	@echo "  make <target>             Compile and run"
	@echo "  make <target> PROF=nsys   Run with Nsight Systems (Timeline)"
	@echo "  make <target> PROF=ncu    Run with Nsight Compute (Kernel Metrics)"
	@echo ""
	@echo "Targets: $(TARGETS)"

$(TARGETS): %: $(BUILD_DIR)/%  | $(PROFILE_DIR)
	@echo ""
	@echo "[Executing] $(BUILD_DIR)/$@"
	@echo "------------------------------------------"
	@if [ "$(PROF)" = "nsys" ]; then \
		echo "[Profiling with nsys] Saving to $(PROFILE_DIR)/$@_nsys.nsys-rep"; \
		nsys profile -o $(PROFILE_DIR)/$@_nsys --force-overwrite true --stats=true ./$(BUILD_DIR)/$@; \
	elif [ "$(PROF)" = "ncu" ]; then \
		echo "[Profiling with ncu] Saving to $(PROFILE_DIR)/$@_ncu.ncu-rep"; \
		sudo ncu -o $(PROFILE_DIR)/$@_ncu -f --set full ./$(BUILD_DIR)/$@; \
	else \
		./$(BUILD_DIR)/$@; \
	fi
	@echo "------------------------------------------"

$(BUILD_DIR)/%: %.cu | $(BUILD_DIR)
	@echo " [Compiling] $< -> $@"
	@$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(PROFILE_DIR):
	@mkdir -p $(PROFILE_DIR)

clean:
	@rm -rf $(BUILD_DIR)

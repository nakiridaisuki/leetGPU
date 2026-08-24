NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++17

# NVCCFLAGS += -arch=sm_80

BUILD_DIR := build
SRC_DIRS  := easy medium hard
SRCS      := $(wildcard $(addsuffix /*.cu, $(SRC_DIRS)))
TARGETS   := $(notdir $(basename $(SRCS)))

vpath %.cu $(SRC_DIRS)

.PHONY: all clean help $(TARGETS)

all: help

help:
	@echo "LeetGPU practice compile tool"
	@echo "usage: make <file name without .cu>"

$(TARGETS): %: $(BUILD_DIR)/%
	@echo ""
	@echo "[Executing] $(BUILD_DIR)/$@"
	@echo "------------------------------------------"
	@./$(BUILD_DIR)/$@
	@echo "------------------------------------------"

$(BUILD_DIR)/%: %.cu | $(BUILD_DIR)
	@echo " [Compiling] $< -> $@"
	@$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# 清理 build 目錄
clean:
	@rm -rf $(BUILD_DIR)

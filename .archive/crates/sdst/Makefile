# SDST - Spike-Driven Sparse Temporal Hash
# Makefile for CUDA library build
#
# Targets:
#   make          - Build shared library (libsdst.so)
#   make static   - Build static library (libsdst.a)
#   make test     - Build and run tests
#   make clean    - Clean build artifacts
#
# Requirements:
#   - CUDA Toolkit 13.0+ (tested on 13.1)
#   - SM 120 target (RTX 5080 Blackwell)
#   - GCC 13+

NVCC        := nvcc
AR          := ar
NVCC_FLAGS  := -std=c++17 -O3 --use_fast_math
NVCC_FLAGS  += -gencode arch=compute_120,code=sm_120  # RTX 5080
NVCC_FLAGS  += --compiler-options '-fPIC -Wall'
NVCC_FLAGS  += -I./include

# Debug build
ifdef DEBUG
    NVCC_FLAGS += -G -g -DSDST_DEBUG
endif

SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build
LIB_DIR     := lib

SOURCES     := $(wildcard $(SRC_DIR)/*.cu)
OBJECTS     := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SOURCES))
HEADERS     := $(wildcard $(INC_DIR)/*.h)

SHARED_LIB  := $(LIB_DIR)/libsdst.so
STATIC_LIB  := $(LIB_DIR)/libsdst.a

.PHONY: all shared static test clean dirs

all: shared

shared: dirs $(SHARED_LIB)

static: dirs $(STATIC_LIB)

dirs:
	@mkdir -p $(BUILD_DIR) $(LIB_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(SHARED_LIB): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) --shared $(OBJECTS) -o $@ -lcudart

$(STATIC_LIB): $(OBJECTS)
	$(AR) rcs $@ $(OBJECTS)

# Test build
TEST_SRC    := test/test_sdst.cu
TEST_BIN    := $(BUILD_DIR)/test_sdst

test: shared $(TEST_BIN)
	@echo "Running SDST tests..."
	LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(SHARED_LIB)
	$(NVCC) $(NVCC_FLAGS) -DSDST_DEBUG $(TEST_SRC) -o $@ -L$(LIB_DIR) -lsdst -lcudart

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

# Install to system (optional)
PREFIX ?= /usr/local
install: shared
	install -d $(PREFIX)/lib $(PREFIX)/include/sdst
	install -m 644 $(SHARED_LIB) $(PREFIX)/lib/
	install -m 644 $(INC_DIR)/sdst_api.h $(PREFIX)/include/sdst/
	ldconfig

# Print info
info:
	@echo "Sources:  $(SOURCES)"
	@echo "Objects:  $(OBJECTS)"
	@echo "Headers:  $(HEADERS)"
	@echo "NVCC:     $(NVCC)"
	@echo "Flags:    $(NVCC_FLAGS)"

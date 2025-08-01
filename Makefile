# ─────────────────────────────────────────────────────────────────────────────
# Makefile for Tiger HLM Runoff (Runoff5 only) - Versioned Build
# ─────────────────────────────────────────────────────────────────────────────

NVCC ?= $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
    NVCC := /usr/local/cuda/bin/nvcc
endif

# Directories
SRC_DIR     := src
OBJ_DIR     := build
BIN_DIR     := bin
IO_DIR      := $(SRC_DIR)/I_O
SOLVER_DIR  := $(SRC_DIR)/solver
MODELS_DIR  := $(SRC_DIR)/models
BIN         := $(BIN_DIR)/runoff

# Version Info
VERSION := 0.0.0
BUILD_INFO := $(OBJ_DIR)/build_info.txt

# Build modes
DEBUG   ?= 0
VERBOSE ?= 0

HOST_INCLUDES   := -I$(SRC_DIR) -I$(IO_DIR) -I$(SOLVER_DIR) -I$(MODELS_DIR)
DEVICE_INCLUDES := -I$(SRC_DIR) -I$(SOLVER_DIR) -I$(MODELS_DIR) -I$(IO_DIR)

NVCCFLAGS := -std=c++17 -rdc=true \
  -gencode arch=compute_90,code=compute_90 \
  -gencode arch=compute_90,code=sm_90 \
  -Xcudafe --diag_suppress=177 \
  -Xcudafe --diag_suppress=20091 \
  $(DEVICE_INCLUDES) -DUSE_MODEL_5 -Wno-deprecated-gpu-targets

# Embed version and date into the binary
NVCCFLAGS += -DPROJECT_VERSION="\"Tiger-HLM Runoff version $(VERSION)\"" \
             -DBUILD_DATE="\"$(shell date)\""

HOSTFLAGS := -std=c++17 $(HOST_INCLUDES) -DUSE_MODEL_5 \
             -L${NETCDF_PATH}/lib64 -lnetcdf

ifeq ($(DEBUG),1)
    NVCCFLAGS += -g -G -O0
else
    NVCCFLAGS += -O2
endif

ifeq ($(VERBOSE),1)
    Q :=
else
    Q := @
endif

# NetCDF & MPI
NETCDF_PATH := "${NETCDFDIR:-$${NETCDF_ROOT}}/include"
OUT_CPP     := $(IO_DIR)/output_series.cpp
NVCCFLAGS   += -L${NETCDF_PATH}/lib64 -lnetcdf $(MPI_LIB)

MPI_INC     := $(shell mpicxx --showme:compile)
RAW_MPI_LIB := $(shell mpicxx --showme:link)
MPI_LIB     := $(shell echo $(RAW_MPI_LIB) | sed 's/-Wl,/ -Xlinker /g' | sed 's/,/ -Xlinker /g')
HOSTFLAGS  += $(MPI_INC) $(MPI_LIB)
NVCCFLAGS  += $(MPI_INC) $(MPI_LIB)

# ─────────────────────────────────────────
# Sources
SRC_MAIN_CU  := $(SRC_DIR)/main.cpp
SRC_HOST_CPP := $(MODELS_DIR)/model_registry.cpp $(IO_DIR)/config_loader.cpp
SRC_IO_CPP   := $(IO_DIR)/parameters_loader.cpp $(IO_DIR)/forcing_loader.cpp $(OUT_CPP)
SRC_UTIL_CPP := $(MODELS_DIR)/ETmethods.cpp $(MODELS_DIR)/soiltemp.cpp
SRC_CU       := $(SOLVER_DIR)/rk45_kernel.cu $(SOLVER_DIR)/radau_kernel.cu \
                $(MODELS_DIR)/model_Runoff5_global.cu $(IO_DIR)/forcing_data.cu

# ─────────────────────────────────────────
# Object files
OBJ_MAIN     := $(OBJ_DIR)/main.o
OBJ_HOST     := $(OBJ_DIR)/models/model_registry.o $(OBJ_DIR)/I_O/config_loader.o
OBJ_IO       := $(patsubst $(IO_DIR)/%.cpp,$(OBJ_DIR)/I_O/%.o,$(SRC_IO_CPP))
OBJ_UTIL     := $(patsubst $(MODELS_DIR)/%.cpp,$(OBJ_DIR)/models/%.o,$(SRC_UTIL_CPP))
OBJ_DEVICE   := $(patsubst $(SOLVER_DIR)/%.cu,$(OBJ_DIR)/solver/%.o,$(filter $(SOLVER_DIR)/%.cu,$(SRC_CU))) \
                $(OBJ_DIR)/models/model_Runoff5_global.o $(OBJ_DIR)/I_O/forcing_data.o $(OBJ_MAIN)
DEVICE_LINK  := $(OBJ_DIR)/device_link.o

# ─────────────────────────────────────────
.PHONY: all clean

all: $(BIN)
	@echo "Built $(BIN) [Version $(VERSION), DEBUG=$(DEBUG)]"
	@echo "Build completed on: $$(date)"
	@echo "Tiger-HLM Runoff version $(VERSION)" > $(BUILD_INFO)
	@echo "Build timestamp: $$(date)" >> $(BUILD_INFO)
	@echo "Build info saved to $(BUILD_INFO)"

# ─────────────────────────────────────────
# Directories
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR) $(OBJ_DIR)/I_O $(OBJ_DIR)/models $(OBJ_DIR)/solver
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# ─────────────────────────────────────────
# Compilation rules

$(OBJ_MAIN): $(SRC_MAIN_CU) | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc -x cu -c $< -o $@

$(OBJ_DIR)/I_O/config_loader.o: $(IO_DIR)/config_loader.cpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(HOSTFLAGS) -c $< -o $@

$(OBJ_DIR)/models/model_registry.o: $(MODELS_DIR)/model_registry.cpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(HOSTFLAGS) -c $< -o $@

$(OBJ_DIR)/I_O/%.o: $(IO_DIR)/%.cpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(HOSTFLAGS) -c $< -o $@

$(OBJ_DIR)/models/%.o: $(MODELS_DIR)/%.cpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -x cu -dc -c $< -o $@

$(OBJ_DIR)/solver/%.o: $(SOLVER_DIR)/%.cu | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(OBJ_DIR)/models/%.o: $(MODELS_DIR)/%.cu | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(OBJ_DIR)/I_O/%.o: $(IO_DIR)/%.cu | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(DEVICE_LINK): $(OBJ_DEVICE) $(OBJ_UTIL)
	$(Q)$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@

$(BIN): $(OBJ_HOST) $(OBJ_IO) $(OBJ_DEVICE) $(OBJ_UTIL) $(DEVICE_LINK) | $(BIN_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) --relocatable-device-code=true $^ -lcudadevrt -o $@

# ─────────────────────────────────────────
clean:
	$(Q)rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Cleaned build/ and bin/ successfully."

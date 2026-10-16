# Builds the plugin library and the C++ runner inside the dev image
# (docker/dev/Dockerfile). Outputs go to build/, which is gitignored.
#   make            -> build/libgaia_trt_layout_plugins.so build/runner
# The plugins are compiled for sm_120 (RTX 5090), the only GPU this
# experiment measures. Host symbols are hidden except the registration entry
# point, so this library and the baseline experiment's library (same Gaia
# kernels, other namespace) can be loaded into one process without
# interposing each other.
NVCC ?= nvcc
CXX ?= g++
ARCH ?= -gencode arch=compute_120,code=sm_120
TRT_INC ?= /usr/include/x86_64-linux-gnu
TRT_LIB ?= /usr/lib/x86_64-linux-gnu
CUDA_HOME ?= /usr/local/cuda
BUILD ?= build

PLUGIN_SRC = plugins/gaia_plugins.cu
PLUGIN_HDRS = plugins/gaia_plugins.h plugins/gaia_device_code.cuh plugins/gaia_adapted_epilogue.cuh plugins/gaia_hwc_kernels.cuh
LIB = libgaia_trt_layout_plugins.so

all: $(BUILD)/$(LIB) $(BUILD)/runner

$(BUILD)/$(LIB): $(PLUGIN_SRC) $(PLUGIN_HDRS)
	mkdir -p $(BUILD)
	$(NVCC) -O3 -std=c++17 $(ARCH) -Xcompiler -fPIC,-fvisibility=hidden -shared -I$(TRT_INC) -Iplugins -o $@ $(PLUGIN_SRC) -L$(TRT_LIB) -lnvinfer

$(BUILD)/runner: runner/runner.cpp plugins/gaia_plugins.h $(BUILD)/$(LIB)
	$(CXX) -O2 -std=c++17 -Wall -I$(TRT_INC) -Iplugins -I$(CUDA_HOME)/include -o $@ runner/runner.cpp \
	    -L$(BUILD) -lgaia_trt_layout_plugins -L$(TRT_LIB) -lnvinfer -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath,'$$ORIGIN'

clean:
	rm -rf $(BUILD)

.PHONY: all clean

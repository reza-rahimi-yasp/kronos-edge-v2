# Builds the whole-model plugin library and a runner that registers both
# plugin libraries, inside the dev image (../docker/dev/Dockerfile, tagged
# gaia-trt-whole-model-dev:local). Outputs go to build/, which is gitignored.
#   make            -> build/libgaia_trt_whole_model.so build/runner_whole
# cuDNN and cuBLAS come from the dev image's venv: the very builds torch
# 2.14.1+cu130 links (nvidia-cudnn-cu13 9.24.0.43 and nvidia-cublas
# 13.1.1.3 wheels), so the plugin's plan selection sees what kernel.py's
# convolutions and matmuls see. Compiled for sm_120 (RTX 5090) only.
# Symbols are hidden except the registration entry point, so the library
# co-loads with the hybrid experiment's libgaia_trt_plugins.so.
# The library carries a runpath to those venv directories so it loads in the
# dev image without torch; in the runtime image that path does not exist and
# the loader takes the apt-installed libraries of the same builds.
NVCC ?= nvcc
CXX ?= g++
ARCH ?= -gencode arch=compute_120,code=sm_120
TRT_INC ?= /usr/include/x86_64-linux-gnu
TRT_LIB ?= /usr/lib/x86_64-linux-gnu
CUDA_HOME ?= /usr/local/cuda
VENV_NVIDIA ?= /opt/venv/lib/python3.12/site-packages/nvidia
CUDNN_INC ?= $(VENV_NVIDIA)/cudnn/include
CUDNN_LIB ?= $(VENV_NVIDIA)/cudnn/lib
CUBLAS_INC ?= $(VENV_NVIDIA)/cu13/include
CUBLAS_LIB ?= $(VENV_NVIDIA)/cu13/lib
BUILD ?= build
HYBRID ?= ..
HYBRID_BUILD ?= $(HYBRID)/build

PLUGIN_SRC = plugin/gaia_whole_model_plugin.cu
PLUGIN_HDRS = plugin/gaia_whole_model_plugin.h $(HYBRID)/plugins/gaia_device_code.cuh

all: $(BUILD)/libgaia_trt_whole_model.so $(BUILD)/runner_whole

$(BUILD)/libgaia_trt_whole_model.so: $(PLUGIN_SRC) $(PLUGIN_HDRS)
	mkdir -p $(BUILD)
	$(NVCC) -O3 -std=c++17 $(ARCH) -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -shared \
	    -I$(TRT_INC) -I$(CUDNN_INC) -I$(CUBLAS_INC) -Iplugin -I$(HYBRID)/plugins -o $@ $(PLUGIN_SRC) \
	    -L$(TRT_LIB) -lnvinfer -L$(CUDNN_LIB) -l:libcudnn.so.9 -L$(CUBLAS_LIB) -l:libcublas.so.13 \
	    -Xlinker -rpath=$(CUDNN_LIB) -Xlinker -rpath=$(CUBLAS_LIB)

# The hybrid experiment's runner, built with its second-library hook: it registers the hybrid
# plugins and the whole-model plugin, so one process can time any pair of engines.
$(BUILD)/runner_whole: $(HYBRID)/runner/runner.cpp $(HYBRID)/plugins/gaia_plugins.h $(BUILD)/libgaia_trt_whole_model.so $(HYBRID_BUILD)/libgaia_trt_plugins.so
	$(CXX) -O2 -std=c++17 -Wall -DGAIA_TRT_EXTRA_REGISTER=gaia_trt_whole_model_register_plugins \
	    -I$(TRT_INC) -I$(HYBRID)/plugins -I$(CUDA_HOME)/include -o $@ $(HYBRID)/runner/runner.cpp \
	    -L$(BUILD) -lgaia_trt_whole_model -L$(HYBRID_BUILD) -lgaia_trt_plugins -L$(TRT_LIB) -lnvinfer -L$(CUDA_HOME)/lib64 -lcudart \
	    -Wl,-rpath-link,$(CUDNN_LIB) -Wl,-rpath-link,$(CUBLAS_LIB) -Wl,-rpath,'$$ORIGIN'

clean:
	rm -rf $(BUILD)

.PHONY: all clean

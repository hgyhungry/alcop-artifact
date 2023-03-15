Auto-pipelining Test
===

# Requirement
NVIDIA GPU Ampere generation or later (compute capability >= 80)

# Build TVM framework

## Build from Source
You can follow the standard TVM installation flow. 
Because auto-pipelining feature requires changes to TVM source, you need to 
build TVM from source following the instructions. 

You can follow the steps below, and refer to  https://tvm.apache.org/docs/install/from_source.html and https://tvm.apache.org/docs/install/from_source.html#python-package-installation 
for trouble-shooting. If you encounter errors, please file an issue.

Key steps
```bash
# start from an NVIDIA docker 
docker run -it --gpus all -v /path/to/this/repo:/tvm -w /tvm nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04 bash

# inside the docker, install all the dependencies
apt-get update
apt-get install -y python3 python3-pip python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev git llvm
pip install numpy decorator attrs tornado psutil 'xgboost>=1.1.0' cloudpickle matplotlib torch pytest

# create build directory
mkdir build
cp cmake/config.cmake.template build/config.cmake

# build the TVM shared library
cd build
cmake ..
make -j8

# expose the TVM python directory to PYTHONPATH
export TVM_HOME=/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

```

## Start from the prepared docker image
We prepare a docker image where everything is installed and you can directly start running the following steps. 
The image is build just using the scripts in the previous part.

```bash
docker run -it --gpus all -w /tvm hguyue1/alcop:latest bash
```

# Check the auto-pipelining unit test
```bash
cd ${TVM_HOME}/auto_pipeline_exp/single_op
python3 dense_tensorcore_in_topi.py # baseline
python3 dense_tensorcore_autopipeline_example.py # optimized
```

# Run the autotvm tuning script with auto pipelining integrated
```bash
cd ${TVM_HOME}/auto_pipeline_exp/single_op
sh run.sh
```

# Design Details
* TOPI template with support for pipelining and shared memory swizzling: tvm/python/tvm/topi/exp_cuda/*
* Auto-pipelining program transformation pass: tvm/python/tvm/contrib/auto_pipeline.py

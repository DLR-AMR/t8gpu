## Introduction

A C++ Finite Volume solver using t8code targetting GPUs.

## Build

```
cmake .. -DCMAKE_BUILD_TYPE=Debug/RelWithDebInfo \
	 -DCMAKE_CXX_FLAGS="your flags here"     \
	 -DCMAKE_INSTALL_PREFIX=path/to/install  \
	 -DT8CODE_ROOT=path/to/t8code/install    \
	 -DT8GPU_BUILD_DOCUMENTATION=ON/OFF

cmake --build . -j
```

## Run

For better performance, it is recommended to use the MPS server.

```
export CUDA_VISIBLE_DEVICES=0        # select GPU device
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # set exclusive mode if possible
nvidia-cuda-mps-control -d           # start the MPS daemon
mpirun -n 8 ./src/t8gpu
```

## Documentation

## License and contributing

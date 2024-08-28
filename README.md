# t8gpu

A C++ finite volume framework using t8code targetting GPUs.

> [!CAUTION]
> This repository is still work in progress. Therefore, the API is subject to changes and the user guide is still missing. Feel free to leave feedback/suggestions.

## Build

### Requirements

- The CMake build system generator (version 3.16 or higher).
- T8code installed using the CMake build system.
- A recent Nvidia CUDA SDK.

### Configure t8gpu

```
cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Debug/RelWithDebInfo \
                     -DCMAKE_INSTALL_PREFIX=path/to/install  \
                     -DCMAKE_CXX_FLAGS="your flags here"     \
                     -DT8CODE_ROOT=path/to/t8code/install    \
                     -DT8GPU_BUILD_DOCUMENTATION=ON/OFF
```

The most common build options are described here:

| Option                             | Description                        |
| ---------------------------------- | ---------------------------------- |
| -DT8GPU_BUILD_DOCUMENTATION=ON/OFF | Build the Doxygen documentation    |
| -DCMAKE_CXX_FLAGS="..."            | Set additional c++ compiler flags  |
| -DCMAKE_CUDA_FLAGS="..."           | Set additional cuda compiler flags |

### Build and install t8gpu

```
cmake --build build/ --parallel
cmake --build build/ --target install
```

## Run

> [!NOTE]
> For better performance using multiple MPI ranks, it is heavily recommended to use the [MPS server](https://docs.nvidia.com/deploy/mps/). To launch the MPS server, simply run:
>
> ```
> export CUDA_VISIBLE_DEVICES=0        # select GPU device
> nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # set exclusive mode if possible
> nvidia-cuda-mps-control -d           # start the MPS daemon
> ```

```
mpirun -n 8 ./build/src/t8gpu
```

## Getting started

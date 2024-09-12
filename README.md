# t8gpu

<p align="center">
  <img width="450px" src=AMR_example.png>
</p>

A C++ finite volume header-only library using t8code targeting GPUs.

> [!CAUTION]
> This repository is still work in progress. Therefore, the API is subject to changes and the user guide is still missing. Feel free to leave feedback/suggestions.

## Install

### Requirements

- The CMake build system generator (version 3.16 or higher).
- T8code installed using the CMake build system.
- A recent Nvidia CUDA SDK supporting c++17.

### Configure t8gpu

```
cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Debug/RelWithDebInfo \
                     -DCMAKE_INSTALL_PREFIX=path/to/install  \
                     -DCMAKE_CXX_FLAGS="your flags here"     \
                     -DT8CODE_ROOT=path/to/t8code/install    \
                     -DT8GPU_BUILD_EXAMPLES=ON/OFF           \
                     -DT8GPU_BUILD_DOCUMENTATION=ON/OFF
```

The most common build options are described here:

| Option                             | Description                        |
| ---------------------------------- | ---------------------------------- |
| -DT8GPU_BUILD_DOCUMENTATION=ON/OFF | Build the Doxygen documentation    |
| -DT8GPU_BUILD_EXAMPLES=ON/OFF      | Build the example executables      |
| -DCMAKE_CXX_FLAGS="..."            | Set additional c++ compiler flags  |
| -DCMAKE_CUDA_FLAGS="..."           | Set additional cuda compiler flags |

### Build examples and install t8gpu

```bash
cmake --build build/ --parallel
cmake --build build/ --target install
```

## Run the examples

> [!NOTE]
> For better performance using multiple MPI ranks, it is heavily recommended to use the [MPS server](https://docs.nvidia.com/deploy/mps/). To launch the MPS server, simply run:
>
> ```bash
> export CUDA_VISIBLE_DEVICES=0        # select GPU device
> nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # set exclusive mode if possible
> nvidia-cuda-mps-control -d           # start the MPS daemon
> ```

```
mpirun -n 8 ./build/examples/compressible_euler/kelvin_helmholtz
```

## Getting started

### Including t8gpu in a CMake-based project

As t8gpu is a header-only library, linking against it requires only adding the installed include directory to the compiler include directories. However, for CMake based projects, we recommend doing the following to get all the necessary compiler flags, simply add to your ``CMakeLists.txt``:

```CMake
find_package(t8gpu REQUIRED)
target_link_libraries(your_target PRIVATE t8gpu)
```

You might need to provide the CMake variable ``T8GPU_ROOT`` to let CMake know where t8gpu was installed.

### Quickstart tutorial
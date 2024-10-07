# t8gpu

<p align="center">
  <img width="450px" src=AMR_example.png>
  <img width="350px" src=AMR_subgrid_example.png>
</p>

T8gpu is a C++ finite volume header-only library using t8code targeting GPUs. This library can deal with meshes with multiple element types and also provides support for subgrid elements for cartesian and hexahedral meshes.

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

## Getting started using t8gpu

### Including t8gpu in a CMake-based project

As t8gpu is a header-only library, linking against it requires only adding the installed include directory to the compiler include directories. However, for CMake based projects, we recommend doing the following to get all the necessary compiler flags, simply add to your ``CMakeLists.txt``:

```CMake
find_package(t8gpu REQUIRED)
target_link_libraries(your_target PRIVATE t8gpu)
```

You might need to provide the CMake variable ``T8GPU_ROOT`` to let CMake know where t8gpu was installed.

### Quickstart tutorial

The two main template classes that t8gpu provides is ``t8gpu::MemoryManager`` and ``t8gpu::MeshManager`` as well as their subgrid counterparts equivalent ``t8gpu::SubgridMemoryManager`` and ``t8gpu::SubgridMeshManager``.

- The memory manager class is responsible for storing cell-centered variables on the GPU as well as providing a way to access GPU memory of other MPI rank GPU allocations. To do so, the user needs to specify an enum class ``VariableType`` representing the names of the cell-centered variables to be stored per mesh element. This enum class will later be used to select which GPU array to fetch. Moreover, to handle multiple timestepping schemes, another enum class ``StepType`` is needed to specify the number of variable copies needed for the possible multiple sub-timesteps. This type is also used to select which step to fetch variable data from. Here is an example setup for this class for the 3d compressible Euler equations:

```c++
#include <t8gpu/memory/memory_manager.h>

enum VariableList {
  Rho,     // density.
  Rho_v1,  // x-component of momentum.
  Rho_v2,  // y-component of momentum.
  Rho_v3,  // z-component of momentum.
  Rho_e,   // energy.
  nb_variables // number of variables.
};

enum StepList {
  Step0,   // used for RK3 timestepping.
  Step1,   // used for RK3 timestepping.
  Step2,   // used for RK3 timestepping.
  Step3,   // used for RK3 timestepping.
  Fluxes,  // used to store face fluxes.
  nb_steps // number of steps.
};

// This class is then tailored to our use case:
using MemoryManager = t8gpu::SubgridMemoryManager<VariableList, StepList, Subgrid<4, 4, 4>>;
```

This class then provides methods to access either one are all variables on either the GPU array owned by the MPI rank or the GPU array for all MPI ranks through the ``get_{own,all}_variable(s)`` APIs. These methods either return pointers, array of pointers are custom wrapper types that can be used on both the CPU and GPU to access variables data conveniantly using the user defined enum classes. Here is an example of a GPU kernel setting up the initial condition using one of those conveniant types for subgrid elements and some CPU code to launching this kernel:

```c++
using namespace t8gpu;

__global__ void set_momentum(SubgridMemoryAccessorOwn<VariableList, Subgrid<4, 4, 4>> variables) {
  int const e_idx = blockIdx.x;

  int const i = threadIdx.x;
  int const j = threadIdx.y;
  int const k = threadIdx.z;

  auto [rho_u, rho_v, rho_w] = variables.get(Rho_u, Rho_v, Rho_w);

  // element index
  //    ┌─┴─┐
  rho_u(e_idx, i, j, k) = ...
  //           └──┬──┘
  //     subgrid coordinates
  ...
}

int main(int argc, char* argv[]) {
  ...

  SubgridMemoryManager<VariableList, StepList, Subgrid<4, 4, 4>> memory_manager = ...

  set_momentum<<<num_local_elements, Subgrid<4, 4, 4>::block_size>>>(
              memory_manager.get_own_variables(Step0));
  T8GPU_CUDA_CHECK_LAST_ERROR();
  ...
}
```

- The mesh manager class takes ownership of a t8code mesh and a memory manager class object to handle mesh related operations and GPU data management alltogether. It is templated on the same ``VariableType`` and ``StepType`` enum classes. It provides methods for accessing local connectivity information as well as ghost layer information. This class can also handle mesh adaptation and repartitioning. However, it is important to note that interpolation between two meshes must be done explicitely by the user. This library provides the neccessary API for accessing GPU memory to easily interpolation between two meshes.

For further details, see the doxygen documentation of theses classes or take a look at the examples programs in the [examples folder](examples) of this repository.
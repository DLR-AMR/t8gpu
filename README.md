# Compile

```
module load PrgEnv/gcc11-openmpi-cuda
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../install -DT8CODE_ROOT=../../builds/install -DP4EST_ROOT=../../builds/install
```
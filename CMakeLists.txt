project(CppML)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)

## 注意在Clion的Cmake里,Environment是不包括.bashrc中定义的变量(也不包括其他source xx.sh定义的变量)
## 因此这里的MKLROOT虽然在命令行执行 echo $MKLROOT 可以显示,但Cmake依旧是找不到的. 解决方法只能是:
## 在Clion的Cmake设置里(Preferences/Settings | Build, Execution, Deployment | CMake dialog),添加一个MKLROOT的环境变量.
## 见: https://stackoverflow.com/questions/37662130/clion-or-cmake-does-not-see-environment-variable
include_directories($ENV{MKLROOT}/include)

# set(EXECUTABLE_OUTPUT_PATH bin)
file(GLOB source_files "src/*.cpp" "include/*.h" "include/*.hpp")
add_executable(cpp_ml ${source_files})

# mkl link option reference: https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
# what is difference between ILP64 and LP64?: https://software.intel.com/en-us/node/528524.
# Here i choose ILP64, so MKL_INT is a 64 bit data type.
set(CMAKE_CXX_FLAGS "-DNDEBUG -fopenmp -O3 -DMKL_ILP64 -m64 -mavx2 -msse4.2 -mfma")
target_link_libraries(cpp_ml
        $ENV{MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a
        -Wl,--start-group
        $ENV{MKLROOT}/lib/intel64/libmkl_cdft_core.a
        $ENV{MKLROOT}/lib/intel64/libmkl_intel_ilp64.a
        $ENV{MKLROOT}/lib/intel64/libmkl_gnu_thread.a
        $ENV{MKLROOT}/lib/intel64/libmkl_core.a
        $ENV{MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a
        -Wl,--end-group
        -lgomp -lpthread -lm -ldl)

# ├── build
# ├── CMakeLists.txt
# ├── LICENSE
# └── src
#     ├── kernels
#     │   ├── cut.cuh
#     │   └── naive.cuh
#     ├── kernels.cuh
#     ├── matrix_utils.cuh
#     ├── runner.cuh
#     └── sgemm.cu

cmake_minimum_required(VERSION 3.18)
project(sgemm LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
# change it to higher version if you have a newer GPU (60 is suitable for GTX 1080Ti)
set(CUDA_COMPUTE_CAPABILITY 60)

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(sgemm ${PROJECT_SOURCE_DIR}/src/sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

execute_process(COMMAND cp -r ${PROJECT_SOURCE_DIR}/scripts ${PROJECT_SOURCE_DIR}/build)
#pragma once

#include "kernels/naive.cuh"
#include "kernels/global_mem_coalescing.cuh"
#include "kernels/shared_mem.cuh"
#include "kernels/blocktiling_1d.cuh"
#include "kernels/blocktiling_2d.cuh"
#include "kernels/vectorize.cuh"
# UNIVERSAL SGEMM CUDA

## Results

```
$GEMM_MODE=0/1/2/3/4/5
./sgemm 32 1605632 27 $GEMM_MODE
./sgemm 384 14161 1152 $GEMM_MODE
./sgemm 256 43264 1152 $GEMM_MODE
./sgemm 64 1605632 147 $GEMM_MODE
./sgemm 64 559104 147 $GEMM_MODE
./sgemm 256 50176 1024 $GEMM_MODE
```


| M | N | K    | Algorithm | Time (us) |
|---|---|---   |-----------|-----------|
|32|1605632|27 |naive|123707|
|384|14161|1152|naive|179951|
|256|43264|1152|naive|373887|
|64|1605632|147|naive|702349|
|64|559104|147 |naive|252176|
|256|50176|1024|naive|439462|
|32|1605632|27|global_memory_coalescing|8976.98|
|384|14161|1152|global_memory_coalescing|35116.6|
|256|43264|1152|global_memory_coalescing|65442|
|64|1605632|147|global_memory_coalescing|79366|
|64|559104|147|global_memory_coalescing|28115.3|
|256|50176|1024|global_memory_coalescing|63925.2|
|32|1605632|27|shared_memory|13594.9|
|384|14161|1152|shared_memory|42083.7|
|256|43264|1152|shared_memory|85513.6|
|64|1605632|147|shared_memory|114396|
|64|559104|147|shared_memory|40141.6|
|256|50176|1024|shared_memory|88951.5|
|32|1605632|27|blocktiling_1d|2858.95|
|384|14161|1152|blocktiling_1d|4176.16|
|256|43264|1152|blocktiling_1d|8272.45|
|64|1605632|147|blocktiling_1d|10400.3|
|64|559104|147|blocktiling_1d| 3626.16|
|256|50176|1024|blocktiling_1d|8499.03|

![picture](./imgs/algorithm_performance_plot.png)
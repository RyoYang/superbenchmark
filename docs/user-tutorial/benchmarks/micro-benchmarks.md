---
id: micro-benchmarks
---

# Micro Benchmarks

## Computation Benchmarks

### `kernel-launch`

#### Introduction

Measure GPU kernel launch latency,
which is defined as the time range from the beginning of the launch API call to the beginning of the kernel execution.

#### Metrics

| Name                         | Unit      | Description                          |
|------------------------------|-----------|--------------------------------------|
| kernel-launch/event_overhead | time (ms) | Launch latency measured in GPU time. |
| kernel-launch/wall_overhead  | time (ms) | Launch latency measured in CPU time. |

### `gemm-flops`

#### Introduction

Measure the GPU GEMM FLOPS for different float and int data types, with or without Tensor Core (XDLOPS),
performed by NVIDIA [cutlass](https://github.com/NVIDIA/cutlass/tree/ccb697bac77fcc898e9c897b2c90aa5b60ac72fb)
or AMD [rocblas-bench](https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/develop/clients/benchmarks).

#### Metrics

| Name                   | Unit           | Description                                             |
|------------------------|----------------|---------------------------------------------------------|
| gemm-flops/FP64        | FLOPS (GFLOPS) | GEMM float64 peak FLOPS.                                |
| gemm-flops/FP32        | FLOPS (GFLOPS) | GEMM float32 peak FLOPS.                                |
| gemm-flops/FP16        | FLOPS (GFLOPS) | GEMM float16 peak FLOPS.                                |
| gemm-flops/FP64_TC     | FLOPS (GFLOPS) | GEMM float64 peak FLOPS with NVIDIA Tensor Core.        |
| gemm-flops/TF32_TC     | FLOPS (GFLOPS) | GEMM tensor-float32 peak FLOPS with NVIDIA Tensor Core. |
| gemm-flops/FP16_TC     | FLOPS (GFLOPS) | GEMM float16 peak FLOPS with NVIDIA Tensor Core.        |
| gemm-flops/BF16_TC     | FLOPS (GFLOPS) | GEMM bfloat16 peak FLOPS with NVIDIA Tensor Core.       |
| gemm-flops/INT8_TC     | IOPS (GIOPS)   | GEMM int8 peak IOPS with NVIDIA Tensor Core.            |
| gemm-flops/INT4_TC     | IOPS (GIOPS)   | GEMM int4 peak IOPS with NVIDIA Tensor Core.            |
| gemm-flops/FP32_xDLOPS | FLOPS (GFLOPS) | GEMM tensor-float32 peak FLOPS with AMD XDLOPS.         |
| gemm-flops/FP16_xDLOPS | FLOPS (GFLOPS) | GEMM float16 peak FLOPS with AMD XDLOPS.                |
| gemm-flops/BF16_xDLOPS | FLOPS (GFLOPS) | GEMM bfloat16 peak FLOPS with AMD XDLOPS.               |
| gemm-flops/INT8_xDLOPS | IOPS (GIOPS)   | GEMM int8 peak IOPS with AMD XDLOPS.                    |

### `matmul`

#### Introduction

Large scale matmul operation using `torch.matmul` with one GPU.

#### Metrics

| Name                      | Unit      | Description                    |
|---------------------------|-----------|--------------------------------|
| pytorch-matmul/nosharding | time (ms) | Time of pure matmul operation. |

### `cublas-function`

TODO

### `cudnn-function`

TODO

## Communication Benchmarks

### `mem-bw`

#### Introduction

Measure the memory copy bandwidth across PCI-e and memory copy bandwidth between GPUs,
performed by [NVIDIA](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/bandwidthTest)
or [AMD](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/1_Utils/hipBusBandwidth) bandwidth test tool.

#### Metrics

| Name              | Unit             | Description                      |
|-------------------|------------------|----------------------------------|
| mem-bw/H2D_Mem_BW | bandwidth (GB/s) | Host to device copy bandwidth.   |
| mem-bw/D2H_Mem_BW | bandwidth (GB/s) | Device to host copy bandwidth.   |
| mem-bw/D2D_Mem_BW | bandwidth (GB/s) | Device to device copy bandwidth. |

### `gpu-sm-copy-bw`

Measure the memory copy bandwidth across PCI-e and memory copy bandwidth between GPUs, initialized by GPU SM.

#### Metrics

| Name                | Unit             | Description                                          |
|---------------------|------------------|------------------------------------------------------|
| gpu-sm-copy-bw/htod | bandwidth (GB/s) | Host to device copy bandwidth initialized by GPU SM. |
| gpu-sm-copy-bw/dtoh | bandwidth (GB/s) | Device to host copy bandwidth initialized by GPU SM. |

### `ib-loopback`

#### Introduction

Measure the InfiniBand loopback verbs bandwidth, performed by
[OFED performance tests](https://github.com/linux-rdma/perftest/tree/7504ce48ac396a02f4d00de359257b2cb8458f06).

#### Metrics

| Name                                               | Unit             | Description                                                  |
|----------------------------------------------------|------------------|--------------------------------------------------------------|
| ib-loopback/IB\_write\_${msg\_size}\_Avg_${ib_dev} | bandwidth (MB/s) | InfiniBand loopback write bandwidth with given message size. |
| ib-loopback/IB\_read\_${msg\_size}\_Avg_${ib_dev}  | bandwidth (MB/s) | InfiniBand loopback read bandwidth with given message size.  |
| ib-loopback/IB\_send\_${msg\_size}\_Avg_${ib_dev}  | bandwidth (MB/s) | InfiniBand loopback send bandwidth with given message size.  |

### `nccl-bw` / `rccl-bw`

#### Introduction

Measure the performance of NCCL/RCCL operations,
performed by [nccl-tests](https://github.com/NVIDIA/nccl-tests/tree/44df0bf010dcc95e840ca0fb7466c67cff3f1f0f)
or [rccl-tests](https://github.com/ROCmSoftwarePlatform/rccl-tests/tree/dc1ad4853d7ec738387d42a75a58a98d7af00c7b).
Support the following operations currently: allreduce, allgather, broadcast, reduce, reducescatter, alltoall.

#### Metrics

| Name                                   | Unit             | Description                                                 |
|----------------------------------------|------------------|-------------------------------------------------------------|
| nccl-bw/${operation}_${msg_size}_time  | time (us)        | NCCL operation lantency with given message size.            |
| nccl-bw/${operation}_${msg_size}_algbw | bandwidth (GB/s) | NCCL operation algorithm bandwidth with given message size. |
| nccl-bw/${operation}_${msg_size}_busbw | bandwidth (GB/s) | NCCL operation bus bandwidth with given message size.       |
| rccl-bw/${operation}_${msg_size}_time  | time (us)        | RCCL operation lantency with given message size.            |
| rccl-bw/${operation}_${msg_size}_algbw | bandwidth (GB/s) | RCCL operation algorithm bandwidth with given message size. |
| rccl-bw/${operation}_${msg_size}_busbw | bandwidth (GB/s) | RCCL operation bus bandwidth with given message size.       |

## Computation-communication Benchmarks

### `computation-communication-overlap`

#### Introduction

Test the performance of single node when communication and computation overlap.

#### Metrics

| Name                                                  | Unit      | Description                                                  |
|-------------------------------------------------------|-----------|--------------------------------------------------------------|
| pytorch-computation-communication-overlap/mul_cost    | time (ms) | Time of communication and mul kernel computation overlap.    |
| pytorch-computation-communication-overlap/matmul_cost | time (ms) | Time of communication and matmul kernel computation overlap. |

####

### `sharding-matmul`

#### Introduction

Test the performance of large scale matmul operation with multiple GPUs:
* allreduce: Each GPU will calculate part of the MM calculation, and use AllReduce to merge all data into one tensor.
* allgather: Each GPU will calculate part of the MM calculation, and use AllGather + Concat to merge all data into one tensor.

#### Metrics

| Name                              | Unit      | Description                              |
|-----------------------------------|-----------|------------------------------------------|
| pytorch-sharding-matmul/allreduce | time (ms) | Time of sharding matmul using allreduce. |
| pytorch-sharding-matmul/allgather | time (ms) | Time of sharding matmul using allgather. |

## Storage Benchmarks

### `disk-benchmark`

#### Introduction

Measure the disk performance through [FIO](https://github.com/axboe/fio/tree/0313e938c9c8bb37d71dade239f1f5326677b079).

#### Metrics

| Name                                                               | Unit         | Description                                              |
|--------------------------------------------------------------------|--------------|----------------------------------------------------------|
| disk-benchmark/${disk_name}_rand_read_write_bs                     | size (bytes) | Disk random read write block size.                       |
| disk-benchmark/${disk_name}_rand_read_write_read_iops              | IOPS         | Disk random read write read IOPS.                        |
| disk-benchmark/${disk_name}_rand_read_write_read_lat_ns_95.000000  | time (ns)    | Disk random read write read latency in 95.0 percentile.  |
| disk-benchmark/${disk_name}_rand_read_write_read_lat_ns_99.000000  | time (ns)    | Disk random read write read latency in 99.0 percentile.  |
| disk-benchmark/${disk_name}_rand_read_write_read_lat_ns_99.900000  | time (ns)    | Disk random read write read latency in 99.9 percentile.  |
| disk-benchmark/${disk_name}_rand_read_write_write_iops             | IOPS         | Disk random read write write IOPS.                       |
| disk-benchmark/${disk_name}_rand_read_write_write_lat_ns_95.000000 | time (ns)    | Disk random read write write latency in 95.0 percentile. |
| disk-benchmark/${disk_name}_rand_read_write_write_lat_ns_99.000000 | time (ns)    | Disk random read write write latency in 99.0 percentile. |
| disk-benchmark/${disk_name}_rand_read_write_write_lat_ns_99.900000 | time (ns)    | Disk random read write write latency in 99.9 percentile. |
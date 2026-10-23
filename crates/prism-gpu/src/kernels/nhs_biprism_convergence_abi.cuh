#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Shared unchanged convergence ABI for main and explicit preparation fault channels.
struct __align__(128) BiprismConvergenceRecordFfi {
    uint32_t abi_magic;
    uint32_t leaf_id;
    uint32_t epoch_leaf_fault;
    uint32_t epoch_cluster_fault;
    uint32_t terminal_fault_latch;
    uint32_t committed_substeps;
    uint32_t picard_iters_used;
    uint32_t dec_iters_used;
    float    last_picard_residual;
    float    last_dec_residual;
    float    step_energy_drift;
    float    w_xx_trace;
    float    w_x_lambda_trace;
    float    w_x_tau_trace;
    float    unitarity_defect;
    float    pulay_force_norm;
    uint32_t reserved[16];
};
static_assert(sizeof(BiprismConvergenceRecordFfi) == 128, "BiprismConvergenceRecordFfi must be 128B");
static_assert(alignof(BiprismConvergenceRecordFfi) == 128, "BiprismConvergenceRecordFfi must be align(128)");


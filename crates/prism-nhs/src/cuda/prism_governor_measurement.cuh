#ifndef PRISM_GOVERNOR_MEASUREMENT_CUH
#define PRISM_GOVERNOR_MEASUREMENT_CUH
#include <stdint.h>
#include <stddef.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define PRISM_WINDOW_HD __host__ __device__
#else
#define PRISM_WINDOW_HD
#endif

// Shared by the archive SFA and the separately loaded fused MD PTX. These
// records describe completed physical work; pointer presence is not evidence.
enum PrismWindowStatus : uint32_t {
    PRISM_WINDOW_UNMEASURED = 0, PRISM_WINDOW_OPEN = 1,
    PRISM_WINDOW_COMPLETE = 2, PRISM_WINDOW_NOT_APPLICABLE = 3,
    PRISM_WINDOW_FAILED_DIAGNOSTIC = 4,
    PRISM_WINDOW_BAD_SCHEMA = 5, PRISM_WINDOW_OWNER_MISMATCH = 6,
    PRISM_WINDOW_STALE = 7, PRISM_WINDOW_INCOMPLETE_COVERAGE = 8
};
struct alignas(16) PrismForceWindowV1 {
    uint32_t schema_version, status, source_stream_id, reserved0;
    uint64_t allocation_generation, window_epoch, first_step_epoch, last_step_epoch;
    float force_norm;
    uint32_t producer_phase;
    uint64_t reserved1;
};
struct alignas(16) PrismConstraintWindowV1 {
    uint32_t schema_version, status, source_stream_id, phase_mask;
    uint32_t n_constraint_clusters, expected_physical_steps, completed_physical_steps, fatal_failure_count;
    uint64_t allocation_generation, window_epoch, first_step_epoch, last_step_epoch;
    uint64_t expected_cluster_solves[3], completed_cluster_solves[3];
    uint64_t sealed_epoch;
    uint64_t* phase_cluster_epochs;
};
static_assert(sizeof(PrismForceWindowV1) == 64, "force window ABI");
static_assert(offsetof(PrismForceWindowV1, force_norm) == 48, "force value ABI");
static_assert(sizeof(PrismConstraintWindowV1) == 128, "constraint window ABI");
static_assert(offsetof(PrismConstraintWindowV1, phase_cluster_epochs) == 120, "constraint rows ABI");

PRISM_WINDOW_HD inline uint32_t prism_force_window_status(
    const PrismForceWindowV1* w, uint32_t owner, uint64_t generation, uint64_t epoch) {
    if (!w) return PRISM_WINDOW_UNMEASURED;
    if (w->schema_version != 1) return PRISM_WINDOW_BAD_SCHEMA;
    if (w->source_stream_id != owner || w->allocation_generation != generation)
        return PRISM_WINDOW_OWNER_MISMATCH;
    if (w->status != PRISM_WINDOW_COMPLETE) return w->status;
    if (!epoch || w->window_epoch != epoch || w->first_step_epoch != epoch || w->last_step_epoch != epoch)
        return PRISM_WINDOW_STALE;
    // Measured NaN/Inf deliberately remain COMPLETE: the policy's force
    // emergency gate must see them, rather than an unmeasured substitute.
    return PRISM_WINDOW_COMPLETE;
}
PRISM_WINDOW_HD inline uint32_t prism_constraint_window_status(
    const PrismConstraintWindowV1* w, uint32_t owner, uint64_t generation, uint64_t epoch) {
    if (!w) return PRISM_WINDOW_UNMEASURED;
    if (w->schema_version != 1) return PRISM_WINDOW_BAD_SCHEMA;
    if (w->source_stream_id != owner || w->allocation_generation != generation)
        return PRISM_WINDOW_OWNER_MISMATCH;
    if (w->status != PRISM_WINDOW_COMPLETE) return w->status;
    if (!epoch || w->window_epoch != epoch || w->sealed_epoch != epoch ||
        w->first_step_epoch != epoch || w->last_step_epoch != epoch) return PRISM_WINDOW_STALE;
    if (!w->n_constraint_clusters) return PRISM_WINDOW_NOT_APPLICABLE;
    if (w->phase_mask != 7 || w->expected_physical_steps != 1 || w->completed_physical_steps != 1)
        return PRISM_WINDOW_INCOMPLETE_COVERAGE;
    for (int p = 0; p < 3; ++p)
        if (w->expected_cluster_solves[p] != w->n_constraint_clusters ||
            w->completed_cluster_solves[p] != w->expected_cluster_solves[p])
            return PRISM_WINDOW_INCOMPLETE_COVERAGE;
    return PRISM_WINDOW_COMPLETE;
}

#ifdef __CUDACC__
// Each phase owns disjoint rows; every row is stamped only after the actual
// SHAKE/RATTLE solver returns true. Failed solver traps remain at the call site.
__device__ inline void prism_constraint_success(PrismConstraintWindowV1* w, int phase, int cluster) {
    if (w && w->status == PRISM_WINDOW_OPEN && phase >= 0 && phase < 3 &&
        cluster >= 0 && static_cast<uint32_t>(cluster) < w->n_constraint_clusters && w->phase_cluster_epochs)
        w->phase_cluster_epochs[static_cast<uint64_t>(phase) * w->n_constraint_clusters + cluster] = w->window_epoch;
}
__device__ inline void prism_constraint_failure(PrismConstraintWindowV1* w) {
    if (w) { atomicAdd(&w->fatal_failure_count, 1u); w->status = PRISM_WINDOW_FAILED_DIAGNOSTIC; }
}
#endif
#undef PRISM_WINDOW_HD
#endif

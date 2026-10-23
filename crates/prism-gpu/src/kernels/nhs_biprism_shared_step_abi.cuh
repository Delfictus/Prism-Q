#pragma once
#include <stdint.h>
#include <stddef.h>
// One explicit physical-time owner for the coupled body. This is retained
// control state, never an alias of derived numerical exchange scratch.
struct alignas(128) BiprismSharedStepFfi {
    uint32_t abi_magic;                  //0 BPST
    uint32_t state;                      //4 fresh0,selected1,accepted2,terminal3
    uint32_t sticky_fault;               //8 never cleared in replay
    uint32_t scheduled_step;             //12 current attempted physical body
    uint32_t last_accepted_step;         //16 last collective accepted body
    uint32_t accepted_bodies;            //20 actual accepted count
    uint32_t terminal_frozen_mask;       //24 bit per protocol whose controls are held
    uint32_t policy;                     //28 explicit common-min policy1
    float native_dt_ps;                  //32 exact chosen FP32 native dt
    float outer_dt_au;                   //36 FP32(double(native_dt_ps)*PS_TO_AU)
    float minimum_request_ps;            //40 retained min of all8 requests
    float maximum_request_ps;            //44 retained max of all8 requests
    double cumulative_accepted_time_ps;  //48 sum of accepted represented native dt
    uint64_t generation;                //56 immutable host allocation/run generation
    float requested_dt_ps[8];            //64 original per-leaf requests before common assignment
    uint32_t protocol_control_step[8];   //96 min(scheduled_step,nominal_terminal_step)
};
static_assert(sizeof(BiprismSharedStepFfi)==128,"shared physical step ABI");
static_assert(alignof(BiprismSharedStepFfi)==128,"shared physical step alignment");
static_assert(offsetof(BiprismSharedStepFfi,native_dt_ps)==32,"native dt pointer ABI");
static_assert(offsetof(BiprismSharedStepFfi,outer_dt_au)==36,"AU dt pointer ABI");
static_assert(offsetof(BiprismSharedStepFfi,cumulative_accepted_time_ps)==48,"accepted time ABI");
static_assert(offsetof(BiprismSharedStepFfi,generation)==56,"generation ABI");
static_assert(offsetof(BiprismSharedStepFfi,requested_dt_ps)==64,"requests ABI");
static_assert(offsetof(BiprismSharedStepFfi,protocol_control_step)==96,"held controls ABI");
static constexpr uint32_t BIPRISM_SHARED_STEP_MAGIC=0x42505354u;
static constexpr uint32_t BIPRISM_SHARED_STEP_COMMON_MIN=1u;
static constexpr uint32_t BIPRISM_SHARED_STEP_BOUNDED_COMMON_MIN=2u;
static constexpr uint32_t BIPRISM_SHARED_STEP_SELECTED=1u;
static constexpr uint32_t BIPRISM_SHARED_STEP_ACCEPTED=2u;
// Dedicated protocol continuation snapshots. Every held field is captured from
// the actual Director output at that leaf's nominal terminal step. Physical
// current_step and housekeeping counters continue. Snapshot/restore occurs
// before the ordinary live ASC modifier/decay code, so endogenous feedback is
// not frozen. Terminal baseline UV scheduling fields are not recomputed.
struct alignas(16) BiprismTerminalControlsFfi {
    uint32_t abi_magic,leaf_id,terminal_step,captured_step;
    float temperature,effective_gamma,requested_dt_ps,uv_wavelength_nm;
    int32_t uv_burst_active,uv_target_idx;
    uint32_t phase_bits,reserved;
};
static_assert(sizeof(BiprismTerminalControlsFfi)==48,"terminal controls ABI");

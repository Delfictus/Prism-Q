#ifndef PRISMCODE_CUDA_H
#define PRISMCODE_CUDA_H

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#define PRISMCODE_CUDA_ABI_V0_1 0x00010000u
#define PRISMCODE_CUDA_ISA_V0_1 0x00010000u
#define PRISMCODE_CUDA_LINK_SENTINEL 0x50435531u

#define PRISMCODE_CUDA_THREADS_PER_BLOCK 256u
#define PRISMCODE_CUDA_DEVICE_NAME_BYTES 128u
#define PRISMCODE_CUDA_M1_SELFTEST_ELEMENTS 257u

enum PrismcodeOpcodeCode {
    PRISMCODE_OP_LOAD_NANL_STRAND = 1u,
    PRISMCODE_OP_EMIT_EVENT = 37u,
    PRISMCODE_OP_EMIT_SPIKE = 38u,
    PRISMCODE_OP_EMIT_RUNTIME_EVENT = 49u,
    PRISMCODE_OP_EMIT_DRF_NODE = 62u,
    PRISMCODE_OP_EMIT_DRF_EDGE = 63u,
    PRISMCODE_OP_LOAD_DRF_CHANNEL_VECTOR = 64u,
    PRISMCODE_OP_APPLY_RESONANCE = 65u,
    PRISMCODE_OP_APPLY_REFRACTORY_STATE = 66u,
    PRISMCODE_OP_FIRE_DRF_EVENT = 67u,
    PRISMCODE_OP_SUPPRESS_DRF_EVENT = 68u,
    PRISMCODE_OP_ROUTE_DRF_REPLAY = 69u,
    PRISMCODE_OP_SET_TRANSITION_PROBABILITY = 72u,
    PRISMCODE_OP_SET_DWELL_TIME = 73u,
    PRISMCODE_OP_SET_MFPT = 74u,
    PRISMCODE_OP_SET_COMMITTOR = 75u,
    PRISMCODE_OP_PROJECT_SLOW_MODE = 76u,
    PRISMCODE_OP_PROJECT_SPECTRAL_MODE = 77u,
    PRISMCODE_OP_SET_KINETIC_BARRIER = 78u,
    PRISMCODE_OP_SET_TRANSITION_ENTROPY = 79u,
    PRISMCODE_OP_SET_NRF_RESONANCE_SCORE = 85u,
    PRISMCODE_OP_SET_DTW_TEMPORAL_WAVEFORM_SCORE = 86u,
    PRISMCODE_OP_SET_TRANSITION_STRAIN_SCORE = 87u,
    PRISMCODE_OP_EMIT_NANL_MEMORY_INSTRUCTIONS = 115u
};

enum PrismcodeOpcodeFamily {
    PRISMCODE_FAMILY_MEMORY = 1u,
    PRISMCODE_FAMILY_IDENTITY = 2u,
    PRISMCODE_FAMILY_CHEMISTRY = 3u,
    PRISMCODE_FAMILY_MD_PROVENANCE = 4u,
    PRISMCODE_FAMILY_EVENT = 5u,
    PRISMCODE_FAMILY_TIME = 6u,
    PRISMCODE_FAMILY_DRF = 7u,
    PRISMCODE_FAMILY_MSM_SPECTRAL = 8u,
    PRISMCODE_FAMILY_UNCERTAINTY = 9u,
    PRISMCODE_FAMILY_TEACHER_STUDENT = 10u,
    PRISMCODE_FAMILY_GFLOWNET = 11u,
    PRISMCODE_FAMILY_PRISMAI = 12u,
    PRISMCODE_FAMILY_VALIDATION = 13u
};

enum PrismcodeSemanticStatus {
    PRISMCODE_STATUS_ACTIVE = 1u,
    PRISMCODE_STATUS_PARTIAL = 2u,
    PRISMCODE_STATUS_PENDING = 3u,
    PRISMCODE_STATUS_INVALIDATED = 4u,
    PRISMCODE_STATUS_SUPERSEDED = 5u
};

enum PrismcodeEventKind {
    PRISMCODE_EVENT_KIND_NONE = 0u,
    PRISMCODE_EVENT_KIND_GENERIC = 1u,
    PRISMCODE_EVENT_KIND_SPIKE = 2u,
    PRISMCODE_EVENT_KIND_DRF_FIRE = 3u,
    PRISMCODE_EVENT_KIND_DRF_SUPPRESS = 4u,
    PRISMCODE_EVENT_KIND_RUNTIME = 5u,
    PRISMCODE_EVENT_KIND_VALIDATION = 6u
};

enum PrismcodeEventFlags {
    PRISMCODE_EVENT_FLAG_VALID = 1u << 0,
    PRISMCODE_EVENT_FLAG_INTEGRATED = 1u << 1,
    PRISMCODE_EVENT_FLAG_FIRED = 1u << 2,
    PRISMCODE_EVENT_FLAG_SUPPRESSED = 1u << 3,
    PRISMCODE_EVENT_FLAG_SOURCE_INVALID = 1u << 4
};

enum PrismcodeTensorOp {
    PRISMCODE_TENSOR_OP_COPY = 0u,
    PRISMCODE_TENSOR_OP_ADD = 1u,
    PRISMCODE_TENSOR_OP_MUL = 2u,
    PRISMCODE_TENSOR_OP_AXPBY = 3u,
    PRISMCODE_TENSOR_OP_RELU = 4u,
    PRISMCODE_TENSOR_OP_CLAMP = 5u
};

enum PrismcodeCudaRuntimeState {
    PRISMCODE_CUDA_RUNTIME_UNKNOWN = 0u,
    PRISMCODE_CUDA_RUNTIME_LINKED_READY = 1u,
    PRISMCODE_CUDA_RUNTIME_NO_DEVICE = 2u,
    PRISMCODE_CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR = 3u
};

enum PrismcodeCudaSelfTestStatus {
    PRISMCODE_CUDA_SELFTEST_UNKNOWN = 0u,
    PRISMCODE_CUDA_SELFTEST_PASS = 1u,
    PRISMCODE_CUDA_SELFTEST_FAIL = 2u,
    PRISMCODE_CUDA_SELFTEST_SKIPPED = 3u
};

typedef struct PrismcodeInstruction {
    uint64_t row_index;
    uint64_t subject_index;
    uint64_t payload_hash_lo;
    uint64_t payload_hash_hi;
    uint64_t previous_hash_lo;
    uint64_t previous_hash_hi;
    uint32_t opcode_code;
    uint32_t opcode_family;
    uint32_t semantic_status;
    uint32_t flags;
} PrismcodeInstruction;

typedef struct PrismcodeMemoryRow {
    uint64_t row_index;
    uint64_t nanl_token_u64;
    uint64_t row_hash_lo;
    uint64_t row_hash_hi;
    double drf_resonance_fusion_score;
    double io_replay_priority_score;
    double nrf_resonance_score;
    double dstw_temporal_waveform_score;
    double dstw_durability_score;
    double os_gflownet_reward;
    uint64_t nrf_event_count_compiled;
    uint64_t neuro_event_count_compiled;
    int64_t missing_binding_field_count;
    uint32_t binding_status;
    uint32_t flags;
} PrismcodeMemoryRow;

typedef struct PrismcodeNanlPackedRow {
    uint64_t row_index;
    uint64_t nanl_token_u64;
    uint64_t packed_fingerprint_lo;
    uint64_t packed_fingerprint_hi;
    uint32_t validity_flags;
    uint32_t event_hint;
    uint32_t instruction_opcode_hint;
    uint32_t reserved;
} PrismcodeNanlPackedRow;

typedef struct PrismcodeEventRecord {
    uint64_t row_index;
    uint64_t event_time_tick;
    uint64_t source_hash_lo;
    uint64_t source_hash_hi;
    float intensity;
    float resonance;
    uint32_t opcode_code;
    uint32_t event_kind;
    uint32_t semantic_status;
    uint32_t flags;
} PrismcodeEventRecord;

typedef struct PrismcodeDrfState {
    uint64_t row_index;
    double resonance;
    double refractory;
    double threshold;
    float last_fire_intensity;
    uint32_t fire_count;
    uint32_t flags;
    uint32_t reserved;
} PrismcodeDrfState;

typedef struct PrismcodeThermoSample {
    uint64_t row_index;
    double energy;
    double entropy;
    double temperature;
    float force_norm;
    float weight;
    uint32_t flags;
    uint32_t reserved;
} PrismcodeThermoSample;

typedef struct PrismcodeThermoSummary {
    double energy_sum;
    double entropy_sum;
    double temperature_sum;
    double weighted_energy_sum;
    uint64_t sample_count;
    uint64_t invalid_count;
    uint32_t max_force_q24;
    uint32_t flags;
} PrismcodeThermoSummary;

typedef struct PrismcodeTensorOpParams {
    uint64_t element_count;
    uint32_t op;
    uint32_t flags;
    float alpha;
    float beta;
    float gamma;
    float clamp_min;
    float clamp_max;
} PrismcodeTensorOpParams;

typedef struct PrismcodeCudaRuntimeStatus {
    uint32_t abi_version;
    uint32_t isa_version;
    uint32_t link_sentinel;
    uint32_t runtime_state;
    int32_t runtime_version;
    int32_t driver_version;
    int32_t device_count;
    int32_t active_device;
    int32_t compute_major;
    int32_t compute_minor;
    uint64_t total_global_mem;
    uint32_t error_code;
    uint32_t reserved0;
    char device_name[PRISMCODE_CUDA_DEVICE_NAME_BYTES];
} PrismcodeCudaRuntimeStatus;

typedef struct PrismcodeCudaSelfTestResult {
    uint32_t status;
    uint32_t error_code;
    uint64_t element_count;
    float max_abs_error;
    uint32_t tensor_copy_parity;
    uint64_t mismatch_count;
    uint64_t sentinel_hash;
    uint32_t reserved0;
    uint32_t reserved1;
} PrismcodeCudaSelfTestResult;

#ifdef __cplusplus
static_assert(sizeof(PrismcodeInstruction) == 64, "PrismcodeInstruction ABI drift");
static_assert(sizeof(PrismcodeMemoryRow) == 112, "PrismcodeMemoryRow ABI drift");
static_assert(sizeof(PrismcodeNanlPackedRow) == 48, "PrismcodeNanlPackedRow ABI drift");
static_assert(sizeof(PrismcodeEventRecord) == 56, "PrismcodeEventRecord ABI drift");
static_assert(sizeof(PrismcodeDrfState) == 48, "PrismcodeDrfState ABI drift");
static_assert(sizeof(PrismcodeThermoSample) == 48, "PrismcodeThermoSample ABI drift");
static_assert(sizeof(PrismcodeThermoSummary) == 56, "PrismcodeThermoSummary ABI drift");
static_assert(sizeof(PrismcodeTensorOpParams) == 40, "PrismcodeTensorOpParams ABI drift");
static_assert(sizeof(PrismcodeCudaRuntimeStatus) == 184, "PrismcodeCudaRuntimeStatus ABI drift");
static_assert(sizeof(PrismcodeCudaSelfTestResult) == 48, "PrismcodeCudaSelfTestResult ABI drift");

extern "C" {
#endif

uint32_t prismcode_cuda_abi_version(void);
uint32_t prismcode_cuda_isa_version(void);
uint32_t prismcode_cuda_link_probe(void);

cudaError_t prismcode_cuda_runtime_status(PrismcodeCudaRuntimeStatus* out_status);
cudaError_t prismcode_cuda_m1_selftest(PrismcodeCudaSelfTestResult* out_result, cudaStream_t stream);

cudaError_t prismcode_cuda_nanl_pack(
    const PrismcodeMemoryRow* d_rows,
    PrismcodeNanlPackedRow* d_out,
    uint64_t row_count,
    cudaStream_t stream
);

cudaError_t prismcode_cuda_event_integrate(
    const PrismcodeInstruction* d_instructions,
    const PrismcodeMemoryRow* d_rows,
    PrismcodeEventRecord* d_events,
    uint64_t instruction_count,
    uint64_t row_count,
    uint64_t event_time_base,
    cudaStream_t stream
);

cudaError_t prismcode_cuda_drf_resonate(
    const uint16_t* d_channel_vectors,
    const float* d_channel_weights,
    PrismcodeDrfState* d_states,
    uint64_t row_count,
    uint32_t channels_per_row,
    cudaStream_t stream
);

cudaError_t prismcode_cuda_drf_fire(
    PrismcodeDrfState* d_states,
    PrismcodeEventRecord* d_events,
    uint64_t row_count,
    double threshold_floor,
    double refractory_period,
    uint64_t event_time_tick,
    cudaStream_t stream
);

cudaError_t prismcode_cuda_thermo_reduce(
    const PrismcodeThermoSample* d_samples,
    PrismcodeThermoSummary* d_summary,
    uint64_t sample_count,
    cudaStream_t stream
);

cudaError_t prismcode_cuda_tensor_ops(
    const float* d_lhs,
    const float* d_rhs,
    float* d_out,
    PrismcodeTensorOpParams params,
    cudaStream_t stream
);

int pc_cuda_get_device_count(int32_t* out_count);
int pc_cuda_get_device_info(PrismcodeCudaRuntimeStatus* out_status);
int pc_cuda_selftest(PrismcodeCudaSelfTestResult* out_result);
int pc_cuda_execute_tensor_copy(const float* h_in, float* h_out, uint64_t count);
int pc_cuda_execute_tensor_reduce(const float* h_in, uint64_t count, float* h_sum_out);
int pc_cuda_execute_event_integrate(
    uint64_t instruction_count,
    uint64_t row_count,
    uint64_t* out_event_count,
    uint64_t* out_hash
);
int pc_cuda_execute_nanl_bin_pack(const uint64_t* h_tokens, uint64_t* h_packed, uint64_t count);
int pc_cuda_execute_nanl_q4_validate(
    const uint8_t* h_q4,
    uint64_t len,
    uint32_t* out_pass,
    uint64_t* out_hash
);
int pc_cuda_execute_drf_fire(
    const float* h_resonance,
    float* h_intensity,
    uint64_t count,
    float threshold,
    uint64_t* out_fired
);
const char* pc_cuda_status_string(int code);

#ifdef __cplusplus
}
#endif

#endif

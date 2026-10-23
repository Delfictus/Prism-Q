#include "prismcode_cuda.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../kernels/nanl_bin_pack.cu"
#include "../kernels/nanl_q4_validate.cu"
#include "../kernels/tensor_reduce.cu"

namespace {

cudaError_t launch_shape(uint64_t count, dim3* grid, dim3* block) {
    if (grid == nullptr || block == nullptr || count == 0u) {
        return cudaErrorInvalidValue;
    }
    *block = dim3(PRISMCODE_CUDA_THREADS_PER_BLOCK, 1u, 1u);
    const uint64_t blocks =
        (count + PRISMCODE_CUDA_THREADS_PER_BLOCK - 1u) /
        PRISMCODE_CUDA_THREADS_PER_BLOCK;
    if (blocks > 0x7fffffffu) {
        return cudaErrorInvalidValue;
    }
    *grid = dim3(static_cast<unsigned int>(blocks), 1u, 1u);
    return cudaSuccess;
}

uint64_t mix64(uint64_t hash, uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    hash ^= hash >> 30;
    hash *= 0xbf58476d1ce4e5b9ULL;
    hash ^= hash >> 27;
    hash *= 0x94d049bb133111ebULL;
    hash ^= hash >> 31;
    return hash;
}

int error_to_int(cudaError_t err) {
    return static_cast<int>(err);
}

}  // namespace

extern "C" {

int pc_cuda_get_device_count(int32_t* out_count) {
    if (out_count == nullptr) {
        return error_to_int(cudaErrorInvalidValue);
    }
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaSuccess) {
        *out_count = static_cast<int32_t>(count);
    }
    return error_to_int(err);
}

int pc_cuda_get_device_info(PrismcodeCudaRuntimeStatus* out_status) {
    return error_to_int(prismcode_cuda_runtime_status(out_status));
}

int pc_cuda_selftest(PrismcodeCudaSelfTestResult* out_result) {
    return error_to_int(prismcode_cuda_m1_selftest(out_result, nullptr));
}

int pc_cuda_execute_tensor_copy(const float* h_in, float* h_out, uint64_t count) {
    if (count == 0u) {
        return error_to_int(cudaSuccess);
    }
    if (h_in == nullptr || h_out == nullptr) {
        return error_to_int(cudaErrorInvalidValue);
    }

    const size_t byte_count = static_cast<size_t>(count) * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_in), byte_count);
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_out), byte_count);
    if (err == cudaSuccess) {
        err = cudaMemcpy(d_in, h_in, byte_count, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        PrismcodeTensorOpParams params;
        std::memset(&params, 0, sizeof(params));
        params.element_count = count;
        params.op = PRISMCODE_TENSOR_OP_COPY;
        err = prismcode_cuda_tensor_ops(d_in, nullptr, d_out, params, nullptr);
    }
    if (err == cudaSuccess) {
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(h_out, d_out, byte_count, cudaMemcpyDeviceToHost);
    }
    cudaError_t free_in = cudaFree(d_in);
    cudaError_t free_out = cudaFree(d_out);
    if (err == cudaSuccess && free_in != cudaSuccess) {
        err = free_in;
    }
    if (err == cudaSuccess && free_out != cudaSuccess) {
        err = free_out;
    }
    return error_to_int(err);
}

int pc_cuda_execute_tensor_reduce(const float* h_in, uint64_t count, float* h_sum_out) {
    if (h_sum_out == nullptr) {
        return error_to_int(cudaErrorInvalidValue);
    }
    *h_sum_out = 0.0f;
    if (count == 0u) {
        return error_to_int(cudaSuccess);
    }
    if (h_in == nullptr) {
        return error_to_int(cudaErrorInvalidValue);
    }

    dim3 grid;
    dim3 block;
    cudaError_t err = launch_shape(count, &grid, &block);
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    const size_t in_bytes = static_cast<size_t>(count) * sizeof(float);
    const size_t block_count = static_cast<size_t>(grid.x);
    const size_t block_bytes = block_count * sizeof(float);
    float* d_in = nullptr;
    float* d_block_sums = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&d_in), in_bytes);
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_block_sums), block_bytes);
    if (err == cudaSuccess) {
        err = cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        prismcode_cuda_kernels::prismcode_tensor_reduce_sum_kernel<<<grid, block>>>(
            d_in,
            d_block_sums,
            count
        );
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    std::vector<float> host_block_sums(block_count, 0.0f);
    if (err == cudaSuccess) {
        err = cudaMemcpy(host_block_sums.data(), d_block_sums, block_bytes, cudaMemcpyDeviceToHost);
    }
    cudaError_t free_in = cudaFree(d_in);
    cudaError_t free_sums = cudaFree(d_block_sums);
    if (err == cudaSuccess && free_in != cudaSuccess) {
        err = free_in;
    }
    if (err == cudaSuccess && free_sums != cudaSuccess) {
        err = free_sums;
    }
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    float sum = 0.0f;
    for (float block_sum : host_block_sums) {
        sum += block_sum;
    }
    *h_sum_out = sum;
    return error_to_int(cudaSuccess);
}

int pc_cuda_execute_event_integrate(
    uint64_t instruction_count,
    uint64_t row_count,
    uint64_t* out_event_count,
    uint64_t* out_hash
) {
    if (out_event_count == nullptr || out_hash == nullptr || instruction_count == 0u) {
        return error_to_int(cudaErrorInvalidValue);
    }
    if (row_count == 0u) {
        row_count = instruction_count;
    }

    std::vector<PrismcodeInstruction> instructions(static_cast<size_t>(instruction_count));
    std::vector<PrismcodeMemoryRow> rows(static_cast<size_t>(row_count));
    std::vector<PrismcodeEventRecord> events(static_cast<size_t>(instruction_count));
    for (uint64_t i = 0; i < instruction_count; ++i) {
        PrismcodeInstruction ins;
        std::memset(&ins, 0, sizeof(ins));
        ins.row_index = i % row_count;
        ins.payload_hash_lo = mix64(0x45564e54494e5331ULL, i);
        ins.payload_hash_hi = mix64(ins.payload_hash_lo, i + 17u);
        ins.opcode_code = PRISMCODE_OP_EMIT_EVENT;
        ins.opcode_family = PRISMCODE_FAMILY_EVENT;
        ins.semantic_status = PRISMCODE_STATUS_ACTIVE;
        instructions[static_cast<size_t>(i)] = ins;
    }
    for (uint64_t i = 0; i < row_count; ++i) {
        PrismcodeMemoryRow row;
        std::memset(&row, 0, sizeof(row));
        row.row_index = i;
        row.row_hash_lo = mix64(0x524f574841534831ULL, i);
        row.row_hash_hi = mix64(row.row_hash_lo, i + 3u);
        row.nanl_token_u64 = (i & 3u) + 1u;
        row.drf_resonance_fusion_score = static_cast<double>((i % 7u) + 1u) / 10.0;
        row.nrf_resonance_score = static_cast<double>((i % 5u) + 1u) / 10.0;
        row.dstw_temporal_waveform_score = static_cast<double>((i % 3u) + 1u) / 10.0;
        row.binding_status = PRISMCODE_STATUS_ACTIVE;
        rows[static_cast<size_t>(i)] = row;
    }

    PrismcodeInstruction* d_instructions = nullptr;
    PrismcodeMemoryRow* d_rows = nullptr;
    PrismcodeEventRecord* d_events = nullptr;
    const size_t instruction_bytes = instructions.size() * sizeof(PrismcodeInstruction);
    const size_t row_bytes = rows.size() * sizeof(PrismcodeMemoryRow);
    const size_t event_bytes = events.size() * sizeof(PrismcodeEventRecord);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_instructions), instruction_bytes);
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_rows), row_bytes);
    if (err == cudaSuccess) {
        err = cudaMalloc(reinterpret_cast<void**>(&d_events), event_bytes);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(d_instructions, instructions.data(), instruction_bytes, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(d_rows, rows.data(), row_bytes, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        err = prismcode_cuda_event_integrate(
            d_instructions,
            d_rows,
            d_events,
            instruction_count,
            row_count,
            1000u,
            nullptr
        );
    }
    if (err == cudaSuccess) {
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(events.data(), d_events, event_bytes, cudaMemcpyDeviceToHost);
    }
    cudaError_t free_ins = cudaFree(d_instructions);
    cudaError_t free_rows = cudaFree(d_rows);
    cudaError_t free_events = cudaFree(d_events);
    if (err == cudaSuccess && free_ins != cudaSuccess) {
        err = free_ins;
    }
    if (err == cudaSuccess && free_rows != cudaSuccess) {
        err = free_rows;
    }
    if (err == cudaSuccess && free_events != cudaSuccess) {
        err = free_events;
    }
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    uint64_t hash = 0x4556494e54454731ULL;
    uint64_t valid = 0u;
    for (const PrismcodeEventRecord& event : events) {
        if ((event.flags & PRISMCODE_EVENT_FLAG_VALID) != 0u) {
            valid += 1u;
        }
        hash = mix64(hash, event.row_index);
        hash = mix64(hash, static_cast<uint64_t>(event.opcode_code));
        hash = mix64(hash, static_cast<uint64_t>(event.flags));
    }
    *out_event_count = valid;
    *out_hash = hash;
    return error_to_int(cudaSuccess);
}

int pc_cuda_execute_nanl_bin_pack(const uint64_t* h_tokens, uint64_t* h_packed, uint64_t count) {
    if (count == 0u) {
        return error_to_int(cudaSuccess);
    }
    if (h_tokens == nullptr || h_packed == nullptr) {
        return error_to_int(cudaErrorInvalidValue);
    }
    dim3 grid;
    dim3 block;
    cudaError_t err = launch_shape(count, &grid, &block);
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    const size_t bytes = static_cast<size_t>(count) * sizeof(uint64_t);
    uint64_t* d_tokens = nullptr;
    uint64_t* d_packed = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&d_tokens), bytes);
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_packed), bytes);
    if (err == cudaSuccess) {
        err = cudaMemcpy(d_tokens, h_tokens, bytes, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        prismcode_cuda_kernels::prismcode_nanl_bin_pack_u64_kernel<<<grid, block>>>(
            d_tokens,
            d_packed,
            count
        );
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(h_packed, d_packed, bytes, cudaMemcpyDeviceToHost);
    }
    cudaError_t free_tokens = cudaFree(d_tokens);
    cudaError_t free_packed = cudaFree(d_packed);
    if (err == cudaSuccess && free_tokens != cudaSuccess) {
        err = free_tokens;
    }
    if (err == cudaSuccess && free_packed != cudaSuccess) {
        err = free_packed;
    }
    return error_to_int(err);
}

int pc_cuda_execute_nanl_q4_validate(
    const uint8_t* h_q4,
    uint64_t len,
    uint32_t* out_pass,
    uint64_t* out_hash
) {
    if (out_pass == nullptr || out_hash == nullptr || h_q4 == nullptr || len == 0u) {
        return error_to_int(cudaErrorInvalidValue);
    }
    dim3 grid;
    dim3 block;
    cudaError_t err = launch_shape(len, &grid, &block);
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    const size_t byte_count = static_cast<size_t>(len);
    const size_t block_count = static_cast<size_t>(grid.x);
    uint8_t* d_q4 = nullptr;
    uint64_t* d_hashes = nullptr;
    uint32_t* d_failures = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&d_q4), byte_count);
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_hashes), block_count * sizeof(uint64_t));
    if (err == cudaSuccess) {
        err = cudaMalloc(reinterpret_cast<void**>(&d_failures), block_count * sizeof(uint32_t));
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(d_q4, h_q4, byte_count, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        prismcode_cuda_kernels::prismcode_nanl_q4_validate_kernel<<<grid, block>>>(
            d_q4,
            d_hashes,
            d_failures,
            len
        );
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    std::vector<uint64_t> hashes(block_count, 0u);
    std::vector<uint32_t> failures(block_count, 0u);
    if (err == cudaSuccess) {
        err = cudaMemcpy(hashes.data(), d_hashes, block_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(failures.data(), d_failures, block_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    cudaError_t free_q4 = cudaFree(d_q4);
    cudaError_t free_hash = cudaFree(d_hashes);
    cudaError_t free_fail = cudaFree(d_failures);
    if (err == cudaSuccess && free_q4 != cudaSuccess) {
        err = free_q4;
    }
    if (err == cudaSuccess && free_hash != cudaSuccess) {
        err = free_hash;
    }
    if (err == cudaSuccess && free_fail != cudaSuccess) {
        err = free_fail;
    }
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    uint64_t hash = 0x514456414c524550ULL;
    uint64_t failure_count = 0u;
    for (size_t i = 0; i < hashes.size(); ++i) {
        hash = mix64(hash, hashes[i]);
        failure_count += static_cast<uint64_t>(failures[i]);
    }
    *out_pass = failure_count == 0u ? 1u : 0u;
    *out_hash = hash;
    return error_to_int(cudaSuccess);
}

int pc_cuda_execute_drf_fire(
    const float* h_resonance,
    float* h_intensity,
    uint64_t count,
    float threshold,
    uint64_t* out_fired
) {
    if (count == 0u) {
        if (out_fired != nullptr) {
            *out_fired = 0u;
        }
        return error_to_int(cudaSuccess);
    }
    if (h_resonance == nullptr || h_intensity == nullptr || out_fired == nullptr) {
        return error_to_int(cudaErrorInvalidValue);
    }
    std::vector<PrismcodeDrfState> states(static_cast<size_t>(count));
    std::vector<PrismcodeEventRecord> events(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        PrismcodeDrfState state;
        std::memset(&state, 0, sizeof(state));
        state.row_index = i;
        state.resonance = static_cast<double>(h_resonance[static_cast<size_t>(i)]);
        state.threshold = static_cast<double>(threshold);
        state.refractory = 0.0;
        states[static_cast<size_t>(i)] = state;
    }
    PrismcodeDrfState* d_states = nullptr;
    PrismcodeEventRecord* d_events = nullptr;
    const size_t state_bytes = states.size() * sizeof(PrismcodeDrfState);
    const size_t event_bytes = events.size() * sizeof(PrismcodeEventRecord);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_states), state_bytes);
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_events), event_bytes);
    if (err == cudaSuccess) {
        err = cudaMemcpy(d_states, states.data(), state_bytes, cudaMemcpyHostToDevice);
    }
    if (err == cudaSuccess) {
        err = prismcode_cuda_drf_fire(
            d_states,
            d_events,
            count,
            static_cast<double>(threshold),
            2.0,
            2000u,
            nullptr
        );
    }
    if (err == cudaSuccess) {
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err == cudaSuccess) {
        err = cudaMemcpy(events.data(), d_events, event_bytes, cudaMemcpyDeviceToHost);
    }
    cudaError_t free_states = cudaFree(d_states);
    cudaError_t free_events = cudaFree(d_events);
    if (err == cudaSuccess && free_states != cudaSuccess) {
        err = free_states;
    }
    if (err == cudaSuccess && free_events != cudaSuccess) {
        err = free_events;
    }
    if (err != cudaSuccess) {
        return error_to_int(err);
    }
    uint64_t fired = 0u;
    for (size_t i = 0; i < events.size(); ++i) {
        h_intensity[i] = events[i].intensity;
        if ((events[i].flags & PRISMCODE_EVENT_FLAG_FIRED) != 0u) {
            fired += 1u;
        }
    }
    *out_fired = fired;
    return error_to_int(cudaSuccess);
}

const char* pc_cuda_status_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

}  // extern "C"

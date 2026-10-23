#include "prismcode_cuda.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

void prismcode_cuda_init_runtime_status(PrismcodeCudaRuntimeStatus* status) {
    std::memset(status, 0, sizeof(*status));
    status->abi_version = PRISMCODE_CUDA_ABI_V0_1;
    status->isa_version = PRISMCODE_CUDA_ISA_V0_1;
    status->link_sentinel = PRISMCODE_CUDA_LINK_SENTINEL;
    status->runtime_state = PRISMCODE_CUDA_RUNTIME_UNKNOWN;
    status->runtime_version = -1;
    status->driver_version = -1;
    status->device_count = 0;
    status->active_device = -1;
    status->compute_major = -1;
    status->compute_minor = -1;
}

void prismcode_cuda_init_selftest(PrismcodeCudaSelfTestResult* result) {
    std::memset(result, 0, sizeof(*result));
    result->status = PRISMCODE_CUDA_SELFTEST_UNKNOWN;
    result->element_count = PRISMCODE_CUDA_M1_SELFTEST_ELEMENTS;
}

void prismcode_cuda_copy_device_name(char* out, const char* name) {
    if (out == nullptr) {
        return;
    }
    if (name == nullptr) {
        out[0] = '\0';
        return;
    }
    std::strncpy(out, name, PRISMCODE_CUDA_DEVICE_NAME_BYTES - 1u);
    out[PRISMCODE_CUDA_DEVICE_NAME_BYTES - 1u] = '\0';
}

uint64_t prismcode_cuda_mix_hash(uint64_t hash, uint32_t value) {
    hash ^= static_cast<uint64_t>(value) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    hash ^= hash >> 30;
    hash *= 0xbf58476d1ce4e5b9ULL;
    hash ^= hash >> 27;
    hash *= 0x94d049bb133111ebULL;
    hash ^= hash >> 31;
    return hash;
}

uint64_t prismcode_cuda_hash_f32_vector(const std::vector<float>& values) {
    uint64_t hash = 0x5043494d31544553ULL;
    for (float value : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash = prismcode_cuda_mix_hash(hash, bits);
    }
    return hash;
}

cudaError_t prismcode_cuda_record_selftest_error(
    PrismcodeCudaSelfTestResult* result,
    cudaError_t err
) {
    result->status = PRISMCODE_CUDA_SELFTEST_FAIL;
    result->error_code = static_cast<uint32_t>(err);
    return err;
}

}  // namespace

extern "C" {

uint32_t prismcode_cuda_abi_version(void) {
    return PRISMCODE_CUDA_ABI_V0_1;
}

uint32_t prismcode_cuda_isa_version(void) {
    return PRISMCODE_CUDA_ISA_V0_1;
}

uint32_t prismcode_cuda_link_probe(void) {
    return PRISMCODE_CUDA_LINK_SENTINEL;
}

cudaError_t prismcode_cuda_runtime_status(PrismcodeCudaRuntimeStatus* out_status) {
    if (out_status == nullptr) {
        return cudaErrorInvalidValue;
    }

    prismcode_cuda_init_runtime_status(out_status);

    cudaError_t err = cudaRuntimeGetVersion(&out_status->runtime_version);
    if (err != cudaSuccess) {
        out_status->runtime_state = PRISMCODE_CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR;
        out_status->error_code = static_cast<uint32_t>(err);
        return err;
    }

    err = cudaDriverGetVersion(&out_status->driver_version);
    if (err != cudaSuccess) {
        out_status->runtime_state = PRISMCODE_CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR;
        out_status->error_code = static_cast<uint32_t>(err);
        return err;
    }

    err = cudaGetDeviceCount(&out_status->device_count);
    if (err != cudaSuccess) {
        out_status->runtime_state = err == cudaErrorNoDevice
            ? PRISMCODE_CUDA_RUNTIME_NO_DEVICE
            : PRISMCODE_CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR;
        out_status->error_code = static_cast<uint32_t>(err);
        return err;
    }
    if (out_status->device_count <= 0) {
        out_status->runtime_state = PRISMCODE_CUDA_RUNTIME_NO_DEVICE;
        out_status->error_code = static_cast<uint32_t>(cudaErrorNoDevice);
        return cudaErrorNoDevice;
    }

    err = cudaGetDevice(&out_status->active_device);
    if (err != cudaSuccess) {
        out_status->runtime_state = PRISMCODE_CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR;
        out_status->error_code = static_cast<uint32_t>(err);
        return err;
    }

    err = cudaDeviceGetAttribute(
        &out_status->compute_major,
        cudaDevAttrComputeCapabilityMajor,
        out_status->active_device
    );
    if (err != cudaSuccess) {
        out_status->runtime_state = PRISMCODE_CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR;
        out_status->error_code = static_cast<uint32_t>(err);
        return err;
    }
    err = cudaDeviceGetAttribute(
        &out_status->compute_minor,
        cudaDevAttrComputeCapabilityMinor,
        out_status->active_device
    );
    if (err != cudaSuccess) {
        out_status->runtime_state = PRISMCODE_CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR;
        out_status->error_code = static_cast<uint32_t>(err);
        return err;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        out_status->runtime_state = PRISMCODE_CUDA_RUNTIME_DRIVER_OR_RUNTIME_ERROR;
        out_status->error_code = static_cast<uint32_t>(err);
        return err;
    }

    out_status->total_global_mem = static_cast<uint64_t>(total_bytes);
    prismcode_cuda_copy_device_name(out_status->device_name, "CUDA_DEVICE_ACTIVE");
    out_status->runtime_state = PRISMCODE_CUDA_RUNTIME_LINKED_READY;
    out_status->error_code = static_cast<uint32_t>(cudaSuccess);
    return cudaSuccess;
}

cudaError_t prismcode_cuda_m1_selftest(
    PrismcodeCudaSelfTestResult* out_result,
    cudaStream_t stream
) {
    if (out_result == nullptr) {
        return cudaErrorInvalidValue;
    }
    prismcode_cuda_init_selftest(out_result);

    PrismcodeCudaRuntimeStatus runtime_status;
    cudaError_t err = prismcode_cuda_runtime_status(&runtime_status);
    if (err != cudaSuccess) {
        out_result->status = PRISMCODE_CUDA_SELFTEST_SKIPPED;
        out_result->error_code = static_cast<uint32_t>(err);
        return err;
    }

    const uint64_t count = PRISMCODE_CUDA_M1_SELFTEST_ELEMENTS;
    std::vector<float> host_in(static_cast<size_t>(count));
    std::vector<float> host_out(static_cast<size_t>(count), 0.0f);
    for (uint64_t i = 0; i < count; ++i) {
        host_in[static_cast<size_t>(i)] =
            (static_cast<float>(i) * 17.0f - 33.0f) / 7.0f;
    }

    float* d_lhs = nullptr;
    float* d_out = nullptr;
    const size_t byte_count = static_cast<size_t>(count) * sizeof(float);

    err = cudaMalloc(reinterpret_cast<void**>(&d_lhs), byte_count);
    if (err != cudaSuccess) {
        return prismcode_cuda_record_selftest_error(out_result, err);
    }
    err = cudaMalloc(reinterpret_cast<void**>(&d_out), byte_count);
    if (err != cudaSuccess) {
        cudaFree(d_lhs);
        return prismcode_cuda_record_selftest_error(out_result, err);
    }

    err = cudaMemcpyAsync(d_lhs, host_in.data(), byte_count, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        PrismcodeTensorOpParams params;
        std::memset(&params, 0, sizeof(params));
        params.element_count = count;
        params.op = PRISMCODE_TENSOR_OP_COPY;
        err = prismcode_cuda_tensor_ops(d_lhs, nullptr, d_out, params, stream);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(host_out.data(), d_out, byte_count, cudaMemcpyDeviceToHost, stream);
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }

    cudaError_t free_lhs = cudaFree(d_lhs);
    cudaError_t free_out = cudaFree(d_out);
    if (err == cudaSuccess && free_lhs != cudaSuccess) {
        err = free_lhs;
    }
    if (err == cudaSuccess && free_out != cudaSuccess) {
        err = free_out;
    }
    if (err != cudaSuccess) {
        return prismcode_cuda_record_selftest_error(out_result, err);
    }

    uint64_t mismatches = 0;
    float max_abs_error = 0.0f;
    for (uint64_t i = 0; i < count; ++i) {
        const float expected = host_in[static_cast<size_t>(i)];
        const float actual = host_out[static_cast<size_t>(i)];
        const float abs_error = std::fabs(actual - expected);
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }
        if (actual != expected) {
            mismatches += 1u;
        }
    }

    out_result->mismatch_count = mismatches;
    out_result->max_abs_error = max_abs_error;
    out_result->sentinel_hash = prismcode_cuda_hash_f32_vector(host_out);
    out_result->tensor_copy_parity = mismatches == 0u ? 1u : 0u;
    out_result->status = mismatches == 0u
        ? PRISMCODE_CUDA_SELFTEST_PASS
        : PRISMCODE_CUDA_SELFTEST_FAIL;
    out_result->error_code = static_cast<uint32_t>(
        mismatches == 0u ? cudaSuccess : cudaErrorUnknown
    );
    return mismatches == 0u ? cudaSuccess : cudaErrorUnknown;
}

}  // extern "C"

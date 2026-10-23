#include "prismcode_cuda_device.cuh"

#include <math.h>

namespace prismcode_cuda_kernels {

__global__ void prismcode_thermo_zero_kernel(PrismcodeThermoSummary* summary) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    summary->energy_sum = 0.0;
    summary->entropy_sum = 0.0;
    summary->temperature_sum = 0.0;
    summary->weighted_energy_sum = 0.0;
    summary->sample_count = 0ULL;
    summary->invalid_count = 0ULL;
    summary->max_force_q24 = 0u;
    summary->flags = PRISMCODE_EVENT_FLAG_VALID;
}

__global__ void prismcode_thermo_reduce_kernel(
    const PrismcodeThermoSample* __restrict__ samples,
    PrismcodeThermoSummary* __restrict__ summary,
    uint64_t sample_count
) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    uint64_t local_count = 0ULL;
    uint64_t local_invalid = 0ULL;
    double energy_sum = 0.0;
    double entropy_sum = 0.0;
    double temperature_sum = 0.0;
    double weighted_energy_sum = 0.0;
    uint32_t max_force_q24 = 0u;

    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < sample_count;
         idx += stride) {
        const PrismcodeThermoSample sample = samples[idx];
        const bool valid =
            isfinite(sample.energy) &&
            isfinite(sample.entropy) &&
            isfinite(sample.temperature) &&
            isfinite(sample.force_norm) &&
            isfinite(sample.weight);

        if (!valid) {
            local_invalid += 1ULL;
            continue;
        }

        const double weight = static_cast<double>(sample.weight);
        energy_sum += sample.energy;
        entropy_sum += sample.entropy;
        temperature_sum += sample.temperature;
        weighted_energy_sum += sample.energy * weight;
        local_count += 1ULL;

        const float abs_force = fabsf(sample.force_norm);
        const float scaled = abs_force * 16777216.0f;
        const uint32_t q24 = scaled >= 4294967295.0f
            ? 0xffffffffu
            : static_cast<uint32_t>(scaled);
        if (q24 > max_force_q24) {
            max_force_q24 = q24;
        }
    }

    if (local_count != 0ULL) {
        atomicAdd(&summary->energy_sum, energy_sum);
        atomicAdd(&summary->entropy_sum, entropy_sum);
        atomicAdd(&summary->temperature_sum, temperature_sum);
        atomicAdd(&summary->weighted_energy_sum, weighted_energy_sum);
        atomicAdd(reinterpret_cast<unsigned long long*>(&summary->sample_count), local_count);
    }
    if (local_invalid != 0ULL) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&summary->invalid_count), local_invalid);
    }
    if (max_force_q24 != 0u) {
        atomicMax(&summary->max_force_q24, max_force_q24);
    }
}

}  // namespace prismcode_cuda_kernels

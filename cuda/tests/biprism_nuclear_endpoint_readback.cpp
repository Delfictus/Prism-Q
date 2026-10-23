// Read-only replay of a retained C4 candidate in a source-bound supported scratch ABI.
// This verifies the full-step position expansion, not trajectory acceptance.
// The caller must bind the input to its native dispatch and selected outer dt.
// Build: g++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off \
//   -Icrates/prism-gpu/src/kernels cuda/tests/biprism_nuclear_endpoint_readback.cpp -o READER
// Run: READER retained_epoch.numerical.bin selected_outer_dt_au
#include "nhs_biprism_coordinate_precision.cuh"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

template<class T> T load(const std::vector<char>& bytes, size_t offset) {
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    char* tail = nullptr;
    errno = 0;
    const float dt = std::strtof(argv[2], &tail);
    if (errno || tail == argv[2] || *tail || !bp_coord_finite(double(dt)) || !(dt > 0.f)) return 2;
    const float h = dt * .25f;
    if (!(h > 0.f)) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 3;
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    constexpr size_t leaves = 8, core = 192;
    const size_t stride = bytes.size()/leaves;
    if (bytes.size() != leaves * stride || (stride != 97536 && stride != 106752 && stride != 143872)) return 3;
    unsigned checked = 0, mismatches = 0, half_matches = 0, invalid = 0;
    for (unsigned leaf = 0; leaf < leaves; ++leaf) {
        for (unsigned vertex = 0; vertex < core; ++vertex) {
            for (unsigned axis = 0; axis < 3; ++axis) {
                const size_t base = leaf * stride, component = 24 * vertex + 8 * axis;
                const auto start = load<BiprismCoordinatePairF32>(bytes, base + 72960 + component);
                const auto endpoint = load<BiprismCoordinatePairF32>(bytes, base + 77568 + component);
                const double velocity = load<double>(bytes, base + 61440 + 8 * (core * axis + vertex));
                invalid += !bp_coord_finite(start) || !bp_coord_finite(endpoint) || !bp_coord_finite(velocity);
                const auto expected = bp_coord_fma(start, double(h), velocity);
                const auto wrong_half = bp_coord_fma(start, .5 * double(h), velocity);
                ++checked;
                mismatches += std::memcmp(&endpoint, &expected, sizeof(endpoint)) != 0;
                half_matches += std::memcmp(&endpoint, &wrong_half, sizeof(endpoint)) == 0;
            }
        }
    }
    std::printf("checked_core_axes=%u full_h_bit_mismatches=%u half_h_matches=%u invalid=%u h=%a "
                "scope=retained_native_C4_position_CPU_replay trajectory_accepted=false\n",
                checked, mismatches, half_matches, invalid, double(h));
    // Require a nonzero-displacement control, so a zeroed scratch buffer cannot
    // qualify the distinction between midpoint and endpoint.
    return mismatches || invalid || half_matches == checked ? 4 : 0;
}

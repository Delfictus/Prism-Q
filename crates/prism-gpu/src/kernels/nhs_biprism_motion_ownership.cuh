#pragma once
#include <stdint.h>

// Per-atom ownership for explicitly selected Biprism/AMBER phases only.
// 0: AMBER motion and constraints; 1: external motion, AMBER constraints;
// 3: external motion and excluded AMBER constraints (FlexibleCore).
// Unknown bits and constraint-only exclusion (2) are invalid.
enum PrismBiprismMotionOwnershipV1 : uint8_t {
    PRISM_BIPRISM_EXTERNAL_MOTION_V1 = 1u,
    PRISM_BIPRISM_SKIP_AMBER_CONSTRAINTS_V1 = 2u
};
#ifdef __CUDACC__
#define PRISM_BIPRISM_OWNERSHIP_HD __host__ __device__ __forceinline__
#else
#define PRISM_BIPRISM_OWNERSHIP_HD inline
#endif

PRISM_BIPRISM_OWNERSHIP_HD bool prism_biprism_ownership_valid_v1(uint8_t flags) {
    return flags == 0u || flags == PRISM_BIPRISM_EXTERNAL_MOTION_V1 ||
           flags == (PRISM_BIPRISM_EXTERNAL_MOTION_V1 | PRISM_BIPRISM_SKIP_AMBER_CONSTRAINTS_V1);
}
PRISM_BIPRISM_OWNERSHIP_HD bool prism_biprism_external_motion_v1(uint8_t flags) {
    return (flags & PRISM_BIPRISM_EXTERNAL_MOTION_V1) != 0u;
}

// Return 0 to project, 1 for explicitly excluded whole components, -1 for
// malformed membership or mixed exclusion. No solver or coverage stamp is
// performed here. Every overlapping cluster must pass this same check so
// FlexibleCore exclusion is closed over the full constraint component.
// The caller owns an immutable n_atoms-byte mask and valid cluster storage.
template<class Cluster>
PRISM_BIPRISM_OWNERSHIP_HD int prism_biprism_constraint_action_v1(
    const uint8_t* ownership, int n_atoms, const Cluster& cluster) {
    if (!ownership || n_atoms < 0 || cluster.central_atom < 0 ||
        cluster.central_atom >= n_atoms || cluster.n_hydrogens < 0 ||
        cluster.n_hydrogens > 3) return -1;
    const uint8_t central = ownership[cluster.central_atom];
    if (!prism_biprism_ownership_valid_v1(central)) return -1;
    const uint8_t excluded = central & PRISM_BIPRISM_SKIP_AMBER_CONSTRAINTS_V1;
    for (int h = 0; h < cluster.n_hydrogens; ++h) {
        const int atom = cluster.hydrogen_atoms[h];
        if (atom < 0) continue; // Matches the native unused-hydrogen sentinel.
        if (atom >= n_atoms) return -1;
        const uint8_t flags = ownership[atom];
        if (!prism_biprism_ownership_valid_v1(flags) ||
            (flags & PRISM_BIPRISM_SKIP_AMBER_CONSTRAINTS_V1) != excluded) return -1;
    }
    return excluded ? 1 : 0;
}
#undef PRISM_BIPRISM_OWNERSHIP_HD

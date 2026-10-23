#pragma once

// Operator-selected raw-population coupling. This proposal contains the exact
// action algebra and exchange ABI; it does not select a native nonlinear solver.
// omega is frozen during a coupled sub-step. Changing omega has separate work.
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BIPRISM_FOLIATED_HD __host__ __device__ __forceinline__
#else
#define BIPRISM_FOLIATED_HD inline
#endif

constexpr std::uint32_t BIPRISM_FOLIATED_COUPLING_ABI = 0x46435032u; // FCP2
constexpr std::uint32_t BIPRISM_FOLIATED_LEAVES = 8u;
constexpr std::uint32_t BIPRISM_FOLIATED_CORE = 192u;
constexpr std::uint32_t BIPRISM_FOLIATED_EDGES = 1536u;

// Values describe one actual candidate, not independent per-leaf commit votes.
struct FoliatedCouplingFooterFfi {
    double base_energy_delta;                  // +0
    double owned_pair_energy_delta;            // +8, leaf owns bond (leaf,leaf+1)
    double attributed_coupling_energy_delta;   // +16, half each adjacent bond
    double coupling_discrete_work;             // +24, sum h_numerator * delta_s
    double physical_transfer_in;               // +32, antisymmetric bond transfer
    double balanced_local_defect;              // +40, base_delta + discrete_work
    double global_energy_delta;                // +48, all8 base+ownedpair delta
    double global_chain_defect;                // +56, all8 pairdelta-work
    float max_coordinate_residual;             // +64
    float max_population_residual;             // +68
    float max_electronic_equation_residual;     // +72
    float max_norm_defect;                     // +76
    std::uint32_t abi_magic;                   // +80
    std::uint32_t leaf_id;                     // +84
    std::uint32_t accepted_substeps_at_entry;   // +88, actual convergence counter
    std::uint32_t substep;                     // +92
    std::uint32_t iteration;                   // +96
    std::uint32_t local_fault_mask;            // +100
    std::uint32_t convergence_vote;            // +104, input to collective decision
    std::uint32_t publication_phase;           // +108, provenance, not a barrier
    std::uint32_t reserved[4];                 // +112..127, zero
};

// Dedicated per-leaf global replay scratch. Existing physical scientific arrays
// are not borrowed. FP64 populations are evaluated from the authoritative FP64 spinor owner.
struct alignas(128) FoliatedCouplingLeafFfi {
    double start_population[BIPRISM_FOLIATED_CORE];       // 0..1535
    double trial_population[BIPRISM_FOLIATED_CORE];       // 1536..3071
    float energy_weight[BIPRISM_FOLIATED_CORE];           // 3072..3839
    double reciprocal_response[BIPRISM_FOLIATED_CORE];    // 3840..5375
    double live_weights[BIPRISM_FOLIATED_EDGES];          // 5376..17663
    FoliatedCouplingFooterFfi footer;                    // 17664..17791
};

static_assert(sizeof(FoliatedCouplingFooterFfi) == 128, "FCP2 footer ABI");
static_assert(offsetof(FoliatedCouplingFooterFfi, max_coordinate_residual) == 64, "FCP2 residual offset");
static_assert(offsetof(FoliatedCouplingFooterFfi, abi_magic) == 80, "FCP2 identity offset");
static_assert(offsetof(FoliatedCouplingFooterFfi, reserved) == 112, "FCP2 reserved offset");
static_assert(alignof(FoliatedCouplingLeafFfi) == 128, "FCP2 alignment");
static_assert(sizeof(FoliatedCouplingLeafFfi) == 17792, "FCP2 allocation size");
static_assert(offsetof(FoliatedCouplingLeafFfi, trial_population) == 1536, "FCP2 trial offset");
static_assert(offsetof(FoliatedCouplingLeafFfi, energy_weight) == 3072, "FCP2 weight offset");
static_assert(offsetof(FoliatedCouplingLeafFfi, reciprocal_response) == 3840, "FCP2 response offset");
static_assert(offsetof(FoliatedCouplingLeafFfi, live_weights) == 5376, "FCP2 DEC weight offset");
static_assert(offsetof(FoliatedCouplingLeafFfi, footer) == 17664, "FCP2 footer offset");

namespace biprism_foliated {

BIPRISM_FOLIATED_HD double raw_population_f32(
    float psi0_re, float psi0_im, float psi1_re, float psi1_im
) {
    const double a = static_cast<double>(psi0_re);
    const double b = static_cast<double>(psi0_im);
    const double c = static_cast<double>(psi1_re);
    const double d = static_cast<double>(psi1_im);
    return (a * a + b * b) - (c * c + d * d);
}

BIPRISM_FOLIATED_HD double raw_norm_f32(
    float psi0_re, float psi0_im, float psi1_re, float psi1_im
) {
    const double a = static_cast<double>(psi0_re);
    const double b = static_cast<double>(psi0_im);
    const double c = static_cast<double>(psi1_re);
    const double d = static_cast<double>(psi1_im);
    return (a * a + b * b) + (c * c + d * d);
}

BIPRISM_FOLIATED_HD double pair_weight(double kappa, double omega_i, double omega_j) {
    return 0.5 * kappa * (omega_i + omega_j);
}

// This is the HAMILTONIAN NUMERATOR. Divide by the actual Berry mass to
// obtain the sigma_z coefficient in the evolution generator; do not cancel a
// floored Berry mass against omega. sbar=(s_start+s_trial)/2, not s(psi_bar).
BIPRISM_FOLIATED_HD double endpoint_numerator(
    double kappa, double omega_prev, double omega_i, double omega_next,
    double s0_prev, double s1_prev, double s0_next, double s1_next
) {
    const double j_prev = pair_weight(kappa, omega_prev, omega_i);
    const double j_next = pair_weight(kappa, omega_i, omega_next);
    return -0.5 * (j_prev * (s0_prev + s1_prev) + j_next * (s0_next + s1_next));
}

BIPRISM_FOLIATED_HD double evolution_generator_z(double numerator, double berry_mass) {
    // Caller must collectively refuse nonfinite inputs or berry_mass <= 0.
    return numerator / berry_mass;
}

struct EndpointLedger {
    double numerator;
    double owned_energy_start;
    double owned_energy_end;
    double owned_energy_delta;
    double attributed_energy_delta;
    double discrete_work;
    double transfer_in;
    double local_chain_defect;
};

BIPRISM_FOLIATED_HD EndpointLedger endpoint_ledger(
    double kappa, double omega_prev, double omega_i, double omega_next,
    double s0_prev, double s1_prev, double s0_i, double s1_i,
    double s0_next, double s1_next
) {
    const double j_prev = pair_weight(kappa, omega_prev, omega_i);
    const double j_next = pair_weight(kappa, omega_i, omega_next);
    const double sp = 0.5 * (s0_prev + s1_prev);
    const double si = 0.5 * (s0_i + s1_i);
    const double sn = 0.5 * (s0_next + s1_next);
    const double dp = s1_prev - s0_prev;
    const double di = s1_i - s0_i;
    const double dn = s1_next - s0_next;
    const double numerator = -(j_prev * sp + j_next * sn);
    const double owned_delta = -j_next * (sn * di + si * dn);
    const double attributed_delta = -0.5 * (
        j_prev * (sp * di + si * dp) + j_next * (sn * di + si * dn));
    const double work = numerator * di;
    const double transfer = 0.5 * (
        j_prev * (sp * di - si * dp) + j_next * (sn * di - si * dn));
    return {
        numerator,
        kappa * omega_i - j_next * s0_i * s0_next,
        kappa * omega_i - j_next * s1_i * s1_next,
        owned_delta,
        attributed_delta,
        work,
        transfer,
        attributed_delta - transfer - work
    };
}

} // namespace biprism_foliated

#undef BIPRISM_FOLIATED_HD

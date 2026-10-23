#pragma once
#include <stddef.h>
#include <stdint.h>

// Independent foliated version 1 ABI. Never cast the active legacy adjudicator
// to this type: its pointer-bearing layout is different despite equal size.
namespace prism_biprism_v1 {

static constexpr uint32_t FOLIATED_ADJUDICATOR_ABI_VERSION = 1;
static constexpr size_t FOLIATED_LEAF_COUNT = 8;

struct alignas(128) InterferometricAdjudicatorFfi {
    uint32_t leaf_id;
    uint32_t step_tau;
    uint32_t active_tile_id;
    uint32_t pachner_flip_count;
    float u_tau_re[4];
    float u_tau_im[4];
    float v_lambda_re[4];
    float v_lambda_im[4];
    float state_populations[4];
    float energy_eigenvalues[4];
    float gauge_clamp_omega[4];
    float w_xx_trace[4];
    float w_x_lambda_trace[4];
    float w_x_tau_trace[4];
    float w_lambda_tau_trace[4];
    float bianchi_residual_l2;
    // Version 1: ||U^H S U - S||_F; S is separate source-bound boundary evidence.
    float unitarity_defect_frob;
    float pulay_force_norm;
    uint32_t consensus_branch_flag;
    uint32_t _pad_line1[12];
};

static_assert(sizeof(float) == 4, "f32 ABI requires four-byte float");
static_assert(sizeof(InterferometricAdjudicatorFfi) == 256, "foliated ABI size");
static_assert(alignof(InterferometricAdjudicatorFfi) == 128, "foliated ABI alignment");
static_assert(sizeof(InterferometricAdjudicatorFfi[8]) == 2048, "foliated ring size");
static_assert(offsetof(InterferometricAdjudicatorFfi, leaf_id) == 0, "leaf_id offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, step_tau) == 4, "step_tau offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, active_tile_id) == 8, "active_tile_id offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, pachner_flip_count) == 12, "pachner_flip_count offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, u_tau_re) == 16, "u_tau_re offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, u_tau_im) == 32, "u_tau_im offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, v_lambda_re) == 48, "v_lambda_re offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, v_lambda_im) == 64, "v_lambda_im offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, state_populations) == 80, "state_populations offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, energy_eigenvalues) == 96, "energy_eigenvalues offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, gauge_clamp_omega) == 112, "gauge_clamp_omega offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, w_xx_trace) == 128, "w_xx_trace offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, w_x_lambda_trace) == 144, "w_x_lambda_trace offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, w_x_tau_trace) == 160, "w_x_tau_trace offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, w_lambda_tau_trace) == 176, "w_lambda_tau_trace offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, bianchi_residual_l2) == 192, "bianchi_residual_l2 offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, unitarity_defect_frob) == 196, "unitarity_defect_frob offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, pulay_force_norm) == 200, "pulay_force_norm offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, consensus_branch_flag) == 204, "consensus_branch_flag offset");
static_assert(offsetof(InterferometricAdjudicatorFfi, _pad_line1) == 208, "reserved offset");

}  // namespace prism_biprism_v1

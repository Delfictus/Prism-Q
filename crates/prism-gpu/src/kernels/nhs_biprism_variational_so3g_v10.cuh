// Coupled v10 implementation. Native acceptance requires the actual runtime gates.
// 192 core +64 fixed halos; 10 kernel args; four accepted substeps;31,616B SMEM.
// FCP2 replay scratch provides all8 population/weight/collective exchange through
// owned global banks in either cluster mode. Native cluster barriers remain; the
// accepted gauge-phase exchange retains actual DSMEM loads with an L2 fallback.
// Integrated coupled source uses the original qualified electronic helpers. Resource
// use and integrated native convergence require new measurements; earlier epochs
// compiled but failed the stack/spill gate. No old resource claim carries over.
// The operator's energy/Picard/norm gates are unchanged. Pure midpoint mode does
// not acquire an exact-H claim. Phase2 refresh work is owned outside this body.
#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include "nhs_biprism_coordinate_precision.cuh"
#include "nhs_biprism_potential_precision.cuh"
#include "nhs_biprism_spinor_precision.cuh"
#include "nhs_biprism_coordinate_geometry.cuh"
#include "nhs_biprism_ordered_edges.cuh"
#include "nhs_biprism_convergence_abi.cuh"
#include "nhs_biprism_electronic_discrete.cuh"
#include "nhs_biprism_tangent_branch.cuh"
#include "nhs_biprism_newton_trust.cuh"
#include "nhs_biprism_nuclear_cayley_f64.cuh"
#include "nhs_biprism_magnetic_field_f64.cuh"
#include "nhs_biprism_foliated_population.cuh"
#ifndef BIPRISM_SHARED_STEP_ABI_ONLY
#define BIPRISM_SHARED_STEP_ABI_ONLY
#define BIPRISM_V10_RESTORE_SHARED_STEP_KERNELS
#endif
#include "nhs_biprism_shared_step.cuh"
#ifdef BIPRISM_V10_RESTORE_SHARED_STEP_KERNELS
#undef BIPRISM_V10_RESTORE_SHARED_STEP_KERNELS
#undef BIPRISM_SHARED_STEP_ABI_ONLY
#endif

#if __has_include("nhs_biprism_simplicial_tile.cuh")
#include "nhs_biprism_simplicial_tile.cuh"
#else
#error "nhs_biprism_simplicial_tile.cuh must be in the same directory or include path"
#endif

#if __has_include("nhs_biprism_xtb_lut.cuh")
#include "nhs_biprism_xtb_lut.cuh"
#else
#error "nhs_biprism_xtb_lut.cuh must be in the same directory or include path"
#endif

// The real adjudicator ABI header (layout pinned by its own static_asserts).
// The former "../../../../../cuda/adjudicator.cuh" path resolved to no file in
// either layout, so this header silently compiled an opaque 256-byte stand-in.
// A missing header is now a hard error; there is no fallback definition.
#if __has_include("../../../prism-nhs/src/cuda/adjudicator.cuh")
#include "../../../prism-nhs/src/cuda/adjudicator.cuh"
#else
#error "prism-nhs/src/cuda/adjudicator.cuh not found relative to prism-gpu/src/kernels (InterferometricAdjudicatorFfi ABI header is required)"
#endif
static_assert(sizeof(InterferometricAdjudicatorFfi) == 256, "InterferometricAdjudicatorFfi must stay 256 B");

#define BIPRISM_INNER_SUBSTEPS 4
#define MAX_VARIATIONAL_PICARD_ITERS 48
#include "nhs_biprism_nuclear_budget.cuh"
#define MAX_CONTINUOUS_ELECTRONIC_ITERS 24
#define VARIATIONAL_PICARD_TOL 1e-7f
#define VARIATIONAL_ENERGY_TOL 1e-6f
#define DEC_PCG_MAX_ITERS BIPRISM_DEC_MAX_ITERS
#define DEC_PCG_TOL_SQ 1e-12f
#define MAX_BARRIER_SPIN_POLLS 100000u

#define FAULT_DEC_NONCONV      1u
#define FAULT_PICARD_NONCONV   2u
#define FAULT_METRIC_NON_SPD   4u
#define FAULT_BARRIER_TIMEOUT  8u
#define ADJ_SOLVER_FAULT_FLAG  0x80000000u
#define BIPRISM_TAIL_ABI_MAGIC 0x42505631u // "BPV1"

#define FP_Q39_SCALE 549755813888.0f
#define FP_Q39_INV   1.8189894035458565e-12f

struct __align__(8) sym3_f32 {
    float xx, yy, zz, xy, xz, yz;
};
struct sym3_f64 { double xx, yy, zz, xy, xz, yz; };

// Typed SoA storage shortens numerical live ranges. It does not change precision
// or confer a performance claim: these explicit global transfers are measured.
// Bytes[0,30976) are disposable replay scratch. The peer map at30976 is immutable
// preparation metadata, uploaded by the owner after complete edge qualification.
enum BiprismV10ScalarSlot : uint32_t {
    BP_NS_DT=0, BP_NS_DT_METRIC, BP_NS_WORK_CONS, BP_NS_WORK_BERRY,
    BP_NS_PROJECTION_DOT, BP_NS_BATH_DELTA, BP_NS_DX_RESIDUAL,
    BP_NS_RECON_RAW, BP_NS_RECON_POSITION, BP_NS_ROTATION_DEFECT,
    BP_NS_MID_WORK_DEFECT, BP_NS_COORD_WORK_DEFECT, BP_NS_ELECTRONIC_DELTA,
    BP_NS_EQUATION, BP_NS_POPULATION_RESIDUAL, BP_NS_GLOBAL_ENERGY,
    BP_NS_BASE_ENERGY, BP_NS_PAIR_DELTA, BP_NS_ATTRIBUTED_DELTA,
    BP_NS_TRANSFER, BP_NS_FOL_WORK, BP_NS_CHAIN, BP_NS_OUTER_ERROR,
    BP_NS_MAX_ENERGY, BP_NS_MAX_NORM, BP_NS_WXL, BP_NS_WXT,
    BP_NS_ATTEMPT_ENERGY, BP_NS_ATTEMPT_EQUATION, BP_NS_C1_START_ENERGY,
    BP_NS_C1_GLOBAL_ERROR, BP_NS_C1_SEARCH_BIAS,
    // C1 consumes these operands before C3. Slot29 retains ideal nuclear work;
    // slot31 holds coordinate-round work transiently in C3, then C4 restores
    // the used search bias so raw C1 epsilon = scalar30 - scalar31 is retained.
    BP_NS_MID_IDEAL_WORK_DEFECT=BP_NS_C1_START_ENERGY,
    BP_NS_COORD_ROUND_WORK=BP_NS_C1_SEARCH_BIAS
};
// Refusal-only interpretation of dead C1 replay slots. No accepted endpoint
// carries this tag: scalar29 is overwritten by its ordinary energy/work owner.
static constexpr uint32_t BP_V10_NORM_REFUSAL_TAG=0x4e524634u; // NRF3: triple norm bounds plus interval completion
static constexpr uint32_t BP_V10_REPRESENTATION_TRACE_TAG=0x52524633u; // RRF3: distinguishes equation-repair rounds
static constexpr uint32_t BP_V10_EQUATION_REPAIR_WINNER_BIT=0x80000000u;
static constexpr uint32_t BP_V10_REPRESENTATION_MAX_UPDATES=4u*8u*TILE_CORE_VERTICES;
static constexpr uint32_t BP_V10_EQUATION_MAX_REPAIRS=4u;
// Each charge-changing repair consumes one update and can reset at most four
// original s0 labels at its one selected vertex. No other visited mask resets.
static constexpr uint32_t BP_V10_REPRESENTATION_TOTAL_UPDATES=
    BP_V10_REPRESENTATION_MAX_UPDATES+5u*BP_V10_EQUATION_MAX_REPAIRS;
// Diagnostic-only, invocation-local history. During represented C1 the final
// 768 bytes of endpoint_v are dead: rank_proposal_s0 owns only its first1536.
// C2 regenerates the entire velocity/tau view after successful admission.
struct BiprismV10RepresentationTraceFfi {
    uint32_t magic,stage,count,round;
    // Scores are CTA-local; the predicted global score is their minimum over
    // all eight leaves (equivalently the score from the global winner's leaf).
    double evaluated_error[17],predicted_local_best[17];
    uint32_t winner[17],evaluation_round[17];
};
// Mutually exclusive C1 diagnostics in the existing endpoint union.
// Continuous history occupies the dead proposal prefix and is consumed only
// on continuous refusal. Represented C1 supersedes it with proposal scratch;
// RRF2 and a terminal EQR2 use the separate tail. No physical
// endpoint, potential, Hamiltonian, or scalar-ledger storage is overwritten.
static constexpr uint32_t BP_V10_EQUATION_REFUSAL_TAG=0x45515233u; // EQR2: raw inputs plus saved native equation
static constexpr uint32_t BP_V10_CONTINUOUS_REFUSAL_TAG=0x43465233u; // CFR2: raw inputs plus saved native change
static constexpr uint32_t BP_V10_CONTINUOUS_HISTORY_TAG=0x43485432u; // CHT2: all24 continuous iterations
struct BiprismV10C1DiagnosticIdentityFfi {
    uint32_t magic,version,leaf,vertex,substep,picard,sequence,iteration;
    uint32_t flags,packed_atom,accepted_at_entry,reserved;
};
struct BiprismV10EquationRefusalFfi {
    BiprismV10C1DiagnosticIdentityFfi identity;
    double start[4];
    union {double candidate[4];BiprismElectronicSpinor<double> candidate_spinor;};
    double continuous[4],h_energy[4];
    double connection_im,berry_mass,h,norm_outer,equation_candidate;
    double published_leaf_equation,phi_mean,charge_mean;
    uint64_t reserved[2];
};
struct BiprismV10ContinuousIterationFfi {
    double max_change,max_equation;
    float dec_residual;
    uint32_t dec_iterations;
};
struct BiprismV10ContinuousHistoryFfi {
    uint32_t magic,version,count,valid_mask;
    BiprismV10ContinuousIterationFfi iteration[MAX_CONTINUOUS_ELECTRONIC_ITERS];
};
struct BiprismV10ContinuousRefusalFfi {
    BiprismV10C1DiagnosticIdentityFfi identity;
    double start[4],trial[4],next[4],h_energy[4];
    double connection_im,berry_mass,h,selected_change,equation_trial,equation_next_fixed_h;
    double published_leaf_change,published_leaf_equation,phi_mean;
    uint64_t reserved[3];
};
enum BiprismV10RepresentationTraceStage : uint32_t {
    BP_REPR_ENTER=1u,BP_REPR_NORM_REFUSAL=2u,BP_REPR_START_DEC_REFUSAL=3u,
    BP_REPR_ENDPOINT_DEC_REFUSAL=4u,BP_REPR_MEAN_DEC_REFUSAL=5u,
    BP_REPR_H_REFUSAL=6u,BP_REPR_EQUATION_WORK_REFUSAL=7u,
    BP_REPR_COLLECTIVE_REFUSAL=8u,BP_REPR_NO_IMPROVEMENT=9u,
    BP_REPR_ROUND_LIMIT=10u,BP_REPR_ACCEPTED=11u,BP_REPR_EVALUATED=12u,
    BP_REPR_EQUATION_REBUILD=13u
};
// Solver-private multisecant columns. These are finite differences of complete
// C1-C4 residual evaluations, not physical state or additional energy credits.
// Columns are orthonormal in (velocity, scaled-alpha) residual space. T stores
// the corresponding input displacement, so the measured secant is J*T=Q.
struct __align__(128) BiprismNuclearQrFfi {
    double dx[64][3][TILE_CORE_VERTICES],df[64][3][TILE_CORE_VERTICES];
    double alpha_dx[64],alpha_df[64];
    double work_dx[3][TILE_CORE_VERTICES],work_df[3][TILE_CORE_VERTICES],direction[3][TILE_CORE_VERTICES];
    double coefficient[64];
    double alpha_work_dx,alpha_work_df,alpha_direction,norm2,coefficient_now,alpha_scale,projected_norm2,actual_norm2;
    uint32_t count,have_previous,valid,iteration;
    // actual residual squared, unresolved secant-space residual squared,
    // new orthogonal secant norm squared, applied trust weight, columns, raw dx.
    double history[48][6];
    // Fixed-base finite-difference Krylov control. Previously alignment padding;
    // no pre-existing numerical/physical member moves or aliases this interval.
    double reference_dot,base_norm2,probe_weight,trial_weight,fd_radius,base_input_norm2,predicted_norm2,last_actual_ratio;
    uint32_t mode,linear_restarts,probes,accepted_newton,target_refresh;
    double trust_radius,initial_trial_weight;
    uint32_t backtracks,direction_checked;
    double model_slope,model_curvature;
    union {
        double trust_gram[4096];
        // Tangent construction returns before Gram assembly/solve. Capture
        // only at an evaluated C4 base; QR never writes this view. Ordinary
        // correction fully assembles its Gram before reading it again.
        double tangent_endpoint_raw_D[TILE_MAX_EDGES];
    };
    double trust_factor[4096],trust_y[64];
    double trust_norm2,regularization;
    uint32_t trust_status;
    double input_basis[64][3][TILE_CORE_VERTICES],input_alpha[64];
    // Previous successful measured inverse action is a right preconditioner
    // only. Current J actions are always freshly measured around one base.
    // The former64th PRECONDITIONER input column is now permanently owned
    // by continuation. It never aliases physical state or the64-column
    // current Jacobian. Every preconditioner writer/reader is bounded by63.
    double pre_dx[63][3][TILE_CORE_VERTICES];
    double continuation_previous_velocity[3][TILE_CORE_VERTICES];
    double pre_df[64][3][TILE_CORE_VERTICES];
    double pre_alpha_dx[63],continuation_previous_alpha;
    double pre_alpha_df[63],continuation_previous_stage;
    uint32_t pre_count;
    double direction_error_best;
    // Former8-byte tail padding; only last accepted PRIVATE-trial merit ratio.
    double accepted_trial_ratio;
};
static_assert(sizeof(BiprismNuclearQrFfi)==1560064,"Dedicated multisecant storage");
static_assert(offsetof(BiprismNuclearQrFfi,df)==294912,"Newton private df offset");
static_assert(offsetof(BiprismNuclearQrFfi,alpha_df)==590336,"Newton private alpha_df offset");
static_assert(offsetof(BiprismNuclearQrFfi,count)==605248,"Newton private count offset");
static_assert(offsetof(BiprismNuclearQrFfi,history)==605264,"Newton private history offset");
static_assert(offsetof(BiprismNuclearQrFfi,reference_dot)==607568,"Newton private reference_dot offset");
static_assert(offsetof(BiprismNuclearQrFfi,mode)==607632,"Newton private mode offset");
static_assert(offsetof(BiprismNuclearQrFfi,input_basis)==673768,"Newton private input_basis offset");
static_assert(offsetof(BiprismNuclearQrFfi,input_alpha)==968680,"Newton private input_alpha offset");
static_assert(offsetof(BiprismNuclearQrFfi,pre_dx)==969192,"Disjoint preconditioner input actions");
static_assert(offsetof(BiprismNuclearQrFfi,continuation_previous_velocity)==1259496,"Dedicated continuation velocity");
static_assert(offsetof(BiprismNuclearQrFfi,pre_df)==1264104,"Disjoint preconditioner output actions");
static_assert(offsetof(BiprismNuclearQrFfi,continuation_previous_alpha)==1559520,"Dedicated continuation alpha");
static_assert(offsetof(BiprismNuclearQrFfi,continuation_previous_stage)==1560032,"Dedicated continuation stage");
static_assert(offsetof(BiprismNuclearQrFfi,pre_count)==1560040,"Preconditioner validity extent");
static_assert(offsetof(BiprismNuclearQrFfi,direction_error_best)==1560048,"Independent action accuracy");
static_assert(offsetof(BiprismNuclearQrFfi,accepted_trial_ratio)==1560056,"Private merit ratio; unchanged extent");
struct __align__(128) BiprismNuclearEvaluationFfi {
    uint32_t magic,substep,count,nonlinear_trials;
    uint32_t derivative_evaluations,action_checks,target_replays,base_evaluations;
    uint32_t accepted_newton,rejected_newton;
    float original_h;
    uint32_t continuation_stage,next_stage,reserved[19];
    // Full C4 history (dx, energy, alpha), followed by the six Newton fields.
    // NaN C4 means this evaluation refused before C4 publication; NaN Newton
    // means the final physical gate accepted without another proposal.
    double rows[192][9];
};
static_assert(sizeof(BiprismNuclearEvaluationFfi)==13952,"Dedicated complete evaluation trace");
static_assert(offsetof(BiprismNuclearEvaluationFfi,rows)==128,"Versioned evaluation header");
static_assert(offsetof(BiprismNuclearEvaluationFfi,original_h)==40,"Immutable target timestep");
// A separate tail owner for pseudo-arclength initialization. No existing
// physical, operator, QR, preconditioner, or continuation byte is borrowed.
// h is one GLOBAL unknown, replicated on eight leaves. Its inner products
// carry weight1/8 so replication does not create eight independent timesteps.
struct __align__(128) BiprismNuclearArcFfi {
    double anchor_v[3][TILE_CORE_VERTICES],tangent_v[3][TILE_CORE_VERTICES];
    double anchor_alpha,tangent_alpha,anchor_h,tangent_h,step;
    double current_h,base_h,base_defect,work_dx,work_df,direction;
    double dx[64],df[64],input[64];
    double defect,norm,dot,scale,reference_row;
    uint32_t active,roots,last_root_trial,reserved;
    // Evaluated h, arc residual, requested arclength, active flag. The next
    // proposed h must never overwrite the retained evaluated candidate's h.
    double history[192][4];
    // Dedicated counters in the existing alignment tail. No preconditioner,
    // continuation vector, physical state, or trace row is overlaid.
    uint32_t slow_trial_streak,segment_accepted_trials;
    uint32_t tangent_mode; // 0=secant,1=new-root tangent,2=retry tangent; owns alignment padding.
    uint32_t tangent_probe_refinements; // Failed tangent actions only; independent of QR restarts.
};
static_assert(sizeof(BiprismNuclearArcFfi)==17152,"Dedicated arclength tail");
static_assert(offsetof(BiprismNuclearArcFfi,slow_trial_streak)==17040,"Arc progress counters own former padding");
static_assert(offsetof(BiprismNuclearArcFfi,tangent_mode)==17048,"Measured tangent control owns former padding");
static_assert(offsetof(BiprismNuclearArcFfi,tangent_probe_refinements)==17052,"Tangent probe lifetime owns former padding");
__device__ __forceinline__ void bp_v10_reset_tangent_probes(volatile BiprismNuclearQrFfi& q,volatile BiprismNuclearArcFfi& a) {
    q.linear_restarts=0u;a.tangent_probe_refinements=0u;
}
__device__ __forceinline__ void bp_v10_refine_tangent_probes(volatile BiprismNuclearQrFfi& q,volatile BiprismNuclearArcFfi& a) {
    // Native163's exact two-word coordinate control resolves its full C1-C4
    // action at paired1e-8 (relative error.009). Smaller1e-10 amplifies the
    // coordinate packing floor to.879; the unrounded CPU map missed this.
    // Keep the independent native action check; this is not an admission.
    // Input-direction breakdowns must not advance this action-failure state.
    if(a.tangent_probe_refinements==0u)q.fd_radius=1e-8;
    else if(a.tangent_probe_refinements==1u)q.fd_radius=1e-10;
    // Preserve successful167 probes. Native168's global1e-9 replacement
    // changed earlier continuation before its3072-evaluation refusal.
    // If1e-10,5e-11,2.5e-11 all fail their independent actual actions, test
    // the wider interval resolved by167's full-response control (.00736)
    // before continuing below the measured derivative noise floor.
    else if(a.tangent_probe_refinements==4u)q.fd_radius=1e-9;
    // Capture172: symmetric columns crossed different endpoint overlap
    // branches. Preserve earlier successful probes; only this retry uses
    // a consistent base-branch derivative, checked against physical actions.
    else if(a.tangent_probe_refinements==5u)q.fd_radius=1e-7;
    else q.fd_radius*=.5;
    ++a.tangent_probe_refinements;
}
struct __align__(128) BiprismV10NumericalScratchFfi {
    float x_start[3][TILE_CORE_VERTICES];
    float v_start[3][TILE_CORE_VERTICES];
    float p_start[3][TILE_CORE_VERTICES];
    // After Picard copies the old midpoint to shared x/v, represented C1
    // owns these exact typed aliases. C3 regenerates all midpoint components.
    union {
        float next_mid_x[3][TILE_CORE_VERTICES];
        double potential_e0[TILE_TOTAL_VERTICES];
        struct { double score[TILE_CORE_VERTICES]; uint32_t index[TILE_CORE_VERTICES]; } rank_choice;
        double edge_force_x[TILE_TOTAL_VERTICES];
        float2 edge_berry_x[TILE_TOTAL_VERTICES];
    };
    union {
        double potential_e1[TILE_CORE_VERTICES];
        float next_mid_v[3][TILE_CORE_VERTICES];double rank_e00[TILE_CORE_VERTICES];
        double edge_force_y[TILE_TOTAL_VERTICES];float2 edge_berry_y[TILE_TOTAL_VERTICES];
        struct {
            double coefficient_prefix[TILE_CORE_VERTICES];
            uint32_t visited_corners[TILE_CORE_VERTICES];
        } rank_search;
    };
    // This row retains an FP32 diagnostic tau mirror until C4 exports velocity.
    // C3 consumes exact tau in nuclear_rotation_output, then replaces that
    // same-lane owner with vp. Only matched final acceptance qualifies v.
    // C1's energy rank keeps only proposed s0 here; its bounded equation repair
    // may change s1 only with a complete charge/DEC/H rebuild. C2 then writes
    // all three tau components before any use.
    // C2's edge worker retains three completed original-helper components in
    // the x/y/z aliases, each bounded by256 lanes, after every C1 reader. The
    // force consumers finish before tau rewrites this region and C3 rewrites x/y.
    // Once the electronic components enter Q39, their storage holds the exact
    // FP64 Berry partial sum a then a+b, before the final (a+b)+d result.
    // Keep the prior double's8-byte lane stride: unequal edge loops permit no
    // barrier, and a lane must never overwrite another lane.
    union {
        float endpoint_v[3][TILE_CORE_VERTICES];
        BiprismElectronicComplex<float> rank_proposal_s0[TILE_CORE_VERTICES];
        double edge_force_z[TILE_TOTAL_VERTICES];float2 edge_berry_z[TILE_TOTAL_VERTICES];
        struct {
            BiprismElectronicComplex<float> proposal_prefix[TILE_CORE_VERTICES];
            BiprismV10RepresentationTraceFfi trace;
            BiprismV10EquationRefusalFfi equation_refusal;
        } rank_diagnostic;
        struct {
            BiprismV10ContinuousHistoryFfi history;
            uint8_t unused_to_refusal[sizeof(BiprismElectronicComplex<float>)*TILE_CORE_VERTICES-sizeof(BiprismV10ContinuousHistoryFfi)];
            BiprismV10ContinuousRefusalFfi refusal;
        } continuous_diagnostic;
    };
    double kinetic_start[TILE_CORE_VERTICES];
    // Continuous C1 uses these three rows plus reciprocal row0 for its exact
    // next spinor until collective votes finish. Represented admission follows:
    // exact promoted phi_endpoint/phi_mean/norm_outer.
    // C2 overwrites conservative rows after all C1 consumers.
    // After C3 consumes forces, row2 holds the original used_force dot dx;
    // rows0/1 hold Tmid0/1
    // until C4 computes metric-end work; there are no later force consumers.
    double conservative_force[3][TILE_CORE_VERTICES];
    // At represented-round entry rows0/1/2 hold work/equation/pair delta,
    // with foliation work in rank_e00, until all four reductions finish.
    // C1 local rank then uses rows0/1/2 for exact FP64 drho/dcross/work_neighbors. These are private
    // per-vertex replay operands, never the separately published crossleaf z.
    // C2 first computes/retains the original helper reciprocal force here;
    // after C3 last-force reads, rows0/1/2 hold
    // conservative work/reciprocal work/projection dot for immediate reduction.
    // Potential calls use row0[0..191] and row1[0..63] for all256 exact FP64
    // per-lane energy accumulators, after previous contents' last consumers.
    double reciprocal_force[3][TILE_CORE_VERTICES];
    float accepted_phase[TILE_MAX_EDGES];
    double scalar[32];
    uint16_t peer_edge_index[TILE_MAX_EDGES];
    // C1 initializes/consumes the connection plane. C2 then caches the exact
    // energy-only pairing/b before its incident-edge force gather.
    // C3 next retains the authoritative midpoint covariance, after every z
    // consumer. C4 rewrites all six metric planes. Only matched final acceptance qualifies
    // the endpoint_metric view; an early refusal leaves attempted scratch here.
    union {
        float endpoint_metric[6][TILE_CORE_VERTICES];
        float midpoint_covariance[6][TILE_CORE_VERTICES];
        double electronic_connection_im[TILE_CORE_VERTICES];
        // The live C1 connection occupies only the first1536 bytes. Rank owns
        // the remaining two planes until C2/C3 reuse this entire union.
        struct {
            double connection_prefix[TILE_CORE_VERTICES];
            double e01[TILE_CORE_VERTICES],delta_h[TILE_CORE_VERTICES];
        } rank_metric;
        double reciprocal_connection_z[TILE_CORE_VERTICES];
    };
    // Per-invocation PCG/energy/H/force potential. This is replay storage;
    // accepted physical values live in the descriptor-owned potential bank.
    // All previous numerical/peer-map/metric offsets remain unchanged.
    double working_phi[TILE_TOTAL_VERTICES];
    // Retired Aitken storage owns diagnostic history exclusively. No solver,
    // spinor, force, operator or endpoint reads this region. All48 iterations
    // survive refusal; each record is invalidated before its C1 evaluation.
    union {
        float nuclear_previous_residual[3][TILE_CORE_VERTICES];
        struct { uint32_t magic,substep,count,reserved; double rows[48][3]; } nuclear_history;
    };
    // Dedicated FP64 nuclear transit. None of these owners aliases electronic,
    // operator, physical endpoint, or Picard history storage. The final exported
    // velocity remains FP32 and its realized energy is checked separately in C4.
    double nuclear_start_momentum[3][TILE_CORE_VERTICES];
    double nuclear_rotation_input[3][TILE_CORE_VERTICES];
    // C2 tau -> same-lane C3 rotated velocity. Neither owner crosses C4.
    double nuclear_rotation_output[3][TILE_CORE_VERTICES];
    double nuclear_endpoint_momentum[3][TILE_CORE_VERTICES];
    double nuclear_midpoint_velocity[3][TILE_CORE_VERTICES];
    double nuclear_used_force[3][TILE_CORE_VERTICES];
    // Last fully evaluated numerical iterate, never a committed physical state.
    // Dedicated storage survives C1/C2/PCG scratch reuse during backtracking.
    union { float nuclear_base_velocity[3][TILE_CORE_VERTICES]; double potential_v01[TILE_TOTAL_VERTICES]; };
    // Actual represented coordinates used by every stage and the final commit.
    // These own full normalized pairs; legacy float offsets above are diagnostics.
    BiprismCoordinateExpansion3 nuclear_position_start[TILE_CORE_VERTICES];
    BiprismCoordinateExpansion3 nuclear_position_live[TILE_TOTAL_VERTICES];
    double nuclear_trial_velocity[3][TILE_CORE_VERTICES];
    double nuclear_base_velocity_f64[3][TILE_CORE_VERTICES];
    double nuclear_base_direction[3][TILE_CORE_VERTICES];
    double nuclear_metric[6][TILE_CORE_VERTICES];
    BiprismNuclearQrFfi nuclear_secant;
    BiprismNuclearEvaluationFfi nuclear_evaluations;
    BiprismNuclearArcFfi nuclear_arc;
    double nuclear_evaluation_extension[832][9];
    double nuclear_arc_history_extension[832][4];
    double nuclear_evaluation_tail[512][9];
    double nuclear_arc_history_tail[512][4];
    double nuclear_evaluation_completion[BIPRISM_MAX_NUCLEAR_EVALUATIONS-1536][9];
    double nuclear_arc_history_completion[BIPRISM_MAX_NUCLEAR_EVALUATIONS-1536][4];
};
// Preserve the complete historical prefix and append the bounded experiment's
// extra rows. These accessors never touch the63-column preconditioner owners.
__host__ __device__ __forceinline__ volatile double* bp_v10_evaluation_row(volatile BiprismV10NumericalScratchFfi* ns,uint32_t i) {
    return i<192u?ns->nuclear_evaluations.rows[i]:(i<1024u?ns->nuclear_evaluation_extension[i-192u]:(i<1536u?ns->nuclear_evaluation_tail[i-1024u]:ns->nuclear_evaluation_completion[i-1536u]));
}
__host__ __device__ __forceinline__ volatile double* bp_v10_arc_history_row(volatile BiprismV10NumericalScratchFfi* ns,uint32_t i) {
    return i<192u?ns->nuclear_arc.history[i]:(i<1024u?ns->nuclear_arc_history_extension[i-192u]:(i<1536u?ns->nuclear_arc_history_tail[i-1024u]:ns->nuclear_arc_history_completion[i-1536u]));
}
// C1 keeps visited_corners live across potential replay. The core E1 plane
// ends exactly before that mask; halo E1 uses row2 after its work reduction.
// No potential caller has a remaining consumer of that reduced row.
__host__ __device__ __forceinline__ volatile double* bp_v10_potential_e1_slot(
    volatile BiprismV10NumericalScratchFfi* ns,uint32_t tid) {
    return tid<TILE_CORE_VERTICES?&ns->potential_e1[tid]:&ns->reciprocal_force[2][tid-TILE_CORE_VERTICES];
}
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_search.visited_corners)==10752,"Persistent C1 visited masks");
static_assert(alignof(BiprismV10NumericalScratchFfi)==128,"Numerical scratch alignment");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_start_momentum)==43008,"FP64 nuclear transit suffix");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_midpoint_velocity)==61440,"FP64 nuclear midpoint suffix");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_used_force)==66048,"FP64 nuclear used-force suffix");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_position_start)==72960,"Owned start coordinate pairs");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_position_live)==77568,"Owned live coordinate pairs");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_trial_velocity)==83712,"FP64 nonlinear iterate");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_evaluations)==1666816,"Appended complete evaluation trace");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_arc)==1680768,"Arclength cannot alias the retained prefix");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_evaluation_extension)==1697920,"Original prefix remains disjoint");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_arc_history_extension)==1757824,"Extended arc chronology");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_evaluation_tail)==1784448,"Bounded work tail preserves NHEA");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_arc_history_tail)==1821312,"Bounded arc chronology tail");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_evaluation_completion)==1837696,"Measured work extension preserves NHED");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_arc_history_completion)==1948288,"Measured continuation chronology extension");
static_assert(sizeof(BiprismV10NumericalScratchFfi)==1997440,"Numerical scratch with disjoint arclength owner");
static_assert(sizeof(float2)==sizeof(double),"C2 component and partial-sum lane strides match");
static_assert(offsetof(BiprismV10NumericalScratchFfi,v_start)==2304,"v start");
static_assert(offsetof(BiprismV10NumericalScratchFfi,p_start)==4608,"p start");
static_assert(offsetof(BiprismV10NumericalScratchFfi,next_mid_x)==6912,"midpoint x");
static_assert(offsetof(BiprismV10NumericalScratchFfi,next_mid_v)==9216,"midpoint v");
static_assert(offsetof(BiprismV10NumericalScratchFfi,endpoint_v)==11520,"endpoint v");
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_choice.score)==6912,"C1 score alias");
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_choice.index)==8448,"C1 index alias");
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_e00)==9216,"C1 endpoint coefficient alias");
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_search.visited_corners)==10752,"C1 visited corners in dead midpoint tail");
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_proposal_s0)==11520,"C1 proposal alias");
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_diagnostic.trace)==13056,"C1 dead-tail trace alias");
static_assert(sizeof(BiprismV10RepresentationTraceFfi)==424,"C1 trace exact layout");
static_assert(sizeof(BiprismV10RepresentationTraceFfi)<=768,"C1 trace fits dead endpoint tail");
static_assert(sizeof(BiprismV10C1DiagnosticIdentityFfi)==48,"C1 diagnostic identity");
static_assert(sizeof(BiprismV10EquationRefusalFfi)==256,"EQR1/2 exact layout");
static_assert(sizeof(BiprismV10ContinuousRefusalFfi)==272,"CFR1/2 exact layout");
static_assert(sizeof(BiprismV10ContinuousIterationFfi)==24,"CHT1/2 iteration layout");
static_assert(sizeof(BiprismV10ContinuousHistoryFfi)==592,"CHT2 exact24-iteration history");
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_diagnostic.equation_refusal)==13480,"EQR1/2 after immutable RRF2 trace");
static_assert(offsetof(BiprismV10NumericalScratchFfi,continuous_diagnostic.refusal)==13056,"CFR1/2 replaces dead pre-representation trace");
static_assert(offsetof(BiprismV10NumericalScratchFfi,continuous_diagnostic.history)==11520,"CHT2 in dead continuous proposal prefix");
static_assert(13480+sizeof(BiprismV10EquationRefusalFfi)<=13824,"EQR1/2 within endpoint union");
static_assert(11520+sizeof(BiprismV10ContinuousHistoryFfi)<=13056,"CHT2 disjoint from CFR1/2");
static_assert(offsetof(BiprismV10NumericalScratchFfi,edge_force_x)==6912,"C2 all-lane x component");
static_assert(offsetof(BiprismV10NumericalScratchFfi,edge_force_y)==9216,"C2 all-lane y component");
static_assert(offsetof(BiprismV10NumericalScratchFfi,edge_force_z)==11520,"C2 all-lane z component");
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_metric.e01)==35584,"C1 connection-disjoint rank tail");
static_assert(offsetof(BiprismV10NumericalScratchFfi,rank_metric.delta_h)==37120,"C1 connection-disjoint rank tail");
static_assert(offsetof(BiprismV10NumericalScratchFfi,kinetic_start)==13824,"kinetic start");
static_assert(offsetof(BiprismV10NumericalScratchFfi,conservative_force)==15360,"conservative force");
static_assert(offsetof(BiprismV10NumericalScratchFfi,reciprocal_force)==19968,"reciprocal force");
static_assert(offsetof(BiprismV10NumericalScratchFfi,accepted_phase)==24576,"accepted phase");
static_assert(offsetof(BiprismV10NumericalScratchFfi,scalar)==30720,"CTA scalar footer");
static_assert(offsetof(BiprismV10NumericalScratchFfi,peer_edge_index)==30976,"immutable peer map");
static_assert(offsetof(BiprismV10NumericalScratchFfi,endpoint_metric)==34048,"accepted endpoint metric");
static_assert(offsetof(BiprismV10NumericalScratchFfi,midpoint_covariance)==34048,"C3 covariance lifetime alias");
static_assert(offsetof(BiprismV10NumericalScratchFfi,electronic_connection_im)==34048,"C1 connection lifetime alias");
static_assert(offsetof(BiprismV10NumericalScratchFfi,reciprocal_connection_z)==34048,"C2 exact pairing lifetime alias");
static_assert(offsetof(BiprismV10NumericalScratchFfi,working_phi)==38656,"FP64 potential appends after all prior numerical owners");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_previous_residual)==40704,"Dedicated numerical history after unchanged physical owners");
// Exact FP32 reload from an immutable physical input. This prevents a mass or
// charge value from spanning unrelated provider/solver stages in registers.
__device__ __forceinline__ float bp_v10_readonly_float(const float* p) {
    return *reinterpret_cast<const volatile float*>(p);
}
__device__ __forceinline__ float3 bp_ns_load3(const volatile float a[3][TILE_CORE_VERTICES],uint32_t u) {
    return {a[0][u],a[1][u],a[2][u]};
}
__device__ __forceinline__ void bp_ns_store3(volatile float a[3][TILE_CORE_VERTICES],uint32_t u,float3 v) {
    a[0][u]=v.x;a[1][u]=v.y;a[2][u]=v.z;
}
__device__ __forceinline__ sym3_f32 bp_v10_reload_metric(const volatile sym3_f32* p) {
    return {p->xx,p->yy,p->zz,p->xy,p->xz,p->yz};
}
__device__ __forceinline__ sym3_f32 bp_ns_load_metric(const volatile float a[6][TILE_CORE_VERTICES],uint32_t u) {
    return {a[0][u],a[1][u],a[2][u],a[3][u],a[4][u],a[5][u]};
}
__device__ __forceinline__ sym3_f64 bp_ns_load_metric64(const volatile double a[6][TILE_CORE_VERTICES],uint32_t u) {
    return {a[0][u],a[1][u],a[2][u],a[3][u],a[4][u],a[5][u]};
}
__device__ __forceinline__ void bp_ns_store_metric64(volatile double a[6][TILE_CORE_VERTICES],uint32_t u,const sym3_f64& v) {
    a[0][u]=v.xx;a[1][u]=v.yy;a[2][u]=v.zz;a[3][u]=v.xy;a[4][u]=v.xz;a[5][u]=v.yz;
}
__device__ __forceinline__ void bp_ns_store_metric(volatile float a[6][TILE_CORE_VERTICES],uint32_t u,const sym3_f32& v) {
    a[0][u]=v.xx;a[1][u]=v.yy;a[2][u]=v.zz;a[3][u]=v.xy;a[4][u]=v.xz;a[5][u]=v.yz;
}
__device__ __forceinline__ BiprismElectronicVector<double> bp_ns_load3d(const volatile double a[3][TILE_CORE_VERTICES],uint32_t u) {
    return {a[0][u],a[1][u],a[2][u]};
}
__device__ __forceinline__ void bp_ns_store3d(volatile double a[3][TILE_CORE_VERTICES],uint32_t u,BiprismElectronicVector<double> v) {
    a[0][u]=v.x;a[1][u]=v.y;a[2][u]=v.z;
}

struct __align__(128) FoliatedStreamPtrTableFfi {
    float*          d_coords_xyz[8];
    float*          d_vels_xyz[8];
    float*          d_forces_xyz[8];
    const float*    d_inv_masses[8];
    const float*    d_true_inv_masses[8];
    const uint32_t* d_core_shake_mask[8];
    FoliatedCouplingLeafFfi* d_electronic_exchange[8];
    BiprismV10NumericalScratchFfi* d_numerical_scratch[8];
    BiprismSharedStepFfi* d_shared_step;
    uint32_t physical_step_mode;
    uint32_t reserved_coordinate_alignment;
    BiprismCoordinateOwnerFfi* d_coordinate_owners;
    BiprismOrderedEdgesFfi* d_ordered_edges;
    BiprismPotentialOwnerFfi* d_potential_owners;
    BiprismSpinorOwnerFfi* d_spinor_owners;
    uint8_t reserved_physical_step[80];
};
static_assert(sizeof(FoliatedStreamPtrTableFfi)==640,"Shared physical-step pointer-table ABI");
static_assert(offsetof(FoliatedStreamPtrTableFfi,d_electronic_exchange)==384,"Exchange pointers follow original fields");
static_assert(offsetof(FoliatedStreamPtrTableFfi,d_numerical_scratch)==448,"Numerical scratch replaces former padding");
static_assert(offsetof(FoliatedStreamPtrTableFfi,d_shared_step)==512,"Physical time has its own retained owner");
static_assert(offsetof(FoliatedStreamPtrTableFfi,physical_step_mode)==520,"Explicit production or fixed diagnostic mode");
static_assert(offsetof(FoliatedStreamPtrTableFfi,d_coordinate_owners)==528,"Authoritative full-N coordinate owner array");
static_assert(offsetof(FoliatedStreamPtrTableFfi,d_ordered_edges)==536,"Per-invocation ordered edge scratch array");
static_assert(offsetof(FoliatedStreamPtrTableFfi,d_potential_owners)==544,"Accepted and prepared full-precision potential owners");

static_assert(offsetof(FoliatedStreamPtrTableFfi,d_spinor_owners)==552,"BPS1 dedicated owner pointer");

enum BiprismIntegrationMode : uint32_t {
    ConstrainedShakeSafe   = 0,
    FlexibleCoreVerified   = 1,
    GonzalezExactEnergyBit = 2
};

// The host owns the independent allocation generation and protocol-clock
// association. Its selected record is immutable throughout this body. Validate
// that record and all eight current commit identities before taking its dt.
// Fixed scalar time is available only through the explicit null-owner mode4.
__device__ __forceinline__ bool bp_v10_outer_step_input(
    const FoliatedStreamPtrTableFfi* streams,
    const BiprismConvergenceRecordFfi* records,float scalar_dt,float& outer_dt
) {
    if(streams->physical_step_mode==4u){
        if(streams->d_shared_step||!isfinite(scalar_dt)||scalar_dt<=0.f)return false;
        outer_dt=scalar_dt;return true;
    }
    if(streams->physical_step_mode!=0u||!streams->d_shared_step)return false;
    const auto& selected=*streams->d_shared_step;
    if(!biprism_shared_selected_valid(selected,selected.scheduled_step,selected.generation))return false;
    const uint32_t expected=selected.accepted_bodies*BIPRISM_INNER_SUBSTEPS;
    #pragma unroll 1
    for(uint32_t leaf=0u;leaf<8u;++leaf){
        const auto& r=records[leaf];
        if(r.abi_magic!=BIPRISM_TAIL_ABI_MAGIC||r.leaf_id!=leaf||r.committed_substeps!=expected)return false;
    }
    outer_dt=selected.outer_dt_au;return true;
}

struct __align__(128) AdjudicatorTailContractViewFfi {
    const uint8_t active_abi_do_not_touch[192]; // Bytes 0..191: Strictly const read-only!
    uint32_t tail_abi_magic;       // Bytes 192..195
    uint32_t tail_struct_bytes;    // Bytes 196..199
    float    last_picard_residual; // Bytes 200..203
    float    last_dec_residual;    // Bytes 204..207
    uint32_t picard_iters_used;    // Bytes 208..211
    uint32_t dec_iters_used;       // Bytes 212..215
    uint32_t epoch_leaf_fault;     // Bytes 216..219
    uint32_t epoch_cluster_fault;  // Bytes 220..223
    uint32_t terminal_fault_latch; // Bytes 224..227
    uint32_t committed_substeps;   // Bytes 228..231
    float    w_x_lambda_trace;     // Bytes 232..235
    float    w_x_tau_trace;        // Bytes 236..239
    float    step_energy_drift;    // Bytes 240..243
    uint32_t reserved_tail[3];     // Bytes 244..255
};
static_assert(sizeof(AdjudicatorTailContractViewFfi) == 256, "Must match 256B InterferometricAdjudicatorFfi");
static_assert(alignof(AdjudicatorTailContractViewFfi) == 128, "Must match align(128)");

// One CTA-owned invocation. Volatile members are exact reload points rather
// than per-thread values kept live through every inlined numerical stage.
// Thread0 initializes the fields before the entry CTA barrier and is the sole
// writer of substep/sequence; all consumers follow their publication barriers.
struct __align__(16) BiprismV10CollectiveContext {
    const FoliatedStreamPtrTableFfi* volatile streams;
    BiprismConvergenceRecordFfi* volatile records;
    uint32_t* volatile arrival;
    uint32_t* volatile epoch;
    volatile uint32_t leaf,substep,sequence;
    volatile double kappa,h;
};
static_assert(sizeof(BiprismV10CollectiveContext)==64,"CTA collective context");
struct BiprismV10StoredCollective {
    volatile double sum0,sum1,max0;
};
static_assert(sizeof(BiprismV10StoredCollective)==24,"Exact three-scalar collective cache");
// A next-Picard search predictor, never a physical work or energy credit.
// All values are replicated global reductions. The leader resets the offsets
// every substep, saves raw C1 epsilon before C2, and updates the offsets only
// after complete C4 reductions. CTA barriers publish every change to consumers.
struct BiprismV10ElectronicTarget {
    volatile double epsilon_offset,energy_offset,raw_epsilon,value;
};
static_assert(sizeof(BiprismV10ElectronicTarget)==32,"Electronic target predictor");
struct __align__(128) BiprismV10Invocation {
    BiprismV10CollectiveContext collective;
    ContactShellSimplicialStateFfi* volatile tile;
    const float* volatile amber;
    InterferometricAdjudicatorFfi* volatile adjudicator;
    FoliatedCouplingLeafFfi* volatile bank;
    volatile BiprismV10NumericalScratchFfi* volatile numerical;
    const float* volatile masses;
    volatile uint32_t edges;
    volatile float h,half;
    volatile uint32_t integration_mode;
    // The three C4 collective results have disjoint slots. Thread0 publishes
    // each before the CTA barrier; the later collectives do not overwrite it.
    BiprismV10StoredCollective c4_collective[3];
    BiprismV10ElectronicTarget electronic_target;
    // Stage status and the Picard index are CTA-owned, leader-written values.
    // Publication barriers precede all reads; no returned status or counter
    // must remain in each thread's registers through the numerical stages.
    volatile uint32_t stage_valid,picard;
    // Selected accepted/prepared potential stays immutable during this body.
    // Its fixed halo is the restoration source; working_phi is private replay.
    const double* volatile input_phi;
    volatile uint32_t potential_step;
    volatile uint32_t equation_repairs,equation_winner,equation_repair_valid;
    volatile uint32_t represented_phi_dirty;
    volatile uint32_t representation_round,energy_bank;
    // Solver history only. It is not a work credit or an accepted state.
    volatile double previous_gonzalez_alpha,previous_gonzalez_residual;
    volatile double nuclear_iteration_weight;
    volatile double joint_alpha_scale2,joint_base_alpha,joint_base_merit;
    volatile uint32_t joint_has_base;
    volatile uint32_t joint_base_coordinate_phase;
    volatile double nuclear_secant_alpha,nuclear_secant_residual;
    volatile double nuclear_base_coord_work,nuclear_raw_residual;
    volatile uint32_t nuclear_secant_valid,nuclear_rebase;
    volatile double rounding_work_residual,rounding_old_dt,rounding_old_dtm,potential_result;
    volatile uint32_t rounding_round;
    // A derivative model is not a represented physical candidate. Only paired
    // tangent probes may request unrounded coordinates; base/trial/action
    // evaluations and every accepted state keep the two-word coordinate map.
    volatile uint32_t derivative_coordinate_phase; // 0=physical,1=midpoint,2=endpoint
    uint8_t reserved[96];
};
static_assert(sizeof(BiprismV10Invocation)==512,"Owned invocation storage");
static_assert(offsetof(BiprismV10Invocation,derivative_coordinate_phase)==412,"Derivative phase owns reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,c4_collective)==128,"Collective cache uses reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,electronic_target)==200,"Target uses disjoint reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,stage_valid)==232,"Stage vote uses reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,picard)==236,"Picard index uses reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,input_phi)==240&&offsetof(BiprismV10Invocation,potential_step)==248,"Potential identity uses disjoint invocation bytes");
static_assert(offsetof(BiprismV10Invocation,equation_repairs)==252,"Bounded equation repair count uses reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,equation_winner)==256,"CTA-published equation winner uses reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,equation_repair_valid)==260,"CTA-published equation repair status uses reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,represented_phi_dirty)==264,"CTA-published potential refresh state uses reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,representation_round)==268,"CTA-owned repair loop uses reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,previous_gonzalez_alpha)==280,"Solver history uses disjoint reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,nuclear_iteration_weight)==296,"Numerical relaxation uses reserved invocation bytes");
static_assert(offsetof(BiprismV10Invocation,joint_alpha_scale2)==304,"Joint solver owns previously reserved bytes");
static_assert(offsetof(BiprismV10NumericalScratchFfi,nuclear_base_velocity)==70656,"Dedicated backtracking base");
// Each C3 diagnostic row occupies exactly its own completed covariance row.
// No other vertex reads it. C4 rewrites the covariance after the reductions.
struct BiprismV10C3Diagnostics {
    volatile double rotation,dx_residual,mid_work;
};
static_assert(sizeof(BiprismV10C3Diagnostics)==sizeof(sym3_f32),"Own-row C3 diagnostic alias");

// Coupled shared state: exactly 192 electronic/core vertices and 64 fixed halos.
// The shared FP32 coordinate/velocity rows are diagnostic mirrors. Dedicated
// global coordinate pairs and FP64 trial velocities supply every physical stage;
// the external current/reference planes remain immutable until collective commit.
// Endpoint masses come from the actual
// selected global-atom map. Live DEC weights use the identical represented formula
// through an incident-edge accessor, so their former 6,144-byte cache is absent.
using BiprismVec64 = BiprismElectronicVector<double>;
struct __align__(128) VariationalClusterDsmemV10 {
    float3 x_mid[TILE_TOTAL_VERTICES];
    float3 v_mid[TILE_TOTAL_VERTICES];
    double q_live[TILE_CORE_VERTICES];
    // Retain the shared ABI footprint; all live potential reads use the
    // separately owned FP64 working plane, never a rounded shared shadow.
    float reserved_phi_capacity[TILE_TOTAL_VERTICES];
    BiprismElectronicSpinor<float> retired_psi_n[TILE_CORE_VERTICES];
    BiprismElectronicSpinor<double> psi_trial[TILE_CORE_VERTICES];
    BiprismElectronicSpinor<float> retired_psi_candidate[TILE_CORE_VERTICES];
    uint8_t edge_u8[TILE_MAX_EDGES][2];
    union {
        sym3_f32 g_cov[TILE_CORE_VERTICES];
        struct {
            union {
                sym3_f32 g_cov[TILE_CORE_VERTICES];
                BiprismV10C3Diagnostics diagnostics[TILE_CORE_VERTICES];
            };
            // C2 copied tau before this tail becomes the exact C3 used force.
            // C3 reads only its own slot; C4 starts after the existing CTA barrier.
            float3 used_force[TILE_CORE_VERTICES];
        } nuclear;
        struct {
            float cg_r[TILE_TOTAL_VERTICES],cg_z[TILE_TOTAL_VERTICES];
            float cg_p[TILE_TOTAL_VERTICES],cg_Ap[TILE_TOTAL_VERTICES];
            float diag_inv[TILE_TOTAL_VERTICES];
        } pcg;
        // DEC owns this view only until its completion barrier. Residual and
        // diagonal are lane-local; only direction needs neighboring reads.
        // This2112B view fits inside the existing6240B electronic/force union.
        struct {
            double direction[TILE_TOTAL_VERTICES];
            float reduction[16];
        } precise_pcg;
        struct {
            BiprismElectronicHermitian<double> H_energy[TILE_CORE_VERTICES];
            // Dead tail while force view is inactive; deterministic global
            // candidate selection reuses only this declared electronic view.
            double warp_candidate_score[8];
            uint32_t warp_candidate_index[8];
        } electronic;
        struct {
            unsigned long long f_tot_q39[TILE_CORE_VERTICES][3];
            float3 tau_berry[TILE_CORE_VERTICES];
        } force;
        float accepted_phase[TILE_MAX_EDGES];
    } stage;
    union { float warp_red[8]; double warp_red_f64[8]; };
    double V_pot_n;
    // The previous dH_commit_abs slot had no readers. Keep its eight-byte
    // footprint and move per-substep DEC max telemetry out of caller registers.
    struct __align__(8) { float max_residual; uint32_t max_iters; } dec_aggregate;
    double dg_alpha_global;
    float last_picard_residual,last_dec_residual;
    uint32_t picard_iters_used,dec_iters_used;
    uint32_t local_fault_accum,published_leaf_fault,collective_fault_mask;
    uint32_t _pad_footer[3];
    BiprismV10Invocation invocation;
};
static_assert(sizeof(VariationalClusterDsmemV10)==31616,"Coupled state must be 31,616 bytes");
// Call only on a block-uniform path. The empty read/write pointer operand
// invalidates derived addresses without changing any state or arithmetic.
#define BP_V10_BARRIER_REANCHOR(s_ptr) do { __syncthreads(); asm volatile("" : "+l"(s_ptr) :: "memory"); } while (0)
static_assert(sizeof(VariationalClusterDsmemV10)<=32768,"Operator shared-memory ceiling");
static_assert(offsetof(VariationalClusterDsmemV10,q_live)==6144,"core charge offset");
static_assert(offsetof(VariationalClusterDsmemV10,retired_psi_n)==8704,"represented start offset");
static_assert(offsetof(VariationalClusterDsmemV10,psi_trial)==11776,"continuous trial offset");
static_assert(offsetof(VariationalClusterDsmemV10,retired_psi_candidate)==17920,"represented endpoint offset");
static_assert(offsetof(VariationalClusterDsmemV10,stage)==24064,"aliased stage offset");
static_assert(sizeof(((VariationalClusterDsmemV10*)0)->stage)==6912,"force response bounds");
static_assert(offsetof(VariationalClusterDsmemV10,stage.nuclear.diagnostics)==24064,"C3 diagnostics preserve own metric-row addresses");
static_assert(offsetof(VariationalClusterDsmemV10,stage.nuclear.used_force)==28672,"C3 force follows unchanged metric prefix");
static_assert(offsetof(VariationalClusterDsmemV10,warp_red)==30976,"footer offset");
static_assert(offsetof(VariationalClusterDsmemV10,invocation)==31104,"CTA invocation offset");

// One timestep owner for the complete evaluated C1--C4 map. Private arclength
// corrections are FP64 unknowns; rounding h before evaluating them changes the
// equation and destroys sub-ULP correction actions. The original physical h
// remains the same exact FP32 value promoted to double. Legacy float mirrors
// are diagnostic only and cannot authorize original-timestep acceptance.
__device__ __forceinline__ double bp_v10_half(const VariationalClusterDsmemV10* s) {
    return .5*s->invocation.collective.h;
}
__device__ __forceinline__ void bp_v10_publish_h(VariationalClusterDsmemV10* s,double h) {
    s->invocation.collective.h=h;
    s->invocation.h=float(h);s->invocation.half=float(.5*h);
}
__device__ __forceinline__ void bp_v10_record_evaluated_h(VariationalClusterDsmemV10* s) {
    auto& e=s->invocation.numerical->nuclear_evaluations;
    e.reserved[0]=__float_as_uint(s->invocation.h);
    const unsigned long long bits=__double_as_longlong(s->invocation.collective.h);
    e.reserved[1]=uint32_t(bits);e.reserved[2]=uint32_t(bits>>32u);
}

__device__ __forceinline__ volatile double* bp_v10_phi(const VariationalClusterDsmemV10* s) {
    return s->invocation.numerical->working_phi;
}
__device__ __forceinline__ uint32_t bp_v10_reload_atom_id(
    const VariationalClusterDsmemV10* s,uint32_t vertex
) {
    // The host-admitted topology is immutable throughout this invocation.
    const volatile uint32_t* ids=s->invocation.tile->global_atom_idx;
    return UNPACK_GLOBAL_ATOM_ID(ids[vertex]);
}
__device__ __forceinline__ const BiprismCoordinateOwnerFfi& bp_v10_coordinate_owner(
    const VariationalClusterDsmemV10* s
) {
    return s->invocation.collective.streams->d_coordinate_owners[s->invocation.collective.leaf];
}
__device__ __forceinline__ BiprismCoordinateExpansion3 bp_v10_load_position(
    const volatile BiprismCoordinateExpansion3* p
) {return {{p->x.hi,p->x.lo},{p->y.hi,p->y.lo},{p->z.hi,p->z.lo}};}
__device__ __forceinline__ void bp_v10_store_position(
    volatile BiprismCoordinateExpansion3* p,BiprismCoordinateExpansion3 v
) {
    p->x.hi=v.x.hi;p->x.lo=v.x.lo;p->y.hi=v.y.hi;p->y.lo=v.y.lo;p->z.hi=v.z.hi;p->z.lo=v.z.lo;
}
// External physical planes stay immutable until collective commit. C1--C4
// consume these exact candidate pairs, and commit exports the identical pair.
__device__ __forceinline__ BiprismCoordinateExpansion3 bp_v10_position_pair(
    const VariationalClusterDsmemV10* s,uint32_t u
) {
    return bp_v10_load_position(&s->invocation.numerical->nuclear_position_live[u]);
}
__device__ __forceinline__ BiprismCoordinateExpansion3 bp_v10_start_position(
    const VariationalClusterDsmemV10* s,uint32_t u
) {return bp_v10_load_position(&s->invocation.numerical->nuclear_position_start[u]);}
__device__ __forceinline__ BiprismCoordinateExpansion3 bp_v10_advected_position(
    const VariationalClusterDsmemV10* s,uint32_t u,double dt,BiprismVec64 v
) {
    const auto p=bp_v10_start_position(s,u);
    return {bp_coord_fma(p.x,dt,v.x),bp_coord_fma(p.y,dt,v.y),bp_coord_fma(p.z,dt,v.z)};
}
__device__ __forceinline__ BiprismVec64 bp_v10_live_velocity(const VariationalClusterDsmemV10* s,uint32_t u) {
    return u<TILE_CORE_VERTICES?bp_ns_load3d(s->invocation.numerical->nuclear_trial_velocity,u):BiprismVec64{0.,0.,0.};
}
__host__ __device__ __forceinline__ bool bp_v10_continuous_tangent_probe(uint32_t mode,uint32_t tangent,uint32_t refinements) {
    return mode==1u&&tangent!=0u&&refinements>=2u;
}
__device__ __forceinline__ BiprismCoordinate3F64 bp_v10_displacement(const VariationalClusterDsmemV10* s,uint32_t u) {
    if(s->invocation.derivative_coordinate_phase!=0u&&u<TILE_CORE_VERTICES){
        const auto v=bp_ns_load3d(s->invocation.numerical->nuclear_midpoint_velocity,u);
        const double h=s->invocation.collective.h;
        return {h*v.x,h*v.y,h*v.z};
    }
    return bp_coord_difference(bp_v10_position_pair(s,u),bp_v10_start_position(s,u));
}
__device__ __forceinline__ BiprismVec64 bp_v10_trial_chord(const VariationalClusterDsmemV10* s,uint32_t u) {
    if(s->invocation.derivative_coordinate_phase!=0u&&u<TILE_CORE_VERTICES){
        const auto v=bp_v10_live_velocity(s,u);const double h=s->invocation.collective.h;
        return {h*v.x,h*v.y,h*v.z};
    }
    const auto d=bp_coord_difference(bp_v10_advected_position(s,u,bp_v10_half(s),bp_v10_live_velocity(s,u)),bp_v10_start_position(s,u));
    return {2.*d.x,2.*d.y,2.*d.z};
}
__device__ __forceinline__ void bp_v10_publish_trial_position(VariationalClusterDsmemV10* s,uint32_t u) {
    bp_v10_store_position(&s->invocation.numerical->nuclear_position_live[u],
        bp_v10_advected_position(s,u,bp_v10_half(s),bp_v10_live_velocity(s,u)));
}
__device__ __forceinline__ BiprismElectronicVector<double> bp_v10_position(
    const VariationalClusterDsmemV10* s,uint32_t u
) {
    if(s->invocation.derivative_coordinate_phase!=0u&&u<TILE_CORE_VERTICES){
        const auto x=bp_coord_value(bp_v10_start_position(s,u));
        const bool endpoint=s->invocation.derivative_coordinate_phase==2u;
        const auto v=endpoint?bp_ns_load3d(s->invocation.numerical->nuclear_midpoint_velocity,u):bp_v10_live_velocity(s,u);
        const double h=endpoint?s->invocation.collective.h:bp_v10_half(s);
        return {__dadd_rn(x.x,__dmul_rn(h,v.x)),__dadd_rn(x.y,__dmul_rn(h,v.y)),__dadd_rn(x.z,__dmul_rn(h,v.z))};
    }
    const auto x=bp_coord_value(bp_v10_position_pair(s,u));
    return {x.x,x.y,x.z};
}
__device__ __forceinline__ BiprismCoordinateExpansion3 bp_v10_normalized_position(
    const VariationalClusterDsmemV10* s,uint32_t u
) {
    return bp_coord_add(bp_v10_position_pair(s,u),BiprismCoordinate3F64{0.,0.,0.});
}
__device__ __forceinline__ bool bp_v10_position_exportable(
    const VariationalClusterDsmemV10* s,uint32_t u
) {
    const auto original=bp_v10_position_pair(s,u),normalized=bp_v10_normalized_position(s,u);
    const auto error=bp_coord_difference(normalized,original);
    return bp_coord_finite(normalized)&&error.x==0.&&error.y==0.&&error.z==0.;
}
__device__ __forceinline__ BiprismCoordinate3F64 bp_v10_edge_difference(
    const VariationalClusterDsmemV10* s,uint32_t u,uint32_t v
) {
    if(s->invocation.derivative_coordinate_phase!=0u){
        const auto a=bp_v10_position(s,u),b=bp_v10_position(s,v);
        return {b.x-a.x,b.y-a.y,b.z-a.z};
    }
    return bp_coord_difference(bp_v10_position_pair(s,v),bp_v10_position_pair(s,u));
}
__device__ __forceinline__ BiprismCoordinate3F64 bp_v10_reference_difference(
    const VariationalClusterDsmemV10* s,uint32_t u,uint32_t v
) {
    const auto& owner=bp_v10_coordinate_owner(s);
    const auto* tile=s->invocation.tile;
    return bp_coord_difference(
        bp_coord_load(owner,BiprismCoordinatePlane::MeshReference,UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[v])),
        bp_coord_load(owner,BiprismCoordinatePlane::MeshReference,UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u])));
}

__device__ __forceinline__ uint32_t get_hw_cluster_size_x() {
    uint32_t csize = 1u;
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(csize));
    return csize;
}

__device__ __forceinline__ float dsmem_load_f32_safe(
    const float* local_smem_elem_ptr,
    uint32_t target_leaf_rank,
    const float* l2_fallback_ptr
) {
    if (get_hw_cluster_size_x() >= 8u) {
        uint32_t smem_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(local_smem_elem_ptr));
        float val;
        asm volatile(
            "{\n"
            "  .reg .u32 dsmem_ptr;\n"
            "  mapa.shared::cluster.u32 dsmem_ptr, %1, %2;\n"
            "  ld.shared::cluster.f32 %0, [dsmem_ptr];\n"
            "}\n"
            : "=f"(val)
            : "r"(smem_u32), "r"(target_leaf_rank)
        );
        return val;
    } else {
        return *l2_fallback_ptr;
    }
}

__device__ __forceinline__ uint32_t dsmem_load_u32_safe(
    const uint32_t* local_smem_u32_ptr,
    uint32_t target_leaf_rank,
    const uint32_t* l2_fallback_ptr
) {
    if (get_hw_cluster_size_x() >= 8u) {
        uint32_t smem_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(local_smem_u32_ptr));
        uint32_t val;
        asm volatile(
            "{\n"
            "  .reg .u32 dsmem_ptr;\n"
            "  mapa.shared::cluster.u32 dsmem_ptr, %1, %2;\n"
            "  ld.shared::cluster.u32 %0, [dsmem_ptr];\n"
            "}\n"
            : "=r"(val)
            : "r"(smem_u32), "r"(target_leaf_rank)
        );
        return val;
    } else {
        return *l2_fallback_ptr;
    }
}

// Device-scope acquire/release generations. ABORT is terminal until a quiescent
// host reset; timeout and normal release contend on the same generation CAS.
static constexpr uint32_t BP_V10_BARRIER_ABORT=0x80000000u;
// The owner supplies device-global barrier words. Explicit state-space and
// scope preserve their acquire/release protocol after pointer reloads, avoiding
// generic atomic_ref local/shared dispatch that is invalid for this contract.
__device__ __forceinline__ uint32_t bp_v10_global_load_acquire(const uint32_t* p) {
    uint32_t value;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value)
        : "l"(__cvta_generic_to_global(p)) : "memory");
    return value;
}
__device__ __forceinline__ uint32_t bp_v10_global_add_acq_rel(uint32_t* p,uint32_t value) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old)
        : "l"(__cvta_generic_to_global(p)),"r"(value) : "memory");
    return old;
}
__device__ __forceinline__ uint32_t bp_v10_global_exchange_acq_rel(uint32_t* p,uint32_t value) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.exch.b32 %0, [%1], %2;" : "=r"(old)
        : "l"(__cvta_generic_to_global(p)),"r"(value) : "memory");
    return old;
}
__device__ __forceinline__ uint32_t bp_v10_global_cas_acq_rel(uint32_t* p,uint32_t expected,uint32_t desired) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.cas.b32 %0, [%1], %2, %3;" : "=r"(old)
        : "l"(__cvta_generic_to_global(p)),"r"(expected),"r"(desired) : "memory");
    return old;
}
__device__ __forceinline__ uint32_t bp_v10_global_cas_release(uint32_t* p,uint32_t expected,uint32_t desired) {
    uint32_t old;
    asm volatile("atom.release.gpu.global.cas.b32 %0, [%1], %2, %3;" : "=r"(old)
        : "l"(__cvta_generic_to_global(p)),"r"(expected),"r"(desired) : "memory");
    return old;
}

__device__ __forceinline__ bool bp_v10_barrier_aborted(uint32_t* epoch) {
    if(get_hw_cluster_size_x()>=8u)return false;
    if(!epoch)return true;
    return (bp_v10_global_load_acquire(epoch)&BP_V10_BARRIER_ABORT)!=0u;
}
__device__ __forceinline__ bool cluster_or_grid_sync_8leaves_checked(
    uint32_t* arrival,uint32_t* epoch,uint32_t* local_fault,uint32_t* terminal_fault
) {
    if(gridDim.x<=1u){__syncthreads();return true;}
    if(get_hw_cluster_size_x()>=8u){
        asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
        return true;
    }
    // CTA synchronization orders all payload producers before the leader's
    // acq_rel arrival. The last RMW acquires the preceding arrival chain.
    __syncthreads();
    bool success=true;
    if(threadIdx.x==0u){
        success=false;
        if(arrival&&epoch){
            const uint32_t generation=bp_v10_global_load_acquire(epoch);
            if(!(generation&BP_V10_BARRIER_ABORT)){
                if(generation==BP_V10_BARRIER_ABORT-1u){
                    bp_v10_global_cas_acq_rel(epoch,generation,generation|BP_V10_BARRIER_ABORT);
                }else{
                    const uint32_t count=bp_v10_global_add_acq_rel(arrival,1u)+1u;
                    if(count==gridDim.x){
                        bp_v10_global_exchange_acq_rel(arrival,0u);
                        success=bp_v10_global_cas_release(epoch,generation,generation+1u)==generation;
                        if(!success)success=(bp_v10_global_load_acquire(epoch)==generation+1u);
                    }else if(count<gridDim.x){
                        uint32_t polls=0u,value=generation;
                        while(value==generation&&polls<MAX_BARRIER_SPIN_POLLS){
                            __nanosleep(32);++polls;value=bp_v10_global_load_acquire(epoch);
                        }
                        if(value==generation){
                            bp_v10_global_cas_acq_rel(epoch,generation,generation|BP_V10_BARRIER_ABORT);
                            value=bp_v10_global_load_acquire(epoch);
                        }
                        success=value==generation+1u;
                    }else{
                        bp_v10_global_cas_acq_rel(epoch,generation,generation|BP_V10_BARRIER_ABORT);
                    }
                }
            }
        }
        if(!success){atomicOr(local_fault,FAULT_BARRIER_TIMEOUT);if(terminal_fault)atomicOr(terminal_fault,FAULT_BARRIER_TIMEOUT);}
    }
    __syncthreads();
    // The current generation's decision is broadcast independently of preexisting
    // leaf fault bits. Native/sticky faults must still reach the all8 fault vote.
    return __syncthreads_and(success);
}

__device__ __forceinline__ uint32_t collective_fault_vote_two_barrier_v10(
    VariationalClusterDsmemV10* __restrict__ smem,
    BiprismConvergenceRecordFfi* __restrict__ records,
    uint32_t* __restrict__ arrival,uint32_t* __restrict__ epoch,uint32_t leaf
) {
    if(threadIdx.x==0u){
        smem->_pad_footer[2]=bp_v10_barrier_aborted(epoch)?1u:0u;
        if(smem->_pad_footer[2]){
            atomicOr(&smem->local_fault_accum,FAULT_BARRIER_TIMEOUT);
            atomicOr(&records[leaf].terminal_fault_latch,FAULT_BARRIER_TIMEOUT);
        }
    }
    __syncthreads();
    if(smem->_pad_footer[2])return smem->local_fault_accum;
    if(threadIdx.x==0u){
        const uint32_t fault=smem->local_fault_accum|records[leaf].terminal_fault_latch;
        smem->published_leaf_fault=fault;records[leaf].epoch_leaf_fault=fault;__threadfence();
    }
    const bool first=cluster_or_grid_sync_8leaves_checked(arrival,epoch,&smem->local_fault_accum,&records[leaf].terminal_fault_latch);
    if(threadIdx.x==0u){
        uint32_t fault=smem->published_leaf_fault|smem->local_fault_accum|records[leaf].terminal_fault_latch;
        if(first){
            for(uint32_t r=0u;r<min(gridDim.x,8u);++r)fault|=dsmem_load_u32_safe(&smem->published_leaf_fault,r,&records[r].epoch_leaf_fault);
        }else fault|=FAULT_BARRIER_TIMEOUT;
        smem->collective_fault_mask=fault;if(first)records[leaf].epoch_cluster_fault=fault;
        if(fault)atomicOr(&records[leaf].terminal_fault_latch,fault);
    }
    const bool second=cluster_or_grid_sync_8leaves_checked(arrival,epoch,&smem->local_fault_accum,&records[leaf].terminal_fault_latch);
    if(threadIdx.x==0u){
        uint32_t fault=smem->collective_fault_mask|smem->local_fault_accum|records[leaf].terminal_fault_latch;
        if(!first||!second)fault|=FAULT_BARRIER_TIMEOUT;
        if(second){for(uint32_t r=0u;r<min(gridDim.x,8u);++r)fault|=atomicAdd(&records[r].terminal_fault_latch,0u);}
        smem->collective_fault_mask=fault;if(first&&second)records[leaf].epoch_cluster_fault=fault;
        if(fault)atomicOr(&records[leaf].terminal_fault_latch,fault);
    }
    __syncthreads();return smem->collective_fault_mask;
}

__device__ __forceinline__ float cta_max_reduce_256(float val, float* s_warp_red) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(FULL_WARP_MASK, val, offset));
    }
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (lane == 0) s_warp_red[warp] = val;
    __syncthreads();

    float block_max = (threadIdx.x < 8u) ? s_warp_red[threadIdx.x] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(FULL_WARP_MASK, block_max, offset));
        }
        if (lane == 0) s_warp_red[0] = block_max;
    }
    __syncthreads();
    const float result = s_warp_red[0];
    // Protect readers in slower warps from the next reduction's writes.
    __syncthreads();
    return result;
}

__device__ __forceinline__ void smem_deterministic_add_f32(unsigned long long* addr, float val) {
    long long fixed_val = __float2ll_rn(val * FP_Q39_SCALE);
    atomicAdd(addr, static_cast<unsigned long long>(fixed_val));
}

__device__ __forceinline__ float smem_deterministic_load_f32(const unsigned long long* addr) {
    return static_cast<float>(static_cast<long long>(*addr)) * FP_Q39_INV;
}

__device__ __forceinline__ float3 sym3_mul_vec(const sym3_f32& M, float3 v) {
    return {
        __fmaf_rn(M.xx, v.x, __fmaf_rn(M.xy, v.y, M.xz * v.z)),
        __fmaf_rn(M.xy, v.x, __fmaf_rn(M.yy, v.y, M.yz * v.z)),
        __fmaf_rn(M.xz, v.x, __fmaf_rn(M.yz, v.y, M.zz * v.z))
    };
}

// =============================================================================================
// CORRELATION-EQUILIBRATED 3x3 SPD METRIC INVERSION (`g = m_diag * S_hat * g_corr * S_hat`)
// - Tested controls include diag(10000,1,1) and matched isotropic scales 1e-30/1e30.
// - Extreme anisotropy can underflow scaling ratios or overflow inverse intermediates.
// - out_det_g_inv stores the inverse correlation determinant in this implementation.
// =============================================================================================
__device__ __forceinline__ sym3_f32 invert_sym3_metric_exact(
    const sym3_f32& g,
    float& out_det_g_inv,
    uint32_t* __restrict__ out_spd_fault_mask = nullptr
) {
    if (!(g.xx > 0.0f) || !(g.yy > 0.0f) || !(g.zz > 0.0f) ||
        !isfinite(g.xx) || !isfinite(g.yy) || !isfinite(g.zz)) {
        if (out_spd_fault_mask) atomicOr(out_spd_fault_mask, FAULT_METRIC_NON_SPD);
        out_det_g_inv = 1.0f;
        return {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    }

    const float m_diag = fmaxf(g.xx, fmaxf(g.yy, g.zz));
    const float inv_m  = 1.0f / m_diag;
    const float sx_hat = sqrtf(g.xx * inv_m);
    const float sy_hat = sqrtf(g.yy * inv_m);
    const float sz_hat = sqrtf(g.zz * inv_m);

    // Correlation off-diagonals \rho_{\mu\nu} \in (-1, 1)
    const float r_xy = ((g.xy * inv_m) / sx_hat) / sy_hat;
    const float r_xz = ((g.xz * inv_m) / sx_hat) / sz_hat;
    const float r_yz = ((g.yz * inv_m) / sy_hat) / sz_hat;

    const float tc_xx = __fmaf_rn(-r_yz, r_yz, 1.0f);
    const float tc_yy = __fmaf_rn(-r_xz, r_xz, 1.0f);
    const float tc_zz = __fmaf_rn(-r_xy, r_xy, 1.0f);
    const float tc_xy = __fmaf_rn( r_xz, r_yz, -r_xy);
    const float tc_xz = __fmaf_rn( r_xy, r_yz, -r_xz);
    const float tc_yz = __fmaf_rn( r_xy, r_xz, -r_yz);

    const float det_corr = __fmaf_rn(r_xy, tc_xy, __fmaf_rn(r_xz, tc_xz, tc_xx));
    if (!(tc_zz > 1e-7f) || !(det_corr > 1e-7f) || !isfinite(det_corr)) {
        if (out_spd_fault_mask) atomicOr(out_spd_fault_mask, FAULT_METRIC_NON_SPD);
        out_det_g_inv = 1.0f;
        return {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    }

    const float scale_inv = inv_m / det_corr;
    out_det_g_inv = 1.0f / det_corr;

    return {
        ((tc_xx / sx_hat) / sx_hat) * scale_inv,
        ((tc_yy / sy_hat) / sy_hat) * scale_inv,
        ((tc_zz / sz_hat) / sz_hat) * scale_inv,
        ((tc_xy / sx_hat) / sy_hat) * scale_inv,
        ((tc_xz / sx_hat) / sz_hat) * scale_inv,
        ((tc_yz / sy_hat) / sz_hat) * scale_inv
    };
}

// =============================================================================================
// DIAGONALLY EQUILIBRATED SO(3, g) CAYLEY-BORIS ROTATION
// Equilibration cancels the common metric scale before the cross-product updates.
// Matched isotropic scale controls pass; large tau or extreme anisotropy can still
// overflow intermediates. This routine does not establish a universal finite-output gate.
// =============================================================================================
__device__ __forceinline__ float3 so3_g_metric_cayley_boris(
    float3 v_minus,
    float3 tau_cov,
    const sym3_f32& g_cov,
    const sym3_f32& /*g_inv_unused*/,
    float /*det_g_inv_unused*/
) {
    const float m_diag = fmaxf(g_cov.xx, fmaxf(g_cov.yy, g_cov.zz));
    if (!(m_diag > 0.0f)) return v_minus;

    const float inv_m  = 1.0f / m_diag;
    const float sx_hat = sqrtf(g_cov.xx * inv_m);
    const float sy_hat = sqrtf(g_cov.yy * inv_m);
    const float sz_hat = sqrtf(g_cov.zz * inv_m);

    const float r_xy = ((g_cov.xy * inv_m) / sx_hat) / sy_hat;
    const float r_xz = ((g_cov.xz * inv_m) / sx_hat) / sz_hat;
    const float r_yz = ((g_cov.yz * inv_m) / sy_hat) / sz_hat;

    const float tc_xx = __fmaf_rn(-r_yz, r_yz, 1.0f);
    const float tc_yy = __fmaf_rn(-r_xz, r_xz, 1.0f);
    const float tc_zz = __fmaf_rn(-r_xy, r_xy, 1.0f);
    const float tc_xy = __fmaf_rn( r_xz, r_yz, -r_xy);
    const float tc_xz = __fmaf_rn( r_xy, r_yz, -r_xz);
    const float tc_yz = __fmaf_rn( r_xy, r_xz, -r_yz);
    const float inv_det_corr = 1.0f / fmaxf(__fmaf_rn(r_xy, tc_xy, __fmaf_rn(r_xz, tc_xz, tc_xx)), 1e-12f);

    const sym3_f32 g_corr_inv = {
        tc_xx * inv_det_corr, tc_yy * inv_det_corr, tc_zz * inv_det_corr,
        tc_xy * inv_det_corr, tc_xz * inv_det_corr, tc_yz * inv_det_corr
    };

    // Scale \tau by inv_m = 1/m_diag and relative root-diagonals \hat{s}_\mu \in (0, 1]
    const float3 t_tilde = {
        ((tau_cov.x * inv_m) / sy_hat) / sz_hat,
        ((tau_cov.y * inv_m) / sx_hat) / sz_hat,
        ((tau_cov.z * inv_m) / sx_hat) / sy_hat
    };

    const float3 v_tilde = {sx_hat * v_minus.x, sy_hat * v_minus.y, sz_hat * v_minus.z};

    // \tilde{w}_1 = \tilde{g}^{-1} (\tilde{v}^- \times \tilde{\tau})
    const float3 vt_cross_t = {
        __fmaf_rn(v_tilde.y, t_tilde.z, -v_tilde.z * t_tilde.y),
        __fmaf_rn(v_tilde.z, t_tilde.x, -v_tilde.x * t_tilde.z),
        __fmaf_rn(v_tilde.x, t_tilde.y, -v_tilde.y * t_tilde.x)
    };
    const float3 w1_tilde = sym3_mul_vec(g_corr_inv, vt_cross_t);

    // \tilde{w}_2 = \tilde{g}^{-1} (\tilde{w}_1 \times \tilde{\tau})
    const float3 w1_cross_t = {
        __fmaf_rn(w1_tilde.y, t_tilde.z, -w1_tilde.z * t_tilde.y),
        __fmaf_rn(w1_tilde.z, t_tilde.x, -w1_tilde.x * t_tilde.z),
        __fmaf_rn(w1_tilde.x, t_tilde.y, -w1_tilde.y * t_tilde.x)
    };
    const float3 w2_tilde = sym3_mul_vec(g_corr_inv, w1_cross_t);

    // \omega_g^2 = (\tilde{\tau}^T \tilde{g} \tilde{\tau}) / \det(\tilde{g})
    const float gt_x = __fmaf_rn(r_xy, t_tilde.y, __fmaf_rn(r_xz, t_tilde.z, t_tilde.x));
    const float gt_y = __fmaf_rn(r_xy, t_tilde.x, __fmaf_rn(r_yz, t_tilde.z, t_tilde.y));
    const float gt_z = __fmaf_rn(r_xz, t_tilde.x, __fmaf_rn(r_yz, t_tilde.y, t_tilde.z));
    const float omega_g_sq = inv_det_corr * __fmaf_rn(t_tilde.x, gt_x, __fmaf_rn(t_tilde.y, gt_y, t_tilde.z * gt_z));
    const float scale = 2.0f / (1.0f + fmaxf(omega_g_sq, 0.0f));

    return {
        __fmaf_rn(scale, (w1_tilde.x + w2_tilde.x) / sx_hat, v_minus.x),
        __fmaf_rn(scale, (w1_tilde.y + w2_tilde.y) / sy_hat, v_minus.y),
        __fmaf_rn(scale, (w1_tilde.z + w2_tilde.z) / sz_hat, v_minus.z)
    };
}

__device__ __forceinline__ float3 evaluate_metric_cotangent_force_edge(
    float3 e_uv, float3 r_hat, float r, float l2_cur, float l2_ref,
    float3 v_u,  float3 v_v,   float mass_u, float mass_v,
    float chi_g
) {
    const float inv_l2_ref = 1.0f / fmaxf(l2_ref, 1e-8f);
    const float inv_r      = 1.0f / fmaxf(r, 1e-6f);
    const float strain_fac = (l2_cur - l2_ref) * inv_l2_ref;

    const float xi_u = __fmaf_rn(v_u.x, r_hat.x, __fmaf_rn(v_u.y, r_hat.y, v_u.z * r_hat.z));
    const float xi_v = __fmaf_rn(v_v.x, r_hat.x, __fmaf_rn(v_v.y, r_hat.y, v_v.z * r_hat.z));

    const float rad_coeff = -chi_g * (mass_u * xi_u * xi_u + mass_v * xi_v * xi_v) * inv_l2_ref;
    const float ang_coeff = -chi_g * strain_fac * inv_r;
    const float3 v_perp_weighted = {
        mass_u * xi_u * (v_u.x - xi_u * r_hat.x) + mass_v * xi_v * (v_v.x - xi_v * r_hat.x),
        mass_u * xi_u * (v_u.y - xi_u * r_hat.y) + mass_v * xi_v * (v_v.y - xi_v * r_hat.y),
        mass_u * xi_u * (v_u.z - xi_u * r_hat.z) + mass_v * xi_v * (v_v.z - xi_v * r_hat.z)
    };
    return {
        __fmaf_rn(rad_coeff, e_uv.x, ang_coeff * v_perp_weighted.x),
        __fmaf_rn(rad_coeff, e_uv.y, ang_coeff * v_perp_weighted.y),
        __fmaf_rn(rad_coeff, e_uv.z, ang_coeff * v_perp_weighted.z)
    };
}

// Precision realization of the same metric cotangent expression. Do not feed
// rounded absolute positions or a separately rounded radial length into this
// derivative. The declared force output remains FP32, as in the original map.
__device__ __forceinline__ BiprismVec64 bp_v10_metric_cotangent_compensated(
    BiprismCoordinate3F64 e,BiprismCoordinate3F64 E,
    BiprismVec64 vu,BiprismVec64 vv,float mu,float mv,float chi
) {
    const double l2=bp_geometry_length_squared(e);
    const double l20=bp_geometry_length_squared(E);
    const double r=sqrt(l2),inv_r=1./r;
    const BiprismCoordinate3F64 n={e.x*inv_r,e.y*inv_r,e.z*inv_r};
    const double inv_l20=1./fmax(l20,double(1e-8f));
    const double strain=(l2-l20)*inv_l20;
    const double xu=__fma_rn(double(vu.x),n.x,__fma_rn(double(vu.y),n.y,double(vu.z)*n.z));
    const double xv=__fma_rn(double(vv.x),n.x,__fma_rn(double(vv.y),n.y,double(vv.z)*n.z));
    const double radial=-double(chi)*(double(mu)*xu*xu+double(mv)*xv*xv)*inv_l20;
    const double angular=-double(chi)*strain*(1./fmax(r,double(1e-6f)));
    const BiprismCoordinate3F64 perpendicular={
        double(mu)*xu*(double(vu.x)-xu*n.x)+double(mv)*xv*(double(vv.x)-xv*n.x),
        double(mu)*xu*(double(vu.y)-xu*n.y)+double(mv)*xv*(double(vv.y)-xv*n.y),
        double(mu)*xu*(double(vu.z)-xu*n.z)+double(mv)*xv*(double(vv.z)-xv*n.z)};
    return {__fma_rn(radial,e.x,angular*perpendicular.x),
            __fma_rn(radial,e.y,angular*perpendicular.y),
            __fma_rn(radial,e.z,angular*perpendicular.z)};
}

__device__ __forceinline__ sym3_f32 assemble_metric_to_reg_v10(
    VariationalClusterDsmemV10* __restrict__ smem,
    const ContactShellSimplicialStateFfi* tile,
    uint32_t num_edges,
    float chi_g
) {
    const uint32_t tid = threadIdx.x;
    __syncthreads();
    sym3_f64 my_g = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    if(tid<TILE_CORE_VERTICES){
        const auto* index=&smem->invocation.collective.streams->d_ordered_edges[smem->invocation.collective.leaf];
        // This invocation already validated and published increasing original
        // edge ordinals. One owner sums each metric row in that same order.
        #pragma unroll 1
        for(uint32_t cursor=index->offsets[tid];cursor<index->offsets[tid+1u];++cursor){
            const uint32_t e=index->edge_index[cursor];
        const uint8_t u = smem->edge_u8[e][0];
        const uint8_t v = smem->edge_u8[e][1];
        const auto e_vec=bp_v10_edge_difference(smem,u,v);
        const auto E_ref=bp_v10_reference_difference(smem,u,v);
        const double l2_cur=bp_geometry_length_squared(e_vec);
        const double l2_ref=bp_geometry_length_squared(E_ref);
        const double coeff=double(chi_g)*((l2_cur-l2_ref)/l2_ref)*(1./l2_cur);

        // Preserve FP64 through the ordered sum, the metric maps and their
        // cotangent derivative. The shared FP32 view is diagnostic only.
        const double s_xx = coeff*e_vec.x*e_vec.x;
        const double s_yy = coeff*e_vec.y*e_vec.y;
        const double s_zz = coeff*e_vec.z*e_vec.z;
        const double s_xy = coeff*e_vec.x*e_vec.y;
        const double s_xz = coeff*e_vec.x*e_vec.z;
        const double s_yz = coeff*e_vec.y*e_vec.z;

            my_g.xx+=s_xx;my_g.yy+=s_yy;my_g.zz+=s_zz;
            my_g.xy+=s_xy;my_g.xz+=s_xz;my_g.yz+=s_yz;
        }
        bp_ns_store_metric64(smem->invocation.numerical->nuclear_metric,tid,my_g);
        smem->stage.g_cov[tid]={float(my_g.xx),float(my_g.yy),float(my_g.zz),float(my_g.xy),float(my_g.xz),float(my_g.yz)};
    }
    __syncthreads();
    return {float(my_g.xx),float(my_g.yy),float(my_g.zz),float(my_g.xy),float(my_g.xz),float(my_g.yz)};
}

// The exchange is replay scratch only. No field is authoritative physical state.
// All eight banks are initialized from the current committed n state and current
// candidate before publication. Rejected epochs may leave scratch changed.

__device__ __forceinline__ FoliatedCouplingLeafFfi* bp_v10_bank(const BiprismV10CollectiveContext& c,uint32_t leaf) {
    return c.streams->d_electronic_exchange[leaf];
}
__device__ __forceinline__ bool bp_v10_publication_open(VariationalClusterDsmemV10* s,const BiprismV10CollectiveContext& c) {
    if(threadIdx.x==0u){
        s->_pad_footer[2]=bp_v10_barrier_aborted(c.epoch)?1u:0u;
        if(s->_pad_footer[2]){
            atomicOr(&s->local_fault_accum,FAULT_BARRIER_TIMEOUT);
            atomicOr(&c.records[c.leaf].terminal_fault_latch,FAULT_BARRIER_TIMEOUT);
        }
    }
    __syncthreads();
    return s->_pad_footer[2]==0u;
}
__device__ __forceinline__ bool bp_v10_cluster_barrier(VariationalClusterDsmemV10* s,const BiprismV10CollectiveContext& c) {
    if(get_hw_cluster_size_x()<8u){__threadfence();__syncthreads();}
    return cluster_or_grid_sync_8leaves_checked(c.arrival,c.epoch,&s->local_fault_accum,&c.records[c.leaf].terminal_fault_latch);
}
struct BiprismCollectiveValuesV10 {double sum0,sum1,max0;bool valid;};
// Footer scalar slots are temporary collective operands while publication_phase
// is 1. The final accepted endpoint ledger overwrites them with named quantities.
__device__ __forceinline__ BiprismCollectiveValuesV10 bp_v10_collective_values(
    VariationalClusterDsmemV10* s,BiprismV10CollectiveContext& c,double sum0,double sum1,double max0,bool valid
) {
    if(!bp_v10_publication_open(s,c))return {CUDART_INF,CUDART_INF,CUDART_INF,false};
    auto* bank=bp_v10_bank(c,c.leaf);
    if(threadIdx.x==0u){
        bank->footer.base_energy_delta=sum0;
        bank->footer.owned_pair_energy_delta=sum1;
        bank->footer.attributed_coupling_energy_delta=max0;
        bank->footer.convergence_vote=valid?1u:0u;
        bank->footer.publication_phase=1u;
        __threadfence();
    }
    const bool first=bp_v10_cluster_barrier(s,c);
    if(threadIdx.x==0u){
        double a=0.,b=0.,m=0.;bool ok=first;
        #pragma unroll 1
        for(uint32_t leaf=0u;first&&leaf<8u;++leaf){
            const auto* p=bp_v10_bank(c,leaf);
            a+=p->footer.base_energy_delta;b+=p->footer.owned_pair_energy_delta;
            m=fmax(m,p->footer.attributed_coupling_energy_delta);
            ok=ok&&p->footer.convergence_vote!=0u;
        }
        s->warp_red_f64[0]=a;s->warp_red_f64[1]=b;s->warp_red_f64[2]=m;s->warp_red_f64[3]=ok?1.:0.;
    }
    // Consumers finish before any leaf may reuse its global publication slots.
    const bool second=bp_v10_cluster_barrier(s,c);
    const BiprismCollectiveValuesV10 out={s->warp_red_f64[0],s->warp_red_f64[1],s->warp_red_f64[2],second&&s->warp_red_f64[3]!=0.};
    __syncthreads();return out;
}
// Store only completed, CTA-private result values. Even a refused publication
// writes no remote payload here; the original helper retains every global vote,
// barrier and abort rule. All readers follow this leader-publication barrier.
__device__ __forceinline__ bool bp_v10_collective_store(
    VariationalClusterDsmemV10* s,BiprismV10CollectiveContext& c,uint32_t slot,
    double sum0,double sum1,double max0,bool valid
) {
    const auto out=bp_v10_collective_values(s,c,sum0,sum1,max0,valid);
    if(threadIdx.x==0u){
        s->invocation.c4_collective[slot].sum0=out.sum0;
        s->invocation.c4_collective[slot].sum1=out.sum1;
        s->invocation.c4_collective[slot].max0=out.max0;
    }
    __syncthreads();return out.valid;
}
__device__ __forceinline__ uint32_t bp_v10_collective_candidate(
    VariationalClusterDsmemV10* s,BiprismV10CollectiveContext& c,double score,uint32_t index
) {
    if(!bp_v10_publication_open(s,c))return UINT32_MAX;
    auto* bank=bp_v10_bank(c,c.leaf);
    if(threadIdx.x==0u){bank->footer.base_energy_delta=score;bank->footer.convergence_vote=index;bank->footer.publication_phase=2u;__threadfence();}
    const bool first=bp_v10_cluster_barrier(s,c);
    if(threadIdx.x==0u){
        double best=CUDART_INF;uint32_t selected=UINT32_MAX;
        for(uint32_t leaf=0u;first&&leaf<8u;++leaf){
            const auto* p=bp_v10_bank(c,leaf);const double v=p->footer.base_energy_delta;
            const uint32_t local=p->footer.convergence_vote;
            const uint32_t id=local==UINT32_MAX?UINT32_MAX:leaf*(4u*TILE_CORE_VERTICES)+local;
            if(v<best||(v==best&&id<selected)){best=v;selected=id;}
        }
        s->_pad_footer[0]=first?selected:UINT32_MAX;
    }
    const bool second=bp_v10_cluster_barrier(s,c);
    const uint32_t out=second?s->_pad_footer[0]:UINT32_MAX;__syncthreads();return out;
}

using BiprismPsi64 = BiprismElectronicSpinor<double>;
using BiprismPsi32 = BiprismElectronicSpinor<float>;
using BiprismH64 = BiprismElectronicHermitian<double>;
using BiprismRho64 = BiprismElectronicDensity<double>;

__device__ __forceinline__ BiprismSpinorOwnerFfi& bp_v10_spinor_owner(const VariationalClusterDsmemV10* s) {
    return s->invocation.collective.streams->d_spinor_owners[s->invocation.collective.leaf];
}
__device__ __forceinline__ BiprismPsi64* bp_v10_starts(const VariationalClusterDsmemV10* s) {return bp_v10_spinor_owner(s).substep_start;}
__device__ __forceinline__ BiprismPsi64* bp_v10_candidates(const VariationalClusterDsmemV10* s) {return bp_v10_spinor_owner(s).candidate;}
__device__ __forceinline__ BiprismPsi64 bp_v10_promote(BiprismPsi64 p) {return p;}
__device__ __forceinline__ BiprismPsi64 bp_v10_promote(BiprismPsi32 p) {
    return {{double(p.s0.re),double(p.s0.im)},{double(p.s1.re),double(p.s1.im)}};
}
__device__ __forceinline__ BiprismPsi64 bp_v10_reload_spinor(const volatile BiprismPsi32* p) {
    return {{double(p->s0.re),double(p->s0.im)},{double(p->s1.re),double(p->s1.im)}};
}
__device__ __forceinline__ BiprismPsi64 bp_v10_reload_spinor(const volatile BiprismPsi64* p) {
    return {{p->s0.re,p->s0.im},{p->s1.re,p->s1.im}};
}
__device__ __forceinline__ BiprismH64 bp_v10_reload_h(const volatile BiprismH64* p) {
    return {p->h00,p->h11,p->h01_re,p->h01_im};
}
__device__ __forceinline__ BiprismVec64 bp_v10_vector(float3 p) {
    return {double(p.x),double(p.y),double(p.z)};
}
__device__ __forceinline__ BiprismElectronicElementParams<double> bp_v10_element(uint32_t e) {
    const GfnXtbElementParams p=c_gfn_xtb_lut[e&7u];
    return {double(p.zeta_slater),double(p.h_s_valence),double(p.h_p_excited),double(p.hubbard_eta)};
}
__device__ __forceinline__ double bp_v10_berry_mass(const ContactShellSimplicialStateFfi* tile,uint32_t u) {
    return double(fmaxf(tile->star0_vol[u],1e-4f));
}
__device__ __forceinline__ double bp_v10_spinor_norm(BiprismPsi64 p) {
    return bp_ed_norm2(p.s0)+bp_ed_norm2(p.s1);
}
__device__ __forceinline__ double bp_v10_outer_norm(const VariationalClusterDsmemV10* s,uint32_t u) {
    return u<TILE_CORE_VERTICES?bp_v10_spinor_norm(bp_v10_reload_spinor(&bp_v10_spinor_owner(s).accepted[u])):0.;
}
__device__ __forceinline__ BiprismPsi64 bp_v10_start(const VariationalClusterDsmemV10* s,uint32_t u) {
    return u<TILE_CORE_VERTICES?bp_v10_reload_spinor(&bp_v10_starts(s)[u]):BiprismPsi64{{1.,0.},{0.,0.}};
}
__device__ __forceinline__ BiprismPsi64 bp_v10_endpoint(const VariationalClusterDsmemV10* s,uint32_t u,bool represented) {
    if(u>=TILE_CORE_VERTICES)return {{1.,0.},{0.,0.}};
    return represented?bp_v10_reload_spinor(&bp_v10_candidates(s)[u]):bp_v10_reload_spinor(&s->psi_trial[u]);
}
__device__ __forceinline__ BiprismRho64 bp_v10_density_mean(const VariationalClusterDsmemV10* s,uint32_t u,bool represented) {
    return bp_ed_endpoint_data(bp_v10_start(s,u),bp_v10_endpoint(s,u,represented),0.).rho_bar;
}
__device__ __forceinline__ double bp_v10_charge(const VariationalClusterDsmemV10* s,
    const ContactShellSimplicialStateFfi* tile,const float* amber,uint32_t u) {
    return u<TILE_CORE_VERTICES?s->q_live[u]:double(amber[UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u])]);
}
// Endpoint/charge inputs are immutable throughout each force stage. Explicit
// reloads prevent provider evaluation from retaining the later metric operands.
__device__ __forceinline__ float3 bp_v10_reload_vec3(const volatile float3* p) {
    return {p->x,p->y,p->z};
}
__device__ __forceinline__ double bp_v10_reload_charge(const VariationalClusterDsmemV10* s,
    const ContactShellSimplicialStateFfi* tile,const float* amber,uint32_t u) {
    return u<TILE_CORE_VERTICES?*reinterpret_cast<const volatile double*>(&s->q_live[u])
        :double(bp_v10_readonly_float(amber+UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u])));
}
// The accessor is evaluated only after a row has established incidence. Current
// and reference radial geometry uses full coordinate expansions; only the
// live-weight storage retains the full FP64 radial value.
struct BiprismLiveDecWeightsV10 {
    const VariationalClusterDsmemV10* s;
    const ContactShellSimplicialStateFfi* tile;
    const double* cache=nullptr;
    __device__ __forceinline__ double operator()(uint32_t e) const {
        if(cache)return cache[e];
        const uint32_t u=s->edge_u8[e][0],v=s->edge_u8[e][1];
        const auto x=bp_v10_edge_difference(s,u,v),X=bp_v10_reference_difference(s,u,v);
        return bp_geometry_live_weight_f64(tile->star1_wt[e],x,X);
    }
};
__device__ __forceinline__ double bp_v10_sum(double value,double* warp) {
    const uint32_t lane=threadIdx.x&31u,w=threadIdx.x>>5u;
    #pragma unroll
    for(int o=16;o;o>>=1)value+=__shfl_xor_sync(FULL_WARP_MASK,value,o);
    if(lane==0)warp[w]=value;
    __syncthreads();
    if(w==0){
        double x=lane<8u?warp[lane]:0.;
        #pragma unroll
        for(int o=4;o;o>>=1)x+=__shfl_xor_sync(FULL_WARP_MASK,x,o);
        if(lane==0)warp[0]=x;
    }
    __syncthreads(); const double out=warp[0]; __syncthreads(); return out;
}
__device__ __forceinline__ double bp_v10_max(double value,double* warp) {
    if(!isfinite(value))value=CUDART_INF;
    const uint32_t lane=threadIdx.x&31u,w=threadIdx.x>>5u;
    #pragma unroll
    for(int o=16;o;o>>=1)value=fmax(value,__shfl_xor_sync(FULL_WARP_MASK,value,o));
    if(lane==0)warp[w]=value;
    __syncthreads();
    if(w==0){
        double x=lane<8u?warp[lane]:0.;
        #pragma unroll
        for(int o=4;o;o>>=1)x=fmax(x,__shfl_xor_sync(FULL_WARP_MASK,x,o));
        if(lane==0)warp[0]=x;
    }
    __syncthreads(); const double out=warp[0]; __syncthreads(); return out;
}
__device__ __forceinline__ bool solve_live_dec_poisson_pcg_v10(
    VariationalClusterDsmemV10* s,const ContactShellSimplicialStateFfi* tile,
    uint32_t edges,const double* live_weights,float& residual,uint32_t& iterations
) {
    const double q=threadIdx.x<TILE_CORE_VERTICES?s->q_live[threadIdx.x]:0.;
    const bool converged=biprism_dec_pcg_256_precise_traversal(&s->edge_u8[0][0],&s->edge_u8[0][1],2u,
        BiprismLiveDecWeightsV10{s,tile,live_weights},edges,
        BiprismIncidentEdgeTraversal{&s->invocation.collective.streams->d_ordered_edges[s->invocation.collective.leaf]},
        q,bp_v10_phi(s),s->stage.precise_pcg.direction,s->stage.precise_pcg.reduction,
        residual,iterations,1.0e-20);
    if(threadIdx.x==0u){
        s->last_dec_residual=residual;s->dec_iters_used=iterations;
        s->dec_aggregate.max_residual=fmaxf(s->dec_aggregate.max_residual,residual);
        s->dec_aggregate.max_iters=max(s->dec_aggregate.max_iters,iterations);
        if(!converged)atomicOr(&s->local_fault_accum,FAULT_DEC_NONCONV);
    }
    __syncthreads();return converged;
}
// bank0=start spinor, bank1=endpoint spinor. Bit1 explicitly marks C4's
// endpoint GEOMETRY (C1 also evaluates bank1, but at midpoint geometry).
__device__ __noinline__ void evaluate_total_potential_energy_v10(
    VariationalClusterDsmemV10* s
) {
    const uint32_t bank=s->invocation.energy_bank;
    auto& tile=s->invocation.tile;auto& amber=s->invocation.amber;
    auto& edges=s->invocation.edges;auto& ns=s->invocation.numerical;
    const uint32_t tid=threadIdx.x;
    // The local-ranking operands/force vectors are dead at every potential
    // call. Map all256 lanes into disjoint existing rows without touching H.
    volatile double* energy=tid<TILE_CORE_VERTICES?&ns->reciprocal_force[0][tid]
        :&ns->reciprocal_force[1][tid-TILE_CORE_VERTICES];
    *energy=0.; bool finite=true;
    if(tid<TILE_CORE_VERTICES){
        const auto rho=bp_ed_density(bank?bp_v10_reload_spinor(&bp_v10_candidates(s)[tid]):bp_v10_reload_spinor(&bp_v10_starts(s)[tid]));
        const auto p=bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[tid]));
        *energy=(rho.r00*p.h_s_valence+rho.r11*p.h_p_excited)*double(tile->star0_vol[tid])
            +s->q_live[tid]*double(bp_v10_phi(s)[tid]);
    }
    #pragma unroll 1
    for(uint32_t e=tid;e<edges;e+=256u){
        const uint32_t u=s->edge_u8[e][0],v=s->edge_u8[e][1];
        BiprismElectronicEdgeProvider<double> p{};
        const bool ok=bp_ed_evaluate_edge(bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[u])),
            bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[v])),
            bp_v10_position(s,u),bp_v10_position(s,v),bp_v10_charge(s,tile,amber,u),bp_v10_charge(s,tile,amber,v),p);
        // Store outputs even on a rejected provider evaluation; p is zero
        // initialized for its early-return path, and failed values are never
        // consumed. This lets completed outputs die before the finite vote.
        static_cast<volatile BiprismV10NumericalScratchFfi*>(ns)->potential_e0[tid]=p.E0;
        *static_cast<volatile double*>(bp_v10_potential_e1_slot(ns,tid))=p.E1;
        static_cast<volatile BiprismV10NumericalScratchFfi*>(ns)->potential_v01[tid]=p.V01;
        finite=finite&&ok;
        if(ok){
            if((bank&2u)!=0u&&ns->nuclear_arc.tangent_mode!=0u&&ns->nuclear_secant.mode==0u)
                ns->nuclear_secant.tangent_endpoint_raw_D[e]=1.-p.S*p.S;
            // These three typed stage views own completed canonical provider
            // values only until this thread finishes its edge energy. Ranking
            // and the rounding proposal have no readers at any potential call.
            // All256 lanes use disjoint8-byte slots, including the halo lanes.
            asm volatile("" ::: "memory");
            const auto psi=u<TILE_CORE_VERTICES?(bank?bp_v10_reload_spinor(&bp_v10_candidates(s)[u]):bp_v10_reload_spinor(&bp_v10_starts(s)[u])):BiprismPsi64{{1.,0.},{0.,0.}};
            volatile auto* values=static_cast<volatile BiprismV10NumericalScratchFfi*>(ns);
            const bool branch_model=bp_v10_endpoint_branch_model(bank,s->invocation.derivative_coordinate_phase,
                ns->nuclear_secant.mode,ns->nuclear_arc.tangent_mode,ns->nuclear_arc.tangent_probe_refinements);
            values->potential_e0[tid]=bp_v10_endpoint_branch_energy(values->potential_e0[tid],*static_cast<volatile double*>(bp_v10_potential_e1_slot(ns,tid)),values->potential_v01[tid],p.S,bp_ed_density(psi),
                branch_model?ns->nuclear_secant.tangent_endpoint_raw_D[e]:0.,branch_model);
            asm volatile("" ::: "memory");
            const double dphi=double(bp_v10_phi(s)[u])-double(bp_v10_phi(s)[v]);
            *energy+=values->potential_e0[tid]-.5*double(s->invocation.bank->live_weights[e])*dphi*dphi;
        }
    }
    if(!__syncthreads_and(finite&&isfinite(*energy))){
        if(tid==0){atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);s->invocation.potential_result=CUDART_INF;}
        __syncthreads();return;
    }
    const double total=bp_v10_sum(*energy,s->warp_red_f64);
    if(tid==0u)s->invocation.potential_result=total;
    __syncthreads();
}
__device__ __forceinline__ void bp_v10_publish_charge(
    VariationalClusterDsmemV10* s,double amber,uint32_t bank,bool represented
) {
    if(threadIdx.x<TILE_CORE_VERTICES){
        const auto p0=bp_v10_start(s,threadIdx.x),p1=bp_v10_endpoint(s,threadIdx.x,represented);
        const auto d=bp_ed_endpoint_data(p0,p1,amber);
        s->q_live[threadIdx.x]=bank==0u?d.q_n:(bank==1u?d.q_np1:d.q_bar);
    }
    __syncthreads();
}
__device__ __forceinline__ bool bp_v10_refresh_weights(
    VariationalClusterDsmemV10* s,const ContactShellSimplicialStateFfi* tile,
    uint32_t edges,const BiprismV10CollectiveContext& c
) {
    if(!bp_v10_publication_open(s,c))return false;
    const BiprismLiveDecWeightsV10 evaluate{s,tile};bool valid=true;
    auto* bank=bp_v10_bank(c,c.leaf);
    #pragma unroll 1
    for(uint32_t e=threadIdx.x;e<edges;e+=256u){
        const double w=evaluate(e);bank->live_weights[e]=w;valid=valid&&isfinite(w)&&w>0.f;
    }
    return __syncthreads_and(valid);
}
__device__ __forceinline__ double bp_v10_population(BiprismPsi64 p) {
    return bp_ed_norm2(p.s0)-bp_ed_norm2(p.s1);
}
__device__ __forceinline__ bool bp_v10_publish_populations(
    VariationalClusterDsmemV10* s,const ContactShellSimplicialStateFfi* tile,
    BiprismV10CollectiveContext& c,bool represented
) {
    if(!bp_v10_publication_open(s,c))return false;
    const uint32_t u=threadIdx.x;auto* b=bp_v10_bank(c,c.leaf);bool valid=true;
    if(u<TILE_CORE_VERTICES){
        b->start_population[u]=bp_v10_population(bp_v10_start(s,u));
        b->trial_population[u]=bp_v10_population(bp_v10_endpoint(s,u,represented));
        b->energy_weight[u]=tile->star0_vol[u];
        valid=isfinite(b->start_population[u])&&isfinite(b->trial_population[u])&&isfinite(b->energy_weight[u])&&b->energy_weight[u]>0.f;
    }
    valid=__syncthreads_and(valid);
    if(u==0u)++c.sequence;
    __syncthreads();
    if(u==0u){
        b->footer.abi_magic=BIPRISM_FOLIATED_COUPLING_ABI;b->footer.leaf_id=c.leaf;
        b->footer.accepted_substeps_at_entry=c.records[c.leaf].committed_substeps;
        b->footer.substep=c.substep;b->footer.iteration=c.sequence;
        b->footer.local_fault_mask=s->local_fault_accum;
        b->footer.publication_phase=3u;b->footer.convergence_vote=valid?1u:0u;
        __threadfence();
    }
    valid=bp_v10_cluster_barrier(s,c)&&valid;
    const auto* prev=bp_v10_bank(c,(c.leaf+7u)&7u);const auto* next=bp_v10_bank(c,(c.leaf+1u)&7u);
    valid=valid&&prev->footer.abi_magic==BIPRISM_FOLIATED_COUPLING_ABI&&next->footer.abi_magic==BIPRISM_FOLIATED_COUPLING_ABI
        &&prev->footer.substep==c.substep&&next->footer.substep==c.substep&&prev->footer.iteration==c.sequence&&next->footer.iteration==c.sequence;
    return __syncthreads_and(valid);
}
__device__ __forceinline__ biprism_foliated::EndpointLedger bp_v10_foliation(
    const BiprismV10CollectiveContext& c,uint32_t u
) {
    if(u>=TILE_CORE_VERTICES)return {};
    const auto* b=bp_v10_bank(c,c.leaf);const auto* p=bp_v10_bank(c,(c.leaf+7u)&7u);const auto* n=bp_v10_bank(c,(c.leaf+1u)&7u);
    return biprism_foliated::endpoint_ledger(c.kappa,double(p->energy_weight[u]),double(b->energy_weight[u]),double(n->energy_weight[u]),
        p->start_population[u],p->trial_population[u],b->start_population[u],b->trial_population[u],n->start_population[u],n->trial_population[u]);
}
__device__ __forceinline__ BiprismVec64 bp_v10_kappa(
    const BiprismElectronicEdgeProvider<double>& p,double weight
) {
    const double d=p.E1-p.E0,eps=double(5e-5f);
    const double inv_gap=1./sqrt(__fma_rn(d,d,__fma_rn(4.*p.V01,p.V01,eps*eps)));
    return bp_ed_connection_coefficient(double(weight),p.S,inv_gap,p.grad_V01_u);
}
// Leaves Henergy and connection_im live. A caller must finish every read before
// reusing the union. The six phase scalars preserve this exact evaluated epoch.
__device__ __forceinline__ bool bp_v10_build_electronic_h(
    VariationalClusterDsmemV10* s,bool represented
) {
    auto& c=s->invocation.collective;auto& tile=s->invocation.tile;
    auto& amber=s->invocation.amber;auto& edges=s->invocation.edges;auto& h=c.h;
    const uint32_t tid=threadIdx.x; bool valid=bp_v10_publish_populations(s,tile,c,represented);
    if(tid<TILE_CORE_VERTICES){
        const auto p=bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[tid]));
        const double omega=double(tile->star0_vol[tid]);
        valid=valid&&isfinite(omega)&&omega>0.;
        s->stage.electronic.H_energy[tid]=bp_ed_onsite_dec_hamiltonian(omega,p.h_s_valence,p.h_p_excited,double(bp_v10_phi(s)[tid]));
        const double fol=valid?bp_v10_foliation(c,tid).numerator:0.;
        s->stage.electronic.H_energy[tid].h00+=fol;s->stage.electronic.H_energy[tid].h11-=fol;
        c.streams->d_numerical_scratch[c.leaf]->electronic_connection_im[tid]=0.;
    }
    __syncthreads();
    #pragma unroll 1
    for(uint32_t e=tid;e<edges;e+=256u){
        const uint32_t u=s->edge_u8[e][0],v=s->edge_u8[e][1];
        BiprismElectronicEdgeProvider<double> p;
        const bool ok=bp_ed_evaluate_edge(bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[u])),
            bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[v])),bp_v10_position(s,u),bp_v10_position(s,v),
            bp_v10_charge(s,tile,amber,u),bp_v10_charge(s,tile,amber,v),p);
        valid=valid&&ok;
        if(!ok)continue;
        asm volatile("" ::: "memory");
        const auto edge=bp_ed_edge_discrete(p.E0,p.E1,p.V01,p.S,p.gamma,bp_v10_density_mean(s,u,represented));
        const auto kap=bp_v10_kappa(p,s->invocation.bank->live_weights[e]);
        const auto vu=bp_v10_live_velocity(s,u),vv=bp_v10_live_velocity(s,v);
        const BiprismVec64 vr={vv.x-vu.x,vv.y-vu.y,vv.z-vu.z};
        const double lambda=bp_ed_dot(kap,vr);
        if(u<TILE_CORE_VERTICES){
            atomicAdd(&s->stage.electronic.H_energy[u].h00,edge.owner_h.h00);
            atomicAdd(&s->stage.electronic.H_energy[u].h11,edge.owner_h.h11+double(.15f)*edge.charge_derivative_u);
            atomicAdd(&s->stage.electronic.H_energy[u].h01_re,edge.owner_h.h01_re);
            atomicAdd(&c.streams->d_numerical_scratch[c.leaf]->electronic_connection_im[u],-lambda);
        }
        if(v<TILE_CORE_VERTICES){
            atomicAdd(&s->stage.electronic.H_energy[v].h11,double(.15f)*edge.charge_derivative_v);
            atomicAdd(&c.streams->d_numerical_scratch[c.leaf]->electronic_connection_im[v],lambda);
        }
        // A_omega is retained independently of the star1/(2D) connection factor.
        const double d=p.E1-p.E0,eps=double(5e-5f);
        const float phase=float(h*bp_ed_dot(p.grad_V01_u,vr)/sqrt(__fma_rn(d,d,__fma_rn(4.*p.V01,p.V01,eps*eps))));
        if(valid)static_cast<volatile BiprismV10NumericalScratchFfi*>(c.streams->d_numerical_scratch[c.leaf])->accepted_phase[e]=phase;
    }
    return __syncthreads_and(valid);
}
// True nonlinear equation of a supplied endpoint under H built from that same
// endpoint density. Scaling is identical to the common frozen-H Cayley helper.
__device__ __forceinline__ double bp_v10_electronic_equation(
    BiprismPsi64 p0,BiprismPsi64 p1,BiprismH64 H,double omega,double h
) {
    const double scale=fmax(omega,fmax(fabs(.5*h*H.h00),fmax(fabs(.5*h*H.h11),fmax(fabs(.5*h*H.h01_re),fabs(.5*h*H.h01_im)))));
    const BiprismPsi64 sum={bp_ed_add(p0.s0,p1.s0),bp_ed_add(p0.s1,p1.s1)};
    const auto hs=bp_ed_apply(H,sum);
    const auto d0=bp_ed_sub(p1.s0,p0.s0),d1=bp_ed_sub(p1.s1,p0.s1);
    const double a=omega/scale,b=.5*h/scale;
    return fmax(fmax(fabs(__fma_rn(a,d0.re,-b*hs.s0.im)),fabs(__fma_rn(a,d0.im,b*hs.s0.re))),
        fmax(fabs(__fma_rn(a,d1.re,-b*hs.s1.im)),fabs(__fma_rn(a,d1.im,b*hs.s1.re))));
}
__device__ __forceinline__ float bp_v10_bracket(double x,bool upper) {
    const float n=float(x);
    if(double(n)==x)return n;
    if(upper)return double(n)<x?nextafterf(n,CUDART_INF_F):n;
    return double(n)>x?nextafterf(n,-CUDART_INF_F):n;
}
__device__ __forceinline__ bool bp_v10_norm_admitted(BiprismPsi64 p,double n,double outer) {
    const double np=bp_v10_spinor_norm(p);
    return isfinite(np)&&fabs(np-n)<=1e-7&&fabs(np-outer)<=1e-7&&fabs(np-1.)<=1e-7;
}
__device__ __forceinline__ float bp_v10_shell1_bracket(double x,uint32_t choice) {
    const float b=bp_v10_bracket(x,choice>=2u);
    return choice==0u?nextafterf(b,-CUDART_INF_F):(choice==3u?nextafterf(b,CUDART_INF_F):b);
}
// This finite fallback is called only if every original corner failed the three
// norm bounds. It chooses the nearest admitted represented endpoint
// from one additional outward neighbor on each side of each component bracket.
// No normalization is applied. Charge/DEC/H/equation/work are rebuilt afterward.
__device__ __forceinline__ bool bp_v10_round_norm_shell1(
    BiprismPsi64 continuous,double norm_n,double norm_outer,BiprismPsi32& result
) {
    double best=CUDART_INF;bool found=false;
    #pragma unroll 1
    for(uint32_t bits=0u;bits<256u;++bits){
        const BiprismPsi32 p={{bp_v10_shell1_bracket(continuous.s0.re,bits&3u),bp_v10_shell1_bracket(continuous.s0.im,(bits>>2u)&3u)},
            {bp_v10_shell1_bracket(continuous.s1.re,(bits>>4u)&3u),bp_v10_shell1_bracket(continuous.s1.im,(bits>>6u)&3u)}};
        const auto q=bp_v10_promote(p);
        if(!bp_v10_norm_admitted(q,norm_n,norm_outer))continue;
        // Precision repair: retain the promoted double as the live candidate for the rest of this Picard iteration.
        const double a=q.s0.re-continuous.s0.re,b=q.s0.im-continuous.s0.im;
        const double c=q.s1.re-continuous.s1.re,d=q.s1.im-continuous.s1.im;
        const double error=__fma_rn(a,a,__fma_rn(b,b,__fma_rn(c,c,d*d)));
        if(error<best){best=error;result=p;found=true;}
    }
    return found;
}
__device__ __forceinline__ float bp_v10_spinor_component(BiprismPsi32 p,uint32_t component) {
    return component==0u?p.s0.re:(component==1u?p.s0.im:(component==2u?p.s1.re:p.s1.im));
}
__device__ __forceinline__ BiprismPsi32 bp_v10_with_component(BiprismPsi32 p,uint32_t component,float value) {
    if(component==0u)p.s0.re=value;
    else if(component==1u)p.s0.im=value;
    else if(component==2u)p.s1.re=value;
    else p.s1.im=value;
    return p;
}
__device__ __forceinline__ float bp_v10_slice_component(double value,uint32_t component,uint32_t pivot,uint32_t bits) {
    if(component==pivot)return float(value);
    return bp_v10_bracket(value,(bits>>(component-(component>pivot?1u:0u)))&1u);
}
__device__ __forceinline__ bool bp_v10_norm_below(double np,double n,double outer) {
    return np-n < -1e-7||np-outer < -1e-7||np-1. < -1e-7;
}
__device__ __forceinline__ bool bp_v10_norm_above(double np,double n,double outer) {
    return np-n > 1e-7||np-outer > 1e-7||np-1. > 1e-7;
}
// Complete32 one-dimensional slices of the FP32 norm lattice. The other three
// components use their original brackets; each pivot retains its sign. The
// actual FMA norm and its difference comparisons are monotone in nonnegative
// FP32 magnitude bits, so at most31 bisections locate the closest possible
// boundary to the RN pivot. A final full predicate handles empty slices.
// This is bounded represented-state selection, not normalization or a proof
// of feasibility over all four-dimensional combinations. Every chosen endpoint
// still receives fresh charge/DEC/H/equation/work/energy qualification.
__device__ __forceinline__ bool bp_v10_round_norm_interval(
    BiprismPsi64 continuous,double norm_n,double norm_outer,BiprismPsi32& result
) {
    if(!isfinite(norm_n)||!isfinite(norm_outer)||!isfinite(bp_v10_spinor_norm(continuous)))return false;
    double best=CUDART_INF;bool found=false;
    #pragma unroll 1
    for(uint32_t pivot=0u;pivot<4u;++pivot){
        #pragma unroll 1
        for(uint32_t bits=0u;bits<8u;++bits){
            BiprismPsi32 p={{bp_v10_slice_component(continuous.s0.re,0u,pivot,bits),bp_v10_slice_component(continuous.s0.im,1u,pivot,bits)},
                {bp_v10_slice_component(continuous.s1.re,2u,pivot,bits),bp_v10_slice_component(continuous.s1.im,3u,pivot,bits)}};
            const uint32_t pivot_bits=__float_as_uint(bp_v10_spinor_component(p,pivot));
            const uint32_t magnitude=pivot_bits&0x7fffffffu,sign=pivot_bits&0x80000000u;
            if(magnitude>0x7f7fffffu)continue;
            const double initial_norm=bp_v10_spinor_norm(bp_v10_promote(p));
            if(!isfinite(initial_norm))continue;
            if(!bp_v10_norm_admitted(bp_v10_promote(p),norm_n,norm_outer)){
                const bool increase=bp_v10_norm_below(initial_norm,norm_n,norm_outer);
                uint32_t lo=increase?magnitude:0u,hi=increase?0x7f7fffffu:magnitude;
                #pragma unroll 1
                for(uint32_t iteration=0u;iteration<31u&&lo<hi;++iteration){
                    const uint32_t mid=increase?lo+(hi-lo)/2u:lo+(hi-lo+1u)/2u;
                    p=bp_v10_with_component(p,pivot,__uint_as_float(sign|mid));
                    const double np=bp_v10_spinor_norm(bp_v10_promote(p));
                    if(increase){if(bp_v10_norm_below(np,norm_n,norm_outer))lo=mid+1u;else hi=mid;}
                    else {if(bp_v10_norm_above(np,norm_n,norm_outer))hi=mid-1u;else lo=mid;}
                }
                if(lo!=hi)continue;
                p=bp_v10_with_component(p,pivot,__uint_as_float(sign|lo));
            }
            const auto q=bp_v10_promote(p);
            if(!bp_v10_norm_admitted(q,norm_n,norm_outer))continue;
            const double a=q.s0.re-continuous.s0.re,b=q.s0.im-continuous.s0.im;
            const double c=q.s1.re-continuous.s1.re,d=q.s1.im-continuous.s1.im;
            const double error=__fma_rn(a,a,__fma_rn(b,b,__fma_rn(c,c,d*d)));
            if(error<best){best=error;result=p;found=true;}
        }
    }
    return found;
}
__device__ __forceinline__ bool bp_v10_round_norm(
    BiprismPsi64 continuous,double norm_n,double norm_outer,BiprismPsi32& result
) {
    double best=CUDART_INF; bool found=false;
    #pragma unroll 1
    for(uint32_t bits=0u;bits<16u;++bits){
        const BiprismPsi32 p={{bp_v10_bracket(continuous.s0.re,bits&1u),bp_v10_bracket(continuous.s0.im,bits&2u)},
                             {bp_v10_bracket(continuous.s1.re,bits&4u),bp_v10_bracket(continuous.s1.im,bits&8u)}};
        const auto q=bp_v10_promote(p);
        if(!bp_v10_norm_admitted(q,norm_n,norm_outer))continue;
        // Preserve all three admission bounds, but avoid accumulating norm
        // drift toward their intersection boundary across accepted substeps.
        // Only the initializer's ordering changes; the subsequent nonlinear
        // equation and work/energy checks still qualify the actual FP32 bits.
        const double error=fabs(bp_v10_spinor_norm(q)-1.);
        if(error<best){best=error;result=p;found=true;}
    }
    // Original16-corner enumeration supplies deterministic equal-score ties.
    return found||bp_v10_round_norm_shell1(continuous,norm_n,norm_outer,result)
        ||bp_v10_round_norm_interval(continuous,norm_n,norm_outer,result);
}
// Terminal state-only calls own all derived addresses and wide operands.
// rank_choice is dead until initial norm admission finishes; the per-row flag
// lets every lane enter the same call sites and the same re-anchoring barriers.
__device__ __forceinline__ bool bp_v10_round_norm_from_state(VariationalClusterDsmemV10* s) {
    const uint32_t u=threadIdx.x;if(u>=TILE_CORE_VERTICES)return true;
    bp_v10_candidates(s)[u]=bp_v10_reload_spinor(&s->psi_trial[u]);
    return bp_v10_norm_admitted(bp_v10_reload_spinor(&bp_v10_candidates(s)[u]),
        bp_v10_spinor_norm(bp_v10_start(s,u)),s->invocation.numerical->conservative_force[2][u]);
}
// A finite frozen-H proposal, never nonlinear admission. Prefer the largest
// equation margin among triple-norm-valid shell points, then distance to the
// continuous result, then the original bit enumeration order. Changing s1
// requires the caller to rebuild both DEC snapshots and the complete H/work.
__device__ __forceinline__ bool bp_v10_propose_equation_shell(
    BiprismPsi64 start,BiprismPsi64 continuous,BiprismPsi32 current,
    BiprismH64 H,double omega,double h,double norm_outer,double current_equation,
    volatile BiprismPsi32& result
) {
    if(!bp_ed_finite(start)||!bp_ed_finite(continuous)||!bp_ed_finite(current)
        ||!isfinite(H.h00)||!isfinite(H.h11)||!isfinite(H.h01_re)||!isfinite(H.h01_im)
        ||!isfinite(omega)||!(omega>0.)||!isfinite(h)||!(h>0.)||!isfinite(norm_outer)
        ||!isfinite(current_equation)||!(current_equation>double(VARIATIONAL_PICARD_TOL)))return false;
    const double norm_start=bp_v10_spinor_norm(start);
    double best_equation=CUDART_INF,best_distance=CUDART_INF;bool found=false;
    #pragma unroll 1
    for(uint32_t bits=0u;bits<256u;++bits){
        const BiprismPsi32 p={{bp_v10_shell1_bracket(continuous.s0.re,bits&3u),bp_v10_shell1_bracket(continuous.s0.im,(bits>>2u)&3u)},
            {bp_v10_shell1_bracket(continuous.s1.re,(bits>>4u)&3u),bp_v10_shell1_bracket(continuous.s1.im,(bits>>6u)&3u)}};
        if(__float_as_uint(p.s0.re)==__float_as_uint(current.s0.re)&&__float_as_uint(p.s0.im)==__float_as_uint(current.s0.im)
            &&__float_as_uint(p.s1.re)==__float_as_uint(current.s1.re)&&__float_as_uint(p.s1.im)==__float_as_uint(current.s1.im))continue;
        const auto q=bp_v10_promote(p);
        if(!bp_v10_norm_admitted(q,norm_start,norm_outer))continue;
        const double equation=bp_v10_electronic_equation(start,q,H,omega,h);
        if(!isfinite(equation)||equation>double(VARIATIONAL_PICARD_TOL)||!(equation<current_equation))continue;
        const double a=q.s0.re-continuous.s0.re,b=q.s0.im-continuous.s0.im;
        const double c=q.s1.re-continuous.s1.re,d=q.s1.im-continuous.s1.im;
        const double distance=__fma_rn(a,a,__fma_rn(b,b,__fma_rn(c,c,d*d)));
        if(!isfinite(distance))continue;
        if(equation<best_equation||(equation==best_equation&&distance<best_distance)){
            best_equation=equation;best_distance=distance;
            // A dead EQR slot owns the incumbent bits through the next point.
            // No reader observes this proposal before the all-eight found vote.
            result.s0.re=p.s0.re;result.s0.im=p.s0.im;
            result.s1.re=p.s1.re;result.s1.im=p.s1.im;found=true;
        }
    }
    return found;
}
// Only the globally selected thread calls this state-only helper. The EQR
// candidate field is dead proposal storage until the all-eight found vote.
// Saved native argmax/equation, H, phi and the current candidate stay untouched.
__device__ __noinline__ bool bp_v10_equation_proposal_from_state(VariationalClusterDsmemV10* s) {
    const uint32_t u=threadIdx.x;auto* ns=s->invocation.numerical;
    auto H=bp_v10_reload_h(&s->stage.electronic.H_energy[u]);H.h01_im+=ns->electronic_connection_im[u];
    BiprismElectronicCayleyResult<double> c;
    if(!bp_ed_hermitian_cayley(bp_v10_start(s,u),H,bp_v10_berry_mass(s->invocation.tile,u),s->invocation.collective.h,c)
        ||!bp_v10_norm_admitted(c.psi_np1,bp_v10_spinor_norm(bp_v10_start(s,u)),ns->conservative_force[2][u]))return false;
    const double eq=bp_v10_electronic_equation(bp_v10_start(s,u),c.psi_np1,H,bp_v10_berry_mass(s->invocation.tile,u),s->invocation.collective.h);
    if(!(eq<ns->rank_diagnostic.equation_refusal.equation_candidate))return false;
    auto& out=ns->rank_diagnostic.equation_refusal.candidate_spinor;
    out.s0.re=c.psi_np1.s0.re;out.s0.im=c.psi_np1.s0.im;out.s1.re=c.psi_np1.s1.re;out.s1.im=c.psi_np1.s1.im;
    return true;
}
__device__ __forceinline__ uint32_t bp_v10_matching_s0_labels(BiprismPsi64 continuous,BiprismPsi64 current) {
    uint32_t visited=0u;
    #pragma unroll 1
    for(uint32_t bits=0u;bits<4u;++bits){
        const float re=bp_v10_bracket(continuous.s0.re,bits&1u),im=bp_v10_bracket(continuous.s0.im,bits&2u);
        if(double(re)==current.s0.re&&double(im)==current.s0.im)visited|=1u<<bits;
    }
    return visited;
}
__device__ __forceinline__ double bp_v10_endpoint_charge(
    const VariationalClusterDsmemV10* s,const ContactShellSimplicialStateFfi* tile,
    const float* amber,uint32_t u
) {
    const double q=double(amber[UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u])]);
    return u<TILE_CORE_VERTICES?__fma_rn(double(.15f),bp_ed_norm2(bp_v10_reload_spinor(&bp_v10_candidates(s)[u]).s1),q):q;
}
__device__ __noinline__ uint32_t bp_v10_select_representation_candidate(
    VariationalClusterDsmemV10* s
) {
    const uint32_t tid=threadIdx.x;
    double score=tid<TILE_CORE_VERTICES?s->invocation.numerical->rank_choice.score[tid]:fabs(s->invocation.numerical->scalar[BP_NS_C1_GLOBAL_ERROR]);
    uint32_t index=tid<TILE_CORE_VERTICES?s->invocation.numerical->rank_choice.index[tid]:UINT32_MAX;
    const uint32_t lane=threadIdx.x&31u,warp=threadIdx.x>>5u;
    #pragma unroll
    for(int o=16;o;o>>=1){
        const double q=__shfl_xor_sync(FULL_WARP_MASK,score,o);
        const uint32_t j=__shfl_xor_sync(FULL_WARP_MASK,index,o);
        if(q<score||(q==score&&j<index)){score=q;index=j;}
    }
    if(lane==0){s->stage.electronic.warp_candidate_score[warp]=score;s->stage.electronic.warp_candidate_index[warp]=index;}
    __syncthreads();
    if(warp==0){
        double q=lane<8u?s->stage.electronic.warp_candidate_score[lane]:CUDART_INF;
        uint32_t j=lane<8u?s->stage.electronic.warp_candidate_index[lane]:UINT32_MAX;
        #pragma unroll
        for(int o=4;o;o>>=1){
            const double a=__shfl_xor_sync(FULL_WARP_MASK,q,o);
            const uint32_t b=__shfl_xor_sync(FULL_WARP_MASK,j,o);
            if(a<q||(a==q&&b<j)){q=a;j=b;}
        }
        if(lane==0){s->stage.electronic.warp_candidate_score[0]=q;s->stage.electronic.warp_candidate_index[0]=j;}
    }
    __syncthreads();const uint32_t winner=s->stage.electronic.warp_candidate_index[0];__syncthreads();return winner;
}
// Exact local ranking for a level0-only bracket change at fixed geometry and
// fixed endpoint level1 (hence fixed q_n,q_1,q_bar and every DEC solution).
// Endpoint energy coefficients use q_1, while reciprocal z uses midpoint H.
__device__ __forceinline__ bool bp_v10_rank_rounding_candidate(
    VariationalClusterDsmemV10* s,uint32_t i,BiprismPsi64 proposal,
    double& delta_energy,double& delta_work
) {
    auto& c=s->invocation.collective;auto& tile=s->invocation.tile;
    auto& amber=s->invocation.amber;auto& edges=s->invocation.edges;auto& h=c.h;
    auto& ns=s->invocation.numerical;
    bp_v10_spinor_owner(s).proposed_s0[i].re=proposal.s0.re;bp_v10_spinor_owner(s).proposed_s0[i].im=proposal.s0.im;
    {
        const auto p0=bp_v10_start(s,i),old=bp_v10_reload_spinor(&bp_v10_candidates(s)[i]),next=bp_v10_promote(proposal);
        if(!bp_v10_norm_admitted(next,bp_v10_spinor_norm(p0),ns->conservative_force[2][i]))return false;
        ns->reciprocal_force[0][i]=bp_ed_norm2(next.s0)-bp_ed_norm2(old.s0);
        ns->reciprocal_force[1][i]=bp_ed_density(next).r01_re-bp_ed_density(old).r01_re;
    }
    ns->rank_e00[i]=double(tile->star0_vol[i])*bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[i])).h_s_valence;
    ns->rank_metric.e01[i]=0.;ns->rank_metric.delta_h[i]=0.;ns->reciprocal_force[2][i]=0.;
    const auto* ordered=&c.streams->d_ordered_edges[c.leaf];
    #pragma unroll 1
    for(uint32_t cursor=ordered->offsets[i];cursor<ordered->offsets[i+1u];++cursor){
        const uint32_t e=ordered->edge_index[cursor];
        if(s->edge_u8[e][0]!=i)continue;
        const uint32_t v=s->edge_u8[e][1];
        BiprismElectronicEdgeProvider<double> p;
        if(!bp_ed_evaluate_edge(bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[i])),
            bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[v])),bp_v10_position(s,i),bp_v10_position(s,v),
            bp_v10_endpoint_charge(s,tile,amber,i),bp_v10_endpoint_charge(s,tile,amber,v),p))return false;
        const double D=fmax(1.-p.S*p.S,double(1e-3f));
        ns->rank_e00[i]+=p.E0/D;ns->rank_metric.e01[i]+=p.V01/D;
        const double chain=double(.15f)*p.gamma*ns->reciprocal_force[0][i]/(4.*D);
        ns->rank_metric.delta_h[i]+=chain;
        if(v<TILE_CORE_VERTICES){
            {
                const auto mid=bp_ed_endpoint_data(bp_v10_start(s,v),bp_v10_reload_spinor(&bp_v10_candidates(s)[v]),0.).psi_bar;
                const double dz=-chain*bp_ed_density(mid).r01_re/bp_v10_berry_mass(tile,v);
                ns->reciprocal_force[2][i]+=2.*h*c.streams->d_numerical_scratch[c.leaf]->electronic_connection_im[v]*dz;
            }
            // These arrays are immutable during rank. Do not carry the two
            // spinors and H through the exact FP64 dz division above.
            auto Hv=bp_v10_reload_h(&s->stage.electronic.H_energy[v]);Hv.h11-=chain;
            Hv.h01_im+=c.streams->d_numerical_scratch[c.leaf]->electronic_connection_im[v];
            if(bp_v10_electronic_equation(bp_v10_start(s,v),bp_v10_reload_spinor(&bp_v10_candidates(s)[v]),Hv,bp_v10_berry_mass(tile,v),h)>double(VARIATIONAL_PICARD_TOL))return false;
        }
    }
    const auto p0=bp_v10_reload_spinor(&bp_v10_starts(s)[i]),old=bp_v10_reload_spinor(&bp_v10_candidates(s)[i]);
    const volatile BiprismPsi64* current=&bp_v10_candidates(s)[i];
    const BiprismPsi64 next={{double(bp_v10_spinor_owner(s).proposed_s0[i].re),double(bp_v10_spinor_owner(s).proposed_s0[i].im)},
        {double(current->s1.re),double(current->s1.im)}};
    const double drho=ns->reciprocal_force[0][i],dcross=ns->reciprocal_force[1][i],omega=bp_v10_berry_mass(tile,i);
    const auto oldH=bp_v10_reload_h(&s->stage.electronic.H_energy[i]);auto newH=oldH;newH.h11+=ns->rank_metric.delta_h[i];
    const auto oldmid=bp_ed_endpoint_data(p0,old,0.).psi_bar,newmid=bp_ed_endpoint_data(p0,next,0.).psi_bar;
    const double dz=(bp_ed_connection_energy_pairing(newH,newmid)-bp_ed_connection_energy_pairing(oldH,oldmid))/omega;
    delta_energy=__fma_rn(ns->rank_e00[i],drho,2.*ns->rank_metric.e01[i]*dcross);
    delta_work=__fma_rn(2.*h*c.streams->d_numerical_scratch[c.leaf]->electronic_connection_im[i],dz,ns->reciprocal_force[2][i]);
    // A single global bracket change also changes the neighboring leaves' diagonal
    // generators. Their published response is evaluated at this same endpoint.
    const auto* b=bp_v10_bank(c,c.leaf);
    #pragma unroll 1
    for(uint32_t neighbor_offset=1u;neighbor_offset<=7u;neighbor_offset+=6u){
        const auto* neighbor=bp_v10_bank(c,(c.leaf+neighbor_offset)&7u);
        const double J=biprism_foliated::pair_weight(c.kappa,double(b->energy_weight[i]),double(neighbor->energy_weight[i]));
        delta_energy-=J*neighbor->trial_population[i]*drho;
        delta_work+=J*drho*neighbor->reciprocal_response[i];
    }
    newH.h01_im+=c.streams->d_numerical_scratch[c.leaf]->electronic_connection_im[i];
    return isfinite(delta_energy)&&isfinite(delta_work)
        &&bp_v10_electronic_equation(p0,next,newH,omega,h)<=double(VARIATIONAL_PICARD_TOL);
}
// Called only by a terminal, CTA-uniform refusal branch. Copy the same raw
// immutable H/spinors and saved evaluated scalars; CPU diagnostics derive the
// redundant norms/equations without adding an FP64 expression graph here.
__device__ __forceinline__ double bp_v10_diagnostic_component(BiprismPsi64 p,uint32_t component) {
    return component==0u?p.s0.re:(component==1u?p.s0.im:(component==2u?p.s1.re:p.s1.im));
}
__device__ __forceinline__ void bp_v10_capture_equation_refusal(VariationalClusterDsmemV10* s) {
    auto& c=s->invocation.collective;auto* ns=s->invocation.numerical;
    const uint32_t tid=threadIdx.x;
    __syncthreads();
    if(tid==0u){
        auto& out=ns->rank_diagnostic.equation_refusal;out.identity.magic=0u;
        const uint32_t vertex=out.identity.vertex;
        out.identity.version=3u;out.identity.leaf=c.leaf;out.identity.vertex=vertex;
        out.identity.substep=c.substep;out.identity.picard=s->invocation.picard;out.identity.sequence=c.sequence;
        out.identity.iteration=ns->rank_diagnostic.trace.round;out.identity.flags=isfinite(out.equation_candidate)?0u:1u;
        out.identity.packed_atom=s->invocation.tile->global_atom_idx[vertex];
        out.identity.accepted_at_entry=s->invocation.bank->footer.accepted_substeps_at_entry;out.identity.reserved=0u;
        const auto start=bp_v10_start(s,vertex),candidate=bp_v10_reload_spinor(&bp_v10_candidates(s)[vertex]);
        const auto trial=bp_v10_reload_spinor(&s->psi_trial[vertex]);
        #pragma unroll 1
        for(uint32_t i=0u;i<4u;++i){out.start[i]=bp_v10_diagnostic_component(start,i);out.candidate[i]=bp_v10_diagnostic_component(candidate,i);out.continuous[i]=bp_v10_diagnostic_component(trial,i);}
        const auto H=bp_v10_reload_h(&s->stage.electronic.H_energy[vertex]);
        out.h_energy[0]=H.h00;out.h_energy[1]=H.h11;out.h_energy[2]=H.h01_re;out.h_energy[3]=H.h01_im;
        out.connection_im=ns->electronic_connection_im[vertex];out.berry_mass=bp_v10_berry_mass(s->invocation.tile,vertex);out.h=c.h;
        out.norm_outer=ns->conservative_force[2][vertex];

        out.published_leaf_equation=ns->scalar[BP_NS_EQUATION];out.phi_mean=bp_v10_phi(s)[vertex];out.charge_mean=s->q_live[vertex];
        out.reserved[0]=0u;out.reserved[1]=0u;out.identity.magic=BP_V10_EQUATION_REFUSAL_TAG;
    }
    __syncthreads();
}
// The cap-refusal call precedes the original final psi_trial=next assignment.
// Thus the retained trial, Cayley result and H belong to the same actual
// iteration. This terminal helper then performs the original final update and
// CTA publication; the caller can return false without loop-continuation state.
__device__ __forceinline__ void bp_v10_capture_continuous_refusal(VariationalClusterDsmemV10* s) {
    auto& c=s->invocation.collective;auto* ns=s->invocation.numerical;
    __syncthreads();
    if(threadIdx.x==0u){
        uint32_t vertex=0u;
        #pragma unroll 1
        for(uint32_t i=1u;i<TILE_CORE_VERTICES;++i)if(ns->rank_choice.score[i]>ns->rank_choice.score[vertex])vertex=i;
        auto& out=ns->continuous_diagnostic.refusal;out.identity.magic=0u;
        out.identity.version=3u;out.identity.leaf=c.leaf;out.identity.vertex=vertex;
        out.identity.substep=c.substep;out.identity.picard=s->invocation.picard;out.identity.sequence=c.sequence;
        const uint32_t iteration=ns->continuous_diagnostic.history.count-1u;
        out.identity.iteration=iteration;out.identity.flags=0u;out.identity.packed_atom=s->invocation.tile->global_atom_idx[vertex];
        out.identity.accepted_at_entry=s->invocation.bank->footer.accepted_substeps_at_entry;out.identity.reserved=0u;
        const auto start=bp_v10_start(s,vertex),trial=bp_v10_reload_spinor(&s->psi_trial[vertex]);
        const BiprismPsi64 next={{ns->conservative_force[0][vertex],ns->conservative_force[1][vertex]},
            {ns->conservative_force[2][vertex],ns->reciprocal_force[0][vertex]}};
        #pragma unroll 1
        for(uint32_t i=0u;i<4u;++i){out.start[i]=bp_v10_diagnostic_component(start,i);out.trial[i]=bp_v10_diagnostic_component(trial,i);out.next[i]=bp_v10_diagnostic_component(next,i);}
        const auto H=bp_v10_reload_h(&s->stage.electronic.H_energy[vertex]);
        out.h_energy[0]=H.h00;out.h_energy[1]=H.h11;out.h_energy[2]=H.h01_re;out.h_energy[3]=H.h01_im;
        out.connection_im=ns->electronic_connection_im[vertex];out.berry_mass=bp_v10_berry_mass(s->invocation.tile,vertex);out.h=c.h;
        out.reserved[2]=0u;
        out.selected_change=ns->rank_choice.score[vertex];out.equation_trial=0.;out.equation_next_fixed_h=0.;
        out.published_leaf_change=ns->continuous_diagnostic.history.iteration[iteration].max_change;
        out.published_leaf_equation=ns->continuous_diagnostic.history.iteration[iteration].max_equation;
        out.phi_mean=bp_v10_phi(s)[vertex];out.reserved[0]=0u;out.reserved[1]=0u;out.identity.magic=BP_V10_CONTINUOUS_REFUSAL_TAG;
    }
    __syncthreads();
    const uint32_t tid=threadIdx.x;
    if(tid<TILE_CORE_VERTICES)s->psi_trial[tid]={{ns->conservative_force[0][tid],ns->conservative_force[1][tid]},
        {ns->conservative_force[2][tid],ns->reciprocal_force[0][tid]}};
    __syncthreads();
}
// Keep the unchanged Cayley calculation in its own compiler allocation scope.
// Only the state pointer crosses the device-call boundary; every input is
// reloaded here and all four exact outputs use their existing C1 owners.
// Resource acceptance includes this reachable function and the caller stack.
__device__ __noinline__ bool bp_v10_cayley_vertex_stage(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;auto& ns=s->invocation.numerical;
    auto H=bp_v10_reload_h(&s->stage.electronic.H_energy[tid]);H.h01_im+=ns->electronic_connection_im[tid];
    BiprismElectronicCayleyResult<double> result;
    const bool valid=bp_ed_hermitian_cayley(bp_v10_start(s,tid),H,bp_v10_berry_mass(s->invocation.tile,tid),s->invocation.collective.h,result);
    ns->conservative_force[0][tid]=result.psi_np1.s0.re;
    ns->conservative_force[1][tid]=result.psi_np1.s0.im;
    ns->conservative_force[2][tid]=result.psi_np1.s1.re;
    ns->reciprocal_force[0][tid]=result.psi_np1.s1.im;
    return valid&&result.norm_defect<=double(VARIATIONAL_PICARD_TOL);
}
// Every decision below is collective. Different leaf DEC iteration counts are
// permitted inside the CTA solver; no leaf leaves this coupled loop alone.
__device__ __forceinline__ bool bp_v10_solve_electronic_trial(
    VariationalClusterDsmemV10* s
) {



    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    if(tid==0u){
        s->invocation.numerical->continuous_diagnostic.refusal.identity.magic=0u;
        s->invocation.numerical->continuous_diagnostic.history.magic=BP_V10_CONTINUOUS_HISTORY_TAG;
        s->invocation.numerical->continuous_diagnostic.history.version=2u;s->invocation.numerical->continuous_diagnostic.history.count=0u;s->invocation.numerical->continuous_diagnostic.history.valid_mask=0u;
    }
    __syncthreads();
    #pragma unroll 1
    for(uint32_t k=0u;k<MAX_CONTINUOUS_ELECTRONIC_ITERS;++k){
        bp_v10_publish_charge(s,double(bp_v10_readonly_float(s->invocation.amber+UNPACK_GLOBAL_ATOM_ID(s->invocation.tile->global_atom_idx[tid]))),2u,false);
        float dr=0.;uint32_t di=0u;
        const bool dec=solve_live_dec_poisson_pcg_v10(s,s->invocation.tile,s->invocation.edges,bp_v10_bank(s->invocation.collective,s->invocation.collective.leaf)->live_weights,dr,di);
        // Retain DEC diagnostics at their last use, before the provider/Cayley
        // stage. Neither result needs a register live across that stage.
        if(tid==0u){
            s->invocation.numerical->continuous_diagnostic.history.iteration[k].dec_residual=dr;
            s->invocation.numerical->continuous_diagnostic.history.iteration[k].dec_iterations=di;
        }
        __syncthreads();
        const bool built=bp_v10_build_electronic_h(s,false);
        double change=0.,equation=0.;bool valid=dec&&built;
        BP_V10_BARRIER_REANCHOR(s);
        if(core)valid=bp_v10_cayley_vertex_stage(s)&&valid;
        BP_V10_BARRIER_REANCHOR(s);
        if(core){
            {
                const auto trial=bp_v10_reload_spinor(&s->psi_trial[tid]);
                const BiprismElectronicSpinor<double> next={{s->invocation.numerical->conservative_force[0][tid],s->invocation.numerical->conservative_force[1][tid]},
                    {s->invocation.numerical->conservative_force[2][tid],s->invocation.numerical->reciprocal_force[0][tid]}};
                const auto d0=bp_ed_sub(next.s0,trial.s0),d1=bp_ed_sub(next.s1,trial.s1);
                change=fmax(fmax(fabs(d0.re),fabs(d0.im)),fmax(fabs(d1.re),fabs(d1.im)));
            }
            {
                // n, trial, H and the connection are immutable until both
                // collective votes complete. Reload their exact stored values.
                auto H=bp_v10_reload_h(&s->stage.electronic.H_energy[tid]);H.h01_im+=s->invocation.numerical->electronic_connection_im[tid];
                equation=bp_v10_electronic_equation(bp_v10_start(s,tid),bp_v10_reload_spinor(&s->psi_trial[tid]),H,bp_v10_berry_mass(s->invocation.tile,tid),s->invocation.collective.h);
            }
            valid=valid&&isfinite(change)&&isfinite(equation);
        }
        valid=__syncthreads_and(valid);
        if(core)s->invocation.numerical->rank_choice.score[tid]=change;
        const double max_change=bp_v10_max(change,s->warp_red_f64);
        const double max_equation=bp_v10_max(equation,s->warp_red_f64);
        if(tid==0u){
            auto& history=s->invocation.numerical->continuous_diagnostic.history;
            history.iteration[k].max_change=max_change;history.iteration[k].max_equation=max_equation;
            history.count=k+1u;if(valid)history.valid_mask|=1u<<k;
        }
        const auto collective=bp_v10_collective_values(s,s->invocation.collective,0.,0.,fmax(max_change,max_equation),valid);
        if(!collective.valid)return false;
        // This stringent fixed-point target is an internal convergence target;
        // the true represented equation is independently admitted at1e-7 below.
        const auto target=bp_v10_collective_values(s,s->invocation.collective,0.,0.,max_change,valid);
        if(!target.valid)return false;
        if(k>0u&&target.max0<=1e-12&&collective.max0<=double(VARIATIONAL_PICARD_TOL))return true;
        if(k+1u==MAX_CONTINUOUS_ELECTRONIC_ITERS){bp_v10_capture_continuous_refusal(s);return false;}
        // Publication and remote-consumption barriers have finished before updates.
        if(core)s->psi_trial[tid]={{s->invocation.numerical->conservative_force[0][tid],s->invocation.numerical->conservative_force[1][tid]},
            {s->invocation.numerical->conservative_force[2][tid],s->invocation.numerical->reciprocal_force[0][tid]}};
        __syncthreads();
    }
    return false;
}
// Minimize the largest of the three PREDICTED actual energy defects:
// epsilon+d_e, epsilon+d_H, and A+epsilon+d_H. Their equal-tolerance
// intervals have this midpoint, even if the predicted intersection is empty.
// Offsets can change with the next represented state; only the subsequent full
// C4 equation/work/energy gates establish acceptance.
__host__ __device__ __forceinline__ double bp_v10_joint_electronic_target(
    double prior_outer_error,double epsilon_offset,double energy_offset
) {
    if(!isfinite(prior_outer_error)||!isfinite(epsilon_offset)||!isfinite(energy_offset))return nan("");
    const double cumulative_offset=prior_outer_error+energy_offset;
    return -.5*(fmin(epsilon_offset,fmin(energy_offset,cumulative_offset))
        +fmax(epsilon_offset,fmax(energy_offset,cumulative_offset)));
}
// Endpoint and midpoint potentials depend on rho11. A represented s1 proposal
// invalidates both cached core solutions. Rebuild them with the unchanged
// fixed halo, actual charge tree, live weights, PCG cap and true-residual gate.
// The start potential/energy and the authoritative potential owner are untouched.
__device__ __forceinline__ bool bp_v10_refresh_represented_phi(VariationalClusterDsmemV10* s) {

    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    bp_v10_publish_charge(s,double(bp_v10_readonly_float(s->invocation.amber+UNPACK_GLOBAL_ATOM_ID(s->invocation.tile->global_atom_idx[tid]))),1u,true);
    float residual=0.;uint32_t iterations=0u;
    bool valid=solve_live_dec_poisson_pcg_v10(s,s->invocation.tile,s->invocation.edges,bp_v10_bank(s->invocation.collective,s->invocation.collective.leaf)->live_weights,residual,iterations);
    if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,0.,valid).valid){if(tid==0u)s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_ENDPOINT_DEC_REFUSAL;__syncthreads();return false;}
    if(core)s->invocation.numerical->conservative_force[0][tid]=double(bp_v10_phi(s)[tid]);
    bp_v10_publish_charge(s,double(bp_v10_readonly_float(s->invocation.amber+UNPACK_GLOBAL_ATOM_ID(s->invocation.tile->global_atom_idx[tid]))),2u,true);
    valid=solve_live_dec_poisson_pcg_v10(s,s->invocation.tile,s->invocation.edges,bp_v10_bank(s->invocation.collective,s->invocation.collective.leaf)->live_weights,residual,iterations);
    if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,0.,valid).valid){if(tid==0u)s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_MEAN_DEC_REFUSAL;__syncthreads();return false;}
    if(core)s->invocation.numerical->conservative_force[1][tid]=double(bp_v10_phi(s)[tid]);
    if(tid==0u)s->invocation.represented_phi_dirty=0u;
    __syncthreads();return true;
}
// One globally selected offending row, with deterministic leaf/vertex ties.
// Other leaves vote on the selected row's success; they need no local proposal.
// A successful vote only permits an attempted scratch update. The next round
// rebuilds all-eight H/population/work before any rank or acceptance decision.
__device__ __forceinline__ void bp_v10_repair_represented_equation(VariationalClusterDsmemV10* s) {

    const uint32_t tid=threadIdx.x;
    if(tid==0u)s->invocation.equation_repair_valid=0u;
    __syncthreads();
    if(s->invocation.equation_repairs>=BP_V10_EQUATION_MAX_REPAIRS)return;
    const uint32_t winner=bp_v10_collective_candidate(s,s->invocation.collective,-s->invocation.numerical->scalar[BP_NS_EQUATION],4u*s->invocation.numerical->rank_diagnostic.equation_refusal.identity.vertex);
    if(tid==0u)s->invocation.equation_winner=winner;
    __syncthreads();
    // Reload the exact published index after the proposal and found vote;
    // derived leaf/vertex addresses do not survive either call boundary.
    bool found=s->invocation.equation_winner<8u*4u*TILE_CORE_VERTICES&&(s->invocation.equation_winner&3u)==0u;
    BP_V10_BARRIER_REANCHOR(s);
    if(found&&s->invocation.collective.leaf==s->invocation.equation_winner/(4u*TILE_CORE_VERTICES)
        &&tid==(s->invocation.equation_winner%(4u*TILE_CORE_VERTICES))/4u)found=bp_v10_equation_proposal_from_state(s);
    BP_V10_BARRIER_REANCHOR(s);
    found=__syncthreads_and(found);
    // Preserve the native equation operand in the footer if this terminal
    // proposal fails; EQR2 can still bind its unchanged candidate and H.
    if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,s->invocation.numerical->scalar[BP_NS_EQUATION],found&&s->local_fault_accum==0u).valid)return;
    if(s->invocation.collective.leaf==s->invocation.equation_winner/(4u*TILE_CORE_VERTICES)
        &&tid==(s->invocation.equation_winner%(4u*TILE_CORE_VERTICES))/4u){
        const auto& proposal=s->invocation.numerical->rank_diagnostic.equation_refusal;
        bp_v10_candidates(s)[tid]={{proposal.candidate[0],proposal.candidate[1]},{proposal.candidate[2],proposal.candidate[3]}};
        s->invocation.numerical->rank_search.visited_corners[tid]=bp_v10_matching_s0_labels(s->psi_trial[tid],bp_v10_candidates(s)[tid]);
    }
    if(tid==0u){
        ++s->invocation.equation_repairs;
        const uint32_t round=s->invocation.numerical->rank_diagnostic.trace.round,slot=round==0u?0u:1u+((round-1u)&15u);
        s->invocation.numerical->rank_diagnostic.trace.winner[slot]=BP_V10_EQUATION_REPAIR_WINNER_BIT|s->invocation.equation_winner;
        // No energy-rank prediction exists for a charge-changing proposal.
        s->invocation.numerical->rank_diagnostic.trace.predicted_local_best[slot]=CUDART_NAN;
        s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_EQUATION_REBUILD;
        s->invocation.represented_phi_dirty=1u;
        s->invocation.equation_repair_valid=1u;
    }
    // Publish the candidate and dirty state. The one refresh site at the next
    // loop head rebuilds both solutions before H, work or admission can run.
    __syncthreads();
}
// Select actual FP64 endpoints; tile FP32 fields are projections only. Norm is checked against one, the substep n and
// outer-entry state. A single global bracket update is selected
// each round, preventing simultaneous-neighbor quadratic cross terms.
// Between equation repairs each vertex visits each original s0 label once.
// A fallback initializer may be outside those labels, so its mask may be zero;
// at most4*8*192 original-label updates initially remain. Each of at most four
// single-row equation repairs resets only that row's mask, adding at most four
// labels and one repair update; the total ceiling is6164. This is a
// bounded greedy admission search: exhaustion does not prove that no other
// joint represented state is feasible. Every selected state is re-evaluated
// with the unchanged full nonlinear equation, norm and global energy gates.
__device__ __forceinline__ bool bp_v10_admit_represented_electronic(
    VariationalClusterDsmemV10* s
) {



    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    if(core)s->invocation.numerical->conservative_force[2][tid]=bp_v10_outer_norm(s,tid);
    if(tid<17u){
        s->invocation.numerical->rank_diagnostic.trace.evaluated_error[tid]=CUDART_NAN;
        s->invocation.numerical->rank_diagnostic.trace.predicted_local_best[tid]=CUDART_NAN;
        s->invocation.numerical->rank_diagnostic.trace.winner[tid]=UINT32_MAX;
        s->invocation.numerical->rank_diagnostic.trace.evaluation_round[tid]=UINT32_MAX;
    }
    if(tid==0u){
        s->invocation.equation_repairs=0u;
        s->invocation.representation_round=0u;
        s->invocation.represented_phi_dirty=1u;

        s->invocation.electronic_target.value=(s->invocation.integration_mode&GonzalezExactEnergyBit)
            ?bp_v10_joint_electronic_target(s->invocation.numerical->scalar[BP_NS_OUTER_ERROR],s->invocation.electronic_target.epsilon_offset,s->invocation.electronic_target.energy_offset)
            :-s->invocation.numerical->scalar[BP_NS_OUTER_ERROR];
        // This is distance to a search target, not the actual electronic defect.
        s->invocation.numerical->scalar[BP_NS_C1_SEARCH_BIAS]=-s->invocation.electronic_target.value;
        s->invocation.numerical->rank_diagnostic.trace.magic=BP_V10_REPRESENTATION_TRACE_TAG;
        s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_ENTER;
        s->invocation.numerical->rank_diagnostic.trace.count=0u;s->invocation.numerical->rank_diagnostic.trace.round=UINT32_MAX;
        s->invocation.numerical->rank_diagnostic.equation_refusal.identity.magic=0u;
    }
    __syncthreads();
    bool valid=true;
    valid=bp_v10_round_norm_from_state(s);
    const bool norm_admitted=valid;
    valid=__syncthreads_and(valid);
    if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,0.,valid).valid){
        // The failed initial norm vote is terminal for this substep. Neither
        // the start-energy evaluation nor C2/C3/C4 can consume these aliases.
        // Retain the ACTUAL trial and targets; do not change a candidate or gate.
        if(core){
            const auto p=bp_v10_reload_spinor(&s->psi_trial[tid]);
            const auto n=bp_v10_start(s,tid);
            const double outer=s->invocation.numerical->conservative_force[2][tid];
            s->invocation.numerical->conservative_force[0][tid]=p.s0.re;
            s->invocation.numerical->conservative_force[1][tid]=p.s0.im;
            s->invocation.numerical->conservative_force[2][tid]=p.s1.re;
            s->invocation.numerical->reciprocal_force[0][tid]=p.s1.im;
            s->invocation.numerical->reciprocal_force[1][tid]=bp_v10_spinor_norm(n);
            s->invocation.numerical->reciprocal_force[2][tid]=outer;
            // NRF4 start remains exact in the dedicated BPS1 substep_start plane.
            // NRF4 start remains exact in the dedicated BPS1 substep_start plane.
            // NRF4 start remains exact in the dedicated BPS1 substep_start plane.
            // NRF4 start remains exact in the dedicated BPS1 substep_start plane.
            s->invocation.numerical->rank_choice.index[tid]=norm_admitted?1u:0u;
        }
        if(tid==0u){
            s->invocation.numerical->scalar[BP_NS_C1_START_ENERGY]=double(BP_V10_NORM_REFUSAL_TAG);
            s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_NORM_REFUSAL;
        }
        __syncthreads();return false;
    }
    if(core){
        // Mark every duplicate original label when the initial s0 matches.
        // A fallback result may match none; zero is valid and the
        //4*8*192 update bound includes all four still-unvisited labels.
        s->invocation.numerical->rank_search.visited_corners[tid]=bp_v10_matching_s0_labels(s->psi_trial[tid],bp_v10_candidates(s)[tid]);
    }
    bp_v10_publish_charge(s,double(bp_v10_readonly_float(s->invocation.amber+UNPACK_GLOBAL_ATOM_ID(s->invocation.tile->global_atom_idx[tid]))),0u,true);
    float r=0.;uint32_t it=0u;
    valid=solve_live_dec_poisson_pcg_v10(s,s->invocation.tile,s->invocation.edges,bp_v10_bank(s->invocation.collective,s->invocation.collective.leaf)->live_weights,r,it);
    if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,0.,valid).valid){if(tid==0u)s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_START_DEC_REFUSAL;return false;}
    {if(threadIdx.x==0u)s->invocation.energy_bank=0u;
        BP_V10_BARRIER_REANCHOR(s);
        evaluate_total_potential_energy_v10(s);
        BP_V10_BARRIER_REANCHOR(s);
     if(tid==0u)s->invocation.numerical->scalar[BP_NS_C1_START_ENERGY]=s->invocation.potential_result;}
    __syncthreads();
    #pragma unroll 1
    for(;s->invocation.representation_round<=BP_V10_REPRESENTATION_TOTAL_UPDATES;){
        // Initial rounding and each charge-changing repair share this exact
        // endpoint/midpoint DEC sequence. An s0-only rank move keeps both valid.
        if(s->invocation.represented_phi_dirty){
            if(!bp_v10_refresh_represented_phi(s))return false;
        }
        valid=bp_v10_build_electronic_h(s,true);
        if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,0.,valid).valid){if(tid==0u){s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_H_REFUSAL;s->invocation.numerical->rank_diagnostic.trace.round=s->invocation.representation_round;}return false;}
        // The previous rank's own-row operands are dead here. Materialize the
        // four completed scalars before their reductions; no FP64 spinor or H
        // remains live across another consumer's exact division/equation call.
        // All four reductions finish before potential/ranking reuse these rows.
        if(core){
            {
                const auto mid=bp_ed_endpoint_data(bp_v10_start(s,tid),bp_v10_reload_spinor(&bp_v10_candidates(s)[tid]),0.).psi_bar;
                const auto H=bp_v10_reload_h(&s->stage.electronic.H_energy[tid]);
                const double z=bp_ed_connection_energy_pairing(H,mid)/bp_v10_berry_mass(s->invocation.tile,tid);
                s->invocation.numerical->reciprocal_force[0][tid]=2.*s->invocation.collective.h*s->invocation.numerical->electronic_connection_im[tid]*z;
            }
            {
                const auto mid=bp_ed_endpoint_data(bp_v10_start(s,tid),bp_v10_reload_spinor(&bp_v10_candidates(s)[tid]),0.).psi_bar;
                bp_v10_bank(s->invocation.collective,s->invocation.collective.leaf)->reciprocal_response[tid]=2.*s->invocation.collective.h*s->invocation.numerical->electronic_connection_im[tid]*bp_ed_density(mid).r01_re/bp_v10_berry_mass(s->invocation.tile,tid);
            }
            {
                auto total=bp_v10_reload_h(&s->stage.electronic.H_energy[tid]);total.h01_im+=s->invocation.numerical->electronic_connection_im[tid];
                s->invocation.numerical->reciprocal_force[1][tid]=bp_v10_electronic_equation(bp_v10_start(s,tid),bp_v10_reload_spinor(&bp_v10_candidates(s)[tid]),total,bp_v10_berry_mass(s->invocation.tile,tid),s->invocation.collective.h);
            }
            {
                const auto fol=bp_v10_foliation(s->invocation.collective,tid);
                s->invocation.numerical->reciprocal_force[2][tid]=fol.owned_energy_delta;s->invocation.numerical->rank_e00[tid]=fol.discrete_work;
            }
            valid=valid&&bp_v10_norm_admitted(bp_v10_reload_spinor(&bp_v10_candidates(s)[tid]),bp_v10_spinor_norm(bp_v10_start(s,tid)),s->invocation.numerical->conservative_force[2][tid])
                &&isfinite(s->invocation.numerical->reciprocal_force[0][tid])&&isfinite(s->invocation.numerical->reciprocal_force[1][tid]);
        }
        valid=__syncthreads_and(valid);
        {const double value=bp_v10_sum(core?s->invocation.numerical->reciprocal_force[0][tid]:0.,s->warp_red_f64);if(tid==0u)s->invocation.numerical->scalar[BP_NS_WORK_BERRY]=value;}
        {const double value=bp_v10_sum(core?s->invocation.numerical->reciprocal_force[2][tid]:0.,s->warp_red_f64);if(tid==0u)s->invocation.numerical->scalar[BP_NS_PAIR_DELTA]=value;}
        {const double value=bp_v10_sum(core?s->invocation.numerical->rank_e00[tid]:0.,s->warp_red_f64);if(tid==0u)s->invocation.numerical->scalar[BP_NS_FOL_WORK]=value;}
        {const double value=bp_v10_max(core?s->invocation.numerical->reciprocal_force[1][tid]:0.,s->warp_red_f64);if(tid==0u){s->invocation.numerical->scalar[BP_NS_EQUATION]=value;s->invocation.numerical->scalar[BP_NS_ATTEMPT_EQUATION]=value;}}
        // These are the actual evaluated equation rows. Save their argmax and
        // value before V1 reuses rows0..63; this dead diagnostic record cannot
        // affect the solver or any admission input. The existing CTA barrier
        // completes this reader before the first potential overwrite.
        if(tid==0u){
            uint32_t vertex=0u;
            #pragma unroll 1
            for(uint32_t i=1u;i<TILE_CORE_VERTICES;++i){
                const double previous=s->invocation.numerical->reciprocal_force[1][vertex],value=s->invocation.numerical->reciprocal_force[1][i];
                if(isfinite(previous)&&(!isfinite(value)||value>previous))vertex=i;
            }
            s->invocation.numerical->rank_diagnostic.equation_refusal.identity.vertex=vertex;
            s->invocation.numerical->rank_diagnostic.equation_refusal.equation_candidate=s->invocation.numerical->reciprocal_force[1][vertex];
        }
        __syncthreads();
        bp_v10_publish_charge(s,double(bp_v10_readonly_float(s->invocation.amber+UNPACK_GLOBAL_ATOM_ID(s->invocation.tile->global_atom_idx[tid]))),1u,true);bp_v10_phi(s)[tid]=core?s->invocation.numerical->conservative_force[0][tid]:s->invocation.input_phi[tid];__syncthreads();
        if(threadIdx.x==0u)s->invocation.energy_bank=1u;
        BP_V10_BARRIER_REANCHOR(s);
        evaluate_total_potential_energy_v10(s);
        BP_V10_BARRIER_REANCHOR(s);
        bp_v10_publish_charge(s,double(bp_v10_readonly_float(s->invocation.amber+UNPACK_GLOBAL_ATOM_ID(s->invocation.tile->global_atom_idx[tid]))),2u,true);bp_v10_phi(s)[tid]=core?s->invocation.numerical->conservative_force[1][tid]:s->invocation.input_phi[tid];__syncthreads();
        if(tid==0u)s->invocation.numerical->scalar[BP_NS_ELECTRONIC_DELTA]=s->invocation.potential_result-s->invocation.numerical->scalar[BP_NS_C1_START_ENERGY];
        __syncthreads();
        // The same local and global work operands remain in named replay slots;
        // no bookkeeping value must stay in a register through provider traversal.
        const auto collective=bp_v10_collective_values(s,s->invocation.collective,s->invocation.numerical->scalar[BP_NS_ELECTRONIC_DELTA]+s->invocation.numerical->scalar[BP_NS_WORK_BERRY]+s->invocation.numerical->scalar[BP_NS_PAIR_DELTA],
            s->invocation.numerical->scalar[BP_NS_PAIR_DELTA]-s->invocation.numerical->scalar[BP_NS_FOL_WORK],s->invocation.numerical->scalar[BP_NS_EQUATION],valid&&isfinite(s->invocation.numerical->scalar[BP_NS_ELECTRONIC_DELTA]));
        if(tid==0u){
            s->invocation.electronic_target.raw_epsilon=collective.sum0;
            s->invocation.numerical->scalar[BP_NS_C1_GLOBAL_ERROR]=collective.sum0+s->invocation.numerical->scalar[BP_NS_C1_SEARCH_BIAS];
            const uint32_t slot=s->invocation.representation_round==0u?0u:1u+((s->invocation.representation_round-1u)&15u);
            s->invocation.numerical->rank_diagnostic.trace.evaluated_error[slot]=s->invocation.numerical->scalar[BP_NS_C1_GLOBAL_ERROR];
            s->invocation.numerical->rank_diagnostic.trace.predicted_local_best[slot]=CUDART_NAN;
            s->invocation.numerical->rank_diagnostic.trace.winner[slot]=UINT32_MAX;
            s->invocation.numerical->rank_diagnostic.trace.evaluation_round[slot]=s->invocation.representation_round;
            s->invocation.numerical->rank_diagnostic.trace.round=s->invocation.representation_round;s->invocation.numerical->rank_diagnostic.trace.count=min(s->invocation.representation_round+1u,17u);
            s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_EVALUATED;
        }
        __syncthreads();
        if(!collective.valid||!isfinite(s->invocation.numerical->scalar[BP_NS_C1_GLOBAL_ERROR])||fabs(collective.sum1)>double(VARIATIONAL_ENERGY_TOL)){
            if(tid==0u)s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_EQUATION_WORK_REFUSAL;
            bp_v10_capture_equation_refusal(s);return false;
        }
        if(collective.max0>double(VARIATIONAL_PICARD_TOL)){
            if(tid==0u)s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_EQUATION_WORK_REFUSAL;
            __syncthreads();
            bp_v10_repair_represented_equation(s);
            if(!s->invocation.equation_repair_valid){
                if(s->invocation.numerical->rank_diagnostic.trace.stage==BP_REPR_EQUATION_WORK_REFUSAL)bp_v10_capture_equation_refusal(s);
                return false;
            }
            BP_V10_BARRIER_REANCHOR(s);
            if(threadIdx.x==0u)++s->invocation.representation_round;
            BP_V10_BARRIER_REANCHOR(s);
            continue;
        }
        if(fabs(s->invocation.numerical->scalar[BP_NS_C1_GLOBAL_ERROR])<=1e-8){if(tid==0u)s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_ACCEPTED;return true;}
        if(core){s->invocation.numerical->rank_choice.score[tid]=fabs(s->invocation.numerical->scalar[BP_NS_C1_GLOBAL_ERROR]);s->invocation.numerical->rank_choice.index[tid]=UINT32_MAX;}
        if(core&&s->invocation.representation_round-s->invocation.equation_repairs<BP_V10_REPRESENTATION_MAX_UPDATES+4u*s->invocation.equation_repairs){
            #pragma unroll 1
            for(uint32_t bits=0u;bits<4u;++bits){
                if(s->invocation.numerical->rank_search.visited_corners[tid]&(1u<<bits))continue;
                auto proposal=bp_v10_candidates(s)[tid];
                const volatile BiprismPsi64* continuous=&s->psi_trial[tid];
                proposal.s0={bp_v10_bracket(continuous->s0.re,bits&1u),bp_v10_bracket(continuous->s0.im,bits&2u)};
                double de=0.,dw=0.;
                if(!bp_v10_rank_rounding_candidate(s,tid,proposal,de,dw))continue;
                const double candidate=fabs(s->invocation.numerical->scalar[BP_NS_C1_GLOBAL_ERROR]+de+dw);
                if(candidate<s->invocation.numerical->rank_choice.score[tid]){s->invocation.numerical->rank_choice.score[tid]=candidate;s->invocation.numerical->rank_choice.index[tid]=4u*tid+bits;}
            }
        }
        BP_V10_BARRIER_REANCHOR(s);
        const uint32_t local=bp_v10_select_representation_candidate(s);
        BP_V10_BARRIER_REANCHOR(s);
        const double best=s->stage.electronic.warp_candidate_score[0];
        const uint32_t winner=bp_v10_collective_candidate(s,s->invocation.collective,best,local);
        if(tid==0u){
            const uint32_t slot=s->invocation.representation_round==0u?0u:1u+((s->invocation.representation_round-1u)&15u);
            s->invocation.numerical->rank_diagnostic.trace.predicted_local_best[slot]=best;
            s->invocation.numerical->rank_diagnostic.trace.winner[slot]=winner;
        }
        if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,0.,s->local_fault_accum==0u).valid){if(tid==0u)s->invocation.numerical->rank_diagnostic.trace.stage=BP_REPR_COLLECTIVE_REFUSAL;return false;}
        if(winner==UINT32_MAX||s->invocation.representation_round-s->invocation.equation_repairs>=BP_V10_REPRESENTATION_MAX_UPDATES+4u*s->invocation.equation_repairs){
            const bool admitted=fabs(s->invocation.numerical->scalar[BP_NS_C1_GLOBAL_ERROR])<=double(VARIATIONAL_ENERGY_TOL);
            if(tid==0u)s->invocation.numerical->rank_diagnostic.trace.stage=admitted?BP_REPR_ACCEPTED:(s->invocation.representation_round-s->invocation.equation_repairs>=BP_V10_REPRESENTATION_MAX_UPDATES+4u*s->invocation.equation_repairs?BP_REPR_ROUND_LIMIT:BP_REPR_NO_IMPROVEMENT);
            return admitted;
        }
        const uint32_t leaf=winner/(4u*TILE_CORE_VERTICES),within=winner%(4u*TILE_CORE_VERTICES);
        if(core&&s->invocation.collective.leaf==leaf&&tid==within/4u){
            const auto continuous=s->psi_trial[tid];const uint32_t bits=within&3u;
            bp_v10_candidates(s)[tid].s0={bp_v10_bracket(continuous.s0.re,bits&1u),bp_v10_bracket(continuous.s0.im,bits&2u)};
            s->invocation.numerical->rank_search.visited_corners[tid]|=1u<<bits;
        }
        BP_V10_BARRIER_REANCHOR(s);
        if(threadIdx.x==0u)++s->invocation.representation_round;
        BP_V10_BARRIER_REANCHOR(s);
    }
    return false;
}

__device__ __forceinline__ double bp_v10_dot3(BiprismVec64 a,BiprismVec64 b) {return bp_ed_dot(a,b);}
__device__ __forceinline__ BiprismVec64 bp_v10_sym_mul(const sym3_f64& g,BiprismVec64 v) {
    return {__fma_rn(double(g.xx),v.x,__fma_rn(double(g.xy),v.y,double(g.xz)*v.z)),
            __fma_rn(double(g.xy),v.x,__fma_rn(double(g.yy),v.y,double(g.yz)*v.z)),
            __fma_rn(double(g.xz),v.x,__fma_rn(double(g.yz),v.y,double(g.zz)*v.z))};
}
// G is the ordered FP64 covariance and is the sole metric
// authority. Its original SPD admission still runs at each assembly. Solve
// that same matrix in FP64; do not multiply by an independently rounded inverse.
// No denominator floor changes the admitted linear system.
__device__ __forceinline__ BiprismVec64 bp_v10_metric_solve_f64(const sym3_f64& g,BiprismVec64 b) {
    const double scale=fmax(double(g.xx),fmax(double(g.yy),double(g.zz)));
    if(!(scale>0.)||!isfinite(scale))return {CUDART_NAN,CUDART_NAN,CUDART_NAN};
    const double inverse_scale=1./scale;
    const double sx=sqrt(double(g.xx)*inverse_scale),sy=sqrt(double(g.yy)*inverse_scale),sz=sqrt(double(g.zz)*inverse_scale);
    const double xy=((double(g.xy)*inverse_scale)/sx)/sy;
    const double xz=((double(g.xz)*inverse_scale)/sx)/sz;
    const double yz=((double(g.yz)*inverse_scale)/sy)/sz;
    const double cxx=__fma_rn(-yz,yz,1.),cyy=__fma_rn(-xz,xz,1.),czz=__fma_rn(-xy,xy,1.);
    const double cxy=__fma_rn(xz,yz,-xy),cxz=__fma_rn(xy,yz,-xz),cyz=__fma_rn(xy,xz,-yz);
    const double det=__fma_rn(xy,cxy,__fma_rn(xz,cxz,cxx));
    if(!(det>0.)||!isfinite(det))return {CUDART_NAN,CUDART_NAN,CUDART_NAN};
    const double bx=b.x/sx,by=b.y/sy,bz=b.z/sz,factor=inverse_scale/det;
    return {(__fma_rn(cxx,bx,__fma_rn(cxy,by,cxz*bz))/sx)*factor,
            (__fma_rn(cxy,bx,__fma_rn(cyy,by,cyz*bz))/sy)*factor,
            (__fma_rn(cxz,bx,__fma_rn(cyz,by,czz*bz))/sz)*factor};
}
__device__ __forceinline__ BiprismVec64 bp_v10_metric_velocity_f64(const sym3_f64& g,BiprismVec64 p,double mass) {
    const auto v=bp_v10_metric_solve_f64(g,p);const double inverse_mass=1./mass;
    return {inverse_mass*v.x,inverse_mass*v.y,inverse_mass*v.z};
}
__device__ __forceinline__ BiprismVec64 bp_v10_metric_momentum_f64(const sym3_f64& g,BiprismVec64 v,double mass) {
    const auto p=bp_v10_sym_mul(g,v);return {mass*p.x,mass*p.y,mass*p.z};
}
__device__ __forceinline__ double bp_v10_kinetic_p(const sym3_f64& g,float3 p,double mass) {
    const auto q=bp_v10_vector(p);return .5*(1./mass)*bp_ed_dot(q,bp_v10_metric_solve_f64(g,q));
}
__device__ __forceinline__ double bp_v10_kinetic_p64(const sym3_f64& g,BiprismVec64 p,double mass) {
    return .5*(1./mass)*bp_ed_dot(p,bp_v10_metric_solve_f64(g,p));
}
__device__ __forceinline__ double bp_v10_kinetic_v64(const sym3_f64& g,BiprismVec64 v,double mass) {
    return .5*mass*bp_ed_dot(v,bp_v10_sym_mul(g,v));
}
__device__ __forceinline__ double bp_v10_kinetic_v(const sym3_f64& g,float3 v,double mass) {
    const auto q=bp_v10_vector(v);return .5*mass*bp_ed_dot(q,bp_v10_sym_mul(g,q));
}
// Each vertex receives its CSR incidence count plus one bath term. With M
// such terms, every partial signed sum is bounded by 2^62+M/2 < 2^63,
// including individual RN conversion error. No physical force is clipped.
__device__ __forceinline__ long long bp_v10_q39(double value,uint32_t terms,VariationalClusterDsmemV10* s) {
    const double limit=8388608./double(max(terms,1u));
    if(!isfinite(value)||fabs(value)>limit){atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);return 0ll;}
    return __double2ll_rn(value*549755813888.0);
}
__device__ __forceinline__ double bp_v10_q39_load(unsigned long long value) {return double(static_cast<long long>(value))*1.8189894035458565e-12;}
__device__ __forceinline__ void bp_v10_add_edge_q39(VariationalClusterDsmemV10* s,uint32_t u,uint32_t v,uint32_t edges,BiprismVec64 force) {
    const auto* index=&s->invocation.collective.streams->d_ordered_edges[s->invocation.collective.leaf];
    const uint32_t du=u<TILE_CORE_VERTICES?index->offsets[u+1u]-index->offsets[u]:0u;
    const uint32_t dv=v<TILE_CORE_VERTICES?index->offsets[v+1u]-index->offsets[v]:0u;
    const uint32_t terms=max(du,dv)+1u;
    const long long qx=bp_v10_q39(force.x,terms,s),qy=bp_v10_q39(force.y,terms,s),qz=bp_v10_q39(force.z,terms,s);
    if(u<TILE_CORE_VERTICES){atomicAdd(&s->stage.force.f_tot_q39[u][0],(unsigned long long)qx);atomicAdd(&s->stage.force.f_tot_q39[u][1],(unsigned long long)qy);atomicAdd(&s->stage.force.f_tot_q39[u][2],(unsigned long long)qz);}
    if(v<TILE_CORE_VERTICES){atomicAdd(&s->stage.force.f_tot_q39[v][0],0ull-(unsigned long long)qx);atomicAdd(&s->stage.force.f_tot_q39[v][1],0ull-(unsigned long long)qy);atomicAdd(&s->stage.force.f_tot_q39[v][2],0ull-(unsigned long long)qz);}
}
// Same three represented Gauss nodes/weights and field functional, now using the
// common double-charge provider and original FP32 sine amplitude; this retains the separate frozen-metric Cayley
// field without claiming exact general-field quadrature or a Jacobi identity.
__device__ __forceinline__ float3 bp_v10_berry_node(uint32_t eu,uint32_t ev,double qu,double qv,
    float3 e,float3 de,float3 vr,float h,float node,float weight) {
    const float3 eq={__fmaf_rn(node,de.x,e.x),__fmaf_rn(node,de.y,e.y),__fmaf_rn(node,de.z,e.z)};
    BiprismElectronicEdgeProvider<double> p;
    if(!bp_ed_evaluate_edge(bp_v10_element(eu),bp_v10_element(ev),{0.,0.,0.},bp_v10_vector(eq),qu,qv,p))return {CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F};
    const double gap=p.E1-p.E0;
    const double a=bp_ed_dot(p.grad_V01_u,bp_v10_vector(vr))/sqrt(__fma_rn(gap,gap,__fma_rn(4.*p.V01,p.V01,double(2.5e-9f))));
    const float l2=__fmaf_rn(eq.x,eq.x,__fmaf_rn(eq.y,eq.y,eq.z*eq.z))+1e-12f;
    const float phase=float(a*double(h));
    const float amp=weight*(__sinf(phase)+phase)*(rsqrtf(l2)/l2);
    return {amp*eq.x,amp*eq.y,amp*eq.z};
}
__device__ __forceinline__ float3 bp_v10_berry_chord(uint32_t eu,uint32_t ev,double qu,double qv,float3 e,float3 de,float3 vr,float h) {
    const auto a=bp_v10_berry_node(eu,ev,qu,qv,e,de,vr,h,-0.3872983346f,0.27777778f);
    const auto b=bp_v10_berry_node(eu,ev,qu,qv,e,de,vr,h,0.f,0.44444444f);
    const auto d=bp_v10_berry_node(eu,ev,qu,qv,e,de,vr,h,0.3872983346f,0.27777778f);
    return {a.x+b.x+d.x,a.y+b.y+d.y,a.z+b.z+d.z};
}
// Same quadrature nodes and field functional on compensated edge geometry.
// Retain phase amplitude, node contributions and accumulated rotation in FP64.
// The scalar amplitude is the same phase+sin(phase) functional, independently
// checked against MPFR; it needs no local range-reduction array. Quadrature
// nodes, weights and canonical electronic provider expressions stay unchanged.
__device__ __forceinline__ BiprismVec64 bp_v10_berry_node_compensated(
    uint32_t eu,uint32_t ev,double qu,double qv,BiprismCoordinate3F64 e,
    BiprismVec64 de,BiprismVec64 vr,double h,float node,float weight
) {
    const BiprismVec64 eq={__fma_rn(double(node),double(de.x),e.x),
        __fma_rn(double(node),double(de.y),e.y),__fma_rn(double(node),double(de.z),e.z)};
    BiprismElectronicEdgeProvider<double> p;
    if(!bp_ed_evaluate_edge(bp_v10_element(eu),bp_v10_element(ev),{0.,0.,0.},eq,qu,qv,p))return {CUDART_NAN,CUDART_NAN,CUDART_NAN};
    const double gap=p.E1-p.E0;
    const double a=bp_ed_dot(p.grad_V01_u,vr)/sqrt(__fma_rn(gap,gap,__fma_rn(4.*p.V01,p.V01,double(2.5e-9f))));
    const double l2=__fma_rn(eq.x,eq.x,__fma_rn(eq.y,eq.y,eq.z*eq.z))+double(1e-12f);
    const double phase=a*double(h);
    const double amp=double(weight)*biprism_magnetic_f64::phase_plus_sine(phase)*((1./sqrt(l2))/l2);
    return {amp*eq.x,amp*eq.y,amp*eq.z};
}
// The C2 geometry/charge epoch is immutable over all three calls. Reloading its
// original inputs prevents one node's provider from retaining the next nodes'
// geometry and prior outputs. The original node body is used without changes.
__device__ __forceinline__ BiprismVec64 bp_v10_berry_node_from_state(
    VariationalClusterDsmemV10* s,uint32_t u,uint32_t v,float node,float weight
) {
    auto& tile=s->invocation.tile;auto& amber=s->invocation.amber;
    const auto vu=bp_v10_live_velocity(s,u),vv=bp_v10_live_velocity(s,v);
    const auto e=bp_v10_edge_difference(s,u,v);
    const BiprismVec64 vr={vv.x-vu.x,vv.y-vu.y,vv.z-vu.z};
    const double h=s->invocation.collective.h;
    const BiprismVec64 chord={double(h)*vr.x,double(h)*vr.y,double(h)*vr.z};
    return bp_v10_berry_node_compensated(UNPACK_ELEM_CLASS(tile->global_atom_idx[u]),UNPACK_ELEM_CLASS(tile->global_atom_idx[v]),
        bp_v10_reload_charge(s,tile,amber,u),bp_v10_reload_charge(s,tile,amber,v),e,chord,vr,h,node,weight);
}
__device__ __forceinline__ BiprismVec64 bp_v10_berry_chord_from_state(
    VariationalClusterDsmemV10* s,uint32_t u,uint32_t v
) {
    const uint32_t tid=threadIdx.x;auto& ns=s->invocation.numerical;
    {
        const auto a=bp_v10_berry_node_from_state(s,u,v,-0.3872983346f,0.27777778f);
        ns->edge_force_x[tid]=a.x;ns->edge_force_y[tid]=a.y;ns->edge_force_z[tid]=a.z;
    }
    asm volatile("" ::: "memory");
    {
        const auto b=bp_v10_berry_node_from_state(s,u,v,0.f,0.44444444f);
        ns->edge_force_x[tid]=ns->edge_force_x[tid]+b.x;
        ns->edge_force_y[tid]=ns->edge_force_y[tid]+b.y;
        ns->edge_force_z[tid]=ns->edge_force_z[tid]+b.z;
    }
    asm volatile("" ::: "memory");
    const auto d=bp_v10_berry_node_from_state(s,u,v,0.3872983346f,0.27777778f);
    return {ns->edge_force_x[tid]+d.x,ns->edge_force_y[tid]+d.y,ns->edge_force_z[tid]+d.z};
}
// Literal final contraction from bp_ed_reciprocal_connection_force. Each z
// caches its unchanged energy-only pairing/division at the same state epoch.
// No FMA reassociation or compressed accumulator expression is introduced.
__device__ __forceinline__ BiprismVec64 bp_v10_reciprocal_from_pairings(
    BiprismVec64 kappa,double zu,double zv
) {
    const double coefficient=2.*(zu-zv);
    return {coefficient*kappa.x,coefficient*kappa.y,coefficient*kappa.z};
}
// Select one output of the unchanged vector helper. Production repeats the
// unchanged provider for each component, reducing simultaneous live gradients
// at a computational cost that requires native measurement. No radial formula,
// approximate division or altered force contraction is introduced here.
template<unsigned Component,class R>
__host__ __device__ __forceinline__ R bp_v10_original_force_component(
    const BiprismElectronicEdgeProvider<R>& p,BiprismElectronicDensity<R> rho
) {
    static_assert(Component<3u,"Cartesian component");
    const auto force=bp_ed_edge_coordinate_force(p.E0,p.E1,p.V01,p.S,rho,
        p.grad_E0_u,p.grad_E1_u,p.grad_V01_u,p.grad_S_u);
    if constexpr(Component==0u)return force.x;
    else if constexpr(Component==1u)return force.y;
    else return force.z;
}
template<unsigned Component>
__device__ __forceinline__ bool bp_v10_evaluate_original_force_component(
    VariationalClusterDsmemV10* s,uint32_t u,uint32_t v,double& component
) {
    auto& tile=s->invocation.tile;auto& amber=s->invocation.amber;
    BiprismElectronicEdgeProvider<double> p;
    if(!bp_ed_evaluate_edge(bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[u])),
        bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[v])),
        bp_v10_position(s,u),bp_v10_position(s,v),
        bp_v10_charge(s,tile,amber,u),bp_v10_charge(s,tile,amber,v),p))return false;
    component=bp_v10_original_force_component<Component>(p,bp_v10_density_mean(s,u,true));
    return true;
}
// Store the successful component before merging its status into the caller.
// A refused provider leaves the corresponding existing own-lane slot unchanged.
template<unsigned Component>
__device__ __forceinline__ bool bp_v10_store_original_force_component(
    VariationalClusterDsmemV10* s,uint32_t u,uint32_t v
) {
    auto& tile=s->invocation.tile;auto& amber=s->invocation.amber;
    BiprismElectronicEdgeProvider<double> p;
    if(!bp_ed_evaluate_edge(bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[u])),
        bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[v])),
        bp_v10_position(s,u),bp_v10_position(s,v),
        bp_v10_charge(s,tile,amber,u),bp_v10_charge(s,tile,amber,v),p))return false;
    const double component=bp_v10_original_force_component<Component>(p,bp_v10_density_mean(s,u,true));
    if constexpr(Component==0u)s->invocation.numerical->edge_force_x[threadIdx.x]=component;
    else if constexpr(Component==1u)s->invocation.numerical->edge_force_y[threadIdx.x]=component;
    else s->invocation.numerical->edge_force_z[threadIdx.x]=component;
    return true;
}
__device__ __forceinline__ bool bp_v10_accumulate_reciprocal_edge(
    VariationalClusterDsmemV10* s,uint32_t edge,uint32_t u,uint32_t v
) {
    auto& tile=s->invocation.tile;auto& amber=s->invocation.amber;
    auto& ns=s->invocation.numerical;
    const uint32_t tid=threadIdx.x;
    BiprismVec64 k;
    {
        BiprismElectronicEdgeProvider<double> p;
        const bool ok=bp_ed_evaluate_edge(bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[u])),bp_v10_element(UNPACK_ELEM_CLASS(tile->global_atom_idx[v])),
            bp_v10_position(s,u),bp_v10_position(s,v),bp_v10_charge(s,tile,amber,u),bp_v10_charge(s,tile,amber,v),p);
        if(!ok)return false;
        k=bp_v10_kappa(p,s->invocation.bank->live_weights[edge]);
    }
    const double zu=u<TILE_CORE_VERTICES?ns->reciprocal_connection_z[u]:0.;
    const double zv=v<TILE_CORE_VERTICES?ns->reciprocal_connection_z[v]:0.;
    const auto pair=bp_v10_reciprocal_from_pairings(k,zu,zv);
    const double sign=u==tid?1.:-1.;
    ns->reciprocal_force[0][tid]=__fma_rn(sign,pair.x,ns->reciprocal_force[0][tid]);
    ns->reciprocal_force[1][tid]=__fma_rn(sign,pair.y,ns->reciprocal_force[1][tid]);
    ns->reciprocal_force[2][tid]=__fma_rn(sign,pair.z,ns->reciprocal_force[2][tid]);
    return isfinite(pair.x)&&isfinite(pair.y)&&isfinite(pair.z);
}
__device__ __noinline__ void bp_v10_force_stage(
    VariationalClusterDsmemV10* s
) {
    auto& c=s->invocation.collective;auto& tile=s->invocation.tile;
    auto& amber=s->invocation.amber;auto& edges=s->invocation.edges;
    auto& h=s->invocation.collective.h;auto& inv_masses=s->invocation.masses;auto& ns=s->invocation.numerical;
    const float chi=.12f;
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    bool valid=true;
    // C1 has consumed every connection_im value. Each vertex pairing is
    // invariant over this incident-edge gather and uses the original helper,
    // unnormalized arithmetic midpoint and unchanged propagation weight.
    if(core){
        const auto H=bp_v10_reload_h(&s->stage.electronic.H_energy[tid]);
        const auto midpoint=bp_ed_endpoint_data(bp_v10_start(s,tid),bp_v10_endpoint(s,tid,true),0.).psi_bar;
        const double z=bp_ed_connection_energy_pairing(H,midpoint)/bp_v10_berry_mass(tile,tid);
        ns->reciprocal_connection_z[tid]=z;
        valid=isfinite(z);
    }
    valid=__syncthreads_and(valid);
    asm volatile("" ::: "memory");
    // All192 exact pairings are now published; fixed halos contribute zero.
    // Owned reciprocal rows remain live until C3; no edge-temporary alias exists.
    if(core){
        bp_ns_store3d(ns->reciprocal_force,tid,{0.,0.,0.});
        const auto* ordered=&c.streams->d_ordered_edges[c.leaf];
        #pragma unroll 1
        for(uint32_t cursor=ordered->offsets[tid];cursor<ordered->offsets[tid+1u];++cursor){
            const uint32_t edge=ordered->edge_index[cursor];
            const uint32_t u=s->edge_u8[edge][0],v=s->edge_u8[edge][1];
            if(u!=tid&&v!=tid)continue;
            valid=bp_v10_accumulate_reciprocal_edge(s,edge,u,v)&&valid;
        }
    }
    valid=__syncthreads_and(valid);
    asm volatile("" ::: "memory");
    // Every H/connection read has completed. Reuse only the shared union here;
    // the reciprocal force remains in its separate owned FP64 planes.
    if(core){
        const uint32_t atom=UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[tid]);
        const float3 bath=bp_v10_reload_vec3(reinterpret_cast<const float3*>(c.streams->d_forces_xyz[c.leaf]+3u*atom));
        const auto* index=&c.streams->d_ordered_edges[c.leaf];
        const uint32_t terms=index->offsets[tid+1u]-index->offsets[tid]+1u;
        s->stage.force.f_tot_q39[tid][0]=(unsigned long long)bp_v10_q39(double(bath.x),terms,s);
        s->stage.force.f_tot_q39[tid][1]=(unsigned long long)bp_v10_q39(double(bath.y),terms,s);
        s->stage.force.f_tot_q39[tid][2]=(unsigned long long)bp_v10_q39(double(bath.z),terms,s);
        // The previous C3 output is dead. C2 publishes tau here; the same
        // lane consumes it before replacing it with the new rotated velocity.
        bp_ns_store3d(ns->nuclear_rotation_output,tid,{0.,0.,0.});
    }
    __syncthreads();
    asm volatile("" ::: "memory");
    #pragma unroll 1
    for(uint32_t edge=tid;edge<edges;edge+=256u){
        const uint32_t u=s->edge_u8[edge][0],v=s->edge_u8[edge][1];
        {
            const bool ok=bp_v10_store_original_force_component<0u>(s,u,v);
            valid=valid&&ok;if(!ok)continue;
        }
        asm volatile("" ::: "memory");
        {
            const bool ok=bp_v10_store_original_force_component<1u>(s,u,v);
            valid=valid&&ok;if(!ok)continue;
        }
        asm volatile("" ::: "memory");
        {
            const bool ok=bp_v10_store_original_force_component<2u>(s,u,v);
            valid=valid&&ok;if(!ok)continue;
        }
        asm volatile("" ::: "memory");
        {
            const auto vu=bp_v10_live_velocity(s,u),vv=bp_v10_live_velocity(s,v);
            const auto e=bp_v10_edge_difference(s,u,v),E=bp_v10_reference_difference(s,u,v);
            const double l2=bp_geometry_length_squared(e);
            const float mu=1.f/fmaxf(bp_v10_readonly_float(inv_masses+UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u])),1e-6f);
            const float mv=1.f/fmaxf(bp_v10_readonly_float(inv_masses+UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[v])),1e-6f);
            const auto metric=bp_v10_metric_cotangent_compensated(e,E,vu,vv,mu,mv,chi);
            const double dp=double(bp_v10_phi(s)[u])-double(bp_v10_phi(s)[v]);
            const double dec=.5*dp*dp*double(s->invocation.bank->live_weights[edge])/l2;
            // Reciprocal response already has its ordered per-vertex owner.
            // Deposit this distinct conservative term only to its Q39 owner.
            bp_v10_add_edge_q39(s,u,v,edges,{double(metric.x)+dec*e.x+ns->edge_force_x[tid],double(metric.y)+dec*e.y+ns->edge_force_y[tid],double(metric.z)+dec*e.z+ns->edge_force_z[tid]});
        }
        asm volatile("" ::: "memory");
        // CUDA atomicAdd owns these writes; the volatile observation view
        // is read only after the block-wide publication barrier below.
        const auto B=bp_v10_berry_chord_from_state(s,u,v);
        valid=valid&&isfinite(B.x)&&isfinite(B.y)&&isfinite(B.z);
        if(u<TILE_CORE_VERTICES){const float mu=1.f/fmaxf(bp_v10_readonly_float(inv_masses+UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u])),1e-6f);
            const double a=.5*double(h)/double(mu);atomicAdd(const_cast<double*>(&ns->nuclear_rotation_output[0][u]),a*B.x);atomicAdd(const_cast<double*>(&ns->nuclear_rotation_output[1][u]),a*B.y);atomicAdd(const_cast<double*>(&ns->nuclear_rotation_output[2][u]),a*B.z);}
        if(v<TILE_CORE_VERTICES){const float mv=1.f/fmaxf(bp_v10_readonly_float(inv_masses+UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[v])),1e-6f);
            const double a=-.5*double(h)/double(mv);atomicAdd(const_cast<double*>(&ns->nuclear_rotation_output[0][v]),a*B.x);atomicAdd(const_cast<double*>(&ns->nuclear_rotation_output[1][v]),a*B.y);atomicAdd(const_cast<double*>(&ns->nuclear_rotation_output[2][v]),a*B.z);}
    }
    valid=__syncthreads_and(valid);
    asm volatile("" ::: "memory");
    if(core){
        bp_ns_store3d(ns->conservative_force,tid,{bp_v10_q39_load(s->stage.force.f_tot_q39[tid][0]),bp_v10_q39_load(s->stage.force.f_tot_q39[tid][1]),bp_v10_q39_load(s->stage.force.f_tot_q39[tid][2])});
        // Historical FP32 mirror only; C3 consumes the full tau owner above.
        const auto tau=bp_ns_load3d(ns->nuclear_rotation_output,tid);
        bp_ns_store3(ns->endpoint_v,tid,{float(tau.x),float(tau.y),float(tau.z)});
    }
    valid=__syncthreads_and(valid);
    if(tid==0u)s->invocation.stage_valid=valid?1u:0u;
    __syncthreads();
}

// All reduced C4 scalars are materialized before endpoint PCG; typed volatile
// replay storage prevents their accidental register live range spanning the solve.
__device__ __forceinline__ void bp_ns_reduce(VariationalClusterDsmemV10* s,
    volatile BiprismV10NumericalScratchFfi* n,uint32_t slot,double value,bool maximum=false) {
    const double result=maximum?bp_v10_max(value,s->warp_red_f64):bp_v10_sum(value,s->warp_red_f64);
    if(threadIdx.x==0u)n->scalar[slot]=result;
    __syncthreads();
}
// Admission has finished every snapshot/norm reader of these six rows.
// Publish core ledger primitives first; halo lanes contribute exact zeros.
// C2 consumes H and working_phi, then regenerates these force rows. No ledger
// aggregate remains live across a reduction or the next physical stage.
__device__ __noinline__ void bp_v10_publish_electronic_ledger_stage(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    auto& c=s->invocation.collective;auto& tile=s->invocation.tile;auto* ns=s->invocation.numerical;
    if(core){
        const auto f=bp_v10_foliation(c,tid);
        ns->conservative_force[0][tid]=f.owned_energy_delta;
        ns->conservative_force[1][tid]=f.attributed_energy_delta;
        ns->conservative_force[2][tid]=f.transfer_in;
        ns->reciprocal_force[0][tid]=f.discrete_work;
        const double nn=bp_v10_spinor_norm(bp_v10_endpoint(s,tid,true));
        ns->reciprocal_force[1][tid]=fmax(fabs(nn-1.),fmax(fabs(nn-bp_v10_spinor_norm(bp_v10_start(s,tid))),fabs(nn-bp_v10_outer_norm(s,tid))));
        ns->reciprocal_force[2][tid]=fabs(bp_v10_population(s->psi_trial[tid])-bp_v10_population(bp_v10_endpoint(s,tid,true)));
    }
    __syncthreads();
    bp_ns_reduce(s,ns,BP_NS_PAIR_DELTA,core?ns->conservative_force[0][tid]:0.);
    bp_ns_reduce(s,ns,BP_NS_ATTRIBUTED_DELTA,core?ns->conservative_force[1][tid]:0.);
    bp_ns_reduce(s,ns,BP_NS_TRANSFER,core?ns->conservative_force[2][tid]:0.);
    bp_ns_reduce(s,ns,BP_NS_FOL_WORK,core?ns->reciprocal_force[0][tid]:0.);
    const double maxnorm=bp_v10_max(core?ns->reciprocal_force[1][tid]:0.,s->warp_red_f64);
    if(tid==0u)ns->scalar[BP_NS_MAX_NORM]=fmax(ns->scalar[BP_NS_MAX_NORM],maxnorm);
    bp_ns_reduce(s,ns,BP_NS_POPULATION_RESIDUAL,core?ns->reciprocal_force[2][tid]:0.,true);
}
__device__ __noinline__ void bp_v10_build_ordered_stage(VariationalClusterDsmemV10* s) {
    auto& c=s->invocation.collective;
    const bool valid=biprism_build_ordered_edges_256(&c.streams->d_ordered_edges[c.leaf],
        &s->edge_u8[0][0],&s->edge_u8[0][1],2u,s->invocation.edges,c.leaf,
        reinterpret_cast<uint64_t>(s->invocation.tile),c.records[c.leaf].committed_substeps);
    if(!valid&&threadIdx.x==0u)atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);
}
// The initial metric map uses the same state and mass owners as C3. Keep its
// thread-local address calculations within this compiler allocation scope.
__device__ __noinline__ void bp_v10_initial_nuclear_stage(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    auto& tile=s->invocation.tile;auto& ns=s->invocation.numerical;auto& masses=s->invocation.masses;
    auto& edges=s->invocation.edges;const float chi=.12f;
    const auto metric=assemble_metric_to_reg_v10(s,tile,edges,chi);
    if(core){const auto v=bp_ns_load3(ns->v_start,tid);
        const float mass=1.f/fmaxf(bp_v10_readonly_float(masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
        const auto p=bp_v10_metric_momentum_f64(bp_ns_load_metric64(ns->nuclear_metric,tid),bp_v10_vector(v),double(mass));
        bp_ns_store3d(ns->nuclear_start_momentum,tid,p);
        bp_ns_store3(ns->p_start,tid,{float(p.x),float(p.y),float(p.z)});ns->kinetic_start[tid]=bp_v10_kinetic_v(bp_ns_load_metric64(ns->nuclear_metric,tid),v,double(mass));}
    float det=1.f;(void)invert_sym3_metric_exact(metric,det,&s->local_fault_accum);
}
// Keep the unchanged C3 mapping, work reductions and validity publication in
// one compiler allocation scope. All physical inputs retain their existing
// immutable invocation owners; the existing stage_valid field carries its vote.
__device__ __noinline__ void bp_v10_midpoint_nuclear_stage(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x,leaf=blockIdx.x&7u;const bool core=tid<TILE_CORE_VERTICES;
    auto& d_stream_ptrs=s->invocation.collective.streams;
    auto& tile=s->invocation.tile;auto& ns=s->invocation.numerical;auto& masses=s->invocation.masses;
    auto& edges=s->invocation.edges;auto& h=s->invocation.collective.h;
    auto& integration_mode=s->invocation.integration_mode;
    const float chi=.12f;bool valid=true;
    // C3 retains the FP64 covariance in its dedicated numerical owner; the
    // shared FP32 projection is diagnostic. C2's pairing
    // plane is dead before the six covariance planes replace it.
    (void)assemble_metric_to_reg_v10(s,tile,edges,chi);
    if(core){
        {
            float det=1.f;
            const auto metric=bp_v10_reload_metric(&s->stage.g_cov[tid]);
            (void)invert_sym3_metric_exact(metric,det,&s->local_fault_accum);
            bp_ns_store_metric(ns->midpoint_covariance,tid,metric);
        }
        {
            const auto chord=bp_v10_trial_chord(s,tid);
            const auto fc=bp_ns_load3d(ns->conservative_force,tid),fb=bp_ns_load3d(ns->reciprocal_force,tid);
            const double alpha=(integration_mode&GonzalezExactEnergyBit)?s->dg_alpha_global:0.;
            bp_ns_store3d(ns->nuclear_used_force,tid,{
                __fma_rn(-alpha,double(chord.x),fc.x+fb.x),
                __fma_rn(-alpha,double(chord.y),fc.y+fb.y),
                __fma_rn(-alpha,double(chord.z),fc.z+fb.z)});
        }
        // The previous next_mid arrays were copied to x_mid/v_mid at
        // this Picard entry. They can now retain vm/vp until their last
        // rotation/kinematic readers, then become the next mx/mv.
        {
            const auto p=bp_ns_load3d(ns->nuclear_start_momentum,tid);
            const auto force=bp_ns_load3d(ns->nuclear_used_force,tid);
            const BiprismVec64 pm={__fma_rn(bp_v10_half(s),double(force.x),double(p.x)),
                __fma_rn(bp_v10_half(s),double(force.y),double(p.y)),__fma_rn(bp_v10_half(s),double(force.z),double(p.z))};
            const float mass=1.f/fmaxf(bp_v10_readonly_float(masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const auto vm=bp_v10_metric_velocity_f64(bp_ns_load_metric64(ns->nuclear_metric,tid),pm,double(mass));
            bp_ns_store3d(ns->nuclear_rotation_input,tid,vm);
        }
        {
            const auto vp=biprism_nuclear_f64::cayley(bp_ns_load_metric64(ns->nuclear_metric,tid),
                bp_ns_load3d(ns->nuclear_rotation_input,tid),bp_ns_load3d(ns->nuclear_rotation_output,tid));
            bp_ns_store3d(ns->nuclear_rotation_output,tid,vp);
        }
        {
            const auto force=bp_ns_load3d(ns->nuclear_used_force,tid);
            const float mass=1.f/fmaxf(bp_v10_readonly_float(masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const auto pplus=bp_v10_metric_momentum_f64(bp_ns_load_metric64(ns->nuclear_metric,tid),bp_ns_load3d(ns->nuclear_rotation_output,tid),double(mass));
            const BiprismVec64 raw_p={__fma_rn(bp_v10_half(s),double(force.x),pplus.x),
                __fma_rn(bp_v10_half(s),double(force.y),pplus.y),__fma_rn(bp_v10_half(s),double(force.z),pplus.z)};
            // Cayley has consumed tau; C4 later replaces raw_p by v.
            bp_ns_store3d(ns->nuclear_endpoint_momentum,tid,raw_p);
            valid=valid&&isfinite(raw_p.x)&&isfinite(raw_p.y)&&isfinite(raw_p.z);
        }
        {
            const float mass=1.f/fmaxf(bp_v10_readonly_float(masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const double rotation=bp_v10_kinetic_v64(bp_ns_load_metric64(ns->nuclear_metric,tid),bp_ns_load3d(ns->nuclear_rotation_output,tid),double(mass))
                -bp_v10_kinetic_v64(bp_ns_load_metric64(ns->nuclear_metric,tid),bp_ns_load3d(ns->nuclear_rotation_input,tid),double(mass));
            // This thread has consumed its last covariance component.
            // Its covariance copy remains in the separate six-plane owner.
            s->stage.nuclear.diagnostics[tid].rotation=rotation;
        }
        {
            const auto vm=bp_ns_load3d(ns->nuclear_rotation_input,tid),vp=bp_ns_load3d(ns->nuclear_rotation_output,tid);
            const BiprismVec64 mv={.5*(vm.x+vp.x),.5*(vm.y+vp.y),.5*(vm.z+vp.z)};
            bp_ns_store3d(ns->nuclear_midpoint_velocity,tid,mv);
            bp_ns_store3(ns->next_mid_v,tid,{float(mv.x),float(mv.y),float(mv.z)});
        }
        {
            const auto mv=bp_ns_load3d(ns->nuclear_midpoint_velocity,tid),tv=bp_v10_live_velocity(s,tid);
            const auto delta=bp_coord_difference(bp_v10_advected_position(s,tid,bp_v10_half(s),mv),bp_v10_position_pair(s,tid));
            double dxres=fmax(fabs(delta.x),fmax(fabs(delta.y),fabs(delta.z)));
            dxres=fmax(dxres,bp_v10_half(s)*fmax(fabs(mv.x-tv.x),fmax(fabs(mv.y-tv.y),fabs(mv.z-tv.z))));
            s->stage.nuclear.diagnostics[tid].dx_residual=dxres;
            bp_ns_store3(ns->next_mid_x,tid,{float(tv.x),float(tv.y),float(tv.z)});
        }
        {
            const auto chord=bp_v10_trial_chord(s,tid);
            s->v_mid[tid]={float(chord.x),float(chord.y),float(chord.z)}; // diagnostic mirror only
        }
        {
            const auto ex=bp_v10_advected_position(s,tid,double(h),bp_ns_load3d(ns->nuclear_midpoint_velocity,tid));
            bp_v10_store_position(&ns->nuclear_position_live[tid],ex);
            // Keep the old offset row for historical diagnostics only.
            const uint32_t atom=bp_v10_reload_atom_id(s,tid);const auto& owner=bp_v10_coordinate_owner(s);
            const auto x=bp_coord_value(ex);
            s->x_mid[tid]={float(x.x-double(owner.current_hi[3u*atom])),float(x.y-double(owner.current_hi[3u*atom+1u])),float(x.z-double(owner.current_hi[3u*atom+2u]))};
            valid=valid&&bp_coord_finite(ex)&&isfinite(s->stage.nuclear.diagnostics[tid].dx_residual);
        }
        {
            const auto d=bp_v10_displacement(s,tid);
            const BiprismVec64 dx={d.x,d.y,d.z};
            // Preserve wc while the final reciprocal-force reader runs.
            s->stage.nuclear.diagnostics[tid].mid_work=bp_ed_dot(bp_ns_load3d(ns->conservative_force,tid),dx);
        }
        {
            const auto d=bp_v10_displacement(s,tid);
            const BiprismVec64 dx={d.x,d.y,d.z};
            const double wb=bp_ed_dot(bp_ns_load3d(ns->reciprocal_force,tid),dx);
            ns->reciprocal_force[0][tid]=s->stage.nuclear.diagnostics[tid].mid_work;ns->reciprocal_force[1][tid]=wb;
        }
        {
            const auto d=bp_v10_displacement(s,tid);
            const BiprismVec64 dx={d.x,d.y,d.z};
            ns->reciprocal_force[2][tid]=bp_ed_dot(bp_v10_trial_chord(s,tid),dx);
        }
        {
            const auto d=bp_v10_displacement(s,tid);
            const BiprismVec64 dx={d.x,d.y,d.z};
            ns->conservative_force[2][tid]=bp_ed_dot(bp_ns_load3d(ns->nuclear_used_force,tid),dx);
        }
        {
            const float mass=1.f/fmaxf(bp_v10_readonly_float(masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const double t=bp_v10_kinetic_p64(bp_ns_load_metric64(ns->nuclear_metric,tid),bp_ns_load3d(ns->nuclear_start_momentum,tid),double(mass));
            ns->conservative_force[0][tid]=t;
        }
        {
            const float mass=1.f/fmaxf(bp_v10_readonly_float(masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const double t=bp_v10_kinetic_p64(bp_ns_load_metric64(ns->nuclear_metric,tid),bp_ns_load3d(ns->nuclear_endpoint_momentum,tid),double(mass));
            ns->conservative_force[1][tid]=t;
        }
        s->stage.nuclear.diagnostics[tid].mid_work=ns->conservative_force[1][tid]-ns->conservative_force[0][tid]-ns->conservative_force[2][tid];
    }
    __syncthreads();
    bp_ns_reduce(s,ns,BP_NS_DX_RESIDUAL,core?s->stage.nuclear.diagnostics[tid].dx_residual:0.,true);
    bp_ns_reduce(s,ns,BP_NS_ROTATION_DEFECT,core?s->stage.nuclear.diagnostics[tid].rotation:0.);
    bp_ns_reduce(s,ns,BP_NS_MID_WORK_DEFECT,core?s->stage.nuclear.diagnostics[tid].mid_work:0.);
    // Diagnostic-only split of the represented nuclear work defect.
    // The prior row readers have finished; C4 has not yet reused the
    // metric/used-force stage. C1's slots29/31 are dead until next C1.
    if(core){
        const auto mv=bp_ns_load3d(ns->nuclear_midpoint_velocity,tid);
        const BiprismVec64 ideal_dx={double(h)*mv.x,double(h)*mv.y,double(h)*mv.z};
        const auto d=bp_v10_displacement(s,tid);
        const BiprismVec64 rounding_dx={ideal_dx.x-d.x,ideal_dx.y-d.y,ideal_dx.z-d.z};
        const auto used_force=bp_ns_load3d(ns->nuclear_used_force,tid);
        s->stage.nuclear.diagnostics[tid].rotation=ns->conservative_force[1][tid]-ns->conservative_force[0][tid]-bp_ed_dot(used_force,ideal_dx);
        s->stage.nuclear.diagnostics[tid].mid_work=bp_ed_dot(used_force,rounding_dx);
    }
    __syncthreads();
    bp_ns_reduce(s,ns,BP_NS_MID_IDEAL_WORK_DEFECT,core?s->stage.nuclear.diagnostics[tid].rotation:0.);
    bp_ns_reduce(s,ns,BP_NS_COORD_ROUND_WORK,core?s->stage.nuclear.diagnostics[tid].mid_work:0.);
    bp_ns_reduce(s,ns,BP_NS_WORK_CONS,core?ns->reciprocal_force[0][tid]:0.);
    bp_ns_reduce(s,ns,BP_NS_WORK_BERRY,core?ns->reciprocal_force[1][tid]:0.);
    bp_ns_reduce(s,ns,BP_NS_PROJECTION_DOT,core?ns->reciprocal_force[2][tid]:0.);
    // Freeze the alpha component's norm at the first nonzero evaluated chord.
    // Its scale is the half-kick velocity sensitivity in this exact covariance;
    // it is numerical conditioning only, never an operator or work change.
    if(s->invocation.joint_alpha_scale2==0.){
        double scale2=0.;
        if(core){
            const auto mv=bp_ns_load3d(ns->nuclear_midpoint_velocity,tid);
            const double factor=bp_v10_half(s)*double(h);
            const float mass=1.f/fmaxf(bp_v10_readonly_float(masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const auto response=bp_v10_metric_velocity_f64(bp_ns_load_metric64(ns->nuclear_metric,tid),
                BiprismVec64{factor*mv.x,factor*mv.y,factor*mv.z},double(mass));
            scale2=bp_ed_dot(response,response);
        }
        scale2=bp_v10_sum(scale2,s->warp_red_f64);
        if(tid==0u&&scale2>0.&&isfinite(scale2)){
            s->invocation.joint_alpha_scale2=scale2;
        }
        __syncthreads();
        double reference_dot=0.;
        if(core){const auto mv=bp_ns_load3d(ns->nuclear_midpoint_velocity,tid);
            reference_dot=double(h)*double(h)*bp_ed_dot(mv,mv);}
        reference_dot=bp_v10_sum(reference_dot,s->warp_red_f64);
        if(tid==0u)ns->nuclear_secant.reference_dot=reference_dot>0.?reference_dot:1.;
        __syncthreads();
    }
    {
        double bathdelta=0.;
        if(core){const auto d=bp_v10_displacement(s,tid);
            const BiprismVec64 dx={d.x,d.y,d.z};
            // The admitted packed IDs are immutable for this launch.
            // Reload at the consumer instead of retaining 3*atom over C1/C2.
            const volatile uint32_t* ids=tile->global_atom_idx;
            const uint32_t bath_atom=UNPACK_GLOBAL_ATOM_ID(ids[tid]);
            const float3 bath=*reinterpret_cast<const float3*>(d_stream_ptrs->d_forces_xyz[leaf]+3u*bath_atom);
            bathdelta=-bp_ed_dot(bp_v10_vector(bath),dx);}
        bp_ns_reduce(s,ns,BP_NS_BATH_DELTA,bathdelta);
    }
    valid=__syncthreads_and(valid&&s->invocation.stage_valid!=0u);
    if(tid==0u){s->invocation.stage_valid=valid?1u:0u;
        if(s->invocation.derivative_coordinate_phase!=0u)s->invocation.derivative_coordinate_phase=2u;}
    __syncthreads();
}
// All C3 work/bath reductions precede this stage. Its exact saved raw momentum
// uses only the consumed used_force row. Old Tmid0/1 are consumed together before
// their rows become DT/DT_METRIC; reciprocal rows hold exact dp until the last
// reconstruction solve, then row0 becomes its displacement residual. Every row
// is core-owned and all four reductions finish before endpoint PCG can reuse it.
__device__ __noinline__ void bp_v10_endpoint_nuclear_stage(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    auto& ns=s->invocation.numerical;
    bool valid=s->invocation.stage_valid!=0u;
    {
        const auto metric=assemble_metric_to_reg_v10(s,s->invocation.tile,s->invocation.edges,.12f);
        float det=1.f;(void)invert_sym3_metric_exact(metric,det,&s->local_fault_accum);
        if(core){
            bp_ns_store_metric(ns->endpoint_metric,tid,metric);
            // The raw endpoint momentum remains in its dedicated FP64 owner.
            // Endpoint-metric assembly does not replace or round it.
        }
    }
    if(core){
        {
            const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const auto solved_v=bp_v10_metric_velocity_f64(bp_ns_load_metric64(ns->nuclear_metric,tid),
                bp_ns_load3d(ns->nuclear_endpoint_momentum,tid),double(mass));
            const float3 v={float(solved_v.x),float(solved_v.y),float(solved_v.z)};
            bp_ns_store3(ns->endpoint_v,tid,v);
        }
        {
            // The raw comparison uses G_endpoint, just like Newton's smooth
            // work row. conservative_force[1] still belongs to G_midpoint.
            const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            s->stage.nuclear.diagnostics[tid].mid_work=bp_v10_kinetic_p64(
                bp_ns_load_metric64(ns->nuclear_metric,tid),bp_ns_load3d(ns->nuclear_endpoint_momentum,tid),double(mass));
        }
        {
            const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const auto v=bp_ns_load3(ns->endpoint_v,tid);
            const double te=bp_v10_kinetic_v(bp_ns_load_metric64(ns->nuclear_metric,tid),v,double(mass)),tn=ns->kinetic_start[tid];
            const double dt=te-tn;
            const double dtm=(te-ns->conservative_force[1][tid])-(tn-ns->conservative_force[0][tid]);
            // C3 consumed this own-row diagnostic before the endpoint pass.
            // Retain the initial FP32 endpoint conversion delta without
            // recomputing its metric solve in the rounding optimizer.
            s->stage.nuclear.diagnostics[tid].mid_work=te-s->stage.nuclear.diagnostics[tid].mid_work;
            ns->conservative_force[0][tid]=dt;ns->conservative_force[1][tid]=dtm;
            valid=valid&&isfinite(te)&&isfinite(v.x)&&isfinite(v.y)&&isfinite(v.z);
        }
        {
            const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const auto represented_p=bp_v10_metric_momentum_f64(bp_ns_load_metric64(ns->nuclear_metric,tid),
                bp_v10_vector(bp_ns_load3(ns->endpoint_v,tid)),double(mass));
            const auto raw_p=bp_ns_load3d(ns->nuclear_endpoint_momentum,tid);
            const BiprismVec64 dp={represented_p.x-double(raw_p.x),represented_p.y-double(raw_p.y),represented_p.z-double(raw_p.z)};
            bp_ns_store3d(ns->reciprocal_force,tid,dp);
            ns->conservative_force[2][tid]=fmax(fabs(dp.x),fmax(fabs(dp.y),fabs(dp.z)));
        }
        {
            const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const auto dv=bp_v10_metric_velocity_f64(bp_ns_load_metric64(ns->nuclear_metric,tid),
                bp_ns_load3d(ns->reciprocal_force,tid),double(mass));
            const double recon_pos=bp_v10_half(s)*fmax(fabs(dv.x),fmax(fabs(dv.y),fabs(dv.z)));
            ns->reciprocal_force[0][tid]=recon_pos;
            valid=valid&&isfinite(recon_pos);
        }
    }
    valid=__syncthreads_and(valid);
    if(tid==0u)s->invocation.stage_valid=valid?1u:0u;
    const double rounding=bp_v10_sum(core?s->stage.nuclear.diagnostics[tid].mid_work:0.,s->warp_red_f64);
    if(tid==0u)ns->nuclear_secant.alpha_work_df=valid?rounding:CUDART_NAN;
    __syncthreads();
    bp_ns_reduce(s,ns,BP_NS_DT,core?ns->conservative_force[0][tid]:0.);
    bp_ns_reduce(s,ns,BP_NS_DT_METRIC,core?ns->conservative_force[1][tid]:0.);
    bp_ns_reduce(s,ns,BP_NS_RECON_RAW,core?ns->conservative_force[2][tid]:0.,true);
    bp_ns_reduce(s,ns,BP_NS_RECON_POSITION,core?ns->reciprocal_force[0][tid]:0.,true);
}
__host__ __device__ __forceinline__ double bp_endpoint_rounding_target(double current,double raw,
    double linear,double quadratic,double half,double coordinate_defect) {
    if(!isfinite(current)||!isfinite(raw)||!isfinite(linear)||!isfinite(quadratic)
        ||!(quadratic>0.)||!isfinite(half)||!(half>0.)||!isfinite(coordinate_defect))return NAN;
    const double disc=fma(-4.*quadratic,coordinate_defect,linear*linear);
    double delta=0.;
    if(disc>=0.){
        const double den=linear+copysign(sqrt(disc),linear);
        if(den!=0.)delta=-2.*coordinate_defect/den;
    }else delta=-.5*linear/quadratic;
    // One percent interior margin protects the independent matrix
    // reconstruction (which is evaluated, not replaced by this scalar bound).
    const double radius=.99e-7/half;
    return fmax(raw-radius,fmin(raw+radius,current+delta));
}
// Select actual FP32 endpoint velocities inside the original reconstruction
// ball. Scalar candidates address ONLY coordinate projection work. Every chosen
// velocity is retained, its full kinetic energy and momentum are recomputed,
// and the unchanged C4 all-eight gates decide acceptance. There is no rounding
// work credit and no change to the raw Cayley/force/kinematic state.
__device__ __noinline__ void bp_v10_endpoint_rounding_stage(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    auto* ns=s->invocation.numerical;
    // The endpoint pass retained its FP32 conversion delta in this current
    // output owner. Subtract it before any optimizer or private-root reader.
    // The proposal helper rebuilds/scales the row after C4 has consumed it.
    if(tid==0u)ns->nuclear_secant.alpha_work_df=ns->scalar[BP_NS_COORD_WORK_DEFECT]-ns->nuclear_secant.alpha_work_df;
    __syncthreads();
    if(!(s->invocation.integration_mode&GonzalezExactEnergyBit)||!s->invocation.stage_valid
        ||s->local_fault_accum)return;
    if(!(ns->scalar[BP_NS_DX_RESIDUAL]<=1e-6))return;
    if(tid==0u){
        s->invocation.rounding_work_residual=ns->scalar[BP_NS_COORD_WORK_DEFECT];
        s->invocation.rounding_old_dt=ns->scalar[BP_NS_DT];
        s->invocation.rounding_old_dtm=ns->scalar[BP_NS_DT_METRIC];
        s->invocation.rounding_round=0u;
    }
    if(core){
        const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
        ns->rank_e00[tid]=bp_v10_kinetic_v(bp_ns_load_metric64(ns->nuclear_metric,tid),bp_ns_load3(ns->endpoint_v,tid),double(mass));
    }
    __syncthreads();
    #pragma unroll 1
    for(;s->invocation.rounding_round<32u;){
        if(fabs(s->invocation.rounding_work_residual)<=1e-9)break;
        if(core){
            ns->rank_choice.score[tid]=fabs(s->invocation.rounding_work_residual);
            ns->rank_choice.index[tid]=UINT32_MAX;
            #pragma unroll 1
            for(uint32_t component=0u;component<3u;++component){
                double target;
                {
                    const auto g=bp_ns_load_metric64(ns->nuclear_metric,tid);
                    const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
                    const auto current=bp_ns_load3(ns->endpoint_v,tid);
                    const auto raw=bp_v10_metric_velocity_f64(g,bp_ns_load3d(ns->nuclear_endpoint_momentum,tid),double(mass));
                    const auto momentum=bp_v10_metric_momentum_f64(g,bp_v10_vector(current),double(mass));
                    const double v=component==0u?double(current.x):(component==1u?double(current.y):double(current.z));
                    const double rawv=component==0u?raw.x:(component==1u?raw.y:raw.z);
                    const double linear=component==0u?momentum.x:(component==1u?momentum.y:momentum.z);
                    const double diag=component==0u?g.xx:(component==1u?g.yy:g.zz);
                    target=bp_endpoint_rounding_target(v,rawv,linear,.5*double(mass)*diag,bp_v10_half(s),s->invocation.rounding_work_residual);
                }
                // The scalar target is the sole value crossing this boundary.
                // Reload matrix/raw velocity for checking the actual bits; do
                // not retain their wide structs across both numerical paths.
                asm volatile("" : "+l"(ns) :: "memory");
                #pragma unroll 1
                for(uint32_t upper=0u;upper<2u;++upper){
                    const auto current=bp_ns_load3(ns->endpoint_v,tid);
                    auto proposal=current;const float value=bp_v10_bracket(target,upper!=0u);
                    if(component==0u)proposal.x=value;else if(component==1u)proposal.y=value;else proposal.z=value;
                    const auto g=bp_ns_load_metric64(ns->nuclear_metric,tid);
                    const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
                    const auto raw=bp_v10_metric_velocity_f64(g,bp_ns_load3d(ns->nuclear_endpoint_momentum,tid),double(mass));
                    const double bound=bp_v10_half(s)*fmax(fabs(double(proposal.x)-raw.x),fmax(fabs(double(proposal.y)-raw.y),fabs(double(proposal.z)-raw.z)));
                    if(!isfinite(bound)||bound>double(VARIATIONAL_PICARD_TOL))continue;
                    const double change=bp_v10_kinetic_v(g,proposal,double(mass))-bp_v10_kinetic_v(g,current,double(mass));
                    const double score=fabs(s->invocation.rounding_work_residual+change);
                    if(isfinite(score)&&score<ns->rank_choice.score[tid]){
                        ns->rank_choice.score[tid]=score;ns->rank_choice.index[tid]=component*2u+upper;
                        bp_ns_store3(ns->nuclear_base_velocity,tid,proposal);
                    }
                    asm volatile("" : "+l"(ns) :: "memory");
                }
            }
        }
        __syncthreads();
        if(tid==0u){
            double best=fabs(s->invocation.rounding_work_residual);uint32_t winner=UINT32_MAX;
            for(uint32_t i=0u;i<TILE_CORE_VERTICES;++i)
                if(ns->rank_choice.index[i]!=UINT32_MAX&&ns->rank_choice.score[i]<best){best=ns->rank_choice.score[i];winner=i;}
            s->_pad_footer[0]=winner;
        }
        __syncthreads();
        if(s->_pad_footer[0]==UINT32_MAX)break;
        if(tid==s->_pad_footer[0]){
            const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
            const auto g=bp_ns_load_metric64(ns->nuclear_metric,tid);
            const auto before=bp_ns_load3(ns->endpoint_v,tid),after=bp_ns_load3(ns->nuclear_base_velocity,tid);
            const double delta=bp_v10_kinetic_v(g,after,double(mass))-bp_v10_kinetic_v(g,before,double(mass));
            bp_ns_store3(ns->endpoint_v,tid,after);s->invocation.rounding_work_residual+=delta;
        }
        __syncthreads();
        if(tid==0u)++s->invocation.rounding_round;
        __syncthreads();
    }
    if(core){
        const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
        const auto g=bp_ns_load_metric64(ns->nuclear_metric,tid);const auto v=bp_v10_vector(bp_ns_load3(ns->endpoint_v,tid));
        const double kinetic=bp_v10_kinetic_v64(g,v,double(mass));
        ns->conservative_force[0][tid]=kinetic-ns->kinetic_start[tid];
        ns->conservative_force[1][tid]+=kinetic-ns->rank_e00[tid];
        const auto represented=bp_v10_metric_momentum_f64(g,v,double(mass)),raw=bp_ns_load3d(ns->nuclear_endpoint_momentum,tid);
        const BiprismVec64 dp={represented.x-raw.x,represented.y-raw.y,represented.z-raw.z};
        ns->conservative_force[2][tid]=fmax(fabs(dp.x),fmax(fabs(dp.y),fabs(dp.z)));
        const auto dv=bp_v10_metric_velocity_f64(g,dp,double(mass));
        ns->reciprocal_force[0][tid]=bp_v10_half(s)*fmax(fabs(dv.x),fmax(fabs(dv.y),fabs(dv.z)));
    }
    __syncthreads();
    bp_ns_reduce(s,ns,BP_NS_DT,core?ns->conservative_force[0][tid]:0.);
    bp_ns_reduce(s,ns,BP_NS_DT_METRIC,core?ns->conservative_force[1][tid]:0.);
    bp_ns_reduce(s,ns,BP_NS_RECON_RAW,core?ns->conservative_force[2][tid]:0.,true);
    bp_ns_reduce(s,ns,BP_NS_RECON_POSITION,core?ns->reciprocal_force[0][tid]:0.,true);
    if(tid==0u){
        ns->scalar[BP_NS_COORD_WORK_DEFECT]+=ns->scalar[BP_NS_DT_METRIC]-s->invocation.rounding_old_dtm;
        ns->scalar[BP_NS_BASE_ENERGY]+=ns->scalar[BP_NS_DT]-s->invocation.rounding_old_dt;
    }
    __syncthreads();
}
// Scalar predictor for coordinate work only. A secant is usable only after
// nuclear relaxation, with a finite negative slope comparable to the frozen
// chord. Total, kinetic and electronic work remain separate acceptance gates.
__host__ __device__ __forceinline__ double bp_v10_bounded_alpha_increment(
    double residual,double dot,double alpha,double old_residual,double old_alpha,
    bool have_sample,bool late,double limit
) {
    double step=(dot>1e-24&&isfinite(dot)&&isfinite(residual))?residual/dot:0.;
    const double da=alpha-old_alpha,dr=residual-old_residual;
    if(late&&have_sample&&fabs(da)>1e-12&&isfinite(dr)){
        const double slope=dr/da;
        if(isfinite(slope)&&slope<-.05*dot&&slope>-20.*dot)
            step=-residual/slope;
    }
    return isfinite(step)?fmin(limit,fmax(-limit,step)):0.;
}
// The observed residual along x(w)=x_base+w*r_base is
// r(w) ~= r_base+(w/w_observed)*delta_r. Its least-squares minimum is
// -w_observed*<r_base,delta_r>/||delta_r||^2. A positive directional
// derivative requires a reverse trial, not repeated positive damping.
// This predicts a trial only; C1--C4 re-evaluate every physical equation.
__host__ __device__ __forceinline__ double bp_v10_reverse_trial_weight(
    double observed_weight,double cross,double difference2
) {
    if(!isfinite(observed_weight)||observed_weight<=0.||!isfinite(cross)
        ||cross<=0.||!isfinite(difference2)||difference2<=0.)return 0.;
    const double weight=-observed_weight*(cross/difference2);
    return isfinite(weight)?fmax(-.5,weight):-.5;
}
// Rebuild only a solver-private right preconditioner. Every resulting input
// direction is probed through full C1-C4 before its action enters global QR.
__device__ __forceinline__ void bp_v10_rebuild_block_preconditioner(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    auto* q=&s->invocation.numerical->nuclear_secant;
            // Snapshot before the fresh-base initialization overwrites Q/T.
            // This storage is not a physical endpoint or an operator cache.
            #pragma unroll 1
            for(uint32_t j=0;j<min(q->count,63u);++j){
                if(core){bp_ns_store3d(q->pre_dx[j],tid,bp_ns_load3d(q->dx[j],tid));
                    bp_ns_store3d(q->pre_df[j],tid,bp_ns_load3d(q->df[j],tid));}
                if(tid==0u){q->pre_alpha_dx[j]=q->alpha_dx[j];q->pre_alpha_df[j]=q->alpha_df[j];}
                __syncthreads();
                // Block-Jacobi inverse actions: preserve each leaf's observed
                // input/output pairs while orthogonalizing within that leaf.
                // This approximation only shapes probes; fresh full coupled
                // actions still determine every Newton correction.
                #pragma unroll 1
                for(uint32_t pass=0;pass<2u;++pass){
                    #pragma unroll 1
                    for(uint32_t k=0;k<j;++k){
                        double a=core?bp_ed_dot(bp_ns_load3d(q->pre_df[k],tid),bp_ns_load3d(q->pre_df[j],tid)):0.;
                        a=bp_v10_sum(a,s->warp_red_f64);
                        if(tid==0u)q->coefficient_now=a+q->pre_alpha_df[k]*q->pre_alpha_df[j];
                        __syncthreads();
                        if(core){const auto a=bp_ns_load3d(q->pre_df[j],tid),b=bp_ns_load3d(q->pre_df[k],tid);
                            const auto c=bp_ns_load3d(q->pre_dx[j],tid),d=bp_ns_load3d(q->pre_dx[k],tid);const double w=q->coefficient_now;
                            bp_ns_store3d(q->pre_df[j],tid,{__fma_rn(-w,b.x,a.x),__fma_rn(-w,b.y,a.y),__fma_rn(-w,b.z,a.z)});
                            bp_ns_store3d(q->pre_dx[j],tid,{__fma_rn(-w,d.x,c.x),__fma_rn(-w,d.y,c.y),__fma_rn(-w,d.z,c.z)});}
                        if(tid==0u){q->pre_alpha_df[j]=__fma_rn(-q->coefficient_now,q->pre_alpha_df[k],q->pre_alpha_df[j]);
                            q->pre_alpha_dx[j]=__fma_rn(-q->coefficient_now,q->pre_alpha_dx[k],q->pre_alpha_dx[j]);}
                        __syncthreads();
                    }
                }
                double a=core?bp_ed_dot(bp_ns_load3d(q->pre_df[j],tid),bp_ns_load3d(q->pre_df[j],tid)):0.;
                a=bp_v10_sum(a,s->warp_red_f64);
                if(tid==0u){const double n=a+q->pre_alpha_df[j]*q->pre_alpha_df[j];q->coefficient_now=n>1e-16?1./sqrt(n):0.;}
                __syncthreads();
                if(core){const auto a=bp_ns_load3d(q->pre_df[j],tid),b=bp_ns_load3d(q->pre_dx[j],tid);const double w=q->coefficient_now;
                    bp_ns_store3d(q->pre_df[j],tid,{w*a.x,w*a.y,w*a.z});bp_ns_store3d(q->pre_dx[j],tid,{w*b.x,w*b.y,w*b.z});}
                if(tid==0u){q->pre_alpha_df[j]*=q->coefficient_now;q->pre_alpha_dx[j]*=q->coefficient_now;}
                __syncthreads();
            }
            __syncthreads();if(tid==0u)q->pre_count=min(q->count,63u);__syncthreads();
}
// The ninth continuation unknown is shared across the eight leaves; all
// reductions below account for its eight identical replicas exactly once.
__device__ __forceinline__ double bp_v10_arc_term(VariationalClusterDsmemV10* s,double a,double b) {
    return s->invocation.numerical->nuclear_arc.active?.125*a*b:0.;
}
__device__ __forceinline__ double bp_v10_newton_work_row(VariationalClusterDsmemV10* s) {
    const auto& a=s->invocation.numerical->nuclear_arc;
    return a.active?a.reference_row:5e-9/(bp_v10_half(s)*2e-7);
}
__device__ __forceinline__ void bp_v10_arc_publish_trial(VariationalClusterDsmemV10* s,double weight) {
    if(threadIdx.x==0u){auto& a=s->invocation.numerical->nuclear_arc;
        if(a.active){
            const double proposed=__fma_rn(weight,a.direction,a.base_h);
            const double h=double(s->invocation.numerical->nuclear_evaluations.original_h)*proposed;
            // Trial weights are bounded before this publication. Never clip a
            // failed continuation input into a different represented candidate.
            if(!(h>0.f&&isfinite(h)))atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);
            bp_v10_publish_h(s,h);
            a.current_h=double(h)/double(s->invocation.numerical->nuclear_evaluations.original_h);
        }
    }
    __syncthreads();
}
__device__ __forceinline__ bool bp_v10_arc_evaluate(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;auto& a=s->invocation.numerical->nuclear_arc;
    double dot=0.;
    if(a.active&&tid<TILE_CORE_VERTICES){
        const auto v=bp_v10_live_velocity(s,tid),b=bp_ns_load3d(a.anchor_v,tid),t=bp_ns_load3d(a.tangent_v,tid);
        dot=bp_ed_dot(t,BiprismVec64{v.x-b.x,v.y-b.y,v.z-b.z});
    }
    dot=bp_v10_sum(dot,s->warp_red_f64);
    if(a.active){
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,
            dot+a.tangent_alpha*(a.scale*s->dg_alpha_global-a.anchor_alpha)
            +.125*a.tangent_h*(a.current_h-a.anchor_h),0.,0.,isfinite(dot)))return false;
        if(tid==0u)a.defect=s->invocation.c4_collective[2].sum0-a.step;
    }else if(tid==0u)a.defect=0.;
    __syncthreads();
    if(tid==0u){auto* row=bp_v10_arc_history_row(s->invocation.numerical,s->invocation.picard);
        row[0]=s->invocation.collective.h;row[1]=a.defect;row[2]=a.step;row[3]=double(a.active);
        // Preserve EVALUATED h even after the proposal updates invocation.h.
        bp_v10_record_evaluated_h(s);
    }
    __syncthreads();return true;
}
// Keep the measured residual-space recurrence after returning from private
// continuation. Original-h acceptance still uses all original physical gates.
__device__ __forceinline__ bool bp_v10_residual_arnoldi(const VariationalClusterDsmemV10* s) {
    return s->invocation.numerical->nuclear_arc.active
        ||s->invocation.numerical->nuclear_evaluations.continuation_stage==16u;
}
__device__ __forceinline__ uint32_t bp_v10_arc_slow_trials(uint32_t previous,double ratio) {
    // Only actual accepted corrections enter this counter. Derivative probes
    // and model predictions cannot declare nonlinear stagnation.
    return ratio>.9 ? (previous<4u?previous+1u:4u) : 0u;
}
__device__ __forceinline__ double bp_v10_arc_step_factor(uint32_t accepted_corrections) {
    return accepted_corrections<=6u?1.5:(accepted_corrections>8u?.5:1.);
}
__device__ __forceinline__ double bp_v10_arc_target_intercept(double anchor_h,double tangent_h,double step) {
    // Intersect the already chosen branch predictor with the original h.
    // This only constructs an input for the full original-equation solve;
    // no physical step is accepted or timestep shortened here.
    return anchor_h<1.&&tangent_h>0. ? fmin(step,(1.-anchor_h)/tangent_h) : step;
}
__device__ __forceinline__ double bp_v10_arc_predictor_inverse_step(double linear_norm,double quadratic_norm) {
    // x_mid(lambda,v) is bilinear. Its predictor displacement is ds*L+ds^2*Q,
    // not just ds*L. Solve the conservative norm bound L*ds+Q*ds^2<=.025
    // without subtracting nearly equal square roots or dividing by Q=0.
    return (linear_norm+sqrt(linear_norm*linear_norm+.1*quadratic_norm))/.05;
}
__device__ __forceinline__ double bp_v10_arc_linear_rhs(bool tangent,double measured,bool arc_component) {
    // The tangent RHS is (0,0,-1). The measured F(anchor) remains untouched
    // for every finite difference, including its nonzero private-root error.
    return tangent?(arc_component?-1.:0.):measured;
}
__device__ __forceinline__ void bp_v10_arc_predict_from_tangent(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    auto* ns=s->invocation.numerical;auto& a=ns->nuclear_arc;auto& q=ns->nuclear_secant;auto& e=ns->nuclear_evaluations;
        double dx=0.;
        if(core){const auto v=bp_ns_load3d(a.anchor_v,tid),t=bp_ns_load3d(a.tangent_v,tid);
            const double half=.5*double(e.original_h);
            const BiprismVec64 linear={a.anchor_h*t.x+a.tangent_h*v.x,a.anchor_h*t.y+a.tangent_h*v.y,a.anchor_h*t.z+a.tangent_h*v.z};
            dx=bp_v10_arc_predictor_inverse_step(half*sqrt(bp_ed_dot(linear,linear)),
                half*fabs(a.tangent_h)*sqrt(bp_ed_dot(t,t)));}
        dx=bp_v10_max(dx,s->warp_red_f64);
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,0.,0.,dx,isfinite(dx)))return;
        if(tid==0u){const double ceiling=fmin(1./fmax(s->invocation.c4_collective[2].max0,1e-30),.125/fmax(fabs(a.tangent_h),1e-30));
            a.step=a.roots?fmin(a.step,ceiling):ceiling;
            if(a.tangent_h<0.)a.step=fmin(a.step,-.5*a.anchor_h/a.tangent_h);
            const bool original_target=a.anchor_h<1.&&a.tangent_h>0.
                &&(1.-a.anchor_h)/a.tangent_h<=a.step;
            a.step=bp_v10_arc_target_intercept(a.anchor_h,a.tangent_h,a.step);
            a.active=original_target?0u:1u;
            if(a.tangent_mode!=2u)++a.roots; // A retry retains the same accepted anchor.
            a.tangent_mode=0u;a.last_root_trial=e.nonlinear_trials;
            a.slow_trial_streak=0u;a.segment_accepted_trials=0u;
            // Native151 solved past original h (5.34557 vs5.16767), then
            // returned from the far side and stalled. Its bound CPU control
            // closes the original equations from this intercept in6 trials.
            const double h=double(e.original_h)*(original_target?1.:__fma_rn(a.step,a.tangent_h,a.anchor_h));
            bp_v10_publish_h(s,h);a.current_h=double(h)/double(e.original_h);
            s->dg_alpha_global=__fma_rn(a.step,a.tangent_alpha,a.anchor_alpha)/a.scale;
            e.continuation_stage=original_target?16u:1u;e.next_stage=0u;
            if(original_target)s->invocation.joint_alpha_scale2=0.;}
        __syncthreads();
        if(core){const auto v=bp_ns_load3d(a.anchor_v,tid),t=bp_ns_load3d(a.tangent_v,tid);
            bp_ns_store3d(ns->nuclear_trial_velocity,tid,{__fma_rn(a.step,t.x,v.x),__fma_rn(a.step,t.y,v.y),__fma_rn(a.step,t.z,v.z)});}
    __syncthreads();
    if(core){s->psi_trial[tid]=bp_v10_starts(s)[tid];bp_v10_candidates(s)[tid]=bp_v10_starts(s)[tid];}
    if(tid==0u){q.count=0u;q.have_previous=0u;q.valid=0u;q.pre_count=0u;q.mode=0u;bp_v10_reset_tangent_probes(q,a);
        q.accepted_trial_ratio=0.;q.fd_radius=e.continuation_stage==16u?1e-7:(a.reserved?1e-6:(a.active?fmin(q.fd_radius,1e-4):1e-6));q.trust_radius=.05;q.initial_trial_weight=0.;q.backtracks=0u;q.direction_checked=0u;
        q.target_refresh=0u;q.projected_norm2=0.;q.base_norm2=0.;q.probe_weight=0.;q.trial_weight=0.;
        s->invocation.electronic_target.epsilon_offset=0.;s->invocation.electronic_target.energy_offset=0.;
        s->invocation.electronic_target.raw_epsilon=CUDART_NAN;s->invocation.electronic_target.value=0.;}
    __syncthreads();
}
__device__ __forceinline__ void bp_v10_arc_finish_tangent(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    auto* ns=s->invocation.numerical;auto& a=ns->nuclear_arc;auto& q=ns->nuclear_secant;auto& e=ns->nuclear_evaluations;
    double n=core?bp_ed_dot(bp_ns_load3d(q.direction,tid),bp_ns_load3d(q.direction,tid)):0.;
    double dot=core?bp_ed_dot(bp_ns_load3d(q.direction,tid),bp_ns_load3d(a.tangent_v,tid)):0.;
    n=bp_v10_sum(n,s->warp_red_f64);dot=bp_v10_sum(dot,s->warp_red_f64);
    if(!bp_v10_collective_store(s,s->invocation.collective,2u,n+q.alpha_direction*q.alpha_direction+.125*a.direction*a.direction,
        dot+q.alpha_direction*a.tangent_alpha+.125*a.direction*a.tangent_h,0.,isfinite(n)&&isfinite(dot)))return;
    if(tid==0u){a.norm=s->invocation.c4_collective[2].sum0;
        a.dot=(s->invocation.c4_collective[2].sum1<0.?-1.:1.)/sqrt(fmax(a.norm,1e-30));}
    __syncthreads();
    if(!(isfinite(a.norm)&&a.norm>1e-28)){
        if(tid==0u)s->local_fault_accum|=FAULT_PICARD_NONCONV;
        __syncthreads();return;
    }
    if(core){const auto v=bp_ns_load3d(q.direction,tid);bp_ns_store3d(a.tangent_v,tid,{a.dot*v.x,a.dot*v.y,a.dot*v.z});}
    if(tid==0u){a.tangent_alpha=a.dot*q.alpha_direction;a.tangent_h=a.dot*a.direction;}
    __syncthreads();
    bp_v10_arc_predict_from_tangent(s);
}
__device__ __forceinline__ bool bp_v10_action_probe_pending(
    uint32_t completed,double best_error2,bool tangent
) {
    // Keep the original three independent lengths. Native156's endpoint
    // edge(0,1) crosses the canonical overlap floor at all three, whereas
    // the 1e-7 probe remains on its differentiable branch (relative error
    // 3.58e-4). Try at most two smaller lengths BEFORE rebuilding the basis;
    // reducing construction FD alone drove that run into roundoff at1e-12.
    // Native166's bound complete response resolves another finite window:
    // 1e-8 crosses the endpoint overlap floor, 1e-9 reaches pair-rounding
    // noise, but 3e-9 gives relative action error .0489. Spend two bounded
    // checks in that interval before shrinking construction FD into noise.
    // The independent relative-error-squared gate stays exactly .01, and
    // these checks still consume the existing total evaluation budget.
    // Native169's same-base full response identifies a one-sided window:
    // forward1e-7 crosses the endpoint overlap clamp (relativeerror6.40),
    // reverse1e-7 stays on the admitted branch (.0102; reverse1e-8=.0235).
    // Test that signed action before shrinking construction into roundoff.
    // This does not reverse the predictor or accept a nonlinear correction.
    return completed<3u||(tangent&&best_error2>.01&&completed<9u);
}
__device__ __forceinline__ double bp_v10_action_probe_scale(uint32_t completed) {
    return completed==5u?.3:(completed==6u?1./3.:(completed==7u?-100.:.1));
}
__device__ __forceinline__ double bp_v10_arnoldi_component(
    double output,double residual,double coefficient,double residual_norm2
) {
    // Q_k is orthogonal to preceding output columns, not to the original F.
    // R_k=F-Qc and ||R_{k-1}||^2=||R_k||^2+c_k^2. Project Q_k off
    // span(F,Q_0,...,Q_{k-1}) to obtain the next residual-space Arnoldi vector.
    // This implicit form needs no extra basis or continuation scratch storage.
    const double previous_norm2=__fma_rn(coefficient,coefficient,residual_norm2);
    return __fma_rn(-coefficient/previous_norm2,residual,(residual_norm2/previous_norm2)*output);
}
__device__ __forceinline__ void bp_v10_arc_retry_segment(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;auto* ns=s->invocation.numerical;auto& a=ns->nuclear_arc;auto& q=ns->nuclear_secant;
    if(tid==0u){a.step*=.25;++a.reserved;
        a.slow_trial_streak=0u;a.segment_accepted_trials=0u;a.tangent_mode=2u;
        a.last_root_trial=ns->nuclear_evaluations.nonlinear_trials;
        bp_v10_publish_h(s,double(ns->nuclear_evaluations.original_h)*a.anchor_h);a.current_h=a.anchor_h;
        s->dg_alpha_global=a.anchor_alpha/a.scale;
        q.mode=0u;q.count=0u;q.pre_count=0u;q.valid=0u;q.direction_checked=0u;bp_v10_reset_tangent_probes(q,a);
        q.fd_radius=1e-6;q.trust_radius=.05;q.backtracks=0u;q.target_refresh=0u;
        s->invocation.electronic_target.epsilon_offset=0.;s->invocation.electronic_target.energy_offset=0.;
        s->invocation.electronic_target.raw_epsilon=CUDART_NAN;s->invocation.electronic_target.value=0.;}
    __syncthreads();
    if(tid<TILE_CORE_VERTICES){bp_ns_store3d(ns->nuclear_trial_velocity,tid,bp_ns_load3d(a.anchor_v,tid));
        s->psi_trial[tid]=bp_v10_starts(s)[tid];bp_v10_candidates(s)[tid]=bp_v10_starts(s)[tid];}
    __syncthreads();
}
// Fixed-base matrix-free Newton/Krylov proposal. Every derivative probe runs
// the SAME complete C1-C4 path, including self-consistent electronic and DEC
// response. QR stores paired measured input/residual changes (J*T=Q). A real
// trial must reduce the actual fixed-scaled residual before becoming a new
// solver base. None of these private bases is an accepted physical timestep.
__device__ __noinline__ void bp_v10_joint_nuclear_proposal(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    auto* ns=s->invocation.numerical;auto* q=&ns->nuclear_secant;auto* ar=&ns->nuclear_arc;
    // Newton probes differentiate the continuous endpoint map. Rounding the
    // endpoint velocity to FP32 inserts a staircase into that map's work row
    // (native135 action controls), even when C1 and the nuclear solve are FP64.
    // Remove ONLY its evaluated kinetic difference from the solver-private row.
    // Physical C4 records, represented endpoints and all admission predicates
    // above this helper retain their original values. No work is credited.
    double endpoint_rounding_work=0.;
    if(core){
        const float mass=1.f/fmaxf(bp_v10_readonly_float(s->invocation.masses+bp_v10_reload_atom_id(s,tid)),1e-6f);
        const auto g=bp_ns_load_metric64(ns->nuclear_metric,tid);
        endpoint_rounding_work=bp_v10_kinetic_v(g,bp_ns_load3(ns->endpoint_v,tid),double(mass))
            -bp_v10_kinetic_p64(g,bp_ns_load3d(ns->nuclear_endpoint_momentum,tid),double(mass));
    }
    endpoint_rounding_work=bp_v10_sum(endpoint_rounding_work,s->warp_red_f64);
    if(tid==0u){
        q->alpha_scale=ar->active?ar->scale:sqrt(s->invocation.joint_alpha_scale2>0.?s->invocation.joint_alpha_scale2:1.);
        ar->work_df=ar->defect;
        // Equal component margins in the existing physical stopping gates:
        // half*F_v / 5e-9 and R_coord / 2e-7. This row scaling is fixed for
        // the substep and independent of the current chord/representation.
        q->alpha_work_df=(s->invocation.integration_mode&GonzalezExactEnergyBit)?
            bp_v10_newton_work_row(s)*(ns->scalar[BP_NS_COORD_WORK_DEFECT]-endpoint_rounding_work):0.;
        q->valid=isfinite(q->alpha_work_df)&&q->reference_dot>0.;
        q->iteration=s->invocation.picard;q->have_previous=1u;
    }
    __syncthreads();
    double norm=0.;
    if(core){const auto x=bp_v10_live_velocity(s,tid),g=bp_ns_load3d(ns->nuclear_midpoint_velocity,tid);
        // Solver equations retain velocity units, component signs and rank.
        // The physical position gate was saved at the midpoint in C3; the
        // live position now owns the endpoint and cannot be compared here.
        const BiprismVec64 f={g.x-x.x,g.y-x.y,g.z-x.z};bp_ns_store3d(q->work_df,tid,f);norm=bp_ed_dot(f,f);}
    norm=bp_v10_sum(norm,s->warp_red_f64);
    if(!bp_v10_collective_store(s,s->invocation.collective,2u,norm+q->alpha_work_df*q->alpha_work_df+bp_v10_arc_term(s,ar->work_df,ar->work_df),0.,0.,q->valid&&isfinite(norm)))return;
    if(tid==0u){
        q->actual_norm2=s->invocation.c4_collective[2].sum0;
        auto* h=q->history[s->invocation.picard%48u];h[0]=q->actual_norm2;h[1]=q->mode==2u?q->predicted_norm2:q->projected_norm2;
        h[2]=q->base_norm2;h[3]=q->mode==0u?0.:(q->mode==2u?q->trial_weight:q->probe_weight);
        h[4]=double(1000u*ns->nuclear_evaluations.continuation_stage+100u*(ar->tangent_mode?(q->mode==0u?6u:q->mode==1u?7u:8u):q->mode)+q->count);h[5]=s->invocation.nuclear_raw_residual;
        // A changed C1 target defines a changed residual. Reevaluate the SAME
        // current input before constructing any derivative for the new target.
        if(q->target_refresh){q->mode=0u;q->target_refresh=0u;q->valid=2u;q->pre_count=0u;}
    }
    __syncthreads();if(q->valid==2u)return;
    if(q->mode==4u){
        // Independent action check: this probe is along the assembled Newton
        // direction, not a direction used to construct a QR column. Reevaluate
        // the full native residual at three resolved perturbation lengths.
        double error2=0.,pred2=0.;
        if(core){const auto actual=bp_ns_load3d(q->work_df,tid),base=bp_ns_load3d(ns->nuclear_base_direction,tid);
            const auto jp=bp_ns_load3d(q->work_dx,tid);
            const BiprismVec64 err={(actual.x-base.x)/q->probe_weight-jp.x,(actual.y-base.y)/q->probe_weight-jp.y,(actual.z-base.z)/q->probe_weight-jp.z};
            error2=bp_ed_dot(err,err);pred2=bp_ed_dot(jp,jp);}
        error2=bp_v10_sum(error2,s->warp_red_f64);pred2=bp_v10_sum(pred2,s->warp_red_f64);
        if(tid==0u){const double err=(q->alpha_work_df-s->invocation.previous_gonzalez_residual)/q->probe_weight-q->alpha_work_dx;
            const double ah=(ar->work_df-ar->base_defect)/q->probe_weight-ar->work_dx;
            q->norm2=err*err+bp_v10_arc_term(s,ah,ah);}
        __syncthreads();
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,error2+q->norm2,pred2+q->alpha_work_dx*q->alpha_work_dx+bp_v10_arc_term(s,ar->work_dx,ar->work_dx),0.,isfinite(error2)))return;
        if(tid==0u){q->history[q->iteration%48u][1]=s->invocation.c4_collective[2].sum0;q->history[q->iteration%48u][2]=s->invocation.c4_collective[2].sum1;
            q->direction_error_best=fmin(q->direction_error_best,s->invocation.c4_collective[2].sum0/fmax(s->invocation.c4_collective[2].sum1,1e-30));}
        __syncthreads();
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,pred2,q->norm2,0.,isfinite(pred2)))return;
        if(tid==0u){q->history[q->iteration%48u][0]=s->invocation.c4_collective[2].sum0;
            q->history[q->iteration%48u][5]=s->invocation.c4_collective[2].sum1;
            q->history[q->iteration%48u][4]=double(1000u*ns->nuclear_evaluations.continuation_stage+(ar->tangent_mode?800u:500u)+q->count); // split Jp-check schema
            ++q->backtracks;if(bp_v10_action_probe_pending(q->backtracks,q->direction_error_best,ar->tangent_mode!=0u))q->probe_weight*=bp_v10_action_probe_scale(q->backtracks);
            else if(q->direction_error_best<=.01){q->mode=2u;q->direction_checked=1u;q->backtracks=0u;}
            else{q->mode=3u;q->pre_count=0u;++q->linear_restarts;
                // Native154's measured scalar action bias is first order in
                // construction radius: .191 at1e-6, .0199 at1e-7, .00288 at1e-8.
                // Rebuild the SAME coupled base before cutting its segment.
                // Two refinements bound this work; never repeat native153's
                // seventeen restarts into the finite-difference noise floor.
                // A tangent's failed action selects its own paired refinement;
                // a previous corrector or direction breakdown cannot select it.
                if(ar->tangent_mode)bp_v10_refine_tangent_probes(*q,*ar);
                else if(ar->active&&!ar->tangent_mode&&q->linear_restarts<=2u&&q->fd_radius>1e-8)
                    q->fd_radius=fmax(1e-8,.1*q->fd_radius);
                else q->fd_radius*=.5;}}
        __syncthreads();
        // A bounded derivative rebuild can repair truncation bias (native154).
        // If it cannot, retry the retained root rather than shrinking FD
        // indefinitely without producing a useful correction (native153).
        if(ar->active&&!ar->tangent_mode&&q->mode==3u
            &&(q->linear_restarts>2u||q->fd_radius<1e-8)
            &&ns->nuclear_evaluations.count<BIPRISM_MAX_NUCLEAR_EVALUATIONS
            &&ns->nuclear_evaluations.nonlinear_trials<BIPRISM_MAX_NUCLEAR_TRIALS){
            bp_v10_arc_retry_segment(s);return;
        }
        if(ar->tangent_mode&&q->mode==2u){
            if(ns->nuclear_evaluations.count<BIPRISM_MAX_NUCLEAR_EVALUATIONS
                &&ns->nuclear_evaluations.nonlinear_trials<BIPRISM_MAX_NUCLEAR_TRIALS)bp_v10_arc_finish_tangent(s);
            return;
        }
        if(core){const auto base=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->direction,tid);
            const double w=q->mode==4u?q->probe_weight:(q->mode==3u?0.:q->trial_weight);
            bp_ns_store3d(ns->nuclear_trial_velocity,tid,{__fma_rn(w,p.x,base.x),__fma_rn(w,p.y,base.y),__fma_rn(w,p.z,base.z)});}
        if(tid==0u){const double w=q->mode==4u?q->probe_weight:(q->mode==3u?0.:q->trial_weight);
            s->dg_alpha_global=__fma_rn(w,q->alpha_direction/q->alpha_scale,s->invocation.joint_base_alpha);if(q->mode==3u)q->mode=0u;}
        __syncthreads();bp_v10_arc_publish_trial(s,q->mode==4u?q->probe_weight:(q->mode==0u?0.:q->trial_weight));return;
    }
    if(q->mode==2u){
        if(tid==0u){
            const double prediction=q->base_norm2-q->predicted_norm2;
            q->last_actual_ratio=prediction>0.?(q->base_norm2-q->actual_norm2)/prediction:0.;
            if(q->actual_norm2<(1.-1e-4*q->trial_weight)*q->base_norm2){
                q->accepted_trial_ratio=q->actual_norm2/q->base_norm2;
                // A first-model action check does not qualify later Jacobians.
                // Stagnation at a changed base requires a fresh independent
                // assembled-direction probe before another limited correction.
                // Native143's late private base has raw=3.86e-5, but its
                // assembled action still fails the independent full-response
                // check. A small dimensional raw residual does not validate
                // that Jacobian. Recheck every weak accepted correction.
                if(q->accepted_trial_ratio>.9)
                    q->direction_checked=0u;
                // Retain the radius supported by the ACTUAL accepted trial.
                // A backtracked step must not restart the next model at the
                // same rejected radius. Expand only on good model agreement.
                const double fraction=q->trial_weight/q->initial_trial_weight;
                if(q->backtracks)q->trust_radius=fmax(1e-10,q->trust_radius*fraction);
                if(q->last_actual_ratio>.75&&(q->trial_weight<1.||q->regularization>0.))q->trust_radius=fmin(2.,2.*q->trust_radius);
                else if(q->last_actual_ratio<.25)q->trust_radius=fmax(1e-10,.5*q->trust_radius);
                ++q->accepted_newton;q->mode=0u;q->linear_restarts=0u;
                q->valid=q->last_actual_ratio>=.25?4u:1u;
                if(q->valid!=4u)q->pre_count=0u;
                if(ar->active){
                    ++ar->segment_accepted_trials;
                    ar->slow_trial_streak=bp_v10_arc_slow_trials(ar->slow_trial_streak,q->accepted_trial_ratio);
                    if(ar->slow_trial_streak>=4u)q->mode=3u;
                }
            }else{
                // Directional slope of ||F + w Jp||^2 at this immutable base.
                // The measured full residual supplies curvature for bounded
                // quadratic interpolation; no fixed 1/64 cutoff discards a
                // legitimate short step. Every new weight is evaluated by C1-C4.
                const double slope=q->model_slope;
                const double curvature=q->actual_norm2-q->base_norm2-slope*q->trial_weight;
                const double ratio=curvature>0.?(-.5*slope*q->trial_weight/curvature):.5;
                q->trial_weight*=fmax(.1,fmin(.5,ratio));++q->backtracks;
                if(q->backtracks>=20u||!isfinite(q->trial_weight)){
                    // A non-decreasing actual residual after twenty distinct
                    // probes rejects this model; restore its evaluated base.
                    ++q->linear_restarts;q->fd_radius*=.25;q->trust_radius*=.25;q->mode=3u;
                }
            }
        }
        __syncthreads();
        if(q->mode==3u&&ar->active
            &&ns->nuclear_evaluations.nonlinear_trials<BIPRISM_MAX_NUCLEAR_TRIALS
            &&ns->nuclear_evaluations.count<BIPRISM_MAX_NUCLEAR_EVALUATIONS){
            bp_v10_arc_retry_segment(s);return;
        }
        if(q->valid==4u&&!bp_v10_residual_arnoldi(s)){
            bp_v10_rebuild_block_preconditioner(s);
            if(tid==0u)q->valid=1u;
            __syncthreads();
        }
        if(bp_v10_residual_arnoldi(s)&&tid==0u){q->pre_count=0u;if(q->valid==4u)q->valid=1u;}
        __syncthreads();
        if(q->mode==2u||q->mode==3u){
            if(core){const auto b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->direction,tid);
                const double w=q->mode==3u?0.:q->trial_weight;
                bp_ns_store3d(ns->nuclear_trial_velocity,tid,{__fma_rn(w,p.x,b.x),__fma_rn(w,p.y,b.y),__fma_rn(w,p.z,b.z)});}
            if(tid==0u){const double w=q->mode==3u?0.:q->trial_weight;
                s->dg_alpha_global=__fma_rn(w,q->alpha_direction/q->alpha_scale,s->invocation.joint_base_alpha);
                q->predicted_norm2=q->base_norm2+w*q->model_slope+w*w*q->model_curvature;
                if(q->mode==3u)q->mode=0u;}
            __syncthreads();bp_v10_arc_publish_trial(s,q->mode==0u?0.:q->trial_weight);return;
        }
    }
    if(q->mode==0u){
        double input_norm=0.;
        if(core){const auto x=bp_v10_live_velocity(s,tid),f=bp_ns_load3d(q->work_df,tid);
            bp_ns_store3d(ns->nuclear_base_velocity_f64,tid,x);bp_ns_store3d(ns->nuclear_base_direction,tid,f);
            bp_ns_store3d(q->direction,tid,{-f.x,-f.y,-f.z});input_norm=bp_ed_dot(x,x);}
        input_norm=bp_v10_sum(input_norm,s->warp_red_f64);
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,input_norm+q->alpha_scale*q->alpha_scale*s->dg_alpha_global*s->dg_alpha_global+bp_v10_arc_term(s,ar->current_h,ar->current_h),0.,0.,isfinite(input_norm)))return;
        if(tid==0u){
            ar->base_h=ar->current_h;ar->base_defect=ar->work_df;ar->direction=ar->work_df;
            q->base_input_norm2=s->invocation.c4_collective[2].sum0;
            q->base_norm2=ar->tangent_mode?1.:q->actual_norm2;q->projected_norm2=q->base_norm2;
            s->invocation.joint_base_alpha=s->dg_alpha_global;s->invocation.previous_gonzalez_residual=q->alpha_work_df;
            // Right block preconditioner: J_v starts with -I and the direct
            // coordinate-work derivative is -<trial_chord, output_chord>.
            // Use its immutable positive substep reference at bootstrap. This
            // shapes probe directions only; measured J*T and the original F
            // still define the correction and actual acceptance.
            q->alpha_direction=-q->alpha_work_df*q->alpha_scale/
                (bp_v10_newton_work_row(s)*q->reference_dot);
            q->count=0u;q->mode=1u;
        }
        __syncthreads();
        // Arnoldi's first input vector is the normalized original residual.
        if(core){const auto f=bp_ns_load3d(ns->nuclear_base_direction,tid);const double inv=(bp_v10_residual_arnoldi(s)?-1.:1.)/sqrt(fmax(q->base_norm2,1e-30));
            bp_ns_store3d(q->input_basis[0],tid,{inv*bp_v10_arc_linear_rhs(ar->tangent_mode,f.x,false),inv*bp_v10_arc_linear_rhs(ar->tangent_mode,f.y,false),inv*bp_v10_arc_linear_rhs(ar->tangent_mode,f.z,false)});}
        if(tid==0u){q->input_alpha[0]=(bp_v10_residual_arnoldi(s)?-1.:1.)*bp_v10_arc_linear_rhs(ar->tangent_mode,s->invocation.previous_gonzalez_residual,false)/sqrt(fmax(q->base_norm2,1e-30));
            ar->input[0]=(bp_v10_residual_arnoldi(s)?-1.:1.)*bp_v10_arc_linear_rhs(ar->tangent_mode,ar->work_df,true)/sqrt(fmax(q->base_norm2,1e-30));}
        __syncthreads();
    }else{
        // Difference around one immutable evaluated base, never between unrelated
        // nonlinear states. Both deltas use the actual represented trial bits.
        if(core){const auto x=bp_v10_live_velocity(s,tid),b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid);
            const auto f=bp_ns_load3d(q->work_df,tid),fb=bp_ns_load3d(ns->nuclear_base_direction,tid);
            bp_ns_store3d(q->work_dx,tid,{x.x-b.x,x.y-b.y,x.z-b.z});
            bp_ns_store3d(q->work_df,tid,{f.x-fb.x,f.y-fb.y,f.z-fb.z});}
        if(tid==0u){q->alpha_work_dx=q->alpha_scale*(s->dg_alpha_global-s->invocation.joint_base_alpha);
            q->alpha_work_df-=s->invocation.previous_gonzalez_residual;
            ar->work_dx=ar->current_h-ar->base_h;ar->work_df-=ar->base_defect;++q->probes;}
        __syncthreads();
        // Original-h polish needs a resolved derivative, not ever smaller
        // forward differences. Native160-bound full C1-C4 controls resolve
        // its action with symmetric1e-7 probes (relative error9.1e-4), while
        // forward probes hit a truncation/roundoff floor before convergence.
        // The unfilled CURRENT Jacobian column owns the positive half until
        // the next complete negative evaluation. Retained/preconditioner
        // columns and continuation storage do not overlap this slot.
        if(ns->nuclear_evaluations.continuation_stage==16u||(ar->tangent_mode&&ar->tangent_probe_refinements!=0u)){
            if(q->count>=64u){if(tid==0u)s->local_fault_accum|=FAULT_PICARD_NONCONV;__syncthreads();return;}
            if(q->probe_weight>0.){
                if(core){bp_ns_store3d(q->dx[q->count],tid,bp_ns_load3d(q->work_dx,tid));
                    bp_ns_store3d(q->df[q->count],tid,bp_ns_load3d(q->work_df,tid));}
                if(tid==0u){q->alpha_dx[q->count]=q->alpha_work_dx;q->alpha_df[q->count]=q->alpha_work_df;
                    ar->dx[q->count]=ar->work_dx;ar->df[q->count]=ar->work_df;}
                __syncthreads(); // All warps enter this arm before its sign changes.
                if(tid==0u)q->probe_weight=-q->probe_weight;
                __syncthreads();
                if(core){const auto b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->input_basis[q->count],tid);
                    bp_ns_store3d(ns->nuclear_trial_velocity,tid,{__fma_rn(q->probe_weight,p.x,b.x),__fma_rn(q->probe_weight,p.y,b.y),__fma_rn(q->probe_weight,p.z,b.z)});}
                if(tid==0u){s->dg_alpha_global=__fma_rn(q->probe_weight,q->input_alpha[q->count]/q->alpha_scale,s->invocation.joint_base_alpha);
                    ar->direction=ar->input[q->count];}
                __syncthreads();bp_v10_arc_publish_trial(s,q->probe_weight);return;
            }
            if(core){
                const auto xp=bp_ns_load3d(q->dx[q->count],tid),xm=bp_ns_load3d(q->work_dx,tid);
                const auto fp=bp_ns_load3d(q->df[q->count],tid),fm=bp_ns_load3d(q->work_df,tid);
                bp_ns_store3d(q->work_dx,tid,{.5*(xp.x-xm.x),.5*(xp.y-xm.y),.5*(xp.z-xm.z)});
                bp_ns_store3d(q->work_df,tid,{.5*(fp.x-fm.x),.5*(fp.y-fm.y),.5*(fp.z-fm.z)});}
            if(tid==0u){q->alpha_work_dx=.5*(q->alpha_dx[q->count]-q->alpha_work_dx);
                q->alpha_work_df=.5*(q->alpha_df[q->count]-q->alpha_work_df);
                ar->work_dx=.5*(ar->dx[q->count]-ar->work_dx);ar->work_df=.5*(ar->df[q->count]-ar->work_df);}
            __syncthreads();
        }
        #pragma unroll 1
        for(uint32_t pass=0u;pass<2u;++pass){
            #pragma unroll 1
            for(uint32_t j=0u;j<q->count;++j){
                double value=core?bp_ed_dot(bp_ns_load3d(q->df[j],tid),bp_ns_load3d(q->work_df,tid)):0.;
                value=bp_v10_sum(value,s->warp_red_f64);
                if(!bp_v10_collective_store(s,s->invocation.collective,2u,value+q->alpha_df[j]*q->alpha_work_df+bp_v10_arc_term(s,ar->df[j],ar->work_df),0.,0.,isfinite(value)))return;
                if(tid==0u)q->coefficient_now=s->invocation.c4_collective[2].sum0;
                __syncthreads();
                if(core){const auto a=bp_ns_load3d(q->work_df,tid),b=bp_ns_load3d(q->df[j],tid);
                    bp_ns_store3d(q->work_df,tid,{__fma_rn(-q->coefficient_now,b.x,a.x),__fma_rn(-q->coefficient_now,b.y,a.y),__fma_rn(-q->coefficient_now,b.z,a.z)});
                    const auto c=bp_ns_load3d(q->work_dx,tid),d=bp_ns_load3d(q->dx[j],tid);
                    bp_ns_store3d(q->work_dx,tid,{__fma_rn(-q->coefficient_now,d.x,c.x),__fma_rn(-q->coefficient_now,d.y,c.y),__fma_rn(-q->coefficient_now,d.z,c.z)});}
                if(tid==0u){q->alpha_work_df=__fma_rn(-q->coefficient_now,q->alpha_df[j],q->alpha_work_df);
                    q->alpha_work_dx=__fma_rn(-q->coefficient_now,q->alpha_dx[j],q->alpha_work_dx);
                    ar->work_df=__fma_rn(-q->coefficient_now,ar->df[j],ar->work_df);
                    ar->work_dx=__fma_rn(-q->coefficient_now,ar->dx[j],ar->work_dx);}
                __syncthreads();
            }
        }
        double value=core?bp_ed_dot(bp_ns_load3d(q->work_df,tid),bp_ns_load3d(q->work_df,tid)):0.;
        value=bp_v10_sum(value,s->warp_red_f64);
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,value+q->alpha_work_df*q->alpha_work_df+bp_v10_arc_term(s,ar->work_df,ar->work_df),0.,0.,isfinite(value)))return;
        if(tid==0u)q->norm2=s->invocation.c4_collective[2].sum0;
        __syncthreads();
        if(q->norm2>1e-28){
            const double inv=1./sqrt(q->norm2);const uint32_t j=q->count;
            if(core){const auto a=bp_ns_load3d(q->work_df,tid),b=bp_ns_load3d(q->work_dx,tid);
                bp_ns_store3d(q->df[j],tid,{inv*a.x,inv*a.y,inv*a.z});bp_ns_store3d(q->dx[j],tid,{inv*b.x,inv*b.y,inv*b.z});}
            if(tid==0u){q->alpha_df[j]=inv*q->alpha_work_df;q->alpha_dx[j]=inv*q->alpha_work_dx;
                ar->df[j]=inv*ar->work_df;ar->dx[j]=inv*ar->work_dx;}
            __syncthreads();if(tid==0u)++q->count;__syncthreads();
        }
        if(core){const auto f=bp_ns_load3d(ns->nuclear_base_direction,tid);bp_ns_store3d(q->direction,tid,{0.,0.,0.});
            bp_ns_store3d(q->work_df,tid,{bp_v10_arc_linear_rhs(ar->tangent_mode,f.x,false),bp_v10_arc_linear_rhs(ar->tangent_mode,f.y,false),bp_v10_arc_linear_rhs(ar->tangent_mode,f.z,false)});}
        if(tid==0u){q->alpha_direction=0.;q->alpha_work_df=bp_v10_arc_linear_rhs(ar->tangent_mode,s->invocation.previous_gonzalez_residual,false);
            ar->direction=0.;ar->work_df=bp_v10_arc_linear_rhs(ar->tangent_mode,ar->base_defect,true);}
        __syncthreads();
        #pragma unroll 1
        for(uint32_t j=0u;j<q->count;++j){
            double value=core&&!ar->tangent_mode?bp_ed_dot(bp_ns_load3d(q->df[j],tid),bp_ns_load3d(ns->nuclear_base_direction,tid)):0.;
            value=bp_v10_sum(value,s->warp_red_f64);
            if(!bp_v10_collective_store(s,s->invocation.collective,2u,value+q->alpha_df[j]*bp_v10_arc_linear_rhs(ar->tangent_mode,s->invocation.previous_gonzalez_residual,false)+bp_v10_arc_term(s,ar->df[j],bp_v10_arc_linear_rhs(ar->tangent_mode,ar->base_defect,true)),0.,0.,isfinite(value)))return;
            if(tid==0u)q->coefficient[j]=s->invocation.c4_collective[2].sum0;
            __syncthreads();
            if(core){const auto p=bp_ns_load3d(q->direction,tid),t=bp_ns_load3d(q->dx[j],tid),a=bp_ns_load3d(q->df[j],tid),r=bp_ns_load3d(q->work_df,tid);
                bp_ns_store3d(q->direction,tid,{__fma_rn(-q->coefficient[j],t.x,p.x),__fma_rn(-q->coefficient[j],t.y,p.y),__fma_rn(-q->coefficient[j],t.z,p.z)});
                bp_ns_store3d(q->work_df,tid,{__fma_rn(-q->coefficient[j],a.x,r.x),__fma_rn(-q->coefficient[j],a.y,r.y),__fma_rn(-q->coefficient[j],a.z,r.z)});}
            if(tid==0u){q->alpha_direction=__fma_rn(-q->coefficient[j],q->alpha_dx[j],q->alpha_direction);
                q->alpha_work_df=__fma_rn(-q->coefficient[j],q->alpha_df[j],q->alpha_work_df);
                ar->direction=__fma_rn(-q->coefficient[j],ar->dx[j],ar->direction);
                ar->work_df=__fma_rn(-q->coefficient[j],ar->df[j],ar->work_df);}
            __syncthreads();
        }
        double unresolved=core?bp_ed_dot(bp_ns_load3d(q->work_df,tid),bp_ns_load3d(q->work_df,tid)):0.;
        unresolved=bp_v10_sum(unresolved,s->warp_red_f64);
        double trust_ratio=0.;
        if(core){const auto p=bp_ns_load3d(q->direction,tid),f=bp_ns_load3d(ns->nuclear_base_direction,tid),r=bp_ns_load3d(q->work_df,tid);
            const BiprismVec64 dg={p.x-f.x+r.x,p.y-f.y+r.y,p.z-f.z+r.z};
            trust_ratio=bp_v10_half(s)*sqrt(fmax(bp_ed_dot(p,p),bp_ed_dot(dg,dg)))/q->trust_radius;}
        trust_ratio=bp_v10_max(trust_ratio,s->warp_red_f64);
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,unresolved+q->alpha_work_df*q->alpha_work_df+bp_v10_arc_term(s,ar->work_df,ar->work_df),0.,trust_ratio,isfinite(unresolved)&&isfinite(trust_ratio)))return;
        if(tid==0u){q->projected_norm2=s->invocation.c4_collective[2].sum0;
            // Native121 spent178/192 evaluations refining a full Newton step
            // before restricting it to a0.032bohr trust region. Once a useful
            // >=2-column model reaches that region's boundary, solve the
            // constrained model below and evaluate the actual nonlinear trial.
            // Interior steps retain the ten-percent linear-residual criterion.
            // Neither condition is physical convergence or acceptance.
            const double forcing2=ar->tangent_mode?1e-6:.01;
            // Native141's private fold corrector stopped at a weak boundary
            // model, then spent repeated tiny trials on its inaccurate action.
            // The matched full-response control contracts with the existing
            // interior criterion. A private augmented solve must satisfy that
            // criterion before reshaping its direction to the trust region.
            const bool boundary_model=!bp_v10_residual_arnoldi(s)&&q->count>=2u&&s->invocation.c4_collective[2].max0>=1.
                &&q->projected_norm2<=.64*q->base_norm2;
            q->mode=(boundary_model||q->projected_norm2<=forcing2*q->base_norm2||q->count>=64u||q->norm2<=1e-28)?2u:1u;}
        __syncthreads();
        if(q->mode==1u){
            // The global six-column subspace in native112 exhausted useful
            // progress while leaf-local inverse actions resolved additional
            // directions. Direct block corrections failed the native Jp test
            // in113. Use those blocks ONLY to shape a new probe, measured at
            // this unchanged base and incorporated in the full coupled QR.
            const bool refresh=!bp_v10_residual_arnoldi(s)&&(q->count%6u)==0u;
            if(refresh)bp_v10_rebuild_block_preconditioner(s);
            // Arnoldi input-space orthogonalization. Reusing the current
            // residual here makes successive finite-difference probes nearly
            // parallel and amplifies inner-solve/weight quantization noise.
            // The latest output QR column spans the same new Krylov direction;
            // remove all prior INPUT columns before applying the preconditioner.
            if(bp_v10_residual_arnoldi(s)){
                // Native144's mixed input/output recurrence stagnated at a
                // nonzero coupled residual. Form the next Arnoldi direction
                // in residual space BEFORE the unchanged right preconditioner.
                if(core){const auto v=bp_ns_load3d(q->df[q->count-1u],tid),r=bp_ns_load3d(q->work_df,tid);
                    const double c=q->coefficient[q->count-1u],n=q->projected_norm2;
                    bp_ns_store3d(q->direction,tid,{bp_v10_arnoldi_component(v.x,r.x,c,n),
                        bp_v10_arnoldi_component(v.y,r.y,c,n),bp_v10_arnoldi_component(v.z,r.z,c,n)});}
                if(tid==0u){const double c=q->coefficient[q->count-1u],n=q->projected_norm2;
                    q->alpha_direction=bp_v10_arnoldi_component(q->alpha_df[q->count-1u],q->alpha_work_df,c,n);
                    ar->direction=bp_v10_arnoldi_component(ar->df[q->count-1u],ar->work_df,c,n);}
            }else{
                if(core)bp_ns_store3d(q->direction,tid,bp_ns_load3d(refresh?q->work_df:q->df[q->count-1u],tid));
                if(tid==0u){q->alpha_direction=refresh?q->alpha_work_df:q->alpha_df[q->count-1u];
                    ar->direction=refresh?ar->work_df:ar->df[q->count-1u];}
            }
            __syncthreads();
            #pragma unroll 1
            for(uint32_t pass=0u;!bp_v10_residual_arnoldi(s)&&!refresh&&pass<2u;++pass){
                #pragma unroll 1
                for(uint32_t j=0u;j<q->count;++j){
                    double a=core?bp_ed_dot(bp_ns_load3d(q->input_basis[j],tid),bp_ns_load3d(q->direction,tid)):0.;
                    a=bp_v10_sum(a,s->warp_red_f64);
                    if(!bp_v10_collective_store(s,s->invocation.collective,2u,a+q->input_alpha[j]*q->alpha_direction+bp_v10_arc_term(s,ar->input[j],ar->direction),0.,0.,isfinite(a)))return;
                    if(tid==0u)q->coefficient_now=s->invocation.c4_collective[2].sum0;
                    __syncthreads();
                    if(core){const auto d=bp_ns_load3d(q->direction,tid),v=bp_ns_load3d(q->input_basis[j],tid);
                        bp_ns_store3d(q->direction,tid,{__fma_rn(-q->coefficient_now,v.x,d.x),__fma_rn(-q->coefficient_now,v.y,d.y),__fma_rn(-q->coefficient_now,v.z,d.z)});}
                    if(tid==0u){q->alpha_direction=__fma_rn(-q->coefficient_now,q->input_alpha[j],q->alpha_direction);
                        ar->direction=__fma_rn(-q->coefficient_now,ar->input[j],ar->direction);}
                    __syncthreads();
                }
            }
            double a=core?bp_ed_dot(bp_ns_load3d(q->direction,tid),bp_ns_load3d(q->direction,tid)):0.;
            a=bp_v10_sum(a,s->warp_red_f64);
            if(!bp_v10_collective_store(s,s->invocation.collective,2u,a+q->alpha_direction*q->alpha_direction+bp_v10_arc_term(s,ar->direction,ar->direction),0.,0.,isfinite(a)))return;
            if(tid==0u)q->norm2=s->invocation.c4_collective[2].sum0;
            __syncthreads();
            if(q->norm2<=1e-24){
                // Invariant subspace: use the already measured minimum residual
                // correction; a zero new direction is not convergence.
                if(tid==0u)q->mode=2u;
                if(core){BiprismVec64 p={0.,0.,0.};for(uint32_t j=0u;j<q->count;++j){const auto t=bp_ns_load3d(q->dx[j],tid);const double c=q->coefficient[j];p={__fma_rn(-c,t.x,p.x),__fma_rn(-c,t.y,p.y),__fma_rn(-c,t.z,p.z)};}bp_ns_store3d(q->direction,tid,p);}
                if(tid==0u){q->alpha_direction=0.;ar->direction=0.;for(uint32_t j=0u;j<q->count;++j){q->alpha_direction-=q->coefficient[j]*q->alpha_dx[j];ar->direction-=q->coefficient[j]*ar->dx[j];}}
            }else{
                const double inv=1./sqrt(q->norm2);
                if(core){const auto d=bp_ns_load3d(q->direction,tid);bp_ns_store3d(q->input_basis[q->count],tid,{inv*d.x,inv*d.y,inv*d.z});}
                if(tid==0u){q->input_alpha[q->count]=inv*q->alpha_direction;ar->input[q->count]=inv*ar->direction;}
            }
            __syncthreads();
        }
    }
    if(q->mode==1u){
        if(core){const auto v=bp_ns_load3d(q->input_basis[q->count],tid);bp_ns_store3d(q->direction,tid,{-v.x,-v.y,-v.z});}
        if(tid==0u){q->alpha_direction=-q->input_alpha[q->count]*q->alpha_scale/
            (bp_v10_newton_work_row(s)*q->reference_dot);ar->direction=ar->input[q->count];}
        __syncthreads();
        // H_old = T_old Q_old^T + P(I-Q_old Q_old^T). Only the probe
        // direction is preconditioned; the next C1-C4 evaluation measures its
        // current J action. No old action enters the physical residual.
        #pragma unroll 1
        // These leaf-block pairs omit the shared h component. They are valid
        // probe preconditioning at fixed h, but not an inverse of the augmented
        // arclength action. Retained native138 controls measured the resulting
        // degradation (7 Krylov probes versus 1 without those mixed pairs).
        for(uint32_t j=0;j<q->pre_count&&!bp_v10_residual_arnoldi(s);++j){
            double dot=core?bp_ed_dot(bp_ns_load3d(q->pre_df[j],tid),bp_ns_load3d(q->input_basis[q->count],tid)):0.;
            dot=bp_v10_sum(dot,s->warp_red_f64);
            if(tid==0u)q->coefficient_now=dot+q->pre_alpha_df[j]*q->input_alpha[q->count];
            __syncthreads();
            if(core){const auto p=bp_ns_load3d(q->direction,tid),t=bp_ns_load3d(q->pre_dx[j],tid),a=bp_ns_load3d(q->pre_df[j],tid);const double c=q->coefficient_now;
                bp_ns_store3d(q->direction,tid,{__fma_rn(c,t.x+a.x,p.x),__fma_rn(c,t.y+a.y,p.y),__fma_rn(c,t.z+a.z,p.z)});}
            if(tid==0u)q->alpha_direction+=q->coefficient_now*(q->pre_alpha_dx[j]+q->pre_alpha_df[j]*q->alpha_scale/(bp_v10_newton_work_row(s)*q->reference_dot));
            __syncthreads();
        }
        // Lift the physical block inverse along the retained branch tangent.
        // The arclength row then acts as identity on this probe: t.P(r)=r_h.
        // This is only a right preconditioner; the next full C1-C4 evaluation
        // measures its actual action, including the self-consistent response.
        if(ar->active){
            double dot=core?bp_ed_dot(bp_ns_load3d(ar->tangent_v,tid),bp_ns_load3d(q->direction,tid)):0.;
            dot=bp_v10_sum(dot,s->warp_red_f64);
            if(!bp_v10_collective_store(s,s->invocation.collective,2u,dot+ar->tangent_alpha*q->alpha_direction,0.,0.,isfinite(dot)))return;
            if(tid==0u)q->coefficient_now=ar->input[q->count]-s->invocation.c4_collective[2].sum0;
            __syncthreads();
            if(core){const auto p=bp_ns_load3d(q->direction,tid),t=bp_ns_load3d(ar->tangent_v,tid);const double c=q->coefficient_now;
                bp_ns_store3d(q->direction,tid,{__fma_rn(c,t.x,p.x),__fma_rn(c,t.y,p.y),__fma_rn(c,t.z,p.z)});}
            if(tid==0u){q->alpha_direction=__fma_rn(q->coefficient_now,ar->tangent_alpha,q->alpha_direction);
                ar->direction=q->coefficient_now*ar->tangent_h;}
            __syncthreads();
        }
        // Preconditioning can make distinct Arnoldi vectors nearly parallel.
        // Orthogonalize the ACTUAL perturbations, after H_old, so differencing
        // and output QR never amplify a near-dependent input combination.
        #pragma unroll 1
        for(uint32_t pass=0;pass<2u;++pass){
            #pragma unroll 1
            for(uint32_t j=0;j<q->count;++j){
                double dot=core?bp_ed_dot(bp_ns_load3d(q->input_basis[j],tid),bp_ns_load3d(q->direction,tid)):0.;
                dot=bp_v10_sum(dot,s->warp_red_f64);
                if(!bp_v10_collective_store(s,s->invocation.collective,2u,dot+q->input_alpha[j]*q->alpha_direction+bp_v10_arc_term(s,ar->input[j],ar->direction),0.,0.,isfinite(dot)))return;
                if(tid==0u)q->coefficient_now=s->invocation.c4_collective[2].sum0;
                __syncthreads();
                if(core){const auto p=bp_ns_load3d(q->direction,tid),v=bp_ns_load3d(q->input_basis[j],tid);const double c=q->coefficient_now;
                    bp_ns_store3d(q->direction,tid,{__fma_rn(-c,v.x,p.x),__fma_rn(-c,v.y,p.y),__fma_rn(-c,v.z,p.z)});}
                if(tid==0u){q->alpha_direction=__fma_rn(-q->coefficient_now,q->input_alpha[j],q->alpha_direction);
                        ar->direction=__fma_rn(-q->coefficient_now,ar->input[j],ar->direction);}
                __syncthreads();
            }
        }
        double dn=core?bp_ed_dot(bp_ns_load3d(q->direction,tid),bp_ns_load3d(q->direction,tid)):0.;
        dn=bp_v10_sum(dn,s->warp_red_f64);
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,dn+q->alpha_direction*q->alpha_direction+bp_v10_arc_term(s,ar->direction,ar->direction),0.,0.,isfinite(dn)))return;
        if(tid==0u)q->norm2=s->invocation.c4_collective[2].sum0;
        __syncthreads();
        if(q->norm2<=1e-24){
            if(core)bp_ns_store3d(ns->nuclear_trial_velocity,tid,bp_ns_load3d(ns->nuclear_base_velocity_f64,tid));
            if(tid==0u){s->dg_alpha_global=s->invocation.joint_base_alpha;q->mode=0u;q->pre_count=0u;++q->linear_restarts;}
            __syncthreads();bp_v10_arc_publish_trial(s,0.);return;
        }
        if(core){const auto p=bp_ns_load3d(q->direction,tid);const double inv=1./sqrt(q->norm2);
            bp_ns_store3d(q->direction,tid,{inv*p.x,inv*p.y,inv*p.z});
            bp_ns_store3d(q->input_basis[q->count],tid,{inv*p.x,inv*p.y,inv*p.z});}
        if(tid==0u){q->alpha_direction/=sqrt(q->norm2);q->input_alpha[q->count]=q->alpha_direction;
            ar->direction/=sqrt(q->norm2);ar->input[q->count]=ar->direction;}
        __syncthreads();
        // Fixed physical scaling: velocity and half-kick-equivalent alpha.
        // Central solver state is never moved by these derivative probes.
        double probe_norm=core?bp_ed_dot(bp_ns_load3d(q->direction,tid),bp_ns_load3d(q->direction,tid)):0.;
        probe_norm=bp_v10_sum(probe_norm,s->warp_red_f64);
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,probe_norm+q->alpha_direction*q->alpha_direction+bp_v10_arc_term(s,ar->direction,ar->direction),0.,0.,isfinite(probe_norm)))return;
        if(tid==0u)q->probe_weight=q->fd_radius*sqrt(1.+q->base_input_norm2)/sqrt(fmax(s->invocation.c4_collective[2].sum0,1e-30));
        __syncthreads();
        if(core){const auto b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->direction,tid);
            bp_ns_store3d(ns->nuclear_trial_velocity,tid,{__fma_rn(q->probe_weight,p.x,b.x),__fma_rn(q->probe_weight,p.y,b.y),__fma_rn(q->probe_weight,p.z,b.z)});}
        if(tid==0u)s->dg_alpha_global=__fma_rn(q->probe_weight,q->alpha_direction/q->alpha_scale,s->invocation.joint_base_alpha);
        __syncthreads();bp_v10_arc_publish_trial(s,q->probe_weight);return;
    }
    if(ar->tangent_mode){
        if(!(q->projected_norm2<=1e-6)){
            if(tid==0u)s->local_fault_accum|=FAULT_PICARD_NONCONV;
            __syncthreads();return;
        }
        double n=core?bp_ed_dot(bp_ns_load3d(q->direction,tid),bp_ns_load3d(q->direction,tid)):0.;
        n=bp_v10_sum(n,s->warp_red_f64);
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,n+q->alpha_direction*q->alpha_direction+.125*ar->direction*ar->direction,0.,0.,isfinite(n)))return;
        if(tid==0u){q->mode=4u;q->direction_error_best=CUDART_INF;q->backtracks=0u;
            q->probe_weight=1e-4*sqrt((1.+q->base_input_norm2)/fmax(s->invocation.c4_collective[2].sum0,1e-30));
            q->alpha_work_dx=q->alpha_work_df;ar->work_dx=ar->work_df+1.;}
        __syncthreads();
        if(core){const auto b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->direction,tid);
            bp_ns_store3d(q->work_dx,tid,bp_ns_load3d(q->work_df,tid));
            bp_ns_store3d(ns->nuclear_trial_velocity,tid,{__fma_rn(q->probe_weight,p.x,b.x),__fma_rn(q->probe_weight,p.y,b.y),__fma_rn(q->probe_weight,p.z,b.z)});}
        if(tid==0u)s->dg_alpha_global=__fma_rn(q->probe_weight,q->alpha_direction/q->alpha_scale,s->invocation.joint_base_alpha);
        __syncthreads();bp_v10_arc_publish_trial(s,q->probe_weight);return;
    }
    // Measured input metric on the <=64-dimensional native Krylov space.
    // A trust bound changes the minimizer in this space, rather than shortening
    // every direction equally. Gram storage is dedicated global solver state.
    #pragma unroll 1
    for(uint32_t i=0u;i<q->count;++i){
        #pragma unroll 1
        for(uint32_t j=0u;j<=i;++j){
            double dot=core?bp_ed_dot(bp_ns_load3d(q->dx[i],tid),bp_ns_load3d(q->dx[j],tid)):0.;
            dot=bp_v10_sum(dot,s->warp_red_f64);
            if(!bp_v10_collective_store(s,s->invocation.collective,2u,dot+q->alpha_dx[i]*q->alpha_dx[j]+bp_v10_arc_term(s,ar->dx[i],ar->dx[j]),0.,0.,isfinite(dot)))return;
            if(tid==0u)q->trust_gram[i*64u+j]=q->trust_gram[j*64u+i]=s->invocation.c4_collective[2].sum0;
            __syncthreads();
        }
    }
    // First bound estimates the allowed input norm from the full Newton step.
    double ratio=0.;
    if(core){const auto b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->direction,tid),f=bp_ns_load3d(ns->nuclear_base_direction,tid),r=bp_ns_load3d(q->work_df,tid);
        const BiprismVec64 dg={p.x-f.x+r.x,p.y-f.y+r.y,p.z-f.z+r.z};
        const double distance=bp_v10_half(s)*sqrt(fmax(bp_ed_dot(p,p),bp_ed_dot(dg,dg)));ratio=distance/q->trust_radius;
        const auto base=bp_v10_advected_position(s,tid,bp_v10_half(s),b);
        const auto* index=&s->invocation.collective.streams->d_ordered_edges[s->invocation.collective.leaf];
        #pragma unroll 1
        for(uint32_t i=index->offsets[tid];i<index->offsets[tid+1u];++i){const uint32_t e=index->edge_index[i],u=s->edge_u8[e][0],v=s->edge_u8[e][1],other=u==tid?v:u;
            const auto peer=other<TILE_CORE_VERTICES?bp_v10_advected_position(s,other,bp_v10_half(s),bp_ns_load3d(ns->nuclear_base_velocity_f64,other)):bp_v10_position_pair(s,other);
            const auto r=bp_coord_difference(peer,base);
            const auto pp=other<TILE_CORE_VERTICES?bp_ns_load3d(q->direction,other):BiprismVec64{0.,0.,0.};
            const BiprismVec64 dr={bp_v10_half(s)*(pp.x-p.x),bp_v10_half(s)*(pp.y-p.y),bp_v10_half(s)*(pp.z-p.z)};
            const double r2=bp_ed_dot<double>({r.x,r.y,r.z},{r.x,r.y,r.z}),rd=bp_ed_dot<double>({r.x,r.y,r.z},dr),dd=bp_ed_dot(dr,dr);
            // Exact first intersection of the relative edge's line segment
            // with |r(w)|=|r(0)|/2. Translation and outward motion are free;
            // inward crossings remain guarded throughout the proposed segment.
            const double disc=__fma_rn(rd,rd,-.75*r2*dd);
            if(!(r2>0.))ratio=CUDART_INF;
            else if(rd<0.&&disc>0.){const double hit=.75*r2/(-rd+sqrt(disc));ratio=fmax(ratio,1./(.9*hit));}}
    }
    if(ar->active){
        ratio=fmax(ratio,fabs(ar->direction)/.125);
        if(ar->direction<0.)ratio=fmax(ratio,-ar->direction/(.9*ar->base_h));
        if(core){const auto b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->direction,tid);
            ratio=fmax(ratio,.5*double(ns->nuclear_evaluations.original_h)*fabs(ar->direction)
                *(sqrt(bp_ed_dot(b,b))+sqrt(bp_ed_dot(p,p)))/q->trust_radius);}
    }
    ratio=bp_v10_max(ratio,s->warp_red_f64);
    double correction_norm=core?bp_ed_dot(bp_ns_load3d(q->direction,tid),bp_ns_load3d(q->direction,tid)):0.;
    correction_norm=bp_v10_sum(correction_norm,s->warp_red_f64);
    if(!bp_v10_collective_store(s,s->invocation.collective,2u,correction_norm+q->alpha_direction*q->alpha_direction+bp_v10_arc_term(s,ar->direction,ar->direction),0.,ratio,isfinite(ratio)))return;
    if(tid==0u){const double bound=fmax(1.,s->invocation.c4_collective[2].max0);
        q->trust_norm2=s->invocation.c4_collective[2].sum0/(bound*bound);
        q->trust_status=bp_newton_trust<64>(q->count,q->trust_norm2,q->trust_gram,q->coefficient,q->trust_factor,q->trust_y,q->regularization)?1u:0u;
        q->model_slope=0.;q->model_curvature=0.;
        for(uint32_t i=0u;i<q->count;++i){q->model_slope+=2.*q->coefficient[i]*q->trust_y[i];q->model_curvature+=q->trust_y[i]*q->trust_y[i];}
        q->alpha_direction=0.;q->alpha_work_df=s->invocation.previous_gonzalez_residual;
            ar->direction=0.;ar->work_df=ar->base_defect;
        for(uint32_t i=0u;i<q->count;++i){q->alpha_direction+=q->trust_y[i]*q->alpha_dx[i];q->alpha_work_df+=q->trust_y[i]*q->alpha_df[i];
            ar->direction+=q->trust_y[i]*ar->dx[i];ar->work_df+=q->trust_y[i]*ar->df[i];}
    }
    __syncthreads();
    if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,0.,q->trust_status!=0u).valid)return;
    if(core){BiprismVec64 p={0.,0.,0.},r=bp_ns_load3d(ns->nuclear_base_direction,tid);
        #pragma unroll 1
        for(uint32_t i=0u;i<q->count;++i){const auto t=bp_ns_load3d(q->dx[i],tid),a=bp_ns_load3d(q->df[i],tid);const double y=q->trust_y[i];
            p={__fma_rn(y,t.x,p.x),__fma_rn(y,t.y,p.y),__fma_rn(y,t.z,p.z)};
            r={__fma_rn(y,a.x,r.x),__fma_rn(y,a.y,r.y),__fma_rn(y,a.z,r.z)};}
        bp_ns_store3d(q->direction,tid,p);bp_ns_store3d(q->work_df,tid,r);}
    __syncthreads();
    // Recheck the actual reshaped direction against every physical edge bound.
    ratio=0.;
    if(core){const auto b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->direction,tid),f=bp_ns_load3d(ns->nuclear_base_direction,tid),r=bp_ns_load3d(q->work_df,tid);
        const BiprismVec64 dg={p.x-f.x+r.x,p.y-f.y+r.y,p.z-f.z+r.z};
        const double distance=bp_v10_half(s)*sqrt(fmax(bp_ed_dot(p,p),bp_ed_dot(dg,dg)));ratio=distance/q->trust_radius;
        const auto base=bp_v10_advected_position(s,tid,bp_v10_half(s),b);
        const auto* index=&s->invocation.collective.streams->d_ordered_edges[s->invocation.collective.leaf];
        #pragma unroll 1
        for(uint32_t i=index->offsets[tid];i<index->offsets[tid+1u];++i){const uint32_t e=index->edge_index[i],u=s->edge_u8[e][0],v=s->edge_u8[e][1],other=u==tid?v:u;
            const auto peer=other<TILE_CORE_VERTICES?bp_v10_advected_position(s,other,bp_v10_half(s),bp_ns_load3d(ns->nuclear_base_velocity_f64,other)):bp_v10_position_pair(s,other);
            const auto r=bp_coord_difference(peer,base);
            const auto pp=other<TILE_CORE_VERTICES?bp_ns_load3d(q->direction,other):BiprismVec64{0.,0.,0.};
            const BiprismVec64 dr={bp_v10_half(s)*(pp.x-p.x),bp_v10_half(s)*(pp.y-p.y),bp_v10_half(s)*(pp.z-p.z)};
            const double r2=bp_ed_dot<double>({r.x,r.y,r.z},{r.x,r.y,r.z}),rd=bp_ed_dot<double>({r.x,r.y,r.z},dr),dd=bp_ed_dot(dr,dr);
            // Exact first intersection of the relative edge's line segment
            // with |r(w)|=|r(0)|/2. Translation and outward motion are free;
            // inward crossings remain guarded throughout the proposed segment.
            const double disc=__fma_rn(rd,rd,-.75*r2*dd);
            if(!(r2>0.))ratio=CUDART_INF;
            else if(rd<0.&&disc>0.){const double hit=.75*r2/(-rd+sqrt(disc));ratio=fmax(ratio,1./(.9*hit));}}
    }
    if(ar->active){
        ratio=fmax(ratio,fabs(ar->direction)/.125);
        if(ar->direction<0.)ratio=fmax(ratio,-ar->direction/(.9*ar->base_h));
        if(core){const auto b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->direction,tid);
            ratio=fmax(ratio,.5*double(ns->nuclear_evaluations.original_h)*fabs(ar->direction)
                *(sqrt(bp_ed_dot(b,b))+sqrt(bp_ed_dot(p,p)))/q->trust_radius);}
    }
    ratio=bp_v10_max(ratio,s->warp_red_f64);
    double direction_norm=core?bp_ed_dot(bp_ns_load3d(q->direction,tid),bp_ns_load3d(q->direction,tid)):0.;
    direction_norm=bp_v10_sum(direction_norm,s->warp_red_f64);
    if(!bp_v10_collective_store(s,s->invocation.collective,2u,direction_norm+q->alpha_direction*q->alpha_direction+bp_v10_arc_term(s,ar->direction,ar->direction),0.,ratio,isfinite(ratio)))return;
    if(tid==0u){q->trial_weight=1./fmax(1.,s->invocation.c4_collective[2].max0);
        q->initial_trial_weight=q->trial_weight;q->backtracks=0u;
        if(!q->direction_checked&&(q->count>=3u||ns->nuclear_evaluations.continuation_stage==16u)){q->mode=4u;q->direction_error_best=CUDART_INF;
            q->probe_weight=1e-4*sqrt((1.+q->base_input_norm2)/fmax(s->invocation.c4_collective[2].sum0,1e-30));
            q->alpha_work_dx=q->alpha_work_df-s->invocation.previous_gonzalez_residual;
            ar->work_dx=ar->work_df-ar->base_defect;}
        q->predicted_norm2=q->base_norm2+q->trial_weight*q->model_slope+q->trial_weight*q->trial_weight*q->model_curvature;}
    __syncthreads();
    if(core){const auto b=bp_ns_load3d(ns->nuclear_base_velocity_f64,tid),p=bp_ns_load3d(q->direction,tid);
        if(q->mode==4u){const auto r=bp_ns_load3d(q->work_df,tid),f=bp_ns_load3d(ns->nuclear_base_direction,tid);
            bp_ns_store3d(q->work_dx,tid,{r.x-f.x,r.y-f.y,r.z-f.z});}
        const double w=q->mode==4u?q->probe_weight:q->trial_weight;
        bp_ns_store3d(ns->nuclear_trial_velocity,tid,{__fma_rn(w,p.x,b.x),__fma_rn(w,p.y,b.y),__fma_rn(w,p.z,b.z)});}
    if(tid==0u){const double w=q->mode==4u?q->probe_weight:q->trial_weight;
        s->dg_alpha_global=__fma_rn(w,q->alpha_direction/q->alpha_scale,s->invocation.joint_base_alpha);}
    __syncthreads();bp_v10_arc_publish_trial(s,q->mode==4u?q->probe_weight:q->trial_weight);
}
// Both checked phase-exchange barriers precede this local reduction scope.
__device__ __noinline__ void bp_v10_publish_phase_traces(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;auto& ns=s->invocation.numerical;
    const float wxl=s->stage.accepted_phase[2u*tid],wxt=s->stage.accepted_phase[2u*tid+1u];
    const float sx=cta_dot_product_256(wxl,s->warp_red),st=cta_dot_product_256(wxt,s->warp_red);
    // Publish the final accepted substep's C1-generator phase traces.
    // Each value is an edge sum, normalized once at outer commit; summing
    // four substeps here would give a different range and LIF drive.
    if(tid==0u){ns->scalar[BP_NS_WXL]=double(sx);ns->scalar[BP_NS_WXT]=double(st);}
}
// Solver-private continuation changes the initial guess, never accepted time.
// Every stage starts from the SAME physical z_n. The original h is restored
// exactly before the only branch that can publish a physical substep.
__device__ __forceinline__ void bp_v10_advance_nuclear_continuation(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;
    if(tid<TILE_CORE_VERTICES){
        if(s->invocation.numerical->nuclear_evaluations.continuation_stage==0u)
            bp_ns_store3d(s->invocation.numerical->nuclear_trial_velocity,tid,
                bp_v10_vector(bp_ns_load3(s->invocation.numerical->v_start,tid)));
        else {
            const auto v0=bp_v10_vector(bp_ns_load3(s->invocation.numerical->v_start,tid));
            const auto v=bp_ns_load3d(s->invocation.numerical->nuclear_midpoint_velocity,tid);
            const double ratio=double(s->invocation.numerical->nuclear_evaluations.next_stage)/double(s->invocation.numerical->nuclear_evaluations.continuation_stage);
            bp_ns_store3d(s->invocation.numerical->nuclear_trial_velocity,tid,
                {__fma_rn(ratio,v.x-v0.x,v0.x),__fma_rn(ratio,v.y-v0.y,v0.y),__fma_rn(ratio,v.z-v0.z,v0.z)});
        }
        bp_ns_store3d(s->invocation.numerical->nuclear_secant.continuation_previous_velocity,tid,
            s->invocation.numerical->nuclear_evaluations.continuation_stage?
            bp_ns_load3d(s->invocation.numerical->nuclear_midpoint_velocity,tid):
            bp_v10_vector(bp_ns_load3(s->invocation.numerical->v_start,tid)));
        s->psi_trial[tid]=bp_v10_starts(s)[tid];bp_v10_candidates(s)[tid]=bp_v10_starts(s)[tid];
    }
    __syncthreads();
    if(tid==0u){
        auto& e=s->invocation.numerical->nuclear_evaluations;auto& q=s->invocation.numerical->nuclear_secant;
        // First-order velocity and second-order quadrature predictors. Their
        // errors are corrected by the unchanged full native residual, never
        // credited as physical work or used as acceptance substitutes.
        // Dedicated previous-stage history is retained for diagnosis only.
        // Native127 disproved using its secant as the active predictor.
        q.continuation_previous_alpha=e.continuation_stage?s->dg_alpha_global:0.;
        q.continuation_previous_stage=double(e.continuation_stage);
        if(e.continuation_stage==0u)s->dg_alpha_global=0.;
        else {const double ratio=double(e.next_stage)/double(e.continuation_stage);s->dg_alpha_global*=ratio*ratio;}
        e.continuation_stage=e.next_stage;e.next_stage=0u;
        bp_v10_publish_h(s,double(e.original_h)*(double(e.continuation_stage)*.0625));
        s->invocation.joint_alpha_scale2=0.;s->invocation.joint_base_alpha=0.;
        s->invocation.previous_gonzalez_alpha=0.;s->invocation.previous_gonzalez_residual=0.;
        q.count=0u;q.have_previous=0u;q.valid=0u;q.pre_count=0u;q.mode=0u;
        q.accepted_trial_ratio=0.;q.fd_radius=1e-4;q.trust_radius=.05;q.initial_trial_weight=0.;q.backtracks=0u;q.direction_checked=0u;
        q.target_refresh=0u;q.projected_norm2=0.;q.base_norm2=0.;q.probe_weight=0.;q.trial_weight=0.;
        s->invocation.electronic_target.epsilon_offset=0.;s->invocation.electronic_target.energy_offset=0.;
        s->invocation.electronic_target.raw_epsilon=CUDART_NAN;s->invocation.electronic_target.value=0.;
    }
    __syncthreads();
}
// Turn along the solution branch, including folds in h. Only the restored
// original-h branch in the caller can accept a physical substep. The secant
// sets a tangent/constraint, not an unchecked extrapolation at a larger h.
__device__ __forceinline__ bool bp_v10_private_nuclear_converged(double raw,double h,double original_h) {
    // A small private h must not hide a nonzero velocity fixed-point residual.
    // Keep the existing private tolerance at the fixed original-h scale.
    return isfinite(raw)&&raw>=0.&&isfinite(h)&&h>0.&&isfinite(original_h)&&original_h>0.
        &&raw<=1e-6*(h/original_h);
}
__device__ __forceinline__ bool bp_v10_private_coordinate_work_converged(bool gonzalez,double smooth_work) {
    return !gonzalez||(isfinite(smooth_work)&&fabs(smooth_work)<=1e-4);
}
__device__ __noinline__ void bp_v10_arc_advance_stage(VariationalClusterDsmemV10* s) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    auto* ns=s->invocation.numerical;auto& a=ns->nuclear_arc;auto& q=ns->nuclear_secant;auto& e=ns->nuclear_evaluations;
    if(a.active&&a.current_h>=1.){
        if(tid==0u){a.active=0u;a.current_h=1.;e.continuation_stage=16u;e.next_stage=0u;
            bp_v10_publish_h(s,double(e.original_h));
            s->invocation.joint_alpha_scale2=0.;}
    }else{
        // Preserve the step qualified by the previous corrector. The geometric
        // bound is a ceiling, not a reason to discard a successful cutback and
        // immediately request the same unresolved large segment again.
        if(tid==0u&&a.active)a.step*=bp_v10_arc_step_factor(a.segment_accepted_trials+(q.mode==2u?1u:0u));
        __syncthreads();
        if(!a.active){
            if(core){bp_ns_store3d(a.anchor_v,tid,bp_v10_vector(bp_ns_load3(ns->v_start,tid)));
                bp_ns_store3d(a.tangent_v,tid,{0.,0.,0.});}
            if(tid==0u){a.scale=sqrt(s->invocation.joint_alpha_scale2>0.?s->invocation.joint_alpha_scale2:1.);
                a.anchor_alpha=0.;a.tangent_alpha=0.;a.anchor_h=0.;a.tangent_h=1.;a.roots=0u;
                a.reference_row=5e-9/(.5*double(e.original_h)*2e-7);}
        }
        __syncthreads();
        double n=0.,orientation=0.;
        if(core){const auto v=bp_v10_live_velocity(s,tid),b=bp_ns_load3d(a.anchor_v,tid),t=bp_ns_load3d(a.tangent_v,tid);
            const BiprismVec64 d={v.x-b.x,v.y-b.y,v.z-b.z};n=bp_ed_dot(d,d);orientation=bp_ed_dot(d,t);}
        n=bp_v10_sum(n,s->warp_red_f64);orientation=bp_v10_sum(orientation,s->warp_red_f64);
        const double dh=s->invocation.collective.h/double(e.original_h)-a.anchor_h,da=a.scale*s->dg_alpha_global-a.anchor_alpha;
        if(!bp_v10_collective_store(s,s->invocation.collective,2u,n+da*da+.125*dh*dh,
            orientation+da*a.tangent_alpha+.125*dh*a.tangent_h,0.,isfinite(n)))return;
        if(tid==0u){a.norm=s->invocation.c4_collective[2].sum0;
            a.dot=(a.roots&&s->invocation.c4_collective[2].sum1<0.?-1.:1.)/sqrt(fmax(a.norm,1e-30));}
        __syncthreads();
        // All leaves see the same reduced chord. Do not overwrite the anchor
        // or repeatedly re-enter a failed transition (native142 rows903..1022).
        if(!(isfinite(a.norm)&&a.norm>1e-28)){
            if(tid==0u){s->local_fault_accum|=FAULT_PICARD_NONCONV;e.next_stage=0u;}
            __syncthreads();return;
        }
        if(core){const auto v=bp_v10_live_velocity(s,tid),b=bp_ns_load3d(a.anchor_v,tid);
            bp_ns_store3d(a.tangent_v,tid,{a.dot*(v.x-b.x),a.dot*(v.y-b.y),a.dot*(v.z-b.z)});
            bp_ns_store3d(a.anchor_v,tid,v);}
        if(tid==0u){a.tangent_alpha=a.dot*da;a.tangent_h=a.dot*dh;
            a.anchor_alpha=a.scale*s->dg_alpha_global;a.anchor_h=s->invocation.collective.h/double(e.original_h);}
        __syncthreads();
        if(a.reserved==0u){bp_v10_arc_predict_from_tangent(s);return;}
        if(core)bp_ns_store3d(ns->nuclear_trial_velocity,tid,bp_ns_load3d(a.anchor_v,tid));
        if(tid==0u){a.active=1u;a.tangent_mode=1u;a.current_h=a.anchor_h;
            bp_v10_publish_h(s,double(e.original_h)*a.anchor_h);
            s->dg_alpha_global=a.anchor_alpha/a.scale;e.continuation_stage=1u;e.next_stage=0u;}
    }
    __syncthreads();
    if(core){s->psi_trial[tid]=bp_v10_starts(s)[tid];bp_v10_candidates(s)[tid]=bp_v10_starts(s)[tid];}
    if(tid==0u){q.count=0u;q.have_previous=0u;q.valid=0u;q.pre_count=0u;q.mode=0u;bp_v10_reset_tangent_probes(q,a);
        q.accepted_trial_ratio=0.;q.fd_radius=e.continuation_stage==16u?1e-7:1e-6;q.trust_radius=.05;q.initial_trial_weight=0.;q.backtracks=0u;q.direction_checked=0u;
        q.target_refresh=0u;q.projected_norm2=0.;q.base_norm2=0.;q.probe_weight=0.;q.trial_weight=0.;
        s->invocation.electronic_target.epsilon_offset=0.;s->invocation.electronic_target.energy_offset=0.;
        s->invocation.electronic_target.raw_epsilon=CUDART_NAN;s->invocation.electronic_target.value=0.;}
    __syncthreads();
}
extern "C" __global__ __cluster_dims__(8,1,1) __launch_bounds__(256,2)
void nhs_biprism_variational_so3g_v10(
    const FoliatedStreamPtrTableFfi* __restrict__ arg_d_stream_ptrs,
    const float* __restrict__ arg_d_amber_charges,
    ContactShellSimplicialStateFfi* __restrict__ arg_d_tile_state,
    InterferometricAdjudicatorFfi* __restrict__ arg_d_adjudicator_ring,
    BiprismConvergenceRecordFfi* __restrict__ arg_d_conv_records,
    uint32_t* __restrict__ arg_d_barrier_arrival_count,
    uint32_t* __restrict__ arg_d_barrier_epoch,
    const float arg_dt_outer_au,const float arg_kappa_foliated,const uint32_t arg_integration_mode
) {
    extern __shared__ __align__(128) unsigned char dsmem_raw_bytes[];
    auto* s=reinterpret_cast<VariationalClusterDsmemV10*>(dsmem_raw_bytes);
    const uint32_t tid=threadIdx.x,leaf=blockIdx.x&7u;const bool core=tid<TILE_CORE_VERTICES;
    bool configuration=gridDim.x==8u&&blockDim.x==256u&&arg_d_stream_ptrs&&arg_d_tile_state&&arg_d_conv_records&&arg_d_amber_charges
        &&isfinite(arg_kappa_foliated)&&(arg_integration_mode&~3u)==0u
        &&(get_hw_cluster_size_x()>=8u||(arg_d_barrier_arrival_count&&arg_d_barrier_epoch));
    if(configuration)for(uint32_t l=0;l<8u;++l)configuration=configuration&&arg_d_stream_ptrs->d_electronic_exchange[l]
        &&arg_d_stream_ptrs->d_numerical_scratch[l]&&arg_d_stream_ptrs->d_coords_xyz[l]&&arg_d_stream_ptrs->d_vels_xyz[l]&&arg_d_stream_ptrs->d_forces_xyz[l]
        &&((arg_integration_mode&FlexibleCoreVerified)?arg_d_stream_ptrs->d_true_inv_masses[l]:arg_d_stream_ptrs->d_inv_masses[l]);
    if(configuration){
        configuration=arg_d_stream_ptrs->d_coordinate_owners
            &&(reinterpret_cast<uintptr_t>(arg_d_stream_ptrs->d_coordinate_owners)&15u)==0u
            &&arg_d_stream_ptrs->d_ordered_edges
            &&(reinterpret_cast<uintptr_t>(arg_d_stream_ptrs->d_ordered_edges)&127u)==0u
            &&arg_d_stream_ptrs->d_potential_owners
            &&(reinterpret_cast<uintptr_t>(arg_d_stream_ptrs->d_potential_owners)&127u)==0u
            &&arg_d_stream_ptrs->d_spinor_owners
            &&(reinterpret_cast<uintptr_t>(arg_d_stream_ptrs->d_spinor_owners)&127u)==0u
            &&arg_d_stream_ptrs->reserved_coordinate_alignment==0u;
        #pragma unroll 1
        for(uint32_t l=0u;configuration&&l<8u;++l){
            const auto* owner=&arg_d_stream_ptrs->d_coordinate_owners[l];
            configuration=bp_coord_owner_valid(owner,l,owner->atom_count,owner->generation,
                BP_COORD_ATOMIC_BOHR,BP_COORD_ALL_INITIALIZED,arg_d_stream_ptrs->d_coords_xyz[l]);
        }
    }
    float selected_dt=0.f;
    if(configuration)configuration=bp_v10_outer_step_input(arg_d_stream_ptrs,arg_d_conv_records,arg_dt_outer_au,selected_dt);
    if(!configuration){if(tid==0u&&arg_d_conv_records)atomicOr(&arg_d_conv_records[leaf].terminal_fault_latch,FAULT_PICARD_NONCONV);return;}

    if(tid==0u){
        s->invocation.collective.streams=arg_d_stream_ptrs;s->invocation.collective.records=arg_d_conv_records;
        s->invocation.collective.arrival=arg_d_barrier_arrival_count;s->invocation.collective.epoch=arg_d_barrier_epoch;
        s->invocation.collective.leaf=leaf;s->invocation.collective.substep=0u;s->invocation.collective.sequence=0u;s->invocation.collective.kappa=double(arg_kappa_foliated);
        s->invocation.tile=&arg_d_tile_state[leaf];s->invocation.amber=arg_d_amber_charges;
        s->invocation.adjudicator=arg_d_adjudicator_ring;
        s->invocation.bank=arg_d_stream_ptrs->d_electronic_exchange[leaf];
        s->invocation.numerical=arg_d_stream_ptrs->d_numerical_scratch[leaf];
        s->invocation.masses=(arg_integration_mode&FlexibleCoreVerified)?arg_d_stream_ptrs->d_true_inv_masses[leaf]:arg_d_stream_ptrs->d_inv_masses[leaf];
        s->invocation.edges=arg_d_tile_state[leaf].num_active_edges;
        s->invocation.h=selected_dt*.25f;s->invocation.half=.5f*s->invocation.h;
        s->invocation.collective.h=double(s->invocation.h);
        s->invocation.integration_mode=arg_integration_mode;
        s->invocation.input_phi=nullptr;s->invocation.potential_step=0u;
        s->local_fault_accum=arg_d_conv_records[leaf].terminal_fault_latch;
    }
    __syncthreads();







    if(!bp_v10_publication_open(s,s->invocation.collective))return;
    // The owner supplies the full-N bound missing from the historical entry.
    // Refuse collectively before any ID-derived mass/charge/position access.
    if(bp_v10_reload_atom_id(s,tid)>=bp_v10_coordinate_owner(s).atom_count)atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);
    if(tid==0u){
        const auto& owner=s->invocation.collective.streams->d_potential_owners[leaf];
        const auto& first=s->invocation.collective.streams->d_potential_owners[0];
        const bool production=s->invocation.collective.streams->physical_step_mode==0u;
        const uint32_t step=production?s->invocation.collective.streams->d_shared_step->scheduled_step
            :(first.state==BP_POTENTIAL_PREPARED?first.prepared_step:first.accepted_step+1u);
        const auto& record=s->invocation.collective.records[leaf];
        const bool valid=record.abi_magic==BIPRISM_TAIL_ABI_MAGIC&&record.leaf_id==leaf
            &&bp_potential_owner_valid(&owner,leaf,s->invocation.tile,&bp_v10_coordinate_owner(s))
            &&bp_potential_input_valid(owner,record.committed_substeps,step,production)
            &&bp_spinor_owner_valid(&bp_v10_spinor_owner(s),leaf,s->invocation.tile,&bp_v10_coordinate_owner(s),
                owner.allocation_generation,owner.topology_generation,record.committed_substeps,owner.accepted_step);
        if(valid){s->invocation.input_phi=bp_potential_input_phi(owner);s->invocation.potential_step=step;}
        else atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);
    }
    __syncthreads();
    if(s->invocation.input_phi&&!bp_potential_projection_value_valid(s->invocation.collective.streams->d_potential_owners[leaf],tid,s->invocation.tile->phi_dec[tid]))
        atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);
    __syncthreads();
    if(collective_fault_vote_two_barrier_v10(s,s->invocation.collective.records,s->invocation.collective.arrival,s->invocation.collective.epoch,leaf)!=0u)return;
    {
        const uint32_t atom=bp_v10_reload_atom_id(s,tid);
        const float inv_mass=bp_v10_readonly_float(s->invocation.masses+atom),amber=bp_v10_readonly_float(s->invocation.amber+atom);
        const auto position=bp_coord_load(bp_v10_coordinate_owner(s),BiprismCoordinatePlane::Current,atom);
        const auto reference=bp_coord_load(bp_v10_coordinate_owner(s),BiprismCoordinatePlane::MeshReference,atom);
        const float3 x={position.x.lo,position.y.lo,position.z.lo};
        const float3 v=*reinterpret_cast<const float3*>(s->invocation.collective.streams->d_vels_xyz[leaf]+3u*atom);
        const float3 f=*reinterpret_cast<const float3*>(s->invocation.collective.streams->d_forces_xyz[leaf]+3u*atom);
        s->x_mid[tid]=x;s->v_mid[tid]=core?v:make_float3(0.f,0.f,0.f);bp_v10_phi(s)[tid]=s->invocation.input_phi[tid];
        bp_v10_store_position(&s->invocation.numerical->nuclear_position_live[tid],position);
        if(core){
            bp_v10_store_position(&s->invocation.numerical->nuclear_position_start[tid],position);
            bp_ns_store3d(s->invocation.numerical->nuclear_trial_velocity,tid,{double(v.x),double(v.y),double(v.z)});
            bp_ns_store3(s->invocation.numerical->x_start,tid,x);bp_ns_store3(s->invocation.numerical->v_start,tid,v);
            const auto a=s->invocation.tile->psi_vertex[tid][0],b=s->invocation.tile->psi_vertex[tid][1];
            bp_v10_starts(s)[tid]=bp_v10_reload_spinor(&bp_v10_spinor_owner(s).accepted[tid]);
            if(!bp_spinor_projection(bp_v10_starts(s)[tid],a.re,a.im,b.re,b.im))atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);bp_v10_candidates(s)[tid]=bp_v10_starts(s)[tid];s->psi_trial[tid]=bp_v10_reload_spinor(&bp_v10_starts(s)[tid]);
            s->q_live[tid]=__fma_rn(double(.15f),bp_ed_norm2(s->psi_trial[tid].s1),double(amber));
        }
        const bool finite=bp_coord_finite(position)&&bp_coord_finite(reference)
            &&reference.x.hi==s->invocation.tile->ref_pos_q[tid].x&&reference.y.hi==s->invocation.tile->ref_pos_q[tid].y&&reference.z.hi==s->invocation.tile->ref_pos_q[tid].z
            &&isfinite(x.x)&&isfinite(x.y)&&isfinite(x.z)&&isfinite(v.x)&&isfinite(v.y)&&isfinite(v.z)
            &&isfinite(f.x)&&isfinite(f.y)&&isfinite(f.z)&&isfinite(inv_mass)&&inv_mass>0.f&&isfinite(amber)
            &&(!core||fabs(bp_v10_outer_norm(s,tid)-1.)<=1e-7);
        if(!__syncthreads_and(finite)&&tid==0u)atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);
    }
    if(tid<32u)s->invocation.numerical->scalar[tid]=0.; // Never clear the immutable peer-map tail.
    if(tid==0u){
        s->last_picard_residual=0.f;s->last_dec_residual=0.f;s->picard_iters_used=0u;s->dec_iters_used=0u;
        s->published_leaf_fault=s->local_fault_accum;s->collective_fault_mask=s->local_fault_accum;
        s->V_pot_n=0.;s->dec_aggregate={0.f,0u};s->dg_alpha_global=0.;
        s->invocation.bank->footer={};s->invocation.bank->footer.abi_magic=BIPRISM_FOLIATED_COUPLING_ABI;s->invocation.bank->footer.leaf_id=leaf;
        s->invocation.bank->footer.accepted_substeps_at_entry=s->invocation.collective.records[leaf].committed_substeps;
    }
    bool map_ok=s->invocation.edges<=TILE_MAX_EDGES;
    const auto* peer=&arg_d_tile_state[(leaf+1u)&7u];
    for(uint32_t e=tid;e<min(s->invocation.edges,uint32_t(TILE_MAX_EDGES));e+=256u){
        const uint32_t u=s->invocation.tile->edge_u8[e][0],v=s->invocation.tile->edge_u8[e][1],j=s->invocation.numerical->peer_edge_index[e];
        s->edge_u8[e][0]=u;s->edge_u8[e][1]=v;
        const bool bounded=j<peer->num_active_edges&&j<TILE_MAX_EDGES;
        map_ok=map_ok&&bounded;
        if(bounded)map_ok=map_ok&&UNPACK_GLOBAL_ATOM_ID(s->invocation.tile->global_atom_idx[u])==UNPACK_GLOBAL_ATOM_ID(peer->global_atom_idx[peer->edge_u8[j][0]])
            &&UNPACK_GLOBAL_ATOM_ID(s->invocation.tile->global_atom_idx[v])==UNPACK_GLOBAL_ATOM_ID(peer->global_atom_idx[peer->edge_u8[j][1]]);
    }
    if(!__syncthreads_and(map_ok)&&tid==0u)atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);
    __syncthreads();
    if(collective_fault_vote_two_barrier_v10(s,s->invocation.collective.records,s->invocation.collective.arrival,s->invocation.collective.epoch,leaf)!=0u)return;
    // Rebuild every invocation from this published topology. No cached READY
    // state or prior accepted-step identity can bypass counts/prefix/fill.
    BP_V10_BARRIER_REANCHOR(s);
            bp_v10_build_ordered_stage(s);
            BP_V10_BARRIER_REANCHOR(s);
    __syncthreads();
    if(collective_fault_vote_two_barrier_v10(s,s->invocation.collective.records,s->invocation.collective.arrival,s->invocation.collective.epoch,leaf)!=0u)return;
    const float chi=.12f;
    #pragma unroll 1
    for(;s->invocation.collective.substep<BIPRISM_INNER_SUBSTEPS;){
        if(tid==0u){auto& e=s->invocation.numerical->nuclear_evaluations;
            e.magic=0x4545484eu;e.substep=s->invocation.collective.substep;e.count=0u;e.nonlinear_trials=0u;
            e.derivative_evaluations=0u;e.action_checks=0u;e.target_replays=0u;e.base_evaluations=0u;
            e.accepted_newton=0u;e.rejected_newton=0u;e.original_h=s->invocation.h;e.continuation_stage=0u;e.next_stage=0u;
            for(uint32_t k=0;k<19u;++k)e.reserved[k]=0u;
            auto& arc=s->invocation.numerical->nuclear_arc;
            arc.active=0u;arc.roots=0u;arc.last_root_trial=0u;arc.reserved=0u;arc.current_h=1.;arc.base_h=1.;
            arc.slow_trial_streak=0u;arc.segment_accepted_trials=0u;arc.tangent_mode=0u;arc.tangent_probe_refinements=0u;
            arc.base_defect=0.;arc.work_dx=0.;arc.work_df=0.;arc.direction=0.;arc.defect=0.;arc.step=0.;}
        __syncthreads();
        if(tid==0u){
            s->invocation.numerical->scalar[BP_NS_ATTEMPT_ENERGY]=CUDART_NAN;s->invocation.numerical->scalar[BP_NS_ATTEMPT_EQUATION]=CUDART_NAN;s->dec_aggregate={0.f,0u};
            s->invocation.picard=0u;s->invocation.stage_valid=1u;
            s->invocation.derivative_coordinate_phase=0u;
            s->invocation.previous_gonzalez_alpha=0.;s->invocation.previous_gonzalez_residual=0.;
            s->invocation.nuclear_iteration_weight=.5;
            s->invocation.joint_alpha_scale2=0.;s->invocation.joint_base_alpha=0.;
            s->invocation.numerical->nuclear_secant.count=0u;s->invocation.numerical->nuclear_secant.have_previous=0u;s->invocation.numerical->nuclear_secant.valid=0u;
            s->invocation.numerical->nuclear_secant.pre_count=0u;
            s->invocation.numerical->nuclear_secant.mode=0u;s->invocation.numerical->nuclear_secant.fd_radius=1e-4;
            s->invocation.numerical->nuclear_secant.accepted_trial_ratio=0.;
            s->invocation.numerical->nuclear_secant.trust_radius=.05;
            s->invocation.numerical->nuclear_secant.initial_trial_weight=0.;s->invocation.numerical->nuclear_secant.backtracks=0u;s->invocation.numerical->nuclear_secant.direction_checked=0u;
            s->invocation.numerical->nuclear_secant.target_refresh=0u;s->invocation.numerical->nuclear_secant.accepted_newton=0u;
            s->invocation.numerical->nuclear_secant.linear_restarts=0u;s->invocation.numerical->nuclear_secant.probes=0u;
            s->invocation.numerical->nuclear_secant.projected_norm2=0.;s->invocation.numerical->nuclear_secant.base_norm2=0.;
            s->invocation.numerical->nuclear_secant.probe_weight=0.;s->invocation.numerical->nuclear_secant.trial_weight=0.;
            s->invocation.joint_base_merit=CUDART_INF;s->invocation.joint_has_base=0u;
            s->invocation.joint_base_coordinate_phase=0u;
            s->invocation.nuclear_secant_alpha=0.;s->invocation.nuclear_secant_residual=0.;
            s->invocation.nuclear_secant_valid=0u;s->invocation.nuclear_rebase=0u;
            s->invocation.nuclear_base_coord_work=CUDART_INF;s->invocation.nuclear_raw_residual=CUDART_INF;
            auto& trace=s->invocation.numerical->nuclear_history;
            trace.magic=0x4e485035u;trace.substep=s->invocation.collective.substep;trace.count=0u;trace.reserved=0u;

            s->invocation.electronic_target.epsilon_offset=0.;s->invocation.electronic_target.energy_offset=0.;s->invocation.electronic_target.raw_epsilon=CUDART_NAN;s->invocation.electronic_target.value=0.;
        }
        if(core){
            bp_v10_store_position(&s->invocation.numerical->nuclear_position_live[tid],bp_v10_start_position(s,tid));
            s->x_mid[tid]=bp_ns_load3(s->invocation.numerical->x_start,tid);s->v_mid[tid]=bp_ns_load3(s->invocation.numerical->v_start,tid);
            bp_ns_store3d(s->invocation.numerical->nuclear_trial_velocity,tid,bp_v10_vector(s->v_mid[tid]));
            s->psi_trial[tid]=bp_v10_reload_spinor(&bp_v10_starts(s)[tid]);bp_v10_candidates(s)[tid]=bp_v10_starts(s)[tid];
        }
        __syncthreads();
        bp_v10_publish_charge(s,double(bp_v10_readonly_float(s->invocation.amber+bp_v10_reload_atom_id(s,tid))),0u,true);
        bool valid=bp_v10_refresh_weights(s,s->invocation.tile,s->invocation.edges,s->invocation.collective);
        {float residual=0.f;uint32_t iterations=0u;
         valid=solve_live_dec_poisson_pcg_v10(s,s->invocation.tile,s->invocation.edges,s->invocation.bank->live_weights,residual,iterations)&&valid;}
        if(threadIdx.x==0u)s->invocation.energy_bank=0u;
        BP_V10_BARRIER_REANCHOR(s);
        evaluate_total_potential_energy_v10(s);
        BP_V10_BARRIER_REANCHOR(s);
        BP_V10_BARRIER_REANCHOR(s);
            bp_v10_initial_nuclear_stage(s);
            BP_V10_BARRIER_REANCHOR(s);
        __syncthreads(); // Finish all SPD fault writers before passing validity.
        if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,0.,valid&&s->local_fault_accum==0u&&isfinite(s->invocation.potential_result)).valid)break;
        if(tid==0u){s->V_pot_n=s->invocation.potential_result;s->dg_alpha_global=0.;}
        if(core){const auto x=bp_ns_load3(s->invocation.numerical->x_start,tid),v=bp_ns_load3(s->invocation.numerical->v_start,tid);
            bp_ns_store3(s->invocation.numerical->next_mid_x,tid,{__fmaf_rn(s->invocation.half,v.x,x.x),__fmaf_rn(s->invocation.half,v.y,x.y),__fmaf_rn(s->invocation.half,v.z,x.z)});bp_ns_store3(s->invocation.numerical->next_mid_v,tid,v);}
        bool converged=false;
#if defined(BIPRISM_V10_DIAGNOSTIC_INITIAL_GUESS)
        // Explicit standalone diagnosis only. The ordinary QC and production
        // translation units never define this macro. The hook may change an
        // initial guess after start-state applicability checks; no physical state,
        // force, work term, residual, iteration limit or gate is replaced.
        BP_V10_BARRIER_REANCHOR(s);
        bp_v10_diagnostic_initial_guess(s);
        BP_V10_BARRIER_REANCHOR(s);
#endif
        #pragma unroll 1
        for(;s->invocation.picard<BIPRISM_MAX_NUCLEAR_EVALUATIONS
            &&s->invocation.numerical->nuclear_evaluations.nonlinear_trials<BIPRISM_MAX_NUCLEAR_TRIALS;){
            if(tid==0u){auto& trace=s->invocation.numerical->nuclear_history;
                trace.count=s->invocation.picard+1u;
                for(uint32_t k=0u;k<3u;++k)trace.rows[s->invocation.picard%48u][k]=CUDART_NAN;
                auto& e=s->invocation.numerical->nuclear_evaluations;const auto& q=s->invocation.numerical->nuclear_secant;
                const auto& a=s->invocation.numerical->nuclear_arc;
                s->invocation.derivative_coordinate_phase=bp_v10_continuous_tangent_probe(q.mode,a.tangent_mode,a.tangent_probe_refinements)?1u:0u;
                e.count=s->invocation.picard+1u;bp_v10_record_evaluated_h(s);
                if(q.mode==1u)++e.derivative_evaluations;
                else if(q.mode==2u)++e.nonlinear_trials;
                else if(q.mode==4u)++e.action_checks;
                else if(q.valid==2u)++e.target_replays;
                else ++e.base_evaluations;
                e.rejected_newton=e.nonlinear_trials-e.accepted_newton;
                for(uint32_t k=0u;k<9u;++k)bp_v10_evaluation_row(s->invocation.numerical,s->invocation.picard)[k]=CUDART_NAN;}
            __syncthreads();
            // B/C1/C2 and C3 delta consume this input midpoint. C3 alone
            // publishes the raw output velocity and full-h endpoint for C4.
            if(core){
                bp_v10_publish_trial_position(s,tid);
                s->x_mid[tid]=bp_ns_load3(s->invocation.numerical->next_mid_x,tid);
                const auto v=bp_v10_live_velocity(s,tid);s->v_mid[tid]={float(v.x),float(v.y),float(v.z)};
            }
            __syncthreads();valid=bp_v10_refresh_weights(s,s->invocation.tile,s->invocation.edges,s->invocation.collective);
            if(!bp_v10_collective_values(s,s->invocation.collective,0.,0.,0.,valid).valid)break;
            asm volatile("" ::: "memory");
            valid=bp_v10_solve_electronic_trial(s);
            asm volatile("" ::: "memory");
            if(tid==0u){s->invocation.numerical->scalar[BP_NS_ELECTRONIC_DELTA]=0.;s->invocation.numerical->scalar[BP_NS_EQUATION]=CUDART_INF;s->invocation.numerical->scalar[BP_NS_ATTEMPT_EQUATION]=CUDART_INF;}
            __syncthreads();
            if(valid)valid=bp_v10_admit_represented_electronic(s);
            if(!valid){if(tid==0u)atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);break;}
            BP_V10_BARRIER_REANCHOR(s);
            bp_v10_publish_electronic_ledger_stage(s);
            BP_V10_BARRIER_REANCHOR(s);
            {
                asm volatile("" ::: "memory");
                BP_V10_BARRIER_REANCHOR(s);
            bp_v10_force_stage(s);
            BP_V10_BARRIER_REANCHOR(s);
                // The published force vote is conjoined after the local C3
                // finite checks; it need not survive C3 in a return register.
                valid=true;
                asm volatile("" ::: "memory");
            }
            BP_V10_BARRIER_REANCHOR(s);
            bp_v10_midpoint_nuclear_stage(s);
            BP_V10_BARRIER_REANCHOR(s);
            BP_V10_BARRIER_REANCHOR(s);
            bp_v10_endpoint_nuclear_stage(s);
            BP_V10_BARRIER_REANCHOR(s);
            valid=s->invocation.stage_valid!=0u;
            // No per-vertex nuclear/force/work values survive the endpoint solve.
            bp_v10_publish_charge(s,double(bp_v10_readonly_float(s->invocation.amber+bp_v10_reload_atom_id(s,tid))),1u,true);valid=bp_v10_refresh_weights(s,s->invocation.tile,s->invocation.edges,s->invocation.collective)&&valid;
            {float residual=0.f;uint32_t iterations=0u;valid=solve_live_dec_poisson_pcg_v10(s,s->invocation.tile,s->invocation.edges,s->invocation.bank->live_weights,residual,iterations)&&valid;}
            if(threadIdx.x==0u)s->invocation.energy_bank=3u; // C4 endpoint geometry, explicit capture owner.
        BP_V10_BARRIER_REANCHOR(s);
        evaluate_total_potential_energy_v10(s);
        BP_V10_BARRIER_REANCHOR(s);
            if(tid==0u){
                const double numerator=s->invocation.numerical->scalar[BP_NS_DT_METRIC]+s->invocation.potential_result-s->V_pot_n-s->invocation.numerical->scalar[BP_NS_ELECTRONIC_DELTA]+s->invocation.numerical->scalar[BP_NS_BATH_DELTA]+s->invocation.numerical->scalar[BP_NS_WORK_CONS];
                const double dot=s->invocation.numerical->scalar[BP_NS_PROJECTION_DOT],used=(s->invocation.integration_mode&GonzalezExactEnergyBit)?s->dg_alpha_global:0.;
                s->invocation.numerical->scalar[BP_NS_COORD_WORK_DEFECT]=numerator-used*dot;
                s->invocation.numerical->scalar[BP_NS_BASE_ENERGY]=s->invocation.numerical->scalar[BP_NS_DT]+s->invocation.potential_result-s->V_pot_n+s->invocation.numerical->scalar[BP_NS_BATH_DELTA];
                // The C3 diagnostic's last reader has finished. Retain the
                // used target through accepted or refused C4 readback. The
                // coordinate-round term remains reconstructible as slot10-29,
                // up to their separately evaluated reduction roundoff.
                s->invocation.numerical->scalar[BP_NS_C1_SEARCH_BIAS]=-s->invocation.electronic_target.value;
                // Preserve the actually evaluated coefficient for diagnostics.
                // The joint proposal helper updates alpha only after all C4
                // readers and acceptance tests have consumed this candidate.
                s->invocation.previous_gonzalez_alpha=used;
                s->last_picard_residual=float(s->invocation.numerical->scalar[BP_NS_DX_RESIDUAL]);s->last_dec_residual=s->dec_aggregate.max_residual;s->picard_iters_used=s->invocation.picard+1u;s->dec_iters_used=s->dec_aggregate.max_iters;
            }
            if(tid==0u)s->invocation.stage_valid=valid?1u:0u;
            BP_V10_BARRIER_REANCHOR(s);
            bp_v10_endpoint_rounding_stage(s);
            BP_V10_BARRIER_REANCHOR(s);
            valid=s->invocation.stage_valid!=0u;
            if(!bp_v10_arc_evaluate(s))break;
            const bool energy_valid=bp_v10_collective_store(s,s->invocation.collective,0u,s->invocation.numerical->scalar[BP_NS_BASE_ENERGY]+s->invocation.numerical->scalar[BP_NS_PAIR_DELTA],
                s->invocation.numerical->scalar[BP_NS_ELECTRONIC_DELTA]+s->invocation.numerical->scalar[BP_NS_WORK_BERRY]+s->invocation.numerical->scalar[BP_NS_FOL_WORK],
                // Nuclear displacement has its own all-leaf reduction below.
                // Keeping it in this reconstruction maximum also imposed the
                // physical 1e-7 bound on PRIVATE continuation guesses, despite
                // their separate 1e-4 displacement criterion. Final admission
                // still requires raw<=5e-9 AND reconstruction<=1e-7.
                s->invocation.numerical->scalar[BP_NS_RECON_POSITION],valid&&s->local_fault_accum==0u&&isfinite(s->invocation.potential_result));
            if(tid==0u){s->invocation.numerical->scalar[BP_NS_ATTEMPT_ENERGY]=s->invocation.c4_collective[0].sum0;s->invocation.numerical->scalar[BP_NS_GLOBAL_ENERGY]=s->invocation.c4_collective[0].sum0;}
            if(!energy_valid)break; // Do not overwrite publication after failed consumption.
            if(!bp_v10_collective_store(s,s->invocation.collective,1u,s->invocation.numerical->scalar[BP_NS_PAIR_DELTA]-s->invocation.numerical->scalar[BP_NS_FOL_WORK],s->invocation.numerical->scalar[BP_NS_TRANSFER],s->invocation.numerical->scalar[BP_NS_EQUATION],true))break;
            // sum0 counts leaves whose smooth private work row is unresolved.
            // max0 keeps the existing represented physical work gate intact.
            if(!bp_v10_collective_store(s,s->invocation.collective,2u,
                bp_v10_private_coordinate_work_converged(s->invocation.integration_mode&GonzalezExactEnergyBit,
                    s->invocation.numerical->nuclear_secant.alpha_work_df)?0.:1.,0.,(s->invocation.integration_mode&GonzalezExactEnergyBit)?
                (s->invocation.numerical->scalar[BP_NS_PROJECTION_DOT]>=0.?
                    fmax(fabs(s->invocation.numerical->scalar[BP_NS_COORD_WORK_DEFECT]),fabs(s->invocation.numerical->scalar[BP_NS_MID_WORK_DEFECT])):CUDART_INF):0.,true))break;
            const auto& energy=s->invocation.c4_collective[0];
            const auto& chain=s->invocation.c4_collective[1];
            const auto& works=s->invocation.c4_collective[2];
            if(tid==0u)s->invocation.numerical->scalar[BP_NS_CHAIN]=chain.sum0;
            const double raw=bp_v10_collective_values(s,s->invocation.collective,0.,0.,s->invocation.numerical->scalar[BP_NS_DX_RESIDUAL],true).max0;
            if(tid==0u){
                s->invocation.nuclear_raw_residual=raw;
                auto& trace=s->invocation.numerical->nuclear_history;
                trace.rows[s->invocation.picard%48u][0]=raw;
                trace.rows[s->invocation.picard%48u][1]=energy.sum0;
                trace.rows[s->invocation.picard%48u][2]=s->invocation.previous_gonzalez_alpha;
                for(uint32_t k=0;k<3u;++k)bp_v10_evaluation_row(s->invocation.numerical,s->invocation.picard)[k]=trace.rows[s->invocation.picard%48u][k];
            }
            const bool energy_ok=!(s->invocation.integration_mode&GonzalezExactEnergyBit)||(fabs(energy.sum0)<=double(VARIATIONAL_ENERGY_TOL)
                &&fabs(s->invocation.numerical->scalar[BP_NS_OUTER_ERROR]+energy.sum0)<=double(VARIATIONAL_ENERGY_TOL)&&fabs(energy.sum1)<=double(VARIATIONAL_ENERGY_TOL)&&works.max0<=2.0e-7);
            const bool joint=s->invocation.derivative_coordinate_phase==0u&&raw<=5.0e-9&&energy.max0<=double(VARIATIONAL_PICARD_TOL)&&chain.max0<=double(VARIATIONAL_PICARD_TOL)
                &&fabs(chain.sum0)<=double(VARIATIONAL_ENERGY_TOL)&&fabs(chain.sum1)<=double(VARIATIONAL_ENERGY_TOL)&&energy_ok;
#if defined(BIPRISM_V10_DIAGNOSTIC_TRACE)
            if(tid==0u&&leaf==0u)printf("v10_coupled sub=%u picard=%u dH=%.17g outer=%.17g epsilon_e=%.17g dx=%.9g eq=%.9g alpha=%.17g dec=%.9g work=%.9g recon=%.9g fault=%u\n",s->invocation.collective.substep,s->invocation.picard,energy.sum0,s->invocation.numerical->scalar[BP_NS_OUTER_ERROR]+energy.sum0,energy.sum1,raw,chain.max0,s->dg_alpha_global,s->dec_aggregate.max_residual,works.max0,s->invocation.numerical->scalar[BP_NS_RECON_POSITION],s->local_fault_accum);
#endif
            if(tid==0u){

                // Unconverged nuclear motion is not a rounding residual. Keep
                // the previous (initially zero) predictor until all existing
                // non-target conditions hold; the three global energy tests
                // remain independent and unchanged above.
                // Update a target only at a newly evaluated physical input:
                // the first base or a real Newton trial. Mode0 after target
                // invalidation is the deliberate SAME-input reevaluation.
                // Updating again there from last-bit C1 noise starves every
                // derivative/proposal indefinitely (native100 iterations9..47).
                if(!s->invocation.numerical->nuclear_arc.tangent_mode&&(s->invocation.numerical->nuclear_secant.mode==2u
                    ||(s->invocation.numerical->nuclear_secant.mode==0u&&!s->invocation.numerical->nuclear_secant.have_previous))
                    &&!energy_ok&&raw<=5.0e-9
                    &&energy.max0<=double(VARIATIONAL_PICARD_TOL)&&chain.max0<=double(VARIATIONAL_PICARD_TOL)
                    &&fabs(chain.sum0)<=double(VARIATIONAL_ENERGY_TOL)&&fabs(chain.sum1)<=double(VARIATIONAL_ENERGY_TOL)
                    &&works.max0<=double(VARIATIONAL_ENERGY_TOL)
                    &&isfinite(energy.sum0)&&isfinite(energy.sum1)&&isfinite(s->invocation.electronic_target.raw_epsilon)){
                    if(s->invocation.electronic_target.epsilon_offset!=energy.sum1-s->invocation.electronic_target.raw_epsilon
                        ||s->invocation.electronic_target.energy_offset!=energy.sum0-s->invocation.electronic_target.raw_epsilon)
                        s->invocation.numerical->nuclear_secant.target_refresh=1u;
                    s->invocation.electronic_target.epsilon_offset=energy.sum1-s->invocation.electronic_target.raw_epsilon;
                    s->invocation.electronic_target.energy_offset=energy.sum0-s->invocation.electronic_target.raw_epsilon;
                }
            }
            __syncthreads(); // Replicated C4 offsets precede the next Picard C1.
            if(tid==0u){auto& e=s->invocation.numerical->nuclear_evaluations;
                // Try the original equations first. Native122 reached a
                // nonzero merit basin through many accepted but ineffective
                // trust steps, so rejected-step counts alone missed stagnation.
                const auto& q=s->invocation.numerical->nuclear_secant;
                // A derivative probe deliberately displaces the current base.
                // Its raw residual must not be combined with the last real
                // trial's merit ratio to discard that base (native guess005,
                // evaluation89). Only evaluated correction trials can trigger
                // this stagnation transition.
                const bool slow_trial=q.mode==2u&&e.count>=24u&&raw>1e-3&&q.accepted_trial_ratio>.9;
                if(e.continuation_stage==0u
                    &&((q.mode==2u&&q.backtracks>=6u&&raw>.05)||slow_trial))e.next_stage=2u;
                // Intermediate h values only construct an initial guess.
                // Oversolving them to physical publication tolerances spent
                // the original problem's budget without advancing its clock.
                // The original-h publication branch below still requires joint.
                else if(e.continuation_stage>0u&&e.continuation_stage<16u
                    &&!s->invocation.numerical->nuclear_arc.tangent_mode
                    &&bp_v10_private_nuclear_converged(raw,s->invocation.collective.h,double(e.original_h))&&works.max0<=1e-4&&works.sum0==0.
                    &&fabs(s->invocation.numerical->nuclear_arc.defect)<=1e-6
                    &&energy.max0<=double(VARIATIONAL_PICARD_TOL)
                    &&chain.max0<=double(VARIATIONAL_PICARD_TOL))e.next_stage=17u;
                // Preserve the last evaluated candidate when no further
                // private evaluation is available; advancing resets C1 state.
                if(e.count>=BIPRISM_MAX_NUCLEAR_EVALUATIONS||e.nonlinear_trials>=BIPRISM_MAX_NUCLEAR_TRIALS)e.next_stage=0u;
                if(e.next_stage){
                    for(uint32_t k=3u;k<9u;++k)bp_v10_evaluation_row(s->invocation.numerical,s->invocation.picard)[k]=0.;
                    bp_v10_evaluation_row(s->invocation.numerical,s->invocation.picard)[7]=double(1000u*e.continuation_stage+900u);
                    if(s->invocation.numerical->nuclear_secant.mode==2u){++e.accepted_newton;++s->invocation.numerical->nuclear_secant.accepted_newton;}
                    e.rejected_newton=e.nonlinear_trials-e.accepted_newton;
                }}
            __syncthreads();
            if(s->invocation.numerical->nuclear_evaluations.next_stage){
                if(s->invocation.numerical->nuclear_evaluations.next_stage==17u){
                    BP_V10_BARRIER_REANCHOR(s);
                    bp_v10_arc_advance_stage(s);
                    BP_V10_BARRIER_REANCHOR(s);
                }else bp_v10_advance_nuclear_continuation(s);
                if(s->local_fault_accum!=0u)break;
                if(tid==0u)++s->invocation.picard;
                __syncthreads();continue;
            }
            if(s->invocation.picard>=1u&&joint
                &&(s->invocation.numerical->nuclear_evaluations.continuation_stage==0u||s->invocation.numerical->nuclear_evaluations.continuation_stage==16u)
                &&s->invocation.collective.h==double(s->invocation.numerical->nuclear_evaluations.original_h)){
                if(tid==0u&&s->invocation.numerical->nuclear_secant.mode==2u){
                    auto& e=s->invocation.numerical->nuclear_evaluations;++e.accepted_newton;e.rejected_newton=e.nonlinear_trials-e.accepted_newton;}
                converged=true;break;
            }
            BP_V10_BARRIER_REANCHOR(s);
            bp_v10_joint_nuclear_proposal(s);
            BP_V10_BARRIER_REANCHOR(s);
            if(tid==0u){auto& e=s->invocation.numerical->nuclear_evaluations;const auto& q=s->invocation.numerical->nuclear_secant;
                for(uint32_t k=0;k<6u;++k)bp_v10_evaluation_row(s->invocation.numerical,s->invocation.picard)[k+3u]=q.history[s->invocation.picard%48u][k];
                // Proposal may advance the private stage. History already names the evaluated stage.
                e.accepted_newton=q.accepted_newton;e.rejected_newton=e.nonlinear_trials-e.accepted_newton;}
            __syncthreads(); // Finish every current-index reader before its writer.
#if defined(BIPRISM_V10_DIAGNOSTIC_TANGENT_CAPTURE)
            // Standalone forensic stop, never compiled into an eligible QC or
            // production candidate. Retain the evaluated action and its live
            // paired QR before a later failed model overwrites that evidence.
            // The mode/refinement/counter decisions are already collective.
            if(s->invocation.collective.substep==1u
                &&s->invocation.numerical->nuclear_arc.tangent_mode!=0u
                &&s->invocation.numerical->nuclear_arc.tangent_probe_refinements==5u
                &&s->invocation.numerical->nuclear_secant.mode==4u
                &&s->invocation.numerical->nuclear_secant.backtracks==1u){
                break;
            }
#endif
            if(tid==0u)++s->invocation.picard;
            __syncthreads(); // Publish the next Picard index uniformly.
        }
        if(tid==0u&&!converged)atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);
        __syncthreads();if(collective_fault_vote_two_barrier_v10(s,s->invocation.collective.records,s->invocation.collective.arrival,s->invocation.collective.epoch,leaf)!=0u)break;
        if(tid==0u){
            s->invocation.numerical->scalar[BP_NS_OUTER_ERROR]+=s->invocation.numerical->scalar[BP_NS_GLOBAL_ENERGY];
            s->invocation.numerical->scalar[BP_NS_MAX_ENERGY]=fmax(s->invocation.numerical->scalar[BP_NS_MAX_ENERGY],fmax(fabs(s->invocation.numerical->scalar[BP_NS_GLOBAL_ENERGY]),fabs(s->invocation.numerical->scalar[BP_NS_OUTER_ERROR])));
            s->invocation.bank->footer.base_energy_delta=s->invocation.numerical->scalar[BP_NS_BASE_ENERGY];s->invocation.bank->footer.owned_pair_energy_delta=s->invocation.numerical->scalar[BP_NS_PAIR_DELTA];
            s->invocation.bank->footer.attributed_coupling_energy_delta=s->invocation.numerical->scalar[BP_NS_ATTRIBUTED_DELTA];s->invocation.bank->footer.coupling_discrete_work=s->invocation.numerical->scalar[BP_NS_FOL_WORK];
            s->invocation.bank->footer.physical_transfer_in=s->invocation.numerical->scalar[BP_NS_TRANSFER];s->invocation.bank->footer.balanced_local_defect=s->invocation.numerical->scalar[BP_NS_BASE_ENERGY]+s->invocation.numerical->scalar[BP_NS_FOL_WORK];
            s->invocation.bank->footer.global_energy_delta=s->invocation.numerical->scalar[BP_NS_GLOBAL_ENERGY];s->invocation.bank->footer.global_chain_defect=s->invocation.numerical->scalar[BP_NS_CHAIN];
            s->invocation.bank->footer.max_coordinate_residual=float(s->invocation.numerical->scalar[BP_NS_DX_RESIDUAL]);s->invocation.bank->footer.max_population_residual=float(s->invocation.numerical->scalar[BP_NS_POPULATION_RESIDUAL]);
            s->invocation.bank->footer.max_electronic_equation_residual=float(s->invocation.numerical->scalar[BP_NS_EQUATION]);s->invocation.bank->footer.max_norm_defect=float(s->invocation.numerical->scalar[BP_NS_MAX_NORM]);
            s->invocation.bank->footer.local_fault_mask=0u;s->invocation.bank->footer.convergence_vote=1u;s->invocation.bank->footer.publication_phase=4u;
        }
        #pragma unroll 1
        for(uint32_t e=tid;e<s->invocation.edges;e+=256u){const float p=s->invocation.numerical->accepted_phase[e];s->stage.accepted_phase[e]=p;}
        if(!bp_v10_cluster_barrier(s,s->invocation.collective))break;
        float wxl=0.f,wxt=0.f;
        #pragma unroll 1
        for(uint32_t e=tid;e<s->invocation.edges;e+=256u){const uint32_t j=s->invocation.numerical->peer_edge_index[e];
            const float a=s->stage.accepted_phase[e],b=dsmem_load_f32_safe(&s->stage.accepted_phase[j],(leaf+1u)&7u,&s->invocation.collective.streams->d_numerical_scratch[(leaf+1u)&7u]->accepted_phase[j]);
            wxl+=2.f*__cosf(a-b);wxt+=2.f*__cosf(a);}
        if(!bp_v10_cluster_barrier(s,s->invocation.collective))break;
        s->stage.accepted_phase[2u*tid]=wxl;s->stage.accepted_phase[2u*tid+1u]=wxt;
        BP_V10_BARRIER_REANCHOR(s);
        bp_v10_publish_phase_traces(s);
        BP_V10_BARRIER_REANCHOR(s);
        if(core){
            bp_v10_store_position(&s->invocation.numerical->nuclear_position_start[tid],bp_v10_position_pair(s,tid));
            bp_ns_store3(s->invocation.numerical->x_start,tid,s->x_mid[tid]);
            bp_ns_store3(s->invocation.numerical->v_start,tid,bp_ns_load3(s->invocation.numerical->endpoint_v,tid));bp_v10_starts(s)[tid]=bp_v10_candidates(s)[tid];
        }
        __syncthreads();
        // The current substep has no remaining readers. Publish the next
        // loop index uniformly; early refusal paths never reach this update.
        if(tid==0u)++s->invocation.collective.substep;
        __syncthreads();
    }
    // Normalize the final expansion before the last all-leaf vote. The admitted
    // stored pair must preserve exactly the coordinate used by all endpoint
    // checks. The commit repeats this deterministic normalization on immutable
    // own-atom inputs; it introduces no new admission decision after the vote.
    if(core&&!bp_v10_position_exportable(s,tid))atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);
    // The precise value is physical authority. The compatibility projection
    // must remain finite; fixed halo values must retain this invocation's input.
    {const double phi=bp_v10_phi(s)[tid];
     if(!isfinite(phi)||!isfinite(float(phi))||(!core&&phi!=s->invocation.input_phi[tid]))
        atomicOr(&s->local_fault_accum,FAULT_PICARD_NONCONV);}
    __syncthreads();
    const uint32_t fault=collective_fault_vote_two_barrier_v10(s,s->invocation.collective.records,s->invocation.collective.arrival,s->invocation.collective.epoch,leaf);
    // This is the last global barrier; no subsequent operation can create a new
    // collective generation or scientific fault before the guarded state writes.
    if(fault==0u){
        const double phi=bp_v10_phi(s)[tid];
        s->invocation.collective.streams->d_potential_owners[leaf].accepted_phi[tid]=phi;
        s->invocation.tile->phi_dec[tid]=float(phi);
    }
    if(core&&fault==0u){
        const auto p=bp_v10_starts(s)[tid];bp_v10_spinor_owner(s).accepted[tid]=p;
        s->invocation.tile->psi_vertex[tid][0]={float(p.s0.re),float(p.s0.im)};s->invocation.tile->psi_vertex[tid][1]={float(p.s1.re),float(p.s1.im)};
        const auto v=bp_ns_load3(s->invocation.numerical->v_start,tid);auto* velocities=s->invocation.collective.streams->d_vels_xyz[leaf];
        // The exact normalized pair was checked against C4's evaluated
        // coordinate before the final vote. No high or low changed earlier.
        const volatile uint32_t* ids=s->invocation.tile->global_atom_idx;
        const uint32_t commit_atom=UNPACK_GLOBAL_ATOM_ID(ids[tid]);
        (void)bp_coord_store_current(bp_v10_coordinate_owner(s),commit_atom,bp_v10_normalized_position(s,tid));
        velocities[3u*commit_atom]=v.x;velocities[3u*commit_atom+1u]=v.y;velocities[3u*commit_atom+2u]=v.z;
    }
    __syncthreads(); // All256 precise/projection writes precede their identity.
    if(tid==0u){
        const float wxl=float(s->invocation.numerical->scalar[BP_NS_WXL]/fmax(double(s->invocation.edges),1.)),wxt=float(s->invocation.numerical->scalar[BP_NS_WXT]/fmax(double(s->invocation.edges),1.));
        auto* rec=&s->invocation.collective.records[leaf];rec->abi_magic=BIPRISM_TAIL_ABI_MAGIC;rec->leaf_id=leaf;
        rec->picard_iters_used=s->picard_iters_used;rec->dec_iters_used=s->dec_iters_used;rec->last_picard_residual=s->last_picard_residual;rec->last_dec_residual=s->last_dec_residual;
        rec->step_energy_drift=float(fault?fabs(s->invocation.numerical->scalar[BP_NS_ATTEMPT_ENERGY]):s->invocation.numerical->scalar[BP_NS_MAX_ENERGY]);rec->w_x_lambda_trace=wxl;rec->w_x_tau_trace=wxt;
        rec->unitarity_defect=float(s->invocation.numerical->scalar[BP_NS_MAX_NORM]);if(fault==0u){
            bp_potential_publish_main_commit(s->invocation.collective.streams->d_potential_owners[leaf],rec->committed_substeps+BIPRISM_INNER_SUBSTEPS,s->invocation.potential_step);
            bp_spinor_publish(bp_v10_spinor_owner(s),rec->committed_substeps+BIPRISM_INNER_SUBSTEPS,s->invocation.potential_step);
            rec->committed_substeps+=BIPRISM_INNER_SUBSTEPS;
        }
        if(fault!=0u&&!bp_v10_barrier_aborted(s->invocation.collective.epoch)){s->invocation.bank->footer.publication_phase=5u;s->invocation.bank->footer.local_fault_mask=s->local_fault_accum;s->invocation.bank->footer.convergence_vote=0u;}
        if(s->invocation.adjudicator){auto* tail=&reinterpret_cast<AdjudicatorTailContractViewFfi*>(s->invocation.adjudicator)[leaf];
            tail->tail_abi_magic=BIPRISM_TAIL_ABI_MAGIC;tail->tail_struct_bytes=64u;tail->last_picard_residual=s->last_picard_residual;tail->last_dec_residual=s->last_dec_residual;
            tail->picard_iters_used=s->picard_iters_used;tail->dec_iters_used=s->dec_iters_used;tail->epoch_leaf_fault=s->published_leaf_fault;tail->epoch_cluster_fault=fault;
            tail->terminal_fault_latch=rec->terminal_fault_latch;tail->committed_substeps=rec->committed_substeps;tail->w_x_lambda_trace=wxl;tail->w_x_tau_trace=wxt;tail->step_energy_drift=rec->step_energy_drift;}
    }
}

extern "C" __global__ void nhs_chunk_while_latch_v10(
    uint32_t* __restrict__ d_remaining_steps,
    uint32_t* __restrict__ d_protocol_current_step,
    const BiprismConvergenceRecordFfi* __restrict__ d_conv_records,
    cudaGraphConditionalHandle cond_handle
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t any_fault = 0u;
        #pragma unroll
        for (uint32_t r = 0u; r < 8u; ++r) {
            any_fault |= d_conv_records[r].terminal_fault_latch;
        }
        if (any_fault != 0u) {
            *d_remaining_steps = 0u;
            cudaGraphSetConditional(cond_handle, 0);
            return;
        }
        const uint32_t rem = *d_remaining_steps;
        if (rem > 0u) {
            *d_remaining_steps = rem - 1u;
            atomicAdd(d_protocol_current_step, 1u);
            cudaGraphSetConditional(cond_handle, (rem - 1u) > 0u ? 1 : 0);
        } else {
            cudaGraphSetConditional(cond_handle, 0);
        }
    }
}

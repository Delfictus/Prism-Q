//! Version 1 foliated adjudicator ABI and checked boundary observations.
//!
//! This record has a separate namespace from the active pointer-bearing legacy
//! adjudicator. It does not allocate a ring, produce scientific measurements, or
//! qualify the missing native producer. The caller must acquire the record and
//! its matching boundary evidence after completed device execution.
//!
//! Production boundary validation requires a source-bound overlap metric. A
//! separate physical acceptance campaign checks the full NVE energy window and
//! every topology flip; it is not imposed on thermostatted or UV-driven chunks.
//! These checks establish numerical/identity consistency of supplied observations.
//! They do not establish scientific or ranking eligibility: Bianchi remains a
//! producer-reported scalar until the native coboundary producer is qualified.

/// Number of leaves in the version 1 ring.
pub const FOLIATED_LEAF_COUNT: usize = 8;
/// Version of the independent foliated ABI, not the legacy adjudicator ABI.
pub const FOLIATED_ADJUDICATOR_ABI_VERSION: u32 = 1;
/// Exact size of the contiguous ring, excluding acquisition evidence.
pub const FOLIATED_RING_BYTES: usize = 2048;
/// Strict upper bound for the Bianchi diagnostic.
pub const BIANCHI_TOLERANCE: f64 = 1.0e-5;
/// Strict upper bound for the overlap-metric defect.
pub const S_METRIC_TOLERANCE: f64 = 1.0e-6;
/// Strict upper bound for relative total-energy drift throughout the NVE window.
pub const NVE_RELATIVE_ENERGY_TOLERANCE: f64 = 2.0e-5;
/// Strict upper bound for the magnitude of each trace-flux change.
pub const FLIP_TRACE_FLUX_TOLERANCE: f64 = 5.0e-6;
/// Required number of physical steps in the acceptance window.
pub const FOLIATED_WINDOW_STEPS: u32 = 500;
/// Required physical timestep in femtoseconds for this acceptance contract.
pub const FOLIATED_TIMESTEP_FS: f64 = 0.5;

/// Exact 256-byte C-layout record; the matching C++ namespace is `prism_biprism_v1`.
///
/// All arrays are row-major. `unitarity_defect_frob` denotes the overlap-metric
/// quantity `||U^H S U - S||_F` in version 1. The overlap matrix is supplied in
/// the separately bound acquisition evidence. Reserved words remain zero.
#[repr(C, align(128))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterferometricAdjudicatorFfi {
    /// Index in the contiguous eight-leaf ring.
    pub leaf_id: u32,
    /// Completed physical step, with epoch supplied separately.
    pub step_tau: u32,
    /// Site/tile identity for this leaf's measurements.
    pub active_tile_id: u32,
    /// Number of topology flips at this record's physical step.
    pub pachner_flip_count: u32,
    /// Real part of the temporal 2x2 propagator.
    pub u_tau_re: [f32; 4],
    /// Imaginary part of the temporal 2x2 propagator.
    pub u_tau_im: [f32; 4],
    /// Real part of the coefficient map from this leaf to the next leaf.
    pub v_lambda_re: [f32; 4],
    /// Imaginary part of the link to the next leaf.
    pub v_lambda_im: [f32; 4],
    /// Producer-defined physical state populations; never interpreted as a Euclidean norm.
    pub state_populations: [f32; 4],
    /// Producer-defined surface energies, in the source-bound model's units.
    pub energy_eigenvalues: [f32; 4],
    /// Four finite regularization coefficients in [0, 1].
    pub gauge_clamp_omega: [f32; 4],
    /// Real traces of spatial plaquettes.
    pub w_xx_trace: [f32; 4],
    /// Real traces of space-leaf plaquettes.
    pub w_x_lambda_trace: [f32; 4],
    /// Real traces of space-time plaquettes.
    pub w_x_tau_trace: [f32; 4],
    /// Real traces of leaf-time plaquettes.
    pub w_lambda_tau_trace: [f32; 4],
    /// Nonnegative producer-reported Bianchi norm.
    pub bianchi_residual_l2: f32,
    /// Nonnegative producer-reported overlap-metric defect; independently recomputed.
    pub unitarity_defect_frob: f32,
    /// Nonnegative force norm in the source-bound model's force units.
    pub pulay_force_norm: f32,
    /// Boolean branch request, exactly 0 or 1.
    pub consensus_branch_flag: u32,
    /// Reserved zero words; they are not an implicit production marker.
    pub _pad_line1: [u32; 12],
}

impl InterferometricAdjudicatorFfi {
    /// Empty storage, explicitly inadmissible as produced evidence.
    pub const fn zero() -> Self {
        Self {
            leaf_id: 0,
            step_tau: 0,
            active_tile_id: 0,
            pachner_flip_count: 0,
            u_tau_re: [0.0; 4],
            u_tau_im: [0.0; 4],
            v_lambda_re: [0.0; 4],
            v_lambda_im: [0.0; 4],
            state_populations: [0.0; 4],
            energy_eigenvalues: [0.0; 4],
            gauge_clamp_omega: [0.0; 4],
            w_xx_trace: [0.0; 4],
            w_x_lambda_trace: [0.0; 4],
            w_x_tau_trace: [0.0; 4],
            w_lambda_tau_trace: [0.0; 4],
            bianchi_residual_l2: 0.0,
            unitarity_defect_frob: 0.0,
            pulay_force_norm: 0.0,
            consensus_branch_flag: 0,
            _pad_line1: [0; 12],
        }
    }
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(align_of::<InterferometricAdjudicatorFfi>() == 128);
    assert!(size_of::<InterferometricAdjudicatorFfi>() == 256);
    assert!(size_of::<[InterferometricAdjudicatorFfi; 8]>() == FOLIATED_RING_BYTES);
    assert!(offset_of!(InterferometricAdjudicatorFfi, leaf_id) == 0);
    assert!(offset_of!(InterferometricAdjudicatorFfi, step_tau) == 4);
    assert!(offset_of!(InterferometricAdjudicatorFfi, active_tile_id) == 8);
    assert!(offset_of!(InterferometricAdjudicatorFfi, pachner_flip_count) == 12);
    assert!(offset_of!(InterferometricAdjudicatorFfi, u_tau_re) == 16);
    assert!(offset_of!(InterferometricAdjudicatorFfi, u_tau_im) == 32);
    assert!(offset_of!(InterferometricAdjudicatorFfi, v_lambda_re) == 48);
    assert!(offset_of!(InterferometricAdjudicatorFfi, v_lambda_im) == 64);
    assert!(offset_of!(InterferometricAdjudicatorFfi, state_populations) == 80);
    assert!(offset_of!(InterferometricAdjudicatorFfi, energy_eigenvalues) == 96);
    assert!(offset_of!(InterferometricAdjudicatorFfi, gauge_clamp_omega) == 112);
    assert!(offset_of!(InterferometricAdjudicatorFfi, w_xx_trace) == 128);
    assert!(offset_of!(InterferometricAdjudicatorFfi, w_x_lambda_trace) == 144);
    assert!(offset_of!(InterferometricAdjudicatorFfi, w_x_tau_trace) == 160);
    assert!(offset_of!(InterferometricAdjudicatorFfi, w_lambda_tau_trace) == 176);
    assert!(offset_of!(InterferometricAdjudicatorFfi, bianchi_residual_l2) == 192);
    assert!(offset_of!(InterferometricAdjudicatorFfi, unitarity_defect_frob) == 196);
    assert!(offset_of!(InterferometricAdjudicatorFfi, pulay_force_norm) == 200);
    assert!(offset_of!(InterferometricAdjudicatorFfi, consensus_branch_flag) == 204);
    assert!(offset_of!(InterferometricAdjudicatorFfi, _pad_line1) == 208);
};

/// Double precision complex scalar for independently evaluated boundary evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}
impl Complex64 {
    /// Construct a complex scalar without changing its representation.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
    fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }
}
/// Full complex row-major 2x2 overlap matrix, never inferred from a scalar norm.
pub type OverlapMatrix = [Complex64; 4];

/// Boundary identity independently expected by the acquiring runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoliatedBoundaryExpectation {
    /// Nonzero source/model identity established outside the candidate record.
    pub source_id: [u8; 32],
    /// Nonzero run/continuation epoch identity.
    pub epoch_id: [u8; 32],
    /// Nonzero identity of the coherent physical population.
    pub population_id: [u8; 32],
    /// Nonzero identity of the declared force-unit convention for raw diagnostics.
    pub force_unit_id: [u8; 32],
    /// Exact physical step at the end of the window.
    pub step_tau: u32,
    /// Nonzero identity of each fixed basis and metric generation, indexed by leaf.
    pub basis_generation_ids: [[u8; 32]; FOLIATED_LEAF_COUNT],
    /// Site selected on each leaf; the array is indexed by exact leaf ID.
    pub active_tile_ids: [u32; FOLIATED_LEAF_COUNT],
}

/// Acquisition metadata emitted with one leaf's completed measurements.
///
/// The eventual native producer must bind these fields to the same completed
/// record. Creating this object in host code does not establish native execution.
#[derive(Clone, Copy, Debug)]
pub struct FoliatedLeafEvidence {
    /// Source/model identity of this observation.
    pub source_id: [u8; 32],
    /// Epoch of this observation.
    pub epoch_id: [u8; 32],
    /// Physical population identity of this observation.
    pub population_id: [u8; 32],
    /// Source-bound identity of the force-unit convention.
    pub force_unit_id: [u8; 32],
    /// Explicit completion marker from the producing path, never allocation state.
    pub produced: bool,
    /// Exact completed step associated with the record and metric.
    pub step_tau: u32,
    /// Identity of the basis and metric used at the input of the temporal map.
    pub previous_basis_generation_id: [u8; 32],
    /// Identity of the basis and metric used at the output of the temporal map.
    /// Version 1 supports the same fixed metric at both ends; a moving basis fails.
    pub basis_generation_id: [u8; 32],
    /// Complete Hermitian positive-definite overlap metric for this record's basis.
    pub overlap: OverlapMatrix,
}

/// Complete energy observations at the initial point and all 500 subsequent steps.
#[derive(Clone, Copy, Debug)]
pub struct NveEnergyWindow<'a> {
    /// First physical step represented by `total_energies[0]`.
    pub start_step: u32,
    /// Physical timestep in femtoseconds, exactly 0.5 for this contract.
    pub timestep_fs: f64,
    /// Producer declaration that this complete window is NVE.
    pub is_nve: bool,
    /// Total energies in one consistent unit and energy reference, length 501.
    pub total_energies: &'a [f64],
}

/// One flip's before/after trace-flux pair, acquired in its actual topology context.
#[derive(Clone, Copy, Debug)]
pub struct PachnerTraceFluxObservation {
    /// Exact producing leaf.
    pub leaf_id: u32,
    /// Physical step in the open-closed interval `(window_start, window_end]`.
    pub step_tau: u32,
    /// Zero-based per-leaf ordinal; observations must preserve this order.
    pub flip_ordinal: u32,
    /// Complex trace flux before the flip.
    pub before: Complex64,
    /// Complex trace flux after the flip.
    pub after: Complex64,
}

/// All independently acquired evidence required for one boundary admission.
#[derive(Clone, Copy, Debug)]
pub struct FoliatedBoundaryEvidence<'a> {
    /// Metadata/metric for every leaf, in ring order.
    pub leaves: &'a [FoliatedLeafEvidence; FOLIATED_LEAF_COUNT],
}

/// Evidence for the separate NVE and topology-flip acceptance campaign.
#[derive(Clone, Copy, Debug)]
pub struct FoliatedPhysicalAcceptanceEvidence<'a> {
    /// Identity and boundary of the dedicated campaign observations.
    pub boundary: FoliatedBoundaryExpectation,
    /// Full NVE energy series for every leaf, from the same source and epoch.
    pub nve_windows: &'a [NveEnergyWindow<'a>; FOLIATED_LEAF_COUNT],
    /// Independently recorded complete flip count for every leaf in this window.
    pub window_flip_counts: [u32; FOLIATED_LEAF_COUNT],
    /// Every flip from the window, ordered within each leaf; no truncation allowed.
    pub flips: &'a [PachnerTraceFluxObservation],
}

/// Exact reason a ring or an attempted site contribution was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoliatedConsensusError {
    /// Missing nonzero source, epoch or population identity.
    MissingIdentity,
    /// A tolerance is invalid or would weaken the specified bound.
    InvalidTolerance,
    /// An expected or observed step is absent or inconsistent.
    StepMismatch,
    /// Leaf slots are not the exact identity-ordered population 0 through 7.
    LeafIdentity,
    /// Selected site identity does not match the expected site.
    TileMismatch,
    /// An observation belongs to a different source, epoch or population.
    BoundaryIdentity,
    /// At least one leaf was never produced by the acquisition path.
    Unproduced,
    /// A scalar is nonfinite, a norm is negative, or a Boolean/interval is malformed.
    InvalidField,
    /// Reserved words are nonzero under version 1.
    ReservedWords,
    /// The temporal map changes basis/metric generation; fixed-metric version 1 cannot check it.
    MovingMetricUnsupported,
    /// The Bianchi diagnostic is outside its strict bound.
    BianchiViolation,
    /// The overlap metric is not a finite Hermitian positive-definite matrix.
    InvalidOverlap,
    /// The cross-leaf link fails its separately evaluated metric-transport bound.
    LinkMetricViolation,
    /// The reported or independently computed overlap-metric defect fails its bound.
    SMetricViolation,
    /// Metric/window/identity evidence is absent from the diagnostic-only API.
    MissingBoundaryEvidence,
    /// Energy samples, interval, timestep or NVE classification are incomplete.
    InvalidNveWindow,
    /// Relative energy drift reaches or exceeds the specified bound.
    NveEnergyViolation,
    /// Flip observations are incomplete, duplicated, unordered or belong to another step.
    InvalidFlipCoverage,
    /// A per-flip trace-flux change reaches or exceeds its strict bound.
    FlipTraceFluxViolation,
    /// No leaf in this admitted boundary belongs to the requested site.
    SiteAbsent,
}
impl std::fmt::Display for FoliatedConsensusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "foliated consensus refused: {self:?}")
    }
}
impl std::error::Error for FoliatedConsensusError {}

/// Diagnostic outcome. Ranking fields stay unavailable without full boundary evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoliatedConsensusVerdict {
    /// Whether the ring has passed the scalar Bianchi bound and basic checks.
    pub bianchi_conserved: bool,
    /// False in the diagnostic-only API, which has no overlap matrix.
    pub unitarity_conserved: bool,
    /// Unavailable unless the full boundary is admitted; use the strict API instead.
    pub consensus_branch_mask: Option<u8>,
    /// No uniform ring score is exposed for site ranking.
    pub phase_manifold_holonomy_boost: Option<f32>,
    /// The precise failed or unavailable requirement.
    pub rejection: FoliatedConsensusError,
}

fn valid_norm(x: f32) -> bool {
    x.is_finite() && !x.is_sign_negative()
}

fn validate_record_fields(
    ring: &[InterferometricAdjudicatorFfi; FOLIATED_LEAF_COUNT],
    expected_step: u32,
    bianchi_tol: f64,
    metric_tol: f64,
) -> Result<(), FoliatedConsensusError> {
    use FoliatedConsensusError as E;
    if expected_step == 0 {
        return Err(E::Unproduced);
    }
    for (leaf, adj) in ring.iter().enumerate() {
        if adj.leaf_id != leaf as u32 {
            return Err(E::LeafIdentity);
        }
        if adj.step_tau != expected_step {
            return Err(E::StepMismatch);
        }
        if adj._pad_line1 != [0; 12] {
            return Err(E::ReservedWords);
        }
        if adj.consensus_branch_flag > 1
            || !valid_norm(adj.bianchi_residual_l2)
            || !valid_norm(adj.unitarity_defect_frob)
            || !valid_norm(adj.pulay_force_norm)
        {
            return Err(E::InvalidField);
        }
        for values in [
            adj.u_tau_re,
            adj.u_tau_im,
            adj.v_lambda_re,
            adj.v_lambda_im,
            adj.state_populations,
            adj.energy_eigenvalues,
            adj.gauge_clamp_omega,
            adj.w_xx_trace,
            adj.w_x_lambda_trace,
            adj.w_x_tau_trace,
            adj.w_lambda_tau_trace,
        ] {
            if values.iter().any(|x| !x.is_finite()) {
                return Err(E::InvalidField);
            }
        }
        if adj.state_populations.iter().any(|&x| x < 0.0)
            || adj
                .gauge_clamp_omega
                .iter()
                .any(|&x| !(0.0..=1.0).contains(&x))
        {
            return Err(E::InvalidField);
        }
        if adj.state_populations.iter().all(|&x| x == 0.0) {
            return Err(E::Unproduced);
        }
        for traces in [
            adj.w_xx_trace,
            adj.w_x_lambda_trace,
            adj.w_x_tau_trace,
            adj.w_lambda_tau_trace,
        ] {
            if traces.iter().any(|&x| !(-2.0..=2.0).contains(&x)) {
                return Err(E::InvalidField);
            }
        }
        if adj
            .v_lambda_re
            .iter()
            .chain(adj.v_lambda_im.iter())
            .all(|&x| x == 0.0)
        {
            return Err(E::Unproduced);
        }
        if adj
            .u_tau_re
            .iter()
            .chain(adj.u_tau_im.iter())
            .all(|&x| x == 0.0)
        {
            return Err(E::Unproduced);
        }
        if f64::from(adj.bianchi_residual_l2) >= bianchi_tol {
            return Err(E::BianchiViolation);
        }
        if f64::from(adj.unitarity_defect_frob) >= metric_tol {
            return Err(E::SMetricViolation);
        }
    }
    Ok(())
}

/// Check scalar diagnostics without pretending the ring contains the missing metric.
///
/// The historical three-argument shape is retained, but tolerances may only be
/// tightened. This function never returns an actionable branch mask or score.
/// Use [`adjudicate_foliated_biprism_boundary`] to obtain site-bound contributions.
pub fn adjudicate_foliated_biprism_ring(
    ring: &[InterferometricAdjudicatorFfi; FOLIATED_LEAF_COUNT],
    bianchi_tol: f32,
    unitarity_tol: f32,
) -> FoliatedConsensusVerdict {
    let tol_valid = bianchi_tol.is_finite()
        && bianchi_tol > 0.0
        && bianchi_tol <= BIANCHI_TOLERANCE as f32
        && unitarity_tol.is_finite()
        && unitarity_tol > 0.0
        && unitarity_tol <= S_METRIC_TOLERANCE as f32;
    let result = if tol_valid {
        validate_record_fields(
            ring,
            ring[0].step_tau,
            f64::from(bianchi_tol),
            f64::from(unitarity_tol),
        )
    } else {
        Err(FoliatedConsensusError::InvalidTolerance)
    };
    FoliatedConsensusVerdict {
        bianchi_conserved: result.is_ok(),
        unitarity_conserved: false,
        consensus_branch_mask: None,
        phase_manifold_holonomy_boost: None,
        rejection: result
            .err()
            .unwrap_or(FoliatedConsensusError::MissingBoundaryEvidence),
    }
}

fn validate_overlap(s: &OverlapMatrix) -> Result<(), FoliatedConsensusError> {
    use FoliatedConsensusError as E;
    if s.iter().any(|x| !x.finite())
        || s[0].im != 0.0
        || s[3].im != 0.0
        || s[1] != s[2].conj()
        || s[0].re <= 0.0
        || s[3].re <= 0.0
    {
        return Err(E::InvalidOverlap);
    }
    let determinant = s[0].re * s[3].re - s[1].re * s[1].re - s[1].im * s[1].im;
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(E::InvalidOverlap);
    }
    Ok(())
}

/// Compute the absolute Frobenius defect `||U^H S U - S||_F` in a fixed metric.
///
/// This literal absolute criterion does not establish a relative norm bound or
/// well-conditioned basis. For very small S, a small absolute defect alone may
/// coexist with a large relative defect. It is never scientific eligibility.
pub fn overlap_metric_defect(
    re: &[f32; 4],
    im: &[f32; 4],
    s: &OverlapMatrix,
) -> Result<f64, FoliatedConsensusError> {
    overlap_metric_transport_defect(re, im, s, s)
}

/// Compute `||V^H S_next V - S_current||_F` with the full complex matrices.
///
/// V maps current-leaf coefficient vectors to next-leaf coefficient vectors.
/// Reversing that map requires reversing the metrics; no implicit inversion is made.
pub fn overlap_metric_transport_defect(
    re: &[f32; 4],
    im: &[f32; 4],
    s_current: &OverlapMatrix,
    s_next: &OverlapMatrix,
) -> Result<f64, FoliatedConsensusError> {
    use FoliatedConsensusError as E;
    validate_overlap(s_current)?;
    validate_overlap(s_next)?;
    let u = std::array::from_fn::<_, 4, _>(|i| Complex64::new(f64::from(re[i]), f64::from(im[i])));
    if u.iter().any(|x| !x.finite()) {
        return Err(E::InvalidField);
    }
    let mut defect = 0.0_f64;
    for i in 0..2 {
        for j in 0..2 {
            let mut gram = Complex64::default();
            for k in 0..2 {
                for l in 0..2 {
                    gram = gram.add(u[2 * k + i].conj().mul(s_next[2 * k + l]).mul(u[2 * l + j]));
                }
            }
            defect = defect.hypot(gram.sub(s_current[2 * i + j]).magnitude());
        }
    }
    if !defect.is_finite() {
        return Err(E::SMetricViolation);
    }
    Ok(defect)
}

/// Evidence scope of the supplied observations, not scientific/ranking admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoliatedEvidenceLevel {
    /// Identity, finite values and metric defects checked; Bianchi still producer-reported.
    CheckedObservationsProducerReportedBianchi,
}

/// Checked boundary observations; private fields prevent bypassing the input checks.
///
/// No native execution, coboundary construction, physical model, conditioning or
/// ranking eligibility is certified by this type. It cannot promote or branch the runtime.
#[derive(Debug)]
pub struct CheckedFoliatedObservations {
    expected: FoliatedBoundaryExpectation,
    ring: [InterferometricAdjudicatorFfi; FOLIATED_LEAF_COUNT],
    metric_defects: [f64; FOLIATED_LEAF_COUNT],
    link_metric_defects: [f64; FOLIATED_LEAF_COUNT],
}

/// Site-specific diagnostic measurements from checked observations, not ranking approval.
///
/// Force norms retain their declared units. The requested raw holonomy formula
/// is explicitly unit-dependent; it is not a variational force or a calibrated
/// ranking model. Contributions only use leaves belonging to the requested site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoliatedSiteContribution {
    /// Scope of these diagnostic measurements.
    pub evidence_level: FoliatedEvidenceLevel,
    /// Site whose actual leaves supplied this contribution.
    pub active_tile_id: u32,
    /// Exact completed physical step.
    pub step_tau: u32,
    /// Bound epoch of this contribution.
    pub epoch_id: [u8; 32],
    /// Bound physical population.
    pub population_id: [u8; 32],
    /// Declared force-unit identity required to interpret the raw diagnostic.
    pub force_unit_id: [u8; 32],
    /// Mask of leaves that measured this site.
    pub participating_leaf_mask: u8,
    /// Producer-requested branch diagnostic restricted to this site's leaves.
    /// This field is not runtime branch authorization.
    pub consensus_branch_mask: u8,
    /// Mean real space-leaf Wilson-trace deficit over this site's actual leaves.
    pub mean_space_leaf_trace_deficit: f64,
    /// Mean source-defined force norm over this site's actual leaves.
    pub mean_pulay_force_norm: f64,
    /// Requested mean of `max(2 - trace, 0) * (1 + raw_force_norm)`.
    /// Its numerical value depends on `force_unit_id`; no unit conversion is implied.
    pub unit_dependent_holonomy_diagnostic: f64,
}

impl CheckedFoliatedObservations {
    /// Report the checked-observation scope; no scientific eligibility is asserted.
    pub fn evidence_level(&self) -> FoliatedEvidenceLevel {
        FoliatedEvidenceLevel::CheckedObservationsProducerReportedBianchi
    }
    /// Return absolute metric-transport defects for all eight cross-leaf links.
    pub fn link_metric_defects(&self) -> &[f64; FOLIATED_LEAF_COUNT] {
        &self.link_metric_defects
    }
    /// Return independent S-metric defects for all eight admitted leaves.
    pub fn metric_defects(&self) -> &[f64; FOLIATED_LEAF_COUNT] {
        &self.metric_defects
    }
    /// Bind a contribution to the current expected epoch, population, step and site.
    pub fn site_contribution(
        &self,
        expected: &FoliatedBoundaryExpectation,
        active_tile_id: u32,
    ) -> Result<FoliatedSiteContribution, FoliatedConsensusError> {
        use FoliatedConsensusError as E;
        if expected != &self.expected {
            return Err(E::BoundaryIdentity);
        }
        let mut mask = 0_u8;
        let mut branches = 0_u8;
        let mut deficit = 0.0_f64;
        let mut force_norm = 0.0_f64;
        let mut raw_holonomy = 0.0_f64;
        for (i, adj) in self.ring.iter().enumerate() {
            if adj.active_tile_id != active_tile_id {
                continue;
            }
            mask |= 1 << i;
            if adj.consensus_branch_flag == 1 {
                branches |= 1 << i;
            }
            let leaf_deficit = (2.0 - f64::from(adj.w_x_lambda_trace[0])).max(0.0);
            deficit += leaf_deficit;
            force_norm += f64::from(adj.pulay_force_norm);
            raw_holonomy += leaf_deficit * (1.0 + f64::from(adj.pulay_force_norm));
        }
        if mask == 0 {
            return Err(E::SiteAbsent);
        }
        let count = f64::from(mask.count_ones());
        Ok(FoliatedSiteContribution {
            evidence_level: self.evidence_level(),
            active_tile_id,
            step_tau: self.expected.step_tau,
            epoch_id: self.expected.epoch_id,
            population_id: self.expected.population_id,
            force_unit_id: self.expected.force_unit_id,
            participating_leaf_mask: mask,
            consensus_branch_mask: branches,
            mean_space_leaf_trace_deficit: deficit / count,
            mean_pulay_force_norm: force_norm / count,
            unit_dependent_holonomy_diagnostic: raw_holonomy / count,
        })
    }
}

/// Check supplied boundary observations under the explicit fixed-metric contract.
///
/// The Bianchi scalar is checked against its strict bound but is not recomputed
/// from a coboundary here. Native production and scientific/ranking eligibility
/// remain separate requirements; this API neither changes ranks nor branches.
///
/// All production-boundary checks finish before constructing the result. Inputs must come
/// from an actual coherently acquired producer; this checker cannot authenticate
/// invented metadata or prove the physical adequacy of the source model.
pub fn adjudicate_foliated_biprism_boundary(
    ring: &[InterferometricAdjudicatorFfi; FOLIATED_LEAF_COUNT],
    expected: &FoliatedBoundaryExpectation,
    evidence: &FoliatedBoundaryEvidence<'_>,
) -> Result<CheckedFoliatedObservations, FoliatedConsensusError> {
    use FoliatedConsensusError as E;
    if expected.source_id == [0; 32]
        || expected.epoch_id == [0; 32]
        || expected.population_id == [0; 32]
        || expected.force_unit_id == [0; 32]
        || expected
            .basis_generation_ids
            .iter()
            .any(|id| *id == [0; 32])
    {
        return Err(E::MissingIdentity);
    }
    validate_record_fields(
        ring,
        expected.step_tau,
        BIANCHI_TOLERANCE,
        S_METRIC_TOLERANCE,
    )?;
    let mut metric_defects = [0.0; FOLIATED_LEAF_COUNT];
    for (i, adj) in ring.iter().enumerate() {
        let leaf = &evidence.leaves[i];
        if !leaf.produced {
            return Err(E::Unproduced);
        }
        if leaf.source_id != expected.source_id
            || leaf.epoch_id != expected.epoch_id
            || leaf.population_id != expected.population_id
            || leaf.force_unit_id != expected.force_unit_id
        {
            return Err(E::BoundaryIdentity);
        }
        if leaf.basis_generation_id != expected.basis_generation_ids[i] {
            return Err(E::BoundaryIdentity);
        }
        if leaf.previous_basis_generation_id != leaf.basis_generation_id {
            return Err(E::MovingMetricUnsupported);
        }
        if leaf.step_tau != expected.step_tau {
            return Err(E::StepMismatch);
        }
        if adj.active_tile_id != expected.active_tile_ids[i] {
            return Err(E::TileMismatch);
        }
        metric_defects[i] = overlap_metric_defect(&adj.u_tau_re, &adj.u_tau_im, &leaf.overlap)?;
        if metric_defects[i] >= S_METRIC_TOLERANCE {
            return Err(E::SMetricViolation);
        }
    }
    let mut link_metric_defects = [0.0; FOLIATED_LEAF_COUNT];
    for (i, adj) in ring.iter().enumerate() {
        link_metric_defects[i] = overlap_metric_transport_defect(
            &adj.v_lambda_re,
            &adj.v_lambda_im,
            &evidence.leaves[i].overlap,
            &evidence.leaves[(i + 1) % FOLIATED_LEAF_COUNT].overlap,
        )?;
        if link_metric_defects[i] >= S_METRIC_TOLERANCE {
            return Err(E::LinkMetricViolation);
        }
    }
    Ok(CheckedFoliatedObservations {
        expected: *expected,
        ring: *ring,
        metric_defects,
        link_metric_defects,
    })
}

/// Results from the distinct physical acceptance campaign.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoliatedPhysicalAcceptance {
    /// Identity and completed step of the checked campaign.
    pub boundary: FoliatedBoundaryExpectation,
    /// Maximum relative drift over every NVE energy sample on each leaf.
    pub maximum_relative_energy_drifts: [f64; FOLIATED_LEAF_COUNT],
    /// Complete number of checked flips on every leaf.
    pub checked_flip_counts: [u32; FOLIATED_LEAF_COUNT],
}

/// Check a dedicated 500-step/0.5-fs NVE and per-flip conservation campaign.
///
/// This acceptance gate is separate from ordinary ring admission: the production
/// May profile includes bath/UV work and cannot satisfy uncorrected NVE conservation.
/// A passing campaign covers only the supplied source, epoch, population and step.
pub fn validate_foliated_physical_acceptance(
    consensus: &CheckedFoliatedObservations,
    evidence: &FoliatedPhysicalAcceptanceEvidence<'_>,
) -> Result<FoliatedPhysicalAcceptance, FoliatedConsensusError> {
    use FoliatedConsensusError as E;
    let expected = &consensus.expected;
    let ring = &consensus.ring;
    if evidence.boundary != *expected {
        return Err(E::BoundaryIdentity);
    }
    let start = expected
        .step_tau
        .checked_sub(FOLIATED_WINDOW_STEPS)
        .ok_or(E::StepMismatch)?;
    let mut energy_drifts = [0.0_f64; FOLIATED_LEAF_COUNT];
    for i in 0..FOLIATED_LEAF_COUNT {
        let window = &evidence.nve_windows[i];
        if !window.is_nve
            || window.start_step != start
            || window.timestep_fs != FOLIATED_TIMESTEP_FS
            || window.total_energies.len() != (FOLIATED_WINDOW_STEPS + 1) as usize
            || window.total_energies.iter().any(|x| !x.is_finite())
            || window.total_energies[0] == 0.0
        {
            return Err(E::InvalidNveWindow);
        }
        let initial = window.total_energies[0];
        for &energy in &window.total_energies[1..] {
            let relative = ((energy - initial) / initial).abs();
            if !relative.is_finite() || relative >= NVE_RELATIVE_ENERGY_TOLERANCE {
                return Err(E::NveEnergyViolation);
            }
            energy_drifts[i] = energy_drifts[i].max(relative);
        }
    }
    let mut counts = [0_u32; FOLIATED_LEAF_COUNT];
    let mut last_steps = [start; FOLIATED_LEAF_COUNT];
    let mut terminal_counts = [0_u32; FOLIATED_LEAF_COUNT];
    for flip in evidence.flips {
        let leaf = flip.leaf_id as usize;
        if leaf >= FOLIATED_LEAF_COUNT
            || flip.flip_ordinal != counts[leaf]
            || flip.step_tau <= start
            || flip.step_tau > expected.step_tau
            || flip.step_tau < last_steps[leaf]
        {
            return Err(E::InvalidFlipCoverage);
        }
        if !flip.before.finite() || !flip.after.finite() {
            return Err(E::InvalidField);
        }
        let delta = flip.after.sub(flip.before).magnitude();
        if !delta.is_finite() || delta >= FLIP_TRACE_FLUX_TOLERANCE {
            return Err(E::FlipTraceFluxViolation);
        }
        counts[leaf] = counts[leaf].checked_add(1).ok_or(E::InvalidFlipCoverage)?;
        last_steps[leaf] = flip.step_tau;
        if flip.step_tau == expected.step_tau {
            terminal_counts[leaf] = terminal_counts[leaf]
                .checked_add(1)
                .ok_or(E::InvalidFlipCoverage)?;
        }
    }
    for i in 0..FOLIATED_LEAF_COUNT {
        if counts[i] != evidence.window_flip_counts[i]
            || terminal_counts[i] != ring[i].pachner_flip_count
        {
            return Err(E::InvalidFlipCoverage);
        }
    }
    Ok(FoliatedPhysicalAcceptance {
        boundary: *expected,
        maximum_relative_energy_drifts: energy_drifts,
        checked_flip_counts: counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    type Ring = [InterferometricAdjudicatorFfi; FOLIATED_LEAF_COUNT];
    type Leaves = [FoliatedLeafEvidence; FOLIATED_LEAF_COUNT];

    // Deliberately synthetic contract fixtures; they are not native producer evidence.
    fn fixture() -> (Ring, FoliatedBoundaryExpectation, Leaves) {
        let expected = FoliatedBoundaryExpectation {
            source_id: [1; 32],
            epoch_id: [2; 32],
            population_id: [3; 32],
            force_unit_id: [4; 32],
            basis_generation_ids: [[5; 32]; 8],
            step_tau: 506,
            active_tile_ids: [17, 17, 17, 17, 29, 29, 29, 29],
        };
        let ring = std::array::from_fn(|i| {
            let mut r = InterferometricAdjudicatorFfi::zero();
            r.leaf_id = i as u32;
            r.step_tau = expected.step_tau;
            r.active_tile_id = expected.active_tile_ids[i];
            r.u_tau_re = [1.0, 0.0, 0.0, 1.0];
            r.v_lambda_re = r.u_tau_re;
            r.state_populations = [0.75, 0.25, 0.0, 0.0];
            r.gauge_clamp_omega = [1.0; 4];
            r.w_x_lambda_trace = [if i < 4 { 1.5 } else { 1.0 }; 4];
            r.pulay_force_norm = 3.0;
            r.consensus_branch_flag = (i == 1 || i == 6) as u32;
            r
        });
        let leaf = FoliatedLeafEvidence {
            source_id: expected.source_id,
            epoch_id: expected.epoch_id,
            population_id: expected.population_id,
            force_unit_id: expected.force_unit_id,
            produced: true,
            step_tau: expected.step_tau,
            previous_basis_generation_id: [5; 32],
            basis_generation_id: [5; 32],
            overlap: [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        };
        (ring, expected, [leaf; FOLIATED_LEAF_COUNT])
    }

    fn check(
        r: &Ring,
        e: &FoliatedBoundaryExpectation,
        leaves: &Leaves,
    ) -> Result<CheckedFoliatedObservations, FoliatedConsensusError> {
        adjudicate_foliated_biprism_boundary(r, e, &FoliatedBoundaryEvidence { leaves })
    }

    #[test]
    fn exact_ring_stride_and_alignment() {
        let ring = [InterferometricAdjudicatorFfi::zero(); 8];
        assert_eq!(std::mem::size_of_val(&ring), 2048);
        assert_eq!((&ring as *const _ as usize) % 128, 0);
        for i in 1..8 {
            assert_eq!(
                (&ring[i] as *const _ as usize) - (&ring[i - 1] as *const _ as usize),
                256
            );
        }
    }

    #[test]
    fn all_zero_and_explicit_unproduced_storage_cannot_admit() {
        let (mut r, e, mut leaves) = fixture();
        r.fill(InterferometricAdjudicatorFfi::zero());
        assert!(check(&r, &e, &leaves).is_err());
        let diagnostic = adjudicate_foliated_biprism_ring(&r, 1e-5, 1e-6);
        assert!(!diagnostic.bianchi_conserved);
        assert!(!diagnostic.unitarity_conserved);
        assert_eq!(diagnostic.consensus_branch_mask, None);
        assert_eq!(diagnostic.phase_manifold_holonomy_boost, None);
        let (r, _, _) = fixture();
        for i in 0..8 {
            leaves[i].produced = false;
            assert_eq!(
                check(&r, &e, &leaves).unwrap_err(),
                FoliatedConsensusError::Unproduced
            );
            leaves[i].produced = true;
        }
    }

    #[test]
    fn diagnostic_path_cannot_synthesize_missing_overlap_evidence() {
        let (r, _, _) = fixture();
        let v = adjudicate_foliated_biprism_ring(&r, 1e-5, 1e-6);
        assert!(v.bianchi_conserved);
        assert!(!v.unitarity_conserved);
        assert_eq!(v.rejection, FoliatedConsensusError::MissingBoundaryEvidence);
        assert_eq!(v.consensus_branch_mask, None);
        assert_eq!(v.phase_manifold_holonomy_boost, None);
        for (b, u) in [
            (1e-4, 1e-6),
            (1e-5, 2e-6),
            (0.0, 1e-6),
            (f32::NAN, 1e-6),
            (1e-5, f32::INFINITY),
        ] {
            assert_eq!(
                adjudicate_foliated_biprism_ring(&r, b, u).rejection,
                FoliatedConsensusError::InvalidTolerance
            );
        }
    }

    #[test]
    fn valid_boundary_contributes_only_to_its_actual_site() {
        let (r, e, leaves) = fixture();
        let admitted = check(&r, &e, &leaves).unwrap();
        let a = admitted.site_contribution(&e, 17).unwrap();
        let b = admitted.site_contribution(&e, 29).unwrap();
        assert_eq!(
            (a.participating_leaf_mask, a.consensus_branch_mask),
            (0x0f, 0x02)
        );
        assert_eq!(
            (b.participating_leaf_mask, b.consensus_branch_mask),
            (0xf0, 0x40)
        );
        assert_eq!(a.mean_space_leaf_trace_deficit, 0.5);
        assert_eq!(b.mean_space_leaf_trace_deficit, 1.0);
        assert_eq!(a.unit_dependent_holonomy_diagnostic, 2.0);
        assert_eq!(b.unit_dependent_holonomy_diagnostic, 4.0);
        assert_eq!(a.force_unit_id, e.force_unit_id);
        assert_eq!(
            admitted.site_contribution(&e, 999).unwrap_err(),
            FoliatedConsensusError::SiteAbsent
        );
        let mut later = e;
        later.step_tau += 500;
        assert_eq!(
            admitted.site_contribution(&later, 17).unwrap_err(),
            FoliatedConsensusError::BoundaryIdentity
        );
    }

    #[test]
    fn same_size_legacy_or_changed_boundary_is_never_accepted_by_identity() {
        let (r, e, leaves) = fixture();
        for i in 0..8 {
            let mut bad = r;
            bad[i].leaf_id = ((i + 1) % 8) as u32;
            assert_eq!(
                check(&bad, &e, &leaves).unwrap_err(),
                FoliatedConsensusError::LeafIdentity
            );
            bad = r;
            bad[i].step_tau -= 1;
            assert_eq!(
                check(&bad, &e, &leaves).unwrap_err(),
                FoliatedConsensusError::StepMismatch
            );
            bad = r;
            bad[i].active_tile_id += 1;
            assert_eq!(
                check(&bad, &e, &leaves).unwrap_err(),
                FoliatedConsensusError::TileMismatch
            );
            for mutate in 0..5 {
                let mut mixed = leaves;
                match mutate {
                    0 => mixed[i].epoch_id[0] ^= 1,
                    1 => mixed[i].population_id[0] ^= 1,
                    2 => mixed[i].source_id[0] ^= 1,
                    3 => mixed[i].force_unit_id[0] ^= 1,
                    _ => mixed[i].step_tau += 1,
                }
                assert!(check(&r, &e, &mixed).is_err());
            }
        }
        let mut missing = e;
        missing.epoch_id = [0; 32];
        assert_eq!(
            check(&r, &missing, &leaves).unwrap_err(),
            FoliatedConsensusError::MissingIdentity
        );
    }

    #[test]
    fn every_float_field_rejects_nan_and_infinity_in_every_leaf() {
        let (r, e, leaves) = fixture();
        // ABI words 4 through 50 inclusive are all and only f32 fields.
        for leaf in 0..8 {
            for word in 4..51 {
                for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut bad = r;
                    let mut words: [u32; 64] = unsafe { std::mem::transmute(bad[leaf]) };
                    words[word] = invalid.to_bits();
                    bad[leaf] = unsafe { std::mem::transmute(words) };
                    assert!(
                        check(&bad, &e, &leaves).is_err(),
                        "accepted leaf={leaf} word={word}"
                    );
                }
            }
        }
    }

    #[test]
    fn negative_norms_reserved_words_and_non_boolean_flags_fail() {
        let (r, e, leaves) = fixture();
        for field in 0..3 {
            for value in [-1.0, -0.0] {
                let mut bad = r;
                match field {
                    0 => bad[7].bianchi_residual_l2 = value,
                    1 => bad[7].unitarity_defect_frob = value,
                    _ => bad[7].pulay_force_norm = value,
                }
                assert_eq!(
                    check(&bad, &e, &leaves).unwrap_err(),
                    FoliatedConsensusError::InvalidField
                );
            }
        }
        let mut bad = r;
        bad[7]._pad_line1[11] = 1;
        assert_eq!(
            check(&bad, &e, &leaves).unwrap_err(),
            FoliatedConsensusError::ReservedWords
        );
        bad = r;
        bad[7].consensus_branch_flag = 2;
        assert_eq!(
            check(&bad, &e, &leaves).unwrap_err(),
            FoliatedConsensusError::InvalidField
        );
    }

    #[test]
    fn all_eight_leaves_are_checked_for_strict_bianchi_and_reported_metric_bounds() {
        let (r, e, leaves) = fixture();
        for i in 0..8 {
            let mut bad = r;
            // The next f32 above each mathematical limit is unambiguously outside it.
            bad[i].bianchi_residual_l2 = f32::from_bits((BIANCHI_TOLERANCE as f32).to_bits() + 1);
            assert_eq!(
                check(&bad, &e, &leaves).unwrap_err(),
                FoliatedConsensusError::BianchiViolation
            );
            bad = r;
            bad[i].unitarity_defect_frob =
                f32::from_bits((S_METRIC_TOLERANCE as f32).to_bits() + 1);
            assert_eq!(
                check(&bad, &e, &leaves).unwrap_err(),
                FoliatedConsensusError::SMetricViolation
            );
        }
    }

    #[test]
    fn metric_unitarity_is_not_replaced_with_euclidean_unitarity() {
        let (mut r, e, mut leaves) = fixture();
        for leaf in &mut leaves {
            leaf.overlap[0].re = 4.0;
        }
        r[0].u_tau_re = [0.0, 0.5, 2.0, 0.0];
        assert_eq!(check(&r, &e, &leaves).unwrap().metric_defects()[0], 0.0);
        // Euclidean-unitary swap fails to preserve this S.
        r[0].u_tau_re = [0.0, 1.0, 1.0, 0.0];
        assert_eq!(
            check(&r, &e, &leaves).unwrap_err(),
            FoliatedConsensusError::SMetricViolation
        );
    }

    #[test]
    fn full_complex_overlap_terms_participate() {
        let (mut r, e, mut leaves) = fixture();
        for leaf in &mut leaves {
            leaf.overlap[1] = Complex64::new(0.25, 0.5);
            leaf.overlap[2] = Complex64::new(0.25, -0.5);
        }
        assert_eq!(check(&r, &e, &leaves).unwrap().metric_defects()[0], 0.0);
        // Diagonal phases are Euclidean-unitary but do not preserve the off-diagonal metric.
        r[0].u_tau_re = [1.0, 0.0, 0.0, -1.0];
        assert_eq!(
            check(&r, &e, &leaves).unwrap_err(),
            FoliatedConsensusError::SMetricViolation
        );
    }

    #[test]
    fn overlap_must_be_hermitian_positive_definite_and_finite() {
        let (r, e, leaves) = fixture();
        for invalid in 0..5 {
            let mut changed = leaves;
            match invalid {
                0 => changed[0].overlap[0].re = -1.0,
                1 => changed[0].overlap[1].re = 0.2,
                2 => changed[0].overlap[0].im = 0.1,
                3 => changed[0].overlap[0].re = f64::NAN,
                _ => {
                    changed[0].overlap[1].re = 1.0;
                    changed[0].overlap[2].re = 1.0;
                }
            }
            assert_eq!(
                check(&r, &e, &changed).unwrap_err(),
                FoliatedConsensusError::InvalidOverlap
            );
        }
    }

    #[test]
    fn nonorthogonal_populations_are_not_euclidean_renormalized() {
        let (mut r, e, leaves) = fixture();
        r[0].state_populations = [2.0, 0.5, 0.0, 0.0];
        assert!(check(&r, &e, &leaves).is_ok());
        r[0].state_populations[0] = -0.1;
        assert_eq!(
            check(&r, &e, &leaves).unwrap_err(),
            FoliatedConsensusError::InvalidField
        );
    }

    fn windows<'a>(energies: &'a [[f64; 501]; 8]) -> [NveEnergyWindow<'a>; 8] {
        std::array::from_fn(|i| NveEnergyWindow {
            start_step: 6,
            timestep_fs: 0.5,
            is_nve: true,
            total_energies: &energies[i],
        })
    }

    #[test]
    fn nve_campaign_checks_every_energy_sample_and_is_separate_from_ring_admission() {
        let (r, e, leaves) = fixture();
        let admitted = check(&r, &e, &leaves).unwrap();
        let mut energies = [[-100.0; 501]; 8];
        let w = windows(&energies);
        let evidence = FoliatedPhysicalAcceptanceEvidence {
            boundary: e,
            nve_windows: &w,
            window_flip_counts: [0; 8],
            flips: &[],
        };
        assert!(validate_foliated_physical_acceptance(&admitted, &evidence).is_ok());
        energies[7][250] = -99.99; // The terminal energy still agrees; the interior excursion must fail.
        let w = windows(&energies);
        let evidence = FoliatedPhysicalAcceptanceEvidence {
            boundary: e,
            nve_windows: &w,
            window_flip_counts: [0; 8],
            flips: &[],
        };
        assert_eq!(
            validate_foliated_physical_acceptance(&admitted, &evidence).unwrap_err(),
            FoliatedConsensusError::NveEnergyViolation
        );
        assert!(check(&r, &e, &leaves).is_ok()); // Ring validation does not misclassify a bath/UV workload as NVE.
    }

    #[test]
    fn nve_campaign_rejects_truncation_wrong_timestep_zero_reference_and_non_nve() {
        let (r, e, leaves) = fixture();
        let admitted = check(&r, &e, &leaves).unwrap();
        let energies = [[-100.0; 501]; 8];
        let empty = [0.0; 501];
        for case in 0..6 {
            let mut w = windows(&energies);
            match case {
                0 => w[7].total_energies = &energies[7][..500],
                1 => w[7].timestep_fs = 1.0,
                2 => w[7].is_nve = false,
                3 => w[7].start_step = 5,
                4 => w[7].total_energies = &empty,
                _ => w[7].timestep_fs = f64::NAN,
            }
            let evidence = FoliatedPhysicalAcceptanceEvidence {
                boundary: e,
                nve_windows: &w,
                window_flip_counts: [0; 8],
                flips: &[],
            };
            assert_eq!(
                validate_foliated_physical_acceptance(&admitted, &evidence).unwrap_err(),
                FoliatedConsensusError::InvalidNveWindow
            );
        }
    }

    #[test]
    fn every_flip_is_checked_with_coverage_and_terminal_step_counts() {
        let (mut r, e, leaves) = fixture();
        r[0].pachner_flip_count = 1;
        let admitted = check(&r, &e, &leaves).unwrap();
        let energies = [[-100.0; 501]; 8];
        let w = windows(&energies);
        let base = PachnerTraceFluxObservation {
            leaf_id: 0,
            step_tau: 506,
            flip_ordinal: 0,
            before: Complex64::new(0.0, 0.0),
            after: Complex64::new(1e-6, 2e-6),
        };
        let mut counts = [0; 8];
        counts[0] = 1;
        for case in 0..8 {
            let mut f = base;
            match case {
                0 => {}
                1 => f.flip_ordinal = 1,
                2 => f.step_tau = 6,
                3 => f.after = Complex64::new(5e-6, 0.0),
                4 => f.after = Complex64::new(f64::NAN, 0.0),
                5 => f.leaf_id = 8,
                6 => f.step_tau = 505,
                _ => f.step_tau = 507,
            }
            let flips = [f];
            let evidence = FoliatedPhysicalAcceptanceEvidence {
                boundary: e,
                nve_windows: &w,
                window_flip_counts: counts,
                flips: &flips,
            };
            let result = validate_foliated_physical_acceptance(&admitted, &evidence);
            if case == 0 {
                assert!(result.is_ok());
            } else {
                assert!(result.is_err(), "case {case}");
            }
        }
        let missing = FoliatedPhysicalAcceptanceEvidence {
            boundary: e,
            nve_windows: &w,
            window_flip_counts: counts,
            flips: &[],
        };
        assert_eq!(
            validate_foliated_physical_acceptance(&admitted, &missing).unwrap_err(),
            FoliatedConsensusError::InvalidFlipCoverage
        );
        let flips = [base, base];
        let duplicate = FoliatedPhysicalAcceptanceEvidence {
            flips: &flips,
            ..missing
        };
        assert_eq!(
            validate_foliated_physical_acceptance(&admitted, &duplicate).unwrap_err(),
            FoliatedConsensusError::InvalidFlipCoverage
        );
    }
    #[test]
    fn cross_leaf_links_are_checked_in_the_declared_direction() {
        let (mut r, e, mut leaves) = fixture();
        leaves[0].overlap[0].re = 4.0;
        // Current leaf0 (S=diag(4,1)) -> leaf1 (S=I): V=diag(2,1).
        r[0].v_lambda_re = [2.0, 0.0, 0.0, 1.0];
        // Current leaf7 (S=I) -> leaf0 (S=diag(4,1)): V=diag(0.5,1).
        r[7].v_lambda_re = [0.5, 0.0, 0.0, 1.0];
        let checked = check(&r, &e, &leaves).unwrap();
        assert_eq!(*checked.link_metric_defects(), [0.0; 8]);
        r[0].v_lambda_re[0] = 0.5;
        assert_eq!(
            check(&r, &e, &leaves).unwrap_err(),
            FoliatedConsensusError::LinkMetricViolation
        );
    }

    #[test]
    fn zero_links_empty_occupations_and_impossible_real_traces_are_rejected() {
        let (r, e, leaves) = fixture();
        for case in 0..3 {
            let mut bad = r;
            match case {
                0 => {
                    bad[7].v_lambda_re = [0.0; 4];
                    bad[7].v_lambda_im = [0.0; 4];
                }
                1 => bad[7].state_populations = [0.0; 4],
                _ => bad[7].w_x_lambda_trace[0] = -100.0,
            }
            assert!(check(&bad, &e, &leaves).is_err());
        }
    }

    #[test]
    fn moving_or_mismatched_basis_generations_are_refused() {
        let (r, e, leaves) = fixture();
        let mut moving = leaves;
        moving[7].previous_basis_generation_id[0] ^= 1;
        assert_eq!(
            check(&r, &e, &moving).unwrap_err(),
            FoliatedConsensusError::MovingMetricUnsupported
        );
        moving = leaves;
        moving[7].basis_generation_id[0] ^= 1;
        assert_eq!(
            check(&r, &e, &moving).unwrap_err(),
            FoliatedConsensusError::BoundaryIdentity
        );
    }

    #[test]
    fn small_absolute_metric_defect_does_not_claim_relative_or_scientific_validity() {
        let (mut r, e, mut leaves) = fixture();
        for i in 0..8 {
            leaves[i].overlap[0].re = 1e-20;
            leaves[i].overlap[3].re = 1e-20;
            r[i].u_tau_re = [2.0, 0.0, 0.0, 2.0];
        }
        let observations = check(&r, &e, &leaves).unwrap();
        assert!(observations.metric_defects()[0] < S_METRIC_TOLERANCE);
        assert_eq!(
            observations.evidence_level(),
            FoliatedEvidenceLevel::CheckedObservationsProducerReportedBianchi
        );
    }
}

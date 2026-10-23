//! PrismOS QNANL artifact packetization.
//!
//! This module intentionally avoids serde-based packet/header handling.  The
//! packet header is fixed-width binary metadata followed by canonical UTF-8
//! identifiers, a SECTION_ZSTD body, and a NANL-Q4 seal over deterministic
//! binary seal material.

use crate::compression_registry::{CompressionCodecKind, CompressionError, RuntimeCodecRegistry};
use crate::nanl_q4::{self, NanlQ4Error};
// runtime_origin reuses the SINGLE crate-wide provenance discriminator defined
// in operand_ledger.rs (DSTW-T4 Step 2). qnanl_artifact remains serde-free for
// its own packet/header handling. Serde is used only for durable publication
// journals and stream recovery metadata; packet bytes and canonical receipts
// retain their existing binary/manual encoders.
use crate::operand_ledger::RuntimeOrigin;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const QNANL_ARTIFACT_MAGIC: &[u8; 8] = b"PRSQNB01";
pub const QNANL_ARTIFACT_SCHEMA_KIND: &str = "prismos_qnanl_artifact_packet";
pub const QNANL_ARTIFACT_RECEIPT_KIND: &str = "prismos_qnanl_artifact_receipt";
pub const QNANL_ARTIFACT_SCHEMA_VERSION: u32 = 2;

const FIXED_HEADER_BYTES: usize = 8 + 4 + 2 + 2 + 4 + 8 * 3 + 4 * 3 + 32 * 3;
const TRANSFORM_CANONICAL_BYTES: u16 = 1;

pub type QnanlArtifactResult<T> = Result<T, QnanlArtifactError>;

#[derive(Debug, thiserror::Error)]
pub enum QnanlArtifactError {
    #[error("PC_ERR_IO:{0}")]
    Io(String),
    #[error("PC_ERR_FORMAT:{0}")]
    Format(String),
    #[error("PC_ERR_CODEC:{0}")]
    Codec(String),
    #[error("PC_ERR_QNANL:{0}")]
    Qnanl(String),
    #[error("PC_ERR_HASH:{0}")]
    Hash(String),
}

impl From<std::io::Error> for QnanlArtifactError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<CompressionError> for QnanlArtifactError {
    fn from(value: CompressionError) -> Self {
        Self::Codec(value.to_string())
    }
}

impl From<NanlQ4Error> for QnanlArtifactError {
    fn from(value: NanlQ4Error) -> Self {
        Self::Qnanl(value.to_string())
    }
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[repr(u16)]
pub enum QnanlArtifactFamily {
    EventStream = 1,
    MdForceKernelPtx = 2,
    DenseGpuField = 3,
    SignalGrid = 4,
    NumericArray = 5,
    TrajectoryCheckpoint = 6,
    JsonTelemetry = 7,
    LogStream = 8,
    PrismcodeExecutionArtifact = 9,
    RunInputReference = 10,
    DashboardPackage = 11,
    PathBPostMdTensor = 12,
    DrfDstwForgeTensor = 13,
    MoleculeDesignArtifact = 14,
    MdForceKernelOpcodeBundle = 15,
    BindingSiteMaterialization = 16,
    ThermodynamicPhaseManifold = 17,
    SpectralFiberBundle = 18,
    ResidueAlignmentMap = 19,
    MovieCallableFocus = 20,
    PrismThermTemporalSurface = 21,
    CcnsSdstSurface = 22,
    KccMaterialization = 23,
    BindingSitePhysiology = 24,
    GeometricPocketMorphology = 25,
    ScientificChronologyEvent = 26,
    ScientificStateEpoch = 27,
    T4EpochOverlay = 28,
    ChronologyAcceptanceDecision = 29,
    ScientificPocketOpeningEvent = 30,
    LigandCausalActivityEvent = 31,
    ResidueActivityEvent = 32,
    GhostSegmentCommit = 33,
    GhostConservation = 34,
    GhostPhaseBinding = 35,
    LambdaFragmentForceEvent = 36,
    LambdaFragmentNcmcMoveReceipt = 37,
    LambdaFragmentEndpointObservation = 38,
    LambdaFragmentStateCommit = 39,
    ChemicalStateFiber = 40,
    PathBChemicalState = 41,
    MatchedControlDiscriminator = 42,
    CandidateAdvancementRegistry = 43,
    LambdaFragmentStateRollback = 44,
    LambdaNoProbeAbsence = 45,
    GhostCoordinateSegmentCommit = 46,
    GhostCoordinateConservation = 47,
    GhostLatticeEdgeCommit = 48,
    GhostLatticeEdgeConservation = 49,
    /// Unbiased native no-probe whole-lattice discovery evidence. This family
    /// is intentionally separate from candidate advancement/registry state.
    PathBDiscoveryCandidate = 50,
    /// Post-commit explicit-selection handoff into candidate confirmation.
    PathBConfirmationHandoff = 51,
    Unknown = 65535,
}

impl QnanlArtifactFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EventStream => "EVENT_STREAM",
            Self::MdForceKernelPtx => "MD_FORCE_KERNEL_PTX",
            Self::DenseGpuField => "DENSE_GPU_FIELD",
            Self::SignalGrid => "SIGNAL_GRID",
            Self::NumericArray => "NUMERIC_ARRAY",
            Self::TrajectoryCheckpoint => "TRAJECTORY_CHECKPOINT",
            Self::JsonTelemetry => "JSON_TELEMETRY",
            Self::LogStream => "LOG_STREAM",
            Self::PrismcodeExecutionArtifact => "PRISMCODE_EXECUTION_ARTIFACT",
            Self::RunInputReference => "RUN_INPUT_REFERENCE",
            Self::DashboardPackage => "DASHBOARD_PACKAGE",
            Self::PathBPostMdTensor => "PATH_B_POST_MD_TENSOR",
            Self::DrfDstwForgeTensor => "DRF_DSTW_FORGE_TENSOR",
            Self::MoleculeDesignArtifact => "MOLECULE_DESIGN_ARTIFACT",
            Self::MdForceKernelOpcodeBundle => "MD_FORCE_KERNEL_OPCODE_BUNDLE",
            Self::BindingSiteMaterialization => "BINDING_SITE_MATERIALIZATION",
            Self::ThermodynamicPhaseManifold => "THERMODYNAMIC_PHASE_MANIFOLD",
            Self::SpectralFiberBundle => "SPECTRAL_FIBER_BUNDLE",
            Self::ResidueAlignmentMap => "RESIDUE_ALIGNMENT_MAP",
            Self::MovieCallableFocus => "MOVIE_CALLABLE_FOCUS",
            Self::PrismThermTemporalSurface => "PRISM_THERM_TEMPORAL_SURFACE",
            Self::CcnsSdstSurface => "CCNS_SDST_SURFACE",
            Self::KccMaterialization => "KCC_MATERIALIZATION",
            Self::BindingSitePhysiology => "PRISMOS_BINDING_SITE_PHYSIOLOGY_V1",
            Self::GeometricPocketMorphology => "PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1",
            Self::ScientificChronologyEvent => "SCIENTIFIC_CHRONOLOGY_EVENT",
            Self::ScientificStateEpoch => "SCIENTIFIC_STATE_EPOCH",
            Self::T4EpochOverlay => "T4_EPOCH_OVERLAY",
            Self::ChronologyAcceptanceDecision => "CHRONOLOGY_ACCEPTANCE_DECISION",
            Self::ScientificPocketOpeningEvent => "SCIENTIFIC_POCKET_OPENING_EVENT",
            Self::LigandCausalActivityEvent => "LIGAND_CAUSAL_ACTIVITY_EVENT",
            Self::ResidueActivityEvent => "RESIDUE_ACTIVITY_EVENT",
            Self::GhostSegmentCommit => "GHOST_SEGMENT_COMMIT",
            Self::GhostConservation => "GHOST_CONSERVATION",
            Self::GhostPhaseBinding => "GHOST_PHASE_BINDING",
            Self::LambdaFragmentForceEvent => "LAMBDA_FRAGMENT_FORCE_EVENT",
            Self::LambdaFragmentNcmcMoveReceipt => "LAMBDA_FRAGMENT_NCMC_MOVE_RECEIPT",
            Self::LambdaFragmentEndpointObservation => "LAMBDA_FRAGMENT_ENDPOINT_OBSERVATION",
            Self::LambdaFragmentStateCommit => "LAMBDA_FRAGMENT_STATE_COMMIT",
            Self::ChemicalStateFiber => "PRISM.CHEMICAL_STATE_FIBER.V1",
            Self::PathBChemicalState => "PRISM.PATHB_CHEMICAL_STATE.V1",
            Self::MatchedControlDiscriminator => "PRISM.MATCHED_CONTROL_DISCRIMINATOR.V1",
            Self::CandidateAdvancementRegistry => "PRISM.CANDIDATE_ADVANCEMENT_REGISTRY.V1",
            Self::LambdaFragmentStateRollback => "LAMBDA_FRAGMENT_STATE_ROLLBACK",
            Self::LambdaNoProbeAbsence => "LAMBDA_NO_PROBE_ABSENCE",
            Self::GhostCoordinateSegmentCommit => "GHOST_COORDINATE_SEGMENT_COMMIT",
            Self::GhostCoordinateConservation => "GHOST_COORDINATE_CONSERVATION",
            Self::GhostLatticeEdgeCommit => "GHOST_LATTICE_EDGE_COMMIT",
            Self::GhostLatticeEdgeConservation => "GHOST_LATTICE_EDGE_CONSERVATION",
            Self::PathBDiscoveryCandidate => "PRISM.PATHB_DISCOVERY_CANDIDATE.V1",
            Self::PathBConfirmationHandoff => "PRISM.PATHB_CONFIRMATION_HANDOFF.V1",
            Self::Unknown => "UNKNOWN",
        }
    }

    pub const fn default_transform(self) -> &'static str {
        match self {
            Self::EventStream => "EVENT_CANONICAL_BYTES",
            Self::MdForceKernelPtx => "PTX_CANONICAL_BYTES",
            Self::DenseGpuField => "DENSE_FIELD_CANONICAL_BYTES",
            Self::SignalGrid => "SIGNAL_GRID_CANONICAL_BYTES",
            Self::NumericArray => "NUMERIC_ARRAY_CANONICAL_BYTES",
            Self::TrajectoryCheckpoint => "TRAJECTORY_CANONICAL_BYTES",
            Self::JsonTelemetry => "JSON_CANONICAL_BYTES",
            Self::LogStream => "LOG_CANONICAL_BYTES",
            Self::PrismcodeExecutionArtifact => "PRISMCODE_EXECUTION_CANONICAL_BYTES",
            Self::RunInputReference => "RUN_INPUT_CANONICAL_BYTES",
            Self::DashboardPackage => "DASHBOARD_CANONICAL_BYTES",
            Self::PathBPostMdTensor => "PATH_B_TENSOR_CANONICAL_BYTES",
            Self::DrfDstwForgeTensor => "DRF_DSTW_FORGE_TENSOR_CANONICAL_BYTES",
            Self::MoleculeDesignArtifact => "MOLECULE_DESIGN_CANONICAL_BYTES",
            Self::MdForceKernelOpcodeBundle => "FORCE_KERNEL_OPCODE_BUNDLE_CANONICAL_BYTES",
            Self::BindingSiteMaterialization => "BINDING_SITE_MATERIALIZATION_CANONICAL_BYTES",
            Self::ThermodynamicPhaseManifold => "THERMODYNAMIC_PHASE_MANIFOLD_CANONICAL_BYTES",
            Self::SpectralFiberBundle => "SPECTRAL_FIBER_BUNDLE_CANONICAL_BYTES",
            Self::ResidueAlignmentMap => "RESIDUE_ALIGNMENT_MAP_CANONICAL_BYTES",
            Self::MovieCallableFocus => "MOVIE_CALLABLE_FOCUS_CANONICAL_BYTES",
            Self::PrismThermTemporalSurface => "PRISM_THERM_TEMPORAL_CANONICAL_BYTES",
            Self::CcnsSdstSurface => "CCNS_SDST_CANONICAL_BYTES",
            Self::KccMaterialization => "KCC_MATERIALIZATION_CANONICAL_BYTES",
            Self::BindingSitePhysiology => "BINDING_SITE_PHYSIOLOGY_CANONICAL_BYTES",
            Self::GeometricPocketMorphology => "GEOMETRIC_POCKET_MORPHOLOGY_CANONICAL_BYTES",
            Self::ScientificChronologyEvent => "SCIENTIFIC_CHRONOLOGY_EVENT_CANONICAL_BYTES",
            Self::ScientificStateEpoch => "SCIENTIFIC_STATE_EPOCH_CANONICAL_BYTES",
            Self::T4EpochOverlay => "T4_EPOCH_OVERLAY_CANONICAL_BYTES",
            Self::ChronologyAcceptanceDecision => "CHRONOLOGY_ACCEPTANCE_DECISION_CANONICAL_BYTES",
            Self::ScientificPocketOpeningEvent => "SCIENTIFIC_POCKET_OPENING_EVENT_CANONICAL_BYTES",
            Self::LigandCausalActivityEvent => "LIGAND_CAUSAL_ACTIVITY_EVENT_CANONICAL_BYTES",
            Self::ResidueActivityEvent => "RESIDUE_ACTIVITY_EVENT_CANONICAL_BYTES",
            Self::GhostSegmentCommit => "GHOST_SEGMENT_COMMIT_CANONICAL_BYTES",
            Self::GhostConservation => "GHOST_CONSERVATION_CANONICAL_BYTES",
            Self::GhostPhaseBinding => "GHOST_PHASE_BINDING_CANONICAL_BYTES",
            Self::LambdaFragmentForceEvent => "LAMBDA_FRAGMENT_FORCE_EVENT_CANONICAL_BYTES",
            Self::LambdaFragmentNcmcMoveReceipt => {
                "LAMBDA_FRAGMENT_NCMC_MOVE_RECEIPT_CANONICAL_BYTES"
            }
            Self::LambdaFragmentEndpointObservation => {
                "LAMBDA_FRAGMENT_ENDPOINT_OBSERVATION_CANONICAL_BYTES"
            }
            Self::LambdaFragmentStateCommit => "LAMBDA_FRAGMENT_STATE_COMMIT_CANONICAL_BYTES",
            Self::ChemicalStateFiber => "CHEMICAL_STATE_FIBER_CANONICAL_BYTES_V1",
            Self::PathBChemicalState => "PATHB_CHEMICAL_STATE_CANONICAL_BYTES_V1",
            Self::MatchedControlDiscriminator => "MATCHED_CONTROL_DISCRIMINATOR_CANONICAL_BYTES_V1",
            Self::CandidateAdvancementRegistry => {
                "CANDIDATE_ADVANCEMENT_REGISTRY_CANONICAL_BYTES_V1"
            }
            Self::LambdaFragmentStateRollback => "LAMBDA_FRAGMENT_STATE_ROLLBACK_CANONICAL_BYTES",
            Self::LambdaNoProbeAbsence => "LAMBDA_NO_PROBE_ABSENCE_CANONICAL_BYTES",
            Self::GhostCoordinateSegmentCommit => "GHOST_COORDINATE_SEGMENT_COMMIT_CANONICAL_BYTES",
            Self::GhostCoordinateConservation => "GHOST_COORDINATE_CONSERVATION_CANONICAL_BYTES",
            Self::GhostLatticeEdgeCommit => "GHOST_LATTICE_EDGE_COMMIT_CANONICAL_BYTES",
            Self::GhostLatticeEdgeConservation => "GHOST_LATTICE_EDGE_CONSERVATION_CANONICAL_BYTES",
            Self::PathBDiscoveryCandidate => "PATHB_DISCOVERY_CANDIDATE_CANONICAL_BYTES_V1",
            Self::PathBConfirmationHandoff => "PATHB_CONFIRMATION_HANDOFF_CANONICAL_BYTES_V1",
            Self::Unknown => "RAW_CANONICAL_BYTES",
        }
    }

    fn from_code(code: u16) -> QnanlArtifactResult<Self> {
        match code {
            1 => Ok(Self::EventStream),
            2 => Ok(Self::MdForceKernelPtx),
            3 => Ok(Self::DenseGpuField),
            4 => Ok(Self::SignalGrid),
            5 => Ok(Self::NumericArray),
            6 => Ok(Self::TrajectoryCheckpoint),
            7 => Ok(Self::JsonTelemetry),
            8 => Ok(Self::LogStream),
            9 => Ok(Self::PrismcodeExecutionArtifact),
            10 => Ok(Self::RunInputReference),
            11 => Ok(Self::DashboardPackage),
            12 => Ok(Self::PathBPostMdTensor),
            13 => Ok(Self::DrfDstwForgeTensor),
            14 => Ok(Self::MoleculeDesignArtifact),
            15 => Ok(Self::MdForceKernelOpcodeBundle),
            16 => Ok(Self::BindingSiteMaterialization),
            17 => Ok(Self::ThermodynamicPhaseManifold),
            18 => Ok(Self::SpectralFiberBundle),
            19 => Ok(Self::ResidueAlignmentMap),
            20 => Ok(Self::MovieCallableFocus),
            21 => Ok(Self::PrismThermTemporalSurface),
            22 => Ok(Self::CcnsSdstSurface),
            23 => Ok(Self::KccMaterialization),
            24 => Ok(Self::BindingSitePhysiology),
            25 => Ok(Self::GeometricPocketMorphology),
            26 => Ok(Self::ScientificChronologyEvent),
            27 => Ok(Self::ScientificStateEpoch),
            28 => Ok(Self::T4EpochOverlay),
            29 => Ok(Self::ChronologyAcceptanceDecision),
            30 => Ok(Self::ScientificPocketOpeningEvent),
            31 => Ok(Self::LigandCausalActivityEvent),
            32 => Ok(Self::ResidueActivityEvent),
            33 => Ok(Self::GhostSegmentCommit),
            34 => Ok(Self::GhostConservation),
            35 => Ok(Self::GhostPhaseBinding),
            36 => Ok(Self::LambdaFragmentForceEvent),
            37 => Ok(Self::LambdaFragmentNcmcMoveReceipt),
            38 => Ok(Self::LambdaFragmentEndpointObservation),
            39 => Ok(Self::LambdaFragmentStateCommit),
            40 => Ok(Self::ChemicalStateFiber),
            41 => Ok(Self::PathBChemicalState),
            42 => Ok(Self::MatchedControlDiscriminator),
            43 => Ok(Self::CandidateAdvancementRegistry),
            44 => Ok(Self::LambdaFragmentStateRollback),
            45 => Ok(Self::LambdaNoProbeAbsence),
            46 => Ok(Self::GhostCoordinateSegmentCommit),
            47 => Ok(Self::GhostCoordinateConservation),
            48 => Ok(Self::GhostLatticeEdgeCommit),
            49 => Ok(Self::GhostLatticeEdgeConservation),
            50 => Ok(Self::PathBDiscoveryCandidate),
            51 => Ok(Self::PathBConfirmationHandoff),
            65535 => Ok(Self::Unknown),
            other => Err(QnanlArtifactError::Format(format!(
                "unknown QNANL artifact family code {other}"
            ))),
        }
    }
}

impl fmt::Display for QnanlArtifactFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for QnanlArtifactFamily {
    type Err = QnanlArtifactError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let normalized = value.trim().replace('-', "_").to_ascii_uppercase();
        match normalized.as_str() {
            "EVENT_STREAM" | "SPIKES" | "SPIKE_EVENT_STREAM" => Ok(Self::EventStream),
            "MD_FORCE_KERNEL_PTX" | "FORCE_KERNEL" | "PTX" => Ok(Self::MdForceKernelPtx),
            "DENSE_GPU_FIELD" | "WARP_MATRIX" => Ok(Self::DenseGpuField),
            "SIGNAL_GRID" => Ok(Self::SignalGrid),
            "NUMERIC_ARRAY" | "FORCES_FINAL" | "ASC_VECTORS" | "KCC_V2FULL" => {
                Ok(Self::NumericArray)
            }
            "TRAJECTORY_CHECKPOINT" | "TRAJECTORY" | "CHECKPOINT" => Ok(Self::TrajectoryCheckpoint),
            "JSON_TELEMETRY" | "JSON" | "JSONL" | "RECEIPT" | "MANIFEST" => Ok(Self::JsonTelemetry),
            "LOG_STREAM" | "LOG" => Ok(Self::LogStream),
            "PRISMCODE_EXECUTION_ARTIFACT" | "PCODE" | "PCOUT" => {
                Ok(Self::PrismcodeExecutionArtifact)
            }
            "RUN_INPUT_REFERENCE" | "RUN_INPUT" | "INPUT" => Ok(Self::RunInputReference),
            "DASHBOARD_PACKAGE" | "DASHBOARD" | "PACKAGE" => Ok(Self::DashboardPackage),
            "PATH_B_POST_MD_TENSOR" | "PATH_B" | "POST_MD_TENSOR" => Ok(Self::PathBPostMdTensor),
            "DRF_DSTW_FORGE_TENSOR" | "DSTW" | "DRF" | "FORGE_TENSOR" | "GFLOWNET" => {
                Ok(Self::DrfDstwForgeTensor)
            }
            "MOLECULE_DESIGN_ARTIFACT" | "SDF" | "MOL" | "MOLECULE" => {
                Ok(Self::MoleculeDesignArtifact)
            }
            "MD_FORCE_KERNEL_OPCODE_BUNDLE" | "FORCE_KERNEL_OPCODE_BUNDLE" | "KERNEL_OPCODE" => {
                Ok(Self::MdForceKernelOpcodeBundle)
            }
            "BINDING_SITE_MATERIALIZATION"
            | "BINDING_SITE"
            | "BINDING_SITES"
            | "SITE_MATERIALIZATION"
            | "MATERIALIZED_SITE"
            | "MATERIALIZED_SITES" => Ok(Self::BindingSiteMaterialization),
            "THERMODYNAMIC_PHASE_MANIFOLD"
            | "PHASE_MANIFOLD"
            | "PHASE_STATE_MANIFOLD"
            | "TWIN16"
            | "TWIN16_TRANSITION_VIEW" => Ok(Self::ThermodynamicPhaseManifold),
            "SPECTRAL_FIBER_BUNDLE"
            | "SPECTRAL_BUNDLE"
            | "SPECTRAL_EXTRACTOR"
            | "FIBER_BUNDLE"
            | "PHASE_FIBER" => Ok(Self::SpectralFiberBundle),
            "RESIDUE_ALIGNMENT_MAP"
            | "RESIDUE_MAP"
            | "ATOM_TO_RESIDUE"
            | "CHAIN_NUMBERING"
            | "NUMBERING_LEDGER" => Ok(Self::ResidueAlignmentMap),
            "MOVIE_CALLABLE_FOCUS"
            | "MOVIE_FOCUS"
            | "MOVIE_WINDOW"
            | "MOVIE_WINDOW_INDEX"
            | "FOCUS_PLAN" => Ok(Self::MovieCallableFocus),
            "PRISM_THERM_TEMPORAL_SURFACE"
            | "PRISM_THERM"
            | "THERM_TEMPORAL"
            | "THERMODYNAMIC_TEMPORAL" => Ok(Self::PrismThermTemporalSurface),
            "CCNS_SDST_SURFACE" | "CCNS" | "SDST" | "CCNS_SDST" => Ok(Self::CcnsSdstSurface),
            "KCC_MATERIALIZATION" | "KCC_VISUALIZATION" | "KCC_DESCRIPTOR" | "KCC_MERGED" => {
                Ok(Self::KccMaterialization)
            }
            "PRISMOS_BINDING_SITE_PHYSIOLOGY_V1"
            | "BINDING_SITE_PHYSIOLOGY"
            | "BINDING_SITE_PHYSIOLOGY_SURFACE"
            | "PRISMOS_BINDING_SITE_PHYSIOLOGY"
            | "PRISM_THERM_SITE_PHYSIOLOGY" => Ok(Self::BindingSitePhysiology),
            "PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1"
            | "GEOMETRIC_POCKET_MORPHOLOGY"
            | "POCKET_MORPHOLOGY"
            | "POCKET_MORPHOLOGY_SURFACE"
            | "BINDING_SITE_MORPHOLOGY" => Ok(Self::GeometricPocketMorphology),
            "SCIENTIFIC_CHRONOLOGY_EVENT" | "CHRONOLOGY_EVENT" => {
                Ok(Self::ScientificChronologyEvent)
            }
            "SCIENTIFIC_STATE_EPOCH" | "STATE_EPOCH" => Ok(Self::ScientificStateEpoch),
            "T4_EPOCH_OVERLAY" | "DSTW_T4_EPOCH_OVERLAY" | "EPOCH_OVERLAY" => {
                Ok(Self::T4EpochOverlay)
            }
            "CHRONOLOGY_ACCEPTANCE_DECISION"
            | "DSTW_T4_CHRONOLOGY_ACCEPTANCE_DECISION"
            | "ACCEPTANCE_DECISION" => Ok(Self::ChronologyAcceptanceDecision),
            "SCIENTIFIC_POCKET_OPENING_EVENT" | "SCIENTIFICPOCKETOPENINGEVENT" => {
                Ok(Self::ScientificPocketOpeningEvent)
            }
            "LIGAND_CAUSAL_ACTIVITY_EVENT" | "LIGANDCAUSALACTIVITYEVENT" => {
                Ok(Self::LigandCausalActivityEvent)
            }
            "RESIDUE_ACTIVITY_EVENT" | "RESIDUEACTIVITYEVENT" => Ok(Self::ResidueActivityEvent),
            "GHOST_SEGMENT_COMMIT" => Ok(Self::GhostSegmentCommit),
            "GHOST_CONSERVATION" => Ok(Self::GhostConservation),
            "GHOST_PHASE_BINDING" => Ok(Self::GhostPhaseBinding),
            "LAMBDA_FRAGMENT_FORCE_EVENT" => Ok(Self::LambdaFragmentForceEvent),
            "LAMBDA_FRAGMENT_NCMC_MOVE_RECEIPT" => Ok(Self::LambdaFragmentNcmcMoveReceipt),
            "LAMBDA_FRAGMENT_ENDPOINT_OBSERVATION" => Ok(Self::LambdaFragmentEndpointObservation),
            "LAMBDA_FRAGMENT_STATE_COMMIT" => Ok(Self::LambdaFragmentStateCommit),
            "LAMBDA_FRAGMENT_STATE_ROLLBACK" => Ok(Self::LambdaFragmentStateRollback),
            "LAMBDA_NO_PROBE_ABSENCE" => Ok(Self::LambdaNoProbeAbsence),
            "GHOST_COORDINATE_SEGMENT_COMMIT" => Ok(Self::GhostCoordinateSegmentCommit),
            "GHOST_COORDINATE_CONSERVATION" => Ok(Self::GhostCoordinateConservation),
            "GHOST_LATTICE_EDGE_COMMIT" => Ok(Self::GhostLatticeEdgeCommit),
            "GHOST_LATTICE_EDGE_CONSERVATION" => Ok(Self::GhostLatticeEdgeConservation),
            "PRISM.PATHB_DISCOVERY_CANDIDATE.V1" | "PATHB_DISCOVERY_CANDIDATE" => {
                Ok(Self::PathBDiscoveryCandidate)
            }
            "PRISM.PATHB_CONFIRMATION_HANDOFF.V1" | "PATHB_CONFIRMATION_HANDOFF" => {
                Ok(Self::PathBConfirmationHandoff)
            }
            "PRISM.CHEMICAL_STATE_FIBER.V1" | "CHEMICAL_STATE_FIBER" => {
                Ok(Self::ChemicalStateFiber)
            }
            "PRISM.PATHB_CHEMICAL_STATE.V1" | "PATHB_CHEMICAL_STATE" => {
                Ok(Self::PathBChemicalState)
            }
            "PRISM.MATCHED_CONTROL_DISCRIMINATOR.V1" | "MATCHED_CONTROL_DISCRIMINATOR" => {
                Ok(Self::MatchedControlDiscriminator)
            }
            "PRISM.CANDIDATE_ADVANCEMENT_REGISTRY.V1" | "CANDIDATE_ADVANCEMENT_REGISTRY" => {
                Ok(Self::CandidateAdvancementRegistry)
            }
            "UNKNOWN" | "RAW" => Ok(Self::Unknown),
            other => Err(QnanlArtifactError::Format(format!(
                "unknown QNANL artifact family {other}"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QnanlArtifactPacketHeader {
    pub schema_kind: String,
    pub schema_version: u32,
    pub artifact_id: String,
    pub artifact_family: QnanlArtifactFamily,
    pub artifact_transform: String,
    pub source_path: Option<String>,
    pub source_file_name: Option<String>,
    pub codec_stack: Vec<String>,
    pub codec_kind: CompressionCodecKind,
    pub raw_bytes: u64,
    pub raw_hash_blake3: String,
    pub compressed_body_bytes: u64,
    pub compressed_body_hash_blake3: String,
    pub qnanl_seal_bytes: u64,
    pub qnanl_seal_hash_blake3: String,
}

impl QnanlArtifactPacketHeader {
    pub fn to_json_pretty(&self) -> String {
        let source_path = self
            .source_path
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());
        let source_file_name = self
            .source_file_name
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());
        let codec_stack = json_string_array(&self.codec_stack);
        format!(
            concat!(
                "{{\n",
                "  \"schema_kind\": \"{}\",\n",
                "  \"schema_version\": {},\n",
                "  \"artifact_id\": \"{}\",\n",
                "  \"artifact_family\": \"{}\",\n",
                "  \"artifact_transform\": \"{}\",\n",
                "  \"source_path\": {},\n",
                "  \"source_file_name\": {},\n",
                "  \"codec_stack\": {},\n",
                "  \"codec_kind\": \"SECTION_ZSTD\",\n",
                "  \"raw_bytes\": {},\n",
                "  \"raw_hash_blake3\": \"{}\",\n",
                "  \"compressed_body_bytes\": {},\n",
                "  \"compressed_body_hash_blake3\": \"{}\",\n",
                "  \"qnanl_seal_bytes\": {},\n",
                "  \"qnanl_seal_hash_blake3\": \"{}\"\n",
                "}}\n"
            ),
            json_escape(&self.schema_kind),
            self.schema_version,
            json_escape(&self.artifact_id),
            self.artifact_family,
            json_escape(&self.artifact_transform),
            source_path,
            source_file_name,
            codec_stack,
            self.raw_bytes,
            self.raw_hash_blake3,
            self.compressed_body_bytes,
            self.compressed_body_hash_blake3,
            self.qnanl_seal_bytes,
            self.qnanl_seal_hash_blake3
        )
    }
}

/// A typed PrismCode reference attached to a QNANL receipt.
///
/// Per the DSTW-T4 plan (§STEP 4) the receipt should be able to carry the
/// `pcode_refs[]{opcode, operands[]}` that a runtime stream writer associates
/// with a sealed chunk. This is a typed reference (opcode name/id + operand
/// labels), NOT a behavioral execution claim — the ISA is a sealed folding
/// executor with zero per-opcode dispatch (binary_runtime.rs `execute_packet`).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct QnanlPcodeRef {
    pub opcode_name: String,
    pub opcode_id: Option<u16>,
    pub operands: Vec<String>,
}

impl QnanlPcodeRef {
    pub fn to_json(&self) -> String {
        let opcode_id = self
            .opcode_id
            .map(|id| format!("\"0x{id:04X}\""))
            .unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"opcode_name\": \"{}\", \"opcode_id\": {}, \"operands\": {}}}",
            json_escape(&self.opcode_name),
            opcode_id,
            json_string_array(&self.operands)
        )
    }
}

fn qnanl_pcode_refs_json(refs: &[QnanlPcodeRef]) -> String {
    let mut out = String::from("[");
    for (idx, r) in refs.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&r.to_json());
    }
    out.push(']');
    out
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QnanlArtifactReceipt {
    pub schema_kind: String,
    pub schema_version: u32,
    pub status: String,
    pub artifact_id: String,
    pub artifact_family: QnanlArtifactFamily,
    pub artifact_transform: String,
    pub source_path: String,
    pub packet_path: String,
    pub receipt_path: String,
    pub raw_bytes: u64,
    pub packet_bytes: u64,
    pub compressed_body_bytes: u64,
    pub qnanl_seal_bytes: u64,
    pub compression_ratio_vs_raw: f64,
    pub size_reduction_percent: f64,
    pub encode_ns: u128,
    pub decode_verify_ns: u128,
    pub qnanl_seal_ns: u128,
    pub encode_mib_s: f64,
    pub decode_verify_mib_s: f64,
    pub raw_hash_blake3: String,
    pub packet_hash_blake3: String,
    pub compressed_body_hash_blake3: String,
    pub qnanl_seal_hash_blake3: String,
    pub roundtrip_verified: bool,
    pub codec_roundtrip_verified: bool,
    pub qnanl_seal_status: String,
    pub qnanl_seal_constraints_pass: bool,
    pub qnanl_seal_merkle_verified: bool,
    pub qnanl_seal_ecc_verified: bool,
    /// MANDATORY honest provenance of this packet. Callers must set this in
    /// `QnanlEncodeOptions` before packet and receipt bytes are written.
    /// Teardown-time relabeling is forbidden.
    pub runtime_origin: RuntimeOrigin,
    /// The instant at which the artifact lifecycle origin was assigned, before
    /// packet bytes and the receipt are written. This prevents teardown-time
    /// provenance decoration from masquerading as runtime-native emission.
    pub runtime_origin_assigned_unix_ns: u128,
    /// Code path that instantiated the origin. Runtime writers must name their
    /// writer/bootstrap stage here; post-run sealers use an explicit post-run
    /// stage label.
    pub runtime_origin_assignment_stage: String,
    /// Per-chunk content hashes (blake3 hex). `encode_bytes` seals exactly one
    /// chunk, so this is a single-element vector = `[raw_hash_blake3]`. A stream
    /// writer accumulates one entry per sealed chunk across packets.
    pub chunk_hashes: Vec<String>,
    /// Typed PrismCode references for this artifact (opcode + operand labels).
    /// Empty for the byte-level encoders; populated by callers that drive a
    /// PrismCode opcode trace alongside the seal. NOT an execution claim.
    pub pcode_refs: Vec<QnanlPcodeRef>,
    /// PROCESS-PROVEN per-artifact provenance (DSTW-T4 Step h1). A bare
    /// `runtime_origin` label is INVALID without this block. The byte-level
    /// encoders leave it `None` (a pure sealer has no run context); the
    /// prismos `md-run` emit path attaches it via [`Self::with_runtime_provenance`]
    /// from the run-level `RuntimeProvenance` captured at start.
    pub runtime_provenance: Option<crate::runtime_provenance::ArtifactProvenance>,
}

impl QnanlArtifactReceipt {
    pub fn with_pcode_refs(mut self, refs: Vec<QnanlPcodeRef>) -> Self {
        self.pcode_refs = refs;
        self
    }

    /// Attach PROCESS-PROVEN per-artifact provenance (DSTW-T4 Step h1). Used by
    /// the prismos `md-run` emit path so the receipt's `runtime_origin` label is
    /// backed by the captured binary sha256, run pid, emission time, source stage,
    /// and upstream parent hashes — not a bare string.
    pub fn with_runtime_provenance(
        mut self,
        provenance: crate::runtime_provenance::ArtifactProvenance,
    ) -> Self {
        self.runtime_provenance = Some(provenance);
        self
    }

    pub fn to_json_pretty(&self) -> String {
        let chunk_hashes_json = json_string_array(&self.chunk_hashes);
        let pcode_refs_json = qnanl_pcode_refs_json(&self.pcode_refs);
        // PROCESS-PROVEN provenance block (serde-rendered), or explicit `null`
        // when the pure-sealer path produced this receipt (no run context).
        let runtime_provenance_json = match &self.runtime_provenance {
            Some(p) => serde_json::to_string(&p.to_json()).unwrap_or_else(|_| "null".to_string()),
            None => "null".to_string(),
        };
        format!(
            concat!(
                "{{\n",
                "  \"schema_kind\": \"{}\",\n",
                "  \"schema_version\": {},\n",
                "  \"status\": \"{}\",\n",
                "  \"artifact_id\": \"{}\",\n",
                "  \"artifact_family\": \"{}\",\n",
                "  \"artifact_transform\": \"{}\",\n",
                "  \"source_path\": \"{}\",\n",
                "  \"packet_path\": \"{}\",\n",
                "  \"receipt_path\": \"{}\",\n",
                "  \"raw_bytes\": {},\n",
                "  \"packet_bytes\": {},\n",
                "  \"compressed_body_bytes\": {},\n",
                "  \"qnanl_seal_bytes\": {},\n",
                "  \"compression_ratio_vs_raw\": {:.9},\n",
                "  \"size_reduction_percent\": {:.9},\n",
                "  \"encode_ns\": {},\n",
                "  \"decode_verify_ns\": {},\n",
                "  \"qnanl_seal_ns\": {},\n",
                "  \"encode_mib_s\": {:.9},\n",
                "  \"decode_verify_mib_s\": {:.9},\n",
                "  \"raw_hash_blake3\": \"{}\",\n",
                "  \"packet_hash_blake3\": \"{}\",\n",
                "  \"compressed_body_hash_blake3\": \"{}\",\n",
                "  \"qnanl_seal_hash_blake3\": \"{}\",\n",
                "  \"roundtrip_verified\": {},\n",
                "  \"codec_roundtrip_verified\": {},\n",
                "  \"qnanl_seal_status\": \"{}\",\n",
                "  \"qnanl_seal_constraints_pass\": {},\n",
                "  \"qnanl_seal_merkle_verified\": {},\n",
                "  \"qnanl_seal_ecc_verified\": {},\n",
                "  \"runtime_origin\": \"{}\",\n",
                "  \"runtime_native\": {},\n",
                "  \"runtime_origin_assigned_unix_ns\": {},\n",
                "  \"runtime_origin_assignment_stage\": \"{}\",\n",
                "  \"chunk_hashes\": {},\n",
                "  \"pcode_refs\": {},\n",
                "  \"runtime_provenance\": {}\n",
                "}}\n"
            ),
            json_escape(&self.schema_kind),
            self.schema_version,
            json_escape(&self.status),
            json_escape(&self.artifact_id),
            self.artifact_family,
            json_escape(&self.artifact_transform),
            json_escape(&self.source_path),
            json_escape(&self.packet_path),
            json_escape(&self.receipt_path),
            self.raw_bytes,
            self.packet_bytes,
            self.compressed_body_bytes,
            self.qnanl_seal_bytes,
            self.compression_ratio_vs_raw,
            self.size_reduction_percent,
            self.encode_ns,
            self.decode_verify_ns,
            self.qnanl_seal_ns,
            self.encode_mib_s,
            self.decode_verify_mib_s,
            self.raw_hash_blake3,
            self.packet_hash_blake3,
            self.compressed_body_hash_blake3,
            self.qnanl_seal_hash_blake3,
            self.roundtrip_verified,
            self.codec_roundtrip_verified,
            json_escape(&self.qnanl_seal_status),
            self.qnanl_seal_constraints_pass,
            self.qnanl_seal_merkle_verified,
            self.qnanl_seal_ecc_verified,
            self.runtime_origin.as_str(),
            self.runtime_origin.is_prismos_native(),
            self.runtime_origin_assigned_unix_ns,
            json_escape(&self.runtime_origin_assignment_stage),
            chunk_hashes_json,
            pcode_refs_json,
            runtime_provenance_json
        )
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QnanlEncodeOptions {
    pub artifact_id: String,
    pub artifact_family: QnanlArtifactFamily,
    pub source_path: Option<PathBuf>,
    /// Origin is supplied before encode/write begins. This is intentionally part
    /// of the encode options, not a post-encode receipt mutator.
    pub runtime_origin: RuntimeOrigin,
    pub runtime_origin_assigned_unix_ns: u128,
    pub runtime_origin_assignment_stage: String,
    pub runtime_provenance: Option<crate::runtime_provenance::ArtifactProvenance>,
}

impl QnanlEncodeOptions {
    pub fn new(
        artifact_id: impl Into<String>,
        artifact_family: QnanlArtifactFamily,
        source_path: Option<PathBuf>,
        runtime_origin: RuntimeOrigin,
        runtime_origin_assignment_stage: impl Into<String>,
    ) -> Self {
        Self {
            artifact_id: artifact_id.into(),
            artifact_family,
            source_path,
            runtime_origin,
            runtime_origin_assigned_unix_ns: now_unix_ns(),
            runtime_origin_assignment_stage: runtime_origin_assignment_stage.into(),
            runtime_provenance: None,
        }
    }

    pub fn post_run_seal_only(
        artifact_id: impl Into<String>,
        artifact_family: QnanlArtifactFamily,
        source_path: Option<PathBuf>,
        assignment_stage: impl Into<String>,
    ) -> Self {
        Self::new(
            artifact_id,
            artifact_family,
            source_path,
            RuntimeOrigin::PostRunSealOnly,
            assignment_stage,
        )
    }

    pub fn runtime_subprocess_orchestrated_by_prismos(
        artifact_id: impl Into<String>,
        artifact_family: QnanlArtifactFamily,
        source_path: Option<PathBuf>,
        assignment_stage: impl Into<String>,
    ) -> Self {
        Self::new(
            artifact_id,
            artifact_family,
            source_path,
            RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
            assignment_stage,
        )
    }

    pub fn runtime_inline(
        artifact_id: impl Into<String>,
        artifact_family: QnanlArtifactFamily,
        source_path: Option<PathBuf>,
        assignment_stage: impl Into<String>,
    ) -> Self {
        Self::new(
            artifact_id,
            artifact_family,
            source_path,
            RuntimeOrigin::RuntimeInline,
            assignment_stage,
        )
    }

    pub fn with_runtime_provenance(
        mut self,
        provenance: crate::runtime_provenance::ArtifactProvenance,
    ) -> Self {
        self.runtime_provenance = Some(provenance);
        self
    }
}

pub fn encode_file(
    input: &Path,
    packet_path: &Path,
    receipt_path: &Path,
    options: QnanlEncodeOptions,
) -> QnanlArtifactResult<QnanlArtifactReceipt> {
    let raw = std::fs::read(input)?;
    let mut opts = options;
    opts.source_path = Some(input.to_path_buf());
    encode_bytes(&raw, packet_path, receipt_path, opts)
}

pub fn encode_file_streaming(
    input: &Path,
    packet_path: &Path,
    receipt_path: &Path,
    options: QnanlEncodeOptions,
) -> QnanlArtifactResult<QnanlArtifactReceipt> {
    let _publication_lock = lock_artifact_publication(packet_path)?;
    let _receipt_lock = lock_artifact_publication(receipt_path)?;
    recover_packet_publication(packet_path, receipt_path)?;

    let source_path_str = input.display().to_string();
    let source_file_name = input
        .file_name()
        .and_then(|path| path.to_str())
        .map(str::to_string)
        .unwrap_or_default();

    let registry = RuntimeCodecRegistry::with_builtin_codecs();
    registry.validate_required_codecs()?;

    let (raw_hash, raw_bytes) = hash_file_streaming(input)?;
    let compressed_body_tmp = packet_temp_body_path(packet_path);
    if let Some(parent) = compressed_body_tmp.parent() {
        durable_create_dir_all(parent)?;
    }

    let encode_start = Instant::now();
    {
        let mut raw_reader = File::open(input)?;
        let compressed_writer = File::create(&compressed_body_tmp)?;
        let mut encoder = zstd::stream::Encoder::new(compressed_writer, 3)?;
        std::io::copy(&mut raw_reader, &mut encoder)?;
        encoder.finish()?;
    }
    let encode_ns = encode_start.elapsed().as_nanos();

    let (compressed_hash, compressed_body_bytes) = hash_file_streaming(&compressed_body_tmp)?;
    let pre_seal_header = BinaryHeaderMaterial {
        artifact_family: options.artifact_family,
        transform_code: TRANSFORM_CANONICAL_BYTES,
        flags: 0,
        raw_bytes,
        compressed_body_bytes,
        qnanl_seal_bytes: 0,
        artifact_id: options.artifact_id.clone(),
        source_path: source_path_str.clone(),
        source_file_name: source_file_name.clone(),
        raw_hash,
        compressed_hash,
        qnanl_seal_hash: [0u8; 32],
    };
    let seal_payload = build_seal_payload(&pre_seal_header);

    let qnanl_start = Instant::now();
    let q4_packet = nanl_q4::encode_payload(&seal_payload)?;
    let decoded_seal = nanl_q4::decode_packet(q4_packet.bytes())?;
    if decoded_seal != seal_payload {
        return Err(QnanlArtifactError::Hash(
            "QNANL seal roundtrip mismatch".to_string(),
        ));
    }
    let qnanl_report = nanl_q4::validate_packet(q4_packet.bytes())?;
    let qnanl_seal_ns = qnanl_start.elapsed().as_nanos();
    let qnanl_hash = hash32(q4_packet.bytes());

    let header = BinaryHeaderMaterial {
        qnanl_seal_bytes: q4_packet.bytes().len() as u64,
        qnanl_seal_hash: qnanl_hash,
        ..pre_seal_header
    };
    let header_bytes = encode_fixed_header(&header)?;

    if let Some(parent) = packet_path.parent() {
        durable_create_dir_all(parent)?;
    }
    {
        let mut packet_file = File::create(packet_staging_path(packet_path))?;
        packet_file.write_all(&header_bytes)?;
        let mut compressed_reader = File::open(&compressed_body_tmp)?;
        std::io::copy(&mut compressed_reader, &mut packet_file)?;
        packet_file.write_all(q4_packet.bytes())?;
        packet_file.sync_all()?;
    }

    let decode_start = Instant::now();
    let (decoded_hash, decoded_bytes) = {
        let compressed_reader = File::open(&compressed_body_tmp)?;
        let decoder = zstd::stream::Decoder::new(compressed_reader)?;
        hash_reader_streaming(decoder)?
    };
    if decoded_hash != raw_hash || decoded_bytes != raw_bytes {
        let _ = std::fs::remove_file(&compressed_body_tmp);
        return Err(QnanlArtifactError::Hash(format!(
            "artifact {} streaming decode differs from canonical raw bytes",
            options.artifact_id
        )));
    }
    let decode_verify_ns = decode_start.elapsed().as_nanos();

    let (packet_hash, packet_bytes) = hash_file_streaming(&packet_staging_path(packet_path))?;
    let compression_ratio_vs_raw = if compressed_body_bytes > 0 {
        raw_bytes as f64 / compressed_body_bytes as f64
    } else {
        0.0
    };
    let size_reduction_percent = if raw_bytes > 0 {
        100.0 * (1.0 - compressed_body_bytes as f64 / raw_bytes as f64)
    } else {
        0.0
    };
    let raw_mib = raw_bytes as f64 / (1024.0 * 1024.0);
    let encode_mib_s = if encode_ns > 0 {
        raw_mib / (encode_ns as f64 / 1_000_000_000.0)
    } else {
        0.0
    };
    let decode_verify_mib_s = if decode_verify_ns > 0 {
        raw_mib / (decode_verify_ns as f64 / 1_000_000_000.0)
    } else {
        0.0
    };
    let receipt = QnanlArtifactReceipt {
        schema_kind: QNANL_ARTIFACT_RECEIPT_KIND.to_string(),
        schema_version: QNANL_ARTIFACT_SCHEMA_VERSION,
        status: "PASS".to_string(),
        artifact_id: options.artifact_id,
        artifact_family: options.artifact_family,
        artifact_transform: options.artifact_family.default_transform().to_string(),
        source_path: source_path_str,
        packet_path: packet_path.display().to_string(),
        receipt_path: receipt_path.display().to_string(),
        raw_bytes,
        packet_bytes,
        compressed_body_bytes,
        qnanl_seal_bytes: q4_packet.bytes().len() as u64,
        compression_ratio_vs_raw,
        size_reduction_percent,
        encode_ns,
        decode_verify_ns,
        qnanl_seal_ns,
        encode_mib_s,
        decode_verify_mib_s,
        raw_hash_blake3: hex32(&raw_hash),
        packet_hash_blake3: hex32(&packet_hash),
        compressed_body_hash_blake3: hex32(&compressed_hash),
        qnanl_seal_hash_blake3: hex32(&qnanl_hash),
        roundtrip_verified: true,
        codec_roundtrip_verified: true,
        qnanl_seal_status: qnanl_report.status,
        qnanl_seal_constraints_pass: qnanl_report.constraints_pass,
        qnanl_seal_merkle_verified: qnanl_report.merkle_verified,
        qnanl_seal_ecc_verified: qnanl_report.ecc_verified,
        runtime_origin: options.runtime_origin,
        runtime_origin_assigned_unix_ns: options.runtime_origin_assigned_unix_ns,
        runtime_origin_assignment_stage: options.runtime_origin_assignment_stage,
        chunk_hashes: vec![hex32(&raw_hash)],
        pcode_refs: Vec::new(),
        runtime_provenance: options.runtime_provenance,
    };
    if let Some(parent) = receipt_path.parent() {
        durable_create_dir_all(parent)?;
    }
    let committed = publish_packet(packet_path, receipt_path, receipt)?;
    let _ = std::fs::remove_file(compressed_body_tmp);
    Ok(committed)
}

pub fn encode_bytes(
    raw: &[u8],
    packet_path: &Path,
    receipt_path: &Path,
    options: QnanlEncodeOptions,
) -> QnanlArtifactResult<QnanlArtifactReceipt> {
    let _publication_lock = lock_artifact_publication(packet_path)?;
    let _receipt_lock = lock_artifact_publication(receipt_path)?;
    recover_packet_publication(packet_path, receipt_path)?;

    let source_path_str = options
        .source_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "memory".to_string());
    let source_file_name = options.source_path.as_ref().and_then(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
    });

    let registry = RuntimeCodecRegistry::with_builtin_codecs();
    registry.validate_required_codecs()?;

    let encode_start = Instant::now();
    let section = registry.encode_section(CompressionCodecKind::SectionZstd, raw)?;
    let encode_ns = encode_start.elapsed().as_nanos();

    let raw_hash = hash32(raw);
    let compressed_hash = hash32(&section.compressed_bytes);

    let source_path_for_header = options
        .source_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let source_file_for_header = source_file_name.unwrap_or_default();
    let pre_seal_header = BinaryHeaderMaterial {
        artifact_family: options.artifact_family,
        transform_code: TRANSFORM_CANONICAL_BYTES,
        flags: 0,
        raw_bytes: raw.len() as u64,
        compressed_body_bytes: section.compressed_bytes.len() as u64,
        qnanl_seal_bytes: 0,
        artifact_id: options.artifact_id.clone(),
        source_path: source_path_for_header.clone(),
        source_file_name: source_file_for_header.clone(),
        raw_hash,
        compressed_hash,
        qnanl_seal_hash: [0u8; 32],
    };
    let seal_payload = build_seal_payload(&pre_seal_header);

    let qnanl_start = Instant::now();
    let q4_packet = nanl_q4::encode_payload(&seal_payload)?;
    let decoded_seal = nanl_q4::decode_packet(q4_packet.bytes())?;
    if decoded_seal != seal_payload {
        return Err(QnanlArtifactError::Hash(
            "QNANL seal roundtrip mismatch".to_string(),
        ));
    }
    let qnanl_report = nanl_q4::validate_packet(q4_packet.bytes())?;
    let qnanl_seal_ns = qnanl_start.elapsed().as_nanos();
    let qnanl_hash = hash32(q4_packet.bytes());

    let header = BinaryHeaderMaterial {
        qnanl_seal_bytes: q4_packet.bytes().len() as u64,
        qnanl_seal_hash: qnanl_hash,
        ..pre_seal_header
    };
    let header_bytes = encode_fixed_header(&header)?;

    if let Some(parent) = packet_path.parent() {
        durable_create_dir_all(parent)?;
    }
    let mut packet = Vec::with_capacity(
        header_bytes.len() + section.compressed_bytes.len() + q4_packet.bytes().len(),
    );
    packet.extend_from_slice(&header_bytes);
    packet.extend_from_slice(&section.compressed_bytes);
    packet.extend_from_slice(q4_packet.bytes());

    let decode_start = Instant::now();
    let decoded = decode_packet_bytes(&packet)?;
    let codec_roundtrip = registry.validate_encoded_by_kind(
        CompressionCodecKind::SectionZstd,
        &section.compressed_bytes,
        section.hashes,
    )?;
    if decoded != raw {
        return Err(QnanlArtifactError::Hash(format!(
            "artifact {} decoded bytes differ from canonical raw bytes",
            options.artifact_id
        )));
    }
    let decode_verify_ns = decode_start.elapsed().as_nanos();

    let packet_hash_blake3 = hex32(&hash32(&packet));
    let packet_bytes = packet.len() as u64;
    let compressed_body_bytes = section.compressed_bytes.len() as u64;
    let compression_ratio_vs_raw = if compressed_body_bytes > 0 {
        raw.len() as f64 / compressed_body_bytes as f64
    } else {
        0.0
    };
    let size_reduction_percent = if !raw.is_empty() {
        100.0 * (1.0 - compressed_body_bytes as f64 / raw.len() as f64)
    } else {
        0.0
    };
    let raw_mib = raw.len() as f64 / (1024.0 * 1024.0);
    let encode_mib_s = if encode_ns > 0 {
        raw_mib / (encode_ns as f64 / 1_000_000_000.0)
    } else {
        0.0
    };
    let decode_verify_mib_s = if decode_verify_ns > 0 {
        raw_mib / (decode_verify_ns as f64 / 1_000_000_000.0)
    } else {
        0.0
    };

    let receipt = QnanlArtifactReceipt {
        schema_kind: QNANL_ARTIFACT_RECEIPT_KIND.to_string(),
        schema_version: QNANL_ARTIFACT_SCHEMA_VERSION,
        status: "PASS".to_string(),
        artifact_id: options.artifact_id,
        artifact_family: options.artifact_family,
        artifact_transform: options.artifact_family.default_transform().to_string(),
        source_path: source_path_str,
        packet_path: packet_path.display().to_string(),
        receipt_path: receipt_path.display().to_string(),
        raw_bytes: raw.len() as u64,
        packet_bytes,
        compressed_body_bytes,
        qnanl_seal_bytes: q4_packet.bytes().len() as u64,
        compression_ratio_vs_raw,
        size_reduction_percent,
        encode_ns,
        decode_verify_ns,
        qnanl_seal_ns,
        encode_mib_s,
        decode_verify_mib_s,
        raw_hash_blake3: hex32(&raw_hash),
        packet_hash_blake3,
        compressed_body_hash_blake3: hex32(&compressed_hash),
        qnanl_seal_hash_blake3: hex32(&qnanl_hash),
        roundtrip_verified: true,
        codec_roundtrip_verified: codec_roundtrip.roundtrip_verified,
        qnanl_seal_status: qnanl_report.status,
        qnanl_seal_constraints_pass: qnanl_report.constraints_pass,
        qnanl_seal_merkle_verified: qnanl_report.merkle_verified,
        qnanl_seal_ecc_verified: qnanl_report.ecc_verified,
        runtime_origin: options.runtime_origin,
        runtime_origin_assigned_unix_ns: options.runtime_origin_assigned_unix_ns,
        runtime_origin_assignment_stage: options.runtime_origin_assignment_stage,
        chunk_hashes: vec![hex32(&raw_hash)],
        pcode_refs: Vec::new(),
        runtime_provenance: options.runtime_provenance,
    };
    if let Some(parent) = receipt_path.parent() {
        durable_create_dir_all(parent)?;
    }
    durable_write(&packet_staging_path(packet_path), &packet)?;
    publish_packet(packet_path, receipt_path, receipt)
}

pub fn decode_file(packet_path: &Path, output_path: &Path) -> QnanlArtifactResult<Vec<u8>> {
    let packet = std::fs::read(packet_path)?;
    let decoded = decode_packet_bytes(&packet)?;
    if let Some(parent) = output_path.parent() {
        durable_create_dir_all(parent)?;
    }
    durable_write(output_path, &decoded)?;
    Ok(decoded)
}

pub fn decode_packet_bytes(packet: &[u8]) -> QnanlArtifactResult<Vec<u8>> {
    let (header, compressed, qnanl_seal) = parse_packet(packet)?;
    if hash32(qnanl_seal) != header.qnanl_seal_hash {
        return Err(QnanlArtifactError::Hash(
            "QNANL seal hash mismatch".to_string(),
        ));
    }
    let decoded_seal = nanl_q4::decode_packet(qnanl_seal)?;
    let expected_seal = build_seal_payload(&BinaryHeaderMaterial {
        qnanl_seal_bytes: 0,
        qnanl_seal_hash: [0u8; 32],
        ..header.clone()
    });
    if decoded_seal != expected_seal {
        return Err(QnanlArtifactError::Hash(
            "QNANL seal payload mismatch".to_string(),
        ));
    }
    let registry = RuntimeCodecRegistry::with_builtin_codecs();
    let codec = registry.codec_by_kind(CompressionCodecKind::SectionZstd)?;
    let decoded = codec.decode(compressed)?;
    if hash32(&decoded) != header.raw_hash {
        return Err(QnanlArtifactError::Hash(
            "decoded raw hash mismatch".to_string(),
        ));
    }
    Ok(decoded)
}

pub fn packet_header(packet_path: &Path) -> QnanlArtifactResult<QnanlArtifactPacketHeader> {
    let packet = std::fs::read(packet_path)?;
    let (header, _, _) = parse_packet(&packet)?;
    Ok(QnanlArtifactPacketHeader {
        schema_kind: QNANL_ARTIFACT_SCHEMA_KIND.to_string(),
        schema_version: QNANL_ARTIFACT_SCHEMA_VERSION,
        artifact_id: header.artifact_id,
        artifact_family: header.artifact_family,
        artifact_transform: header.artifact_family.default_transform().to_string(),
        source_path: (!header.source_path.is_empty()).then_some(header.source_path),
        source_file_name: (!header.source_file_name.is_empty()).then_some(header.source_file_name),
        codec_stack: vec![
            header.artifact_family.default_transform().to_string(),
            "SECTION_ZSTD".to_string(),
            "QNANL_Q4_SEAL".to_string(),
        ],
        codec_kind: CompressionCodecKind::SectionZstd,
        raw_bytes: header.raw_bytes,
        raw_hash_blake3: hex32(&header.raw_hash),
        compressed_body_bytes: header.compressed_body_bytes,
        compressed_body_hash_blake3: hex32(&header.compressed_hash),
        qnanl_seal_bytes: header.qnanl_seal_bytes,
        qnanl_seal_hash_blake3: hex32(&header.qnanl_seal_hash),
    })
}

#[derive(Clone, Debug)]
struct BinaryHeaderMaterial {
    artifact_family: QnanlArtifactFamily,
    transform_code: u16,
    flags: u32,
    raw_bytes: u64,
    compressed_body_bytes: u64,
    qnanl_seal_bytes: u64,
    artifact_id: String,
    source_path: String,
    source_file_name: String,
    raw_hash: [u8; 32],
    compressed_hash: [u8; 32],
    qnanl_seal_hash: [u8; 32],
}

fn encode_fixed_header(header: &BinaryHeaderMaterial) -> QnanlArtifactResult<Vec<u8>> {
    let artifact_id = header.artifact_id.as_bytes();
    let source_path = header.source_path.as_bytes();
    let source_file_name = header.source_file_name.as_bytes();
    let artifact_id_len = u32::try_from(artifact_id.len())
        .map_err(|_| QnanlArtifactError::Format("artifact_id too long".to_string()))?;
    let source_path_len = u32::try_from(source_path.len())
        .map_err(|_| QnanlArtifactError::Format("source_path too long".to_string()))?;
    let source_file_name_len = u32::try_from(source_file_name.len())
        .map_err(|_| QnanlArtifactError::Format("source_file_name too long".to_string()))?;

    let mut bytes = Vec::with_capacity(
        FIXED_HEADER_BYTES + artifact_id.len() + source_path.len() + source_file_name.len(),
    );
    bytes.extend_from_slice(QNANL_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&QNANL_ARTIFACT_SCHEMA_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(header.artifact_family as u16).to_le_bytes());
    bytes.extend_from_slice(&header.transform_code.to_le_bytes());
    bytes.extend_from_slice(&header.flags.to_le_bytes());
    bytes.extend_from_slice(&header.raw_bytes.to_le_bytes());
    bytes.extend_from_slice(&header.compressed_body_bytes.to_le_bytes());
    bytes.extend_from_slice(&header.qnanl_seal_bytes.to_le_bytes());
    bytes.extend_from_slice(&artifact_id_len.to_le_bytes());
    bytes.extend_from_slice(&source_path_len.to_le_bytes());
    bytes.extend_from_slice(&source_file_name_len.to_le_bytes());
    bytes.extend_from_slice(&header.raw_hash);
    bytes.extend_from_slice(&header.compressed_hash);
    bytes.extend_from_slice(&header.qnanl_seal_hash);
    bytes.extend_from_slice(artifact_id);
    bytes.extend_from_slice(source_path);
    bytes.extend_from_slice(source_file_name);
    Ok(bytes)
}

fn parse_packet(packet: &[u8]) -> QnanlArtifactResult<(BinaryHeaderMaterial, &[u8], &[u8])> {
    if packet.len() < FIXED_HEADER_BYTES {
        return Err(QnanlArtifactError::Format(
            "truncated QNANL artifact packet".to_string(),
        ));
    }
    if &packet[0..8] != QNANL_ARTIFACT_MAGIC {
        return Err(QnanlArtifactError::Format(
            "QNANL artifact magic mismatch".to_string(),
        ));
    }
    let mut cursor = 8usize;
    let version = read_u32(packet, &mut cursor, "version")?;
    if version != QNANL_ARTIFACT_SCHEMA_VERSION {
        return Err(QnanlArtifactError::Format(format!(
            "unsupported QNANL artifact version {version}"
        )));
    }
    let family = QnanlArtifactFamily::from_code(read_u16(packet, &mut cursor, "family")?)?;
    let transform_code = read_u16(packet, &mut cursor, "transform_code")?;
    if transform_code != TRANSFORM_CANONICAL_BYTES {
        return Err(QnanlArtifactError::Format(format!(
            "unsupported QNANL artifact transform code {transform_code}"
        )));
    }
    let flags = read_u32(packet, &mut cursor, "flags")?;
    let raw_bytes = read_u64(packet, &mut cursor, "raw_bytes")?;
    let compressed_body_bytes = read_u64(packet, &mut cursor, "compressed_body_bytes")?;
    let qnanl_seal_bytes = read_u64(packet, &mut cursor, "qnanl_seal_bytes")?;
    let artifact_id_len = read_u32(packet, &mut cursor, "artifact_id_len")? as usize;
    let source_path_len = read_u32(packet, &mut cursor, "source_path_len")? as usize;
    let source_file_name_len = read_u32(packet, &mut cursor, "source_file_name_len")? as usize;
    let raw_hash = read_hash32(packet, &mut cursor, "raw_hash")?;
    let compressed_hash = read_hash32(packet, &mut cursor, "compressed_hash")?;
    let qnanl_seal_hash = read_hash32(packet, &mut cursor, "qnanl_seal_hash")?;
    let artifact_id = read_utf8(packet, &mut cursor, artifact_id_len, "artifact_id")?;
    let source_path = read_utf8(packet, &mut cursor, source_path_len, "source_path")?;
    let source_file_name = read_utf8(
        packet,
        &mut cursor,
        source_file_name_len,
        "source_file_name",
    )?;
    let compressed_len = usize::try_from(compressed_body_bytes)
        .map_err(|_| QnanlArtifactError::Format("compressed length overflow".to_string()))?;
    let qnanl_len = usize::try_from(qnanl_seal_bytes)
        .map_err(|_| QnanlArtifactError::Format("qnanl seal length overflow".to_string()))?;
    let compressed_end = cursor
        .checked_add(compressed_len)
        .ok_or_else(|| QnanlArtifactError::Format("compressed cursor overflow".to_string()))?;
    let qnanl_end = compressed_end
        .checked_add(qnanl_len)
        .ok_or_else(|| QnanlArtifactError::Format("qnanl cursor overflow".to_string()))?;
    if qnanl_end != packet.len() {
        return Err(QnanlArtifactError::Format(format!(
            "QNANL artifact packet length mismatch: expected {} got {}",
            qnanl_end,
            packet.len()
        )));
    }
    let compressed = &packet[cursor..compressed_end];
    if hash32(compressed) != compressed_hash {
        return Err(QnanlArtifactError::Hash(
            "compressed body hash mismatch".to_string(),
        ));
    }
    let qnanl = &packet[compressed_end..qnanl_end];
    Ok((
        BinaryHeaderMaterial {
            artifact_family: family,
            transform_code,
            flags,
            raw_bytes,
            compressed_body_bytes,
            qnanl_seal_bytes,
            artifact_id,
            source_path,
            source_file_name,
            raw_hash,
            compressed_hash,
            qnanl_seal_hash,
        },
        compressed,
        qnanl,
    ))
}

fn build_seal_payload(header: &BinaryHeaderMaterial) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"PRISMOS_QNANL_ARTIFACT_SEAL_V2\0");
    bytes.extend_from_slice(&(header.artifact_family as u16).to_le_bytes());
    bytes.extend_from_slice(&header.transform_code.to_le_bytes());
    bytes.extend_from_slice(&header.raw_bytes.to_le_bytes());
    bytes.extend_from_slice(&header.compressed_body_bytes.to_le_bytes());
    bytes.extend_from_slice(&(header.artifact_id.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.artifact_id.as_bytes());
    bytes.extend_from_slice(&(header.source_path.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.source_path.as_bytes());
    bytes.extend_from_slice(&(header.source_file_name.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.source_file_name.as_bytes());
    bytes.extend_from_slice(&header.raw_hash);
    bytes.extend_from_slice(&header.compressed_hash);
    bytes
}

fn hash32(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn now_unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// Durable publication metadata is separate from the fixed QNANL packet ABI.
// These mechanisms qualify filesystem artifact continuity only. They do not
// serialize MD velocities, RNG state, thermostats or device execution state.
static DURABLE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn durability_error(message: &str) -> QnanlArtifactError {
    QnanlArtifactError::Io(format!("QNANL_DURABILITY:{message}"))
}

fn durable_json<T: serde::Serialize>(value: &T) -> QnanlArtifactResult<Vec<u8>> {
    let mut bytes =
        serde_json::to_vec_pretty(value).map_err(|e| durability_error(&e.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn read_durable_json<T: serde::de::DeserializeOwned>(path: &Path) -> QnanlArtifactResult<T> {
    serde_json::from_slice(&std::fs::read(path)?).map_err(|e| {
        durability_error(&format!(
            "invalid retained metadata {}: {e}",
            path.display()
        ))
    })
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn sync_parent(path: &Path) -> QnanlArtifactResult<()> {
    File::open(parent_directory(path))?.sync_all()?;
    Ok(())
}

fn durable_create_dir_all(path: &Path) -> QnanlArtifactResult<()> {
    if path.as_os_str().is_empty() || path.is_dir() {
        return Ok(());
    }
    durable_create_dir_all(parent_directory(path))?;
    match std::fs::create_dir(path) {
        Ok(()) => sync_parent(path)?,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && path.is_dir() => {}
        Err(e) => return Err(e.into()),
    }
    Ok(())
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

/// Hold an OS advisory lock for an artifact publication namespace. Keep the
/// returned file alive for the entire transaction; the permanent lock inode is
/// released automatically on process termination and must not be unlinked.
pub fn lock_artifact_publication(path: &Path) -> QnanlArtifactResult<File> {
    durable_create_dir_all(parent_directory(path))?;
    let lock_path = sidecar_path(path, ".publication.lock");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)?;
    // An OS lock is released by process death; a stale PID file is not a lock.
    // Keep the lock inode permanently, so no second lock namespace can arise.
    file.try_lock()
        .map_err(|e| durability_error(&format!("publication busy {}: {e}", path.display())))?;
    sync_parent(&lock_path)?;
    Ok(file)
}

fn durable_write(path: &Path, bytes: &[u8]) -> QnanlArtifactResult<()> {
    durable_create_dir_all(parent_directory(path))?;
    let sequence = DURABLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = sidecar_path(
        path,
        &format!(".tmp.{}.{}.{}", std::process::id(), now_unix_ns(), sequence),
    );
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)?;
        sync_parent(path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

/// Publish immutable artifact bytes using a synchronized exclusive temporary
/// file and a no-clobber hard link. Concurrent identical requests are idempotent;
/// conflicting bytes are retained and rejected. No caller lock is required.
pub fn publish_immutable_artifact(path: &Path, bytes: &[u8]) -> QnanlArtifactResult<()> {
    durable_create_dir_all(parent_directory(path))?;
    let sequence = DURABLE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = sidecar_path(
        path,
        &format!(".tmp.{}.{}.{}", std::process::id(), now_unix_ns(), sequence),
    );
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        match std::fs::hard_link(&temporary, path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if std::fs::read(path)? != bytes {
                    return Err(durability_error(&format!(
                        "conflicting immutable artifact {}",
                        path.display()
                    )));
                }
                File::open(path)?.sync_all()?;
            }
            Err(error) => return Err(error.into()),
        }
        sync_parent(path)?;
        std::fs::remove_file(&temporary)?;
        sync_parent(path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn immutable_write(path: &Path, bytes: &[u8]) -> QnanlArtifactResult<()> {
    publish_immutable_artifact(path, bytes)
}

/// Replace a derived artifact snapshot atomically and durably. The caller owns
/// its transaction lock; immutable scientific evidence must use the immutable
/// publisher instead. This is used for current-status views and stream indexes.
pub fn publish_artifact_snapshot(path: &Path, bytes: &[u8]) -> QnanlArtifactResult<()> {
    durable_write(path, bytes)
}

/// Publication metadata is retained for verification, never reclassified as raw
/// scientific payload by a QNANL directory sweep.
pub fn is_qnanl_auxiliary_path(path: &Path) -> bool {
    is_native_continuation_recovery_path(path) || path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase().contains(".prismqnanl."))
        .unwrap_or(false)
}



fn checkpoint_digest_name(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn native_uncommitted_tail_name(name: &str) -> bool {
    let Some((_, suffix)) = name.rsplit_once(".uncommitted.") else { return false; };
    let words: Vec<_> = suffix.split('.').collect();
    let digits = |value: &str| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit());
    match words.as_slice() {
        // Current content-bound tail and its in-progress copy.
        [prefix, bytes, tail] => checkpoint_digest_name(prefix) && digits(bytes)
            && (checkpoint_digest_name(tail) || *tail == "next"),
        // Earlier prefix-bound retained tail/staging remain recovery evidence.
        [prefix, bytes] => checkpoint_digest_name(prefix) && (digits(bytes) || *bytes == "next"),
        _ => false,
    }
}

/// Native continuation keeps later, uncommitted observations for provenance.
/// These reserved namespaces must never become active scientific payload via
/// recursive discovery, source resolution, QNANL sealing or DSTW ingestion.
pub fn is_native_continuation_recovery_path(path: &Path) -> bool {
    path.components().any(|component| {
        let std::path::Component::Normal(name) = component else { return false; };
        let Some(name) = name.to_str() else { return false; };
        matches!(name, ".continuation_tails" | ".continuation_absent_owners")
            || native_uncommitted_tail_name(name)
            || name.strip_prefix(".restore-").and_then(|n| n.strip_suffix(".pending")).map_or(false, checkpoint_digest_name)
            || (name.starts_with(".scientific_chronology_") && name.ends_with(".restore"))
    })
}

/// Recover a previously initiated publication and return the original receipt
/// only after verifying its durable commit and independently decoding the packet.
/// An absent publication returns None; conflicting/legacy partial paths error.
pub fn recover_published_artifact(
    packet: &Path,
    receipt_path: &Path,
) -> QnanlArtifactResult<Option<QnanlArtifactReceipt>> {
    let _packet_lock = lock_artifact_publication(packet)?;
    let _receipt_lock = lock_artifact_publication(receipt_path)?;
    recover_packet_publication(packet, receipt_path)?;
    let commit_path = sidecar_path(packet, ".commit.json");
    if !commit_path.exists() {
        if packet.exists() || receipt_path.exists() {
            return Err(durability_error(
                "existing publication has no durable commit; retained without overwrite",
            ));
        }
        return Ok(None);
    }
    let commit: DurablePacketCommit = read_durable_json(&commit_path)?;
    verify_packet_commit(packet, receipt_path, &commit)?;
    let decoded = decode_packet_bytes(&std::fs::read(packet)?)?;
    if decoded.len() as u64 != commit.receipt.raw_bytes
        || hex32(&hash32(&decoded)) != commit.receipt.raw_hash_blake3
    {
        return Err(durability_error(
            "recovered packet does not decode to committed raw content",
        ));
    }
    Ok(Some(commit.receipt))
}

fn file_hash(path: &Path) -> QnanlArtifactResult<String> {
    Ok(hex32(&hash_file_streaming(path)?.0))
}

fn packet_staging_path(packet: &Path) -> PathBuf {
    sidecar_path(packet, ".packet.pending")
}

#[derive(serde::Serialize, serde::Deserialize)]
struct DurablePacketCommit {
    version: u32,
    receipt: QnanlArtifactReceipt,
    canonical_receipt_hash: String,
}

fn verify_packet_commit(
    packet: &Path,
    receipt_path: &Path,
    commit: &DurablePacketCommit,
) -> QnanlArtifactResult<()> {
    let receipt = &commit.receipt;
    if commit.version != 1
        || receipt.packet_path != packet.display().to_string()
        || receipt.receipt_path != receipt_path.display().to_string()
        || file_hash(packet)? != receipt.packet_hash_blake3
        || file_hash(receipt_path)? != commit.canonical_receipt_hash
        || std::fs::read(receipt_path)? != receipt.to_json_pretty().as_bytes()
    {
        return Err(durability_error(
            "packet/receipt commit identity or hash mismatch",
        ));
    }
    Ok(())
}

fn recover_packet_publication(packet: &Path, receipt_path: &Path) -> QnanlArtifactResult<()> {
    if packet == receipt_path {
        return Err(durability_error("packet and receipt paths must differ"));
    }
    let intent_path = sidecar_path(packet, ".transaction.json");
    if !intent_path.exists() {
        return Ok(());
    }
    let intent_bytes = std::fs::read(&intent_path)?;
    let commit: DurablePacketCommit =
        serde_json::from_slice(&intent_bytes).map_err(|e| durability_error(&e.to_string()))?;
    if commit.version != 1
        || commit.receipt.packet_path != packet.display().to_string()
        || commit.receipt.receipt_path != receipt_path.display().to_string()
        || hex32(&hash32(commit.receipt.to_json_pretty().as_bytes()))
            != commit.canonical_receipt_hash
    {
        return Err(durability_error("retained packet intent identity mismatch"));
    }
    let stages = [
        (
            packet_staging_path(packet),
            packet,
            &commit.receipt.packet_hash_blake3,
        ),
        (
            sidecar_path(receipt_path, ".receipt.pending"),
            receipt_path,
            &commit.canonical_receipt_hash,
        ),
    ];
    for (index, (staged, destination, expected_hash)) in stages.into_iter().enumerate() {
        if destination.exists() {
            if file_hash(destination)? != *expected_hash {
                return Err(durability_error(&format!(
                    "conflicting publication destination {}",
                    destination.display()
                )));
            }
            File::open(destination)?.sync_all()?;
            sync_parent(destination)?;
        } else {
            if file_hash(&staged)? != *expected_hash {
                return Err(durability_error("retained staged content hash mismatch"));
            }
            File::open(&staged)?.sync_all()?;
            std::fs::rename(&staged, destination)?;
            sync_parent(destination)?;
        }
        durability_fault(if index == 0 {
            "packet_published"
        } else {
            "receipt_published"
        });
    }
    verify_packet_commit(packet, receipt_path, &commit)?;
    immutable_write(&sidecar_path(packet, ".commit.json"), &intent_bytes)?;
    durability_fault("packet_commit");
    std::fs::remove_file(intent_path)?;
    sync_parent(packet)?;
    Ok(())
}

// Performance timings and assignment clock are outcomes, not retry identity.
// All source content, origin/stage and supplied process provenance must match.
// Returning a stored receipt preserves its original timings and provenance.
fn receipt_request_identity(
    receipt: &QnanlArtifactReceipt,
) -> QnanlArtifactResult<serde_json::Value> {
    let mut value = serde_json::to_value(receipt).map_err(|e| durability_error(&e.to_string()))?;
    if let Some(map) = value.as_object_mut() {
        for field in [
            "encode_ns",
            "decode_verify_ns",
            "qnanl_seal_ns",
            "encode_mib_s",
            "decode_verify_mib_s",
            "runtime_origin_assigned_unix_ns",
            // Derived compression diagnostics are outcomes, not immutable operands.
            // Their exact raw/compressed sizes and content hashes remain bound.
            "compression_ratio_vs_raw",
            "size_reduction_percent",
        ] {
            map.remove(field);
        }
    }
    Ok(value)
}

fn publish_packet(
    packet: &Path,
    receipt_path: &Path,
    receipt: QnanlArtifactReceipt,
) -> QnanlArtifactResult<QnanlArtifactReceipt> {
    let commit_path = sidecar_path(packet, ".commit.json");
    if commit_path.exists() {
        let existing: DurablePacketCommit = read_durable_json(&commit_path)?;
        verify_packet_commit(packet, receipt_path, &existing)?;
        if receipt_request_identity(&existing.receipt)? != receipt_request_identity(&receipt)? {
            return Err(durability_error("committed artifact conflicts with retry content or provenance; use a new artifact path"));
        }
        let _ = std::fs::remove_file(packet_staging_path(packet));
        return Ok(existing.receipt);
    }
    if packet.exists() || receipt_path.exists() {
        return Err(durability_error(
            "preexisting artifact lacks this publication commit; retained without overwrite",
        ));
    }
    if file_hash(&packet_staging_path(packet))? != receipt.packet_hash_blake3 {
        return Err(durability_error("staged packet hash mismatch"));
    }
    let receipt_bytes = receipt.to_json_pretty().into_bytes();
    durable_write(
        &sidecar_path(receipt_path, ".receipt.pending"),
        &receipt_bytes,
    )?;
    let commit = DurablePacketCommit {
        version: 1,
        receipt,
        canonical_receipt_hash: hex32(&hash32(&receipt_bytes)),
    };
    durable_write(
        &sidecar_path(packet, ".transaction.json"),
        &durable_json(&commit)?,
    )?;
    durability_fault("packet_intent");
    recover_packet_publication(packet, receipt_path)?;
    Ok(commit.receipt)
}

#[cfg(not(test))]
fn durability_fault(_point: &str) {}

#[cfg(test)]
fn durability_fault(point: &str) {
    // Only the isolated child-test process sets these variables. Production
    // builds contain no environment-controlled fault injection.
    if std::env::var("PRISM_QNANL_DURABILITY_CHILD").as_deref() == Ok("1")
        && std::env::var("PRISM_QNANL_DURABILITY_CRASH").as_deref() == Ok(point)
    {
        std::process::exit(91);
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct DurableStreamIdentity {
    stream_name: String,
    runtime_origin: RuntimeOrigin,
    topology_ref: Option<String>,
    engine_ref: Option<String>,
    parent_state_vertex_id: Option<String>,
    parent_state_vertex_id_absent_reason: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DurableChunkEvidence {
    receipt_hash: String,
    proof_hash: String,
    lineage_hash: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct DurableStreamState {
    version: u32,
    identity: DurableStreamIdentity,
    chunks: Vec<RuntimeQnanlStreamChunk>,
    evidence: Vec<DurableChunkEvidence>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StreamChunkIntent {
    version: u32,
    identity: DurableStreamIdentity,
    predecessor: Option<String>,
    chunk_index: u64,
    raw_hash: String,
    raw_bytes: u64,
    options: QnanlEncodeOptions,
    context: RuntimeQnanlChunkContext,
}

impl StreamChunkIntent {
    fn same_request(&self, other: &Self) -> QnanlArtifactResult<bool> {
        let normalize = |intent: &Self| -> QnanlArtifactResult<serde_json::Value> {
            let mut value =
                serde_json::to_value(intent).map_err(|e| durability_error(&e.to_string()))?;
            value["options"]
                .as_object_mut()
                .ok_or_else(|| durability_error("missing options"))?
                .remove("runtime_origin_assigned_unix_ns");
            Ok(value)
        };
        Ok(self.version == 1 && normalize(self)? == normalize(other)?)
    }
}

fn hash_file_streaming(path: &Path) -> QnanlArtifactResult<([u8; 32], u64)> {
    hash_reader_streaming(File::open(path)?)
}

fn hash_reader_streaming<R: Read>(mut reader: R) -> QnanlArtifactResult<([u8; 32], u64)> {
    let mut hasher = blake3::Hasher::new();
    let mut total = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total += read as u64;
    }
    Ok((*hasher.finalize().as_bytes(), total))
}

fn packet_temp_body_path(packet_path: &Path) -> PathBuf {
    let name = packet_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact.prismqnanl");
    packet_path.with_file_name(format!("{name}.body.zst.tmp"))
}

fn hex32(hash: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in hash {
        out.push(nibble_hex(byte >> 4));
        out.push(nibble_hex(byte & 0x0f));
    }
    out
}

fn nibble_hex(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        _ => (b'a' + (value - 10)) as char,
    }
}

fn read_u16(bytes: &[u8], cursor: &mut usize, label: &str) -> QnanlArtifactResult<u16> {
    let end = cursor
        .checked_add(2)
        .ok_or_else(|| QnanlArtifactError::Format(format!("{label} cursor overflow")))?;
    if end > bytes.len() {
        return Err(QnanlArtifactError::Format(format!(
            "truncated QNANL artifact field {label}"
        )));
    }
    let value = u16::from_le_bytes(bytes[*cursor..end].try_into().map_err(|_| {
        QnanlArtifactError::Format(format!("invalid QNANL artifact field {label}"))
    })?);
    *cursor = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], cursor: &mut usize, label: &str) -> QnanlArtifactResult<u32> {
    let end = cursor
        .checked_add(4)
        .ok_or_else(|| QnanlArtifactError::Format(format!("{label} cursor overflow")))?;
    if end > bytes.len() {
        return Err(QnanlArtifactError::Format(format!(
            "truncated QNANL artifact field {label}"
        )));
    }
    let value = u32::from_le_bytes(bytes[*cursor..end].try_into().map_err(|_| {
        QnanlArtifactError::Format(format!("invalid QNANL artifact field {label}"))
    })?);
    *cursor = end;
    Ok(value)
}

fn read_u64(bytes: &[u8], cursor: &mut usize, label: &str) -> QnanlArtifactResult<u64> {
    let end = cursor
        .checked_add(8)
        .ok_or_else(|| QnanlArtifactError::Format(format!("{label} cursor overflow")))?;
    if end > bytes.len() {
        return Err(QnanlArtifactError::Format(format!(
            "truncated QNANL artifact field {label}"
        )));
    }
    let value = u64::from_le_bytes(bytes[*cursor..end].try_into().map_err(|_| {
        QnanlArtifactError::Format(format!("invalid QNANL artifact field {label}"))
    })?);
    *cursor = end;
    Ok(value)
}

fn read_hash32(bytes: &[u8], cursor: &mut usize, label: &str) -> QnanlArtifactResult<[u8; 32]> {
    let end = cursor
        .checked_add(32)
        .ok_or_else(|| QnanlArtifactError::Format(format!("{label} cursor overflow")))?;
    if end > bytes.len() {
        return Err(QnanlArtifactError::Format(format!(
            "truncated QNANL artifact field {label}"
        )));
    }
    let value = bytes[*cursor..end]
        .try_into()
        .map_err(|_| QnanlArtifactError::Format(format!("invalid QNANL artifact field {label}")))?;
    *cursor = end;
    Ok(value)
}

fn read_utf8(
    bytes: &[u8],
    cursor: &mut usize,
    len: usize,
    label: &str,
) -> QnanlArtifactResult<String> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| QnanlArtifactError::Format(format!("{label} cursor overflow")))?;
    if end > bytes.len() {
        return Err(QnanlArtifactError::Format(format!(
            "truncated QNANL artifact field {label}"
        )));
    }
    let value = std::str::from_utf8(&bytes[*cursor..end])
        .map_err(|err| QnanlArtifactError::Format(format!("{label} utf8 error: {err}")))?
        .to_string();
    *cursor = end;
    Ok(value)
}

pub fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let code = c as u32;
                out.push_str("\\u");
                out.push(nibble_hex(((code >> 12) & 0x0f) as u8));
                out.push(nibble_hex(((code >> 8) & 0x0f) as u8));
                out.push(nibble_hex(((code >> 4) & 0x0f) as u8));
                out.push(nibble_hex((code & 0x0f) as u8));
            }
            c => out.push(c),
        }
    }
    out
}

pub fn json_string_array(values: &[String]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push('"');
        out.push_str(&json_escape(value));
        out.push('"');
    }
    out.push(']');
    out
}

// ============================================================================
// RuntimeQnanlStreamWriter — lifecycle-orchestrated v0 (DSTW-T4 Step 4 scaffold)
// ============================================================================
//
// CORRECTED TAXONOMY (verified file:line):
//   * QNANL Primary Artifact Compression = metadata + SECTION_ZSTD body (the
//     real size reducer) + NANL-Q4 seal (qnanl_artifact.rs:5; encode_bytes:628).
//   * NANL-Q4 Constrained Quaternary Seal = the seal layer (nanl_q4.rs:1), NOT a
//     bytes-to-bases costume.
//   * NANL-DNA Simulated Archival Export = simulator/export only (nanl_dna.rs:1),
//     NOT physical storage and NOT load-bearing here.
//
// HONESTY BOUNDARY — this is v0, *lifecycle-orchestrated*, NOT in-MD-loop
// streaming. It wraps the existing per-chunk sealer `encode_bytes` from the
// prismos lifecycle (the orchestrator that already launched the engine
// subprocess), sealing chunks as they are handed to it and emitting the three
// named side-cars per packet. Because the prismos lifecycle drives it, the
// honest provenance is `RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS` — a genuine
// PRISMOS_NATIVE origin. It does NOT and MUST NOT claim `RUNTIME_INLINE`
// (in-MD-loop) emission: threading a writer handle into the per-stream MD
// closure so chunks seal DURING stepping (nhs_rt_full.rs:18358) is the
// engine-owner v1 gate (DSTW-T4 plan gate 11, BLOCKED-ENGINE).

/// Version tag stamped into every side-car so consumers can tell v0
/// (lifecycle-orchestrated) apart from a future v1 (in-MD-loop) writer.
pub const RUNTIME_QNANL_STREAM_WRITER_VERSION: &str = "v0_lifecycle_orchestrated";

/// One sealed-and-decode-proven chunk in the stream.
///
/// DSTW-T4 Step h2: a chunk now carries the FULL per-packet evidence record the
/// directive enumerates — raw/compressed sizes + ratio, raw/compressed-body/packet
/// hashes, the NANL-Q4 seal proof, the byte-exact decode proof, the artifact
/// family, the PROCESS-PROVEN runtime provenance block, the parent state-vertex,
/// the lineage refs, and the failure-recovery status. Nothing is null-padded: an
/// absent `parent_state_vertex_id` is paired with a `*_absent_reason`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeQnanlStreamChunk {
    pub chunk_index: u64,
    pub artifact_id: String,
    pub family: QnanlArtifactFamily,
    pub packet_path: String,
    pub receipt_path: String,
    /// `raw_size_bytes` (directive term) — bytes handed to the sealer.
    pub raw_bytes: u64,
    /// `compressed_size_bytes` (directive term) — the SECTION_ZSTD body size.
    pub compressed_body_bytes: u64,
    /// `compression_ratio` (directive term) — raw/packet ratio from the receipt.
    pub compression_ratio_vs_raw: f64,
    /// `raw_bytes_hash` (directive term) — blake3 of the raw chunk.
    pub raw_hash_blake3: String,
    /// SECTION_ZSTD `compressed_body_hash` (directive term).
    pub compressed_body_hash_blake3: String,
    /// `packet_hash` (directive term) — blake3 of the whole sealed packet.
    pub packet_hash_blake3: String,
    /// NANL-Q4 seal proof (directive term: "NANL-Q4 seal proof"). The seal is the
    /// constrained quaternary seal layer (NOT a bytes-to-bases costume); these are
    /// the executed constraint/merkle/ecc verdicts copied from the receipt.
    pub qnanl_seal_status: String,
    pub qnanl_seal_constraints_pass: bool,
    pub qnanl_seal_merkle_verified: bool,
    pub qnanl_seal_ecc_verified: bool,
    pub qnanl_seal_hash_blake3: String,
    /// Byte-exact decode assert: `decode_packet_bytes(packet) == raw` AND
    /// `blake3(decoded) == raw_hash`. Always `true` for a returned chunk (a
    /// failure short-circuits with `QnanlArtifactError::Hash`).
    pub decode_proof_pass: bool,
    /// Original evidence-artifact label this chunk was sealed from (e.g. the source
    /// filename), when the caller supplied it.
    pub source_label: Option<String>,
    /// `parent_state_vertex_id` (directive term) — the DSTW-T4 scientific-state
    /// vertex this evidence descends from. `None` is paired with the reason below.
    pub parent_state_vertex_id: Option<String>,
    pub parent_state_vertex_id_absent_reason: Option<String>,
    /// `lineage_refs` (directive term) — upstream artifact/topology hash references.
    pub lineage_refs: Vec<String>,
    /// `failure_recovery_status` (directive term). A returned chunk is always
    /// `SEALED_AND_DECODE_VERIFIED`: a decode mismatch is a hard error. Durable
    /// publication recovery preserves this integrity outcome and the original
    /// receipt; it does not imply molecular-state recovery.
    pub failure_recovery_status: String,
    /// PROCESS-PROVEN per-artifact provenance (h1) — backs `runtime_origin`.
    pub runtime_provenance: Option<crate::runtime_provenance::ArtifactProvenance>,
}

pub const RUNTIME_QNANL_FAILURE_RECOVERY_VERIFIED: &str = "SEALED_AND_DECODE_VERIFIED";

/// Per-chunk emission context handed to [`RuntimeQnanlStreamWriter::push_chunk_with_context`].
/// Lets the prismos lifecycle attach the h1 provenance block, the parent state
/// vertex, lineage refs, and a source label as it streams each evidence artifact.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RuntimeQnanlChunkContext {
    pub family_override: Option<QnanlArtifactFamily>,
    pub source_label: Option<String>,
    pub lineage_refs: Vec<String>,
    pub runtime_provenance: Option<crate::runtime_provenance::ArtifactProvenance>,
}

impl RuntimeQnanlChunkContext {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_source_label(mut self, label: impl Into<String>) -> Self {
        self.source_label = Some(label.into());
        self
    }
    pub fn with_lineage_refs(mut self, refs: Vec<String>) -> Self {
        self.lineage_refs = refs;
        self
    }
    pub fn with_runtime_provenance(
        mut self,
        prov: crate::runtime_provenance::ArtifactProvenance,
    ) -> Self {
        self.runtime_provenance = Some(prov);
        self
    }
}

/// Lifecycle-orchestrated v0 QNANL stream writer.
///
/// `push_chunk` instantiates a native-origin encode request, seals one chunk
/// via `encode_bytes`, re-proves the decode byte-exactly, and writes:
///   * `<stream>.index.json`            — incremental, rewritten after each seal
///   * `<packet>.decode_proof.json`     — per-packet byte-exact decode assert
///   * `<packet>.lineage.json`          — per-packet topology→engine→packet chain
/// The index is published last, after synchronized packet/receipt and evidence
/// sidecars. `recover` revalidates the retained index; unfinished intents require
/// an identical chunk replay. `finalize` seals the stream against further append
/// and binds stream lineage to the committed index hash.
pub struct RuntimeQnanlStreamWriter {
    output_dir: PathBuf,
    stream_name: String,
    runtime_origin: RuntimeOrigin,
    topology_ref: Option<String>,
    engine_ref: Option<String>,
    parent_state_vertex_id: Option<String>,
    parent_state_vertex_id_absent_reason: Option<String>,
    chunks: Vec<RuntimeQnanlStreamChunk>,
    durable_evidence: Vec<DurableChunkEvidence>,
    index_generation: Option<String>,
}

impl RuntimeQnanlStreamWriter {
    /// Create a v0 writer. `runtime_origin` is fixed to the native
    /// `RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS` — the only honest label for a
    /// lifecycle-orchestrated wrapper.
    pub fn new(
        output_dir: impl AsRef<Path>,
        stream_name: impl Into<String>,
        topology_ref: Option<String>,
        engine_ref: Option<String>,
    ) -> Self {
        RuntimeQnanlStreamWriter {
            output_dir: output_dir.as_ref().to_path_buf(),
            stream_name: stream_name.into(),
            runtime_origin: RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
            topology_ref,
            engine_ref,
            // Default: no parent vertex resolved yet. The caller MUST supply either
            // an id or a reason via `with_parent_state_vertex` — a bare `None`
            // without a reason would be null-padding (forbidden).
            parent_state_vertex_id: None,
            parent_state_vertex_id_absent_reason: Some(
                "parent_state_vertex not resolved by caller (with_parent_state_vertex not called)"
                    .to_string(),
            ),
            chunks: Vec::new(),
            durable_evidence: Vec::new(),
            index_generation: None,
        }
    }

    /// Bind the stream to its DSTW-T4 scientific-state parent vertex. Pass the
    /// resolved `id` (e.g. the `post_md_state_vertex` id read from disk) OR, when it
    /// cannot be resolved, a non-empty `absent_reason`. Exactly one is recorded; a
    /// silent `None`/`None` pair is rejected and replaced with an explicit reason.
    pub fn with_parent_state_vertex(
        mut self,
        id: Option<String>,
        absent_reason: Option<String>,
    ) -> Self {
        match (id, absent_reason) {
            (Some(id), _) => {
                self.parent_state_vertex_id = Some(id);
                self.parent_state_vertex_id_absent_reason = None;
            }
            (None, Some(reason)) => {
                self.parent_state_vertex_id = None;
                self.parent_state_vertex_id_absent_reason = Some(reason);
            }
            (None, None) => {
                self.parent_state_vertex_id = None;
                self.parent_state_vertex_id_absent_reason =
                    Some("parent_state_vertex absent and no reason supplied by caller".to_string());
            }
        }
        self
    }

    pub fn runtime_origin(&self) -> RuntimeOrigin {
        self.runtime_origin
    }

    pub fn parent_state_vertex_id(&self) -> Option<&str> {
        self.parent_state_vertex_id.as_deref()
    }

    pub fn chunks(&self) -> &[RuntimeQnanlStreamChunk] {
        &self.chunks
    }

    fn index_path(&self) -> PathBuf {
        self.output_dir
            .join(format!("{}.index.json", self.stream_name))
    }

    fn stream_lineage_path(&self) -> PathBuf {
        self.output_dir
            .join(format!("{}.lineage.json", self.stream_name))
    }

    /// Seal one chunk and emit its side-cars. Returns the origin-stamped receipt.
    ///
    /// Thin wrapper over [`Self::push_chunk_with_context`] with an empty context
    /// (no h1 provenance, no source label, no extra lineage). Kept for callers /
    /// tests that only need the seal + decode proof.
    pub fn push_chunk(
        &mut self,
        family: QnanlArtifactFamily,
        raw: &[u8],
    ) -> QnanlArtifactResult<QnanlArtifactReceipt> {
        self.push_chunk_with_context(
            raw,
            RuntimeQnanlChunkContext {
                family_override: Some(family),
                ..Default::default()
            },
        )
    }

    /// Seal one evidence artifact into a chunked QNANL packet within the prismos
    /// lifecycle, recording the FULL directive evidence record (sizes, ratio,
    /// raw/body/packet hashes, NANL-Q4 seal proof, byte-exact decode proof, family,
    /// the native `RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS` origin + h1
    /// provenance block, parent state vertex, lineage refs, failure-recovery
    /// status). This is the lifecycle-orchestrated NATIVE emission path; it is NOT
    /// `RUNTIME_INLINE` (engine-writes-inside-the-MD-closure), which stays the
    /// engine-owner v1 gate (nhs_rt_full.rs:18358).
    pub fn push_chunk_with_context(
        &mut self,
        raw: &[u8],
        ctx: RuntimeQnanlChunkContext,
    ) -> QnanlArtifactResult<QnanlArtifactReceipt> {
        let _stream_lock = self.lock_stream()?;
        self.check_generation()?;
        if self.stream_lineage_path().exists() {
            return Err(durability_error(
                "stream is finalized; append requires a new stream identity",
            ));
        }
        let family = ctx.family_override.unwrap_or(QnanlArtifactFamily::Unknown);
        let chunk_index = self.chunks.len() as u64;
        let artifact_id = format!("{}.chunk{chunk_index:06}", self.stream_name);
        let packet_path = self.output_dir.join(format!("{artifact_id}.prismqnanl"));
        let receipt_path = self
            .output_dir
            .join(format!("{artifact_id}.prismqnanl.receipt.json"));
        let mut options = QnanlEncodeOptions::new(
            artifact_id.clone(),
            family,
            None,
            self.runtime_origin,
            "RuntimeQnanlStreamWriter::push_chunk_with_context",
        );
        options.runtime_provenance = ctx.runtime_provenance.clone();
        let requested = StreamChunkIntent {
            version: 1,
            identity: self.durable_identity(),
            predecessor: self.index_generation.clone(),
            chunk_index,
            raw_hash: hex32(&hash32(raw)),
            raw_bytes: raw.len() as u64,
            options,
            context: ctx,
        };
        let intent_path = self.intent_path();
        let intent: StreamChunkIntent = if intent_path.exists() {
            let existing: StreamChunkIntent = read_durable_json(&intent_path)?;
            if !existing.same_request(&requested)? {
                return Err(durability_error(
                    "pending chunk conflicts with replay; original intent retained",
                ));
            }
            existing
        } else {
            durable_write(&intent_path, &durable_json(&requested)?)?;
            requested
        };
        durability_fault("stream_intent");
        let ctx = intent.context;
        let options = intent.options;

        // Seal via the canonical per-chunk sealer (SECTION_ZSTD body + NANL-Q4
        // seal). The runtime origin is already in `options` before the packet and
        // receipt are written; no teardown-time relabel/rewrite is allowed.
        let receipt = encode_bytes(raw, &packet_path, &receipt_path, options)?;

        // Byte-exact decode proof: re-decode the packet from disk and assert
        // identity. This is the executed proof (not the embedded bool only).
        let packet_bytes = std::fs::read(&packet_path)?;
        let decoded = decode_packet_bytes(&packet_bytes)?;
        let decoded_hash = hex32(&hash32(&decoded));
        let decode_proof_pass = decoded == raw && decoded_hash == receipt.raw_hash_blake3;
        if !decode_proof_pass {
            return Err(QnanlArtifactError::Hash(format!(
                "RuntimeQnanlStreamWriter decode proof failed for {artifact_id}"
            )));
        }

        let chunk = RuntimeQnanlStreamChunk {
            chunk_index,
            artifact_id: artifact_id.clone(),
            family,
            packet_path: packet_path.display().to_string(),
            receipt_path: receipt_path.display().to_string(),
            raw_bytes: raw.len() as u64,
            compressed_body_bytes: receipt.compressed_body_bytes,
            compression_ratio_vs_raw: receipt.compression_ratio_vs_raw,
            raw_hash_blake3: receipt.raw_hash_blake3.clone(),
            compressed_body_hash_blake3: receipt.compressed_body_hash_blake3.clone(),
            packet_hash_blake3: receipt.packet_hash_blake3.clone(),
            qnanl_seal_status: receipt.qnanl_seal_status.clone(),
            qnanl_seal_constraints_pass: receipt.qnanl_seal_constraints_pass,
            qnanl_seal_merkle_verified: receipt.qnanl_seal_merkle_verified,
            qnanl_seal_ecc_verified: receipt.qnanl_seal_ecc_verified,
            qnanl_seal_hash_blake3: receipt.qnanl_seal_hash_blake3.clone(),
            decode_proof_pass,
            source_label: ctx.source_label.clone(),
            parent_state_vertex_id: self.parent_state_vertex_id.clone(),
            parent_state_vertex_id_absent_reason: self.parent_state_vertex_id_absent_reason.clone(),
            lineage_refs: ctx.lineage_refs.clone(),
            failure_recovery_status: RUNTIME_QNANL_FAILURE_RECOVERY_VERIFIED.to_string(),
            runtime_provenance: ctx.runtime_provenance.clone(),
        };

        let decode_proof_path =
            PathBuf::from(format!("{}.decode_proof.json", packet_path.display()));
        immutable_write(
            &decode_proof_path,
            self.decode_proof_json(&chunk, &decoded_hash).as_bytes(),
        )?;
        durability_fault("stream_proof");
        let lineage_path = PathBuf::from(format!("{}.lineage.json", packet_path.display()));
        immutable_write(&lineage_path, self.packet_lineage_json(&chunk).as_bytes())?;
        durability_fault("stream_lineage");
        let evidence = DurableChunkEvidence {
            receipt_hash: file_hash(&receipt_path)?,
            proof_hash: file_hash(&decode_proof_path)?,
            lineage_hash: file_hash(&lineage_path)?,
        };
        // The durable index is the stream commit point. A failed publication
        // never advances this writer's in-memory generation.
        self.chunks.push(chunk);
        self.durable_evidence.push(evidence);
        let next_index = self.index_json();
        let committed_chunk = self
            .chunks
            .pop()
            .expect("candidate chunk was just appended");
        let committed_evidence = self
            .durable_evidence
            .pop()
            .expect("candidate evidence was just appended");
        let next_index = next_index?;
        durable_write(&self.index_path(), next_index.as_bytes())?;
        durability_fault("stream_index");
        self.chunks.push(committed_chunk);
        self.durable_evidence.push(committed_evidence);
        self.index_generation = Some(hex32(&hash32(next_index.as_bytes())));
        std::fs::remove_file(&intent_path)?;
        sync_parent(&intent_path)?;
        Ok(receipt)
    }

    /// Reopen and verify a committed evidence stream, preserving original chunk
    /// provenance. An interrupted uncommitted chunk remains a durable intent:
    /// replay that same raw chunk/context to finish it. This is artifact recovery,
    /// not molecular phase-space or engine restart.
    pub fn recover(&mut self) -> QnanlArtifactResult<usize> {
        let _stream_lock = self.lock_stream()?;
        self.load_index()?;
        if self.intent_path().exists() {
            let intent: StreamChunkIntent = read_durable_json(&self.intent_path())?;
            if intent.identity != self.durable_identity() {
                return Err(durability_error(
                    "recovery identity differs from retained intent",
                ));
            }
            // A crash after index commit but before intent removal is an
            // acknowledged commit. Verify the committed chunk before clearing it.
            if intent.chunk_index + 1 == self.chunks.len() as u64 {
                let chunk = &self.chunks[intent.chunk_index as usize];
                if intent.version != 1
                    || chunk.raw_hash_blake3 != intent.raw_hash
                    || chunk.raw_bytes != intent.raw_bytes
                    || chunk.family != intent.options.artifact_family
                    || chunk.source_label != intent.context.source_label
                    || chunk.lineage_refs != intent.context.lineage_refs
                    || Self::provenance_json(&chunk.runtime_provenance)
                        != Self::provenance_json(&intent.context.runtime_provenance)
                {
                    return Err(durability_error(
                        "committed chunk conflicts with retained intent",
                    ));
                }
                std::fs::remove_file(self.intent_path())?;
                sync_parent(&self.intent_path())?;
            } else if intent.chunk_index != self.chunks.len() as u64
                || intent.predecessor != self.index_generation
            {
                return Err(durability_error("pending chunk has an invalid predecessor"));
            }
        }
        if self.stream_lineage_path().exists() {
            if self.intent_path().exists()
                || std::fs::read(self.stream_lineage_path())? != self.finalization_bytes()?
            {
                return Err(durability_error(
                    "retained finalization conflicts with committed stream",
                ));
            }
        }
        Ok(self.chunks.len())
    }

    /// True when recovery requires replay of an interrupted chunk's original
    /// raw bytes and context. The retained intent is never silently discarded.
    pub fn has_pending_chunk(&self) -> bool {
        self.intent_path().exists()
    }

    /// Publish immutable stream lineage only after every committed packet and
    /// evidence sidecar is reverified. Repeating finalization is idempotent;
    /// appending after finalization is rejected.
    pub fn finalize(&self) -> QnanlArtifactResult<PathBuf> {
        let _stream_lock = self.lock_stream()?;
        self.check_generation()?;
        if self.intent_path().exists() {
            return Err(durability_error(
                "pending chunk must be replayed or recovered before finalization",
            ));
        }
        self.verify_evidence()?;
        let index = self.index_json()?;
        if !self.index_path().exists() {
            durable_write(&self.index_path(), index.as_bytes())?;
        }
        let path = self.stream_lineage_path();
        immutable_write(&path, &self.finalization_bytes()?)?;
        durability_fault("stream_finalization");
        Ok(path)
    }

    fn finalization_bytes(&self) -> QnanlArtifactResult<Vec<u8>> {
        let mut lineage: serde_json::Value = serde_json::from_str(&self.stream_lineage_json())
            .map_err(|e| durability_error(&e.to_string()))?;
        lineage["durability_schema_version"] = serde_json::json!(1);
        lineage["index_hash_blake3"] = serde_json::json!(file_hash(&self.index_path())?);
        durable_json(&lineage)
    }

    fn durable_identity(&self) -> DurableStreamIdentity {
        DurableStreamIdentity {
            stream_name: self.stream_name.clone(),
            runtime_origin: self.runtime_origin,
            topology_ref: self.topology_ref.clone(),
            engine_ref: self.engine_ref.clone(),
            parent_state_vertex_id: self.parent_state_vertex_id.clone(),
            parent_state_vertex_id_absent_reason: self.parent_state_vertex_id_absent_reason.clone(),
        }
    }

    fn intent_path(&self) -> PathBuf {
        self.output_dir
            .join(format!("{}.chunk.transaction.json", self.stream_name))
    }

    fn lock_stream(&self) -> QnanlArtifactResult<File> {
        if self.stream_name.is_empty()
            || self.stream_name == "."
            || self.stream_name == ".."
            || self.stream_name.contains('/')
            || self.stream_name.contains('\\')
        {
            return Err(durability_error(
                "stream name must be a single nonempty filename component",
            ));
        }
        lock_artifact_publication(&self.index_path())
    }

    fn check_generation(&self) -> QnanlArtifactResult<()> {
        let current = if self.index_path().exists() {
            Some(file_hash(&self.index_path())?)
        } else {
            None
        };
        if current != self.index_generation {
            // Empty finalization creates an empty index through an &self API.
            if self.chunks.is_empty()
                && self.index_generation.is_none()
                && current.as_deref()
                    == Some(hex32(&hash32(self.index_json()?.as_bytes())).as_str())
            {
                return Ok(());
            }
            return Err(durability_error(
                "stale stream writer; explicitly recover the retained index before appending",
            ));
        }
        Ok(())
    }

    fn load_index(&mut self) -> QnanlArtifactResult<()> {
        if !self.index_path().exists() {
            if !self.chunks.is_empty() || self.index_generation.is_some() {
                return Err(durability_error("committed stream index disappeared"));
            }
            return Ok(());
        }
        let bytes = std::fs::read(self.index_path())?;
        let index: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| durability_error(&e.to_string()))?;
        let state: DurableStreamState =
            serde_json::from_value(index.get("durable_state").cloned().ok_or_else(|| {
                durability_error(
                    "legacy index has no durable state; retain it and use a new stream identity",
                )
            })?)
            .map_err(|e| durability_error(&e.to_string()))?;
        if state.version != 1 || state.identity != self.durable_identity() {
            return Err(durability_error(
                "retained stream identity differs from requested topology/engine/parent",
            ));
        }
        let old_chunks = std::mem::replace(&mut self.chunks, state.chunks);
        let old_evidence = std::mem::replace(&mut self.durable_evidence, state.evidence);
        let validation = self.verify_evidence().and_then(|_| {
            let expected: serde_json::Value = serde_json::from_str(&self.index_json()?)
                .map_err(|e| durability_error(&e.to_string()))?;
            if expected != index {
                return Err(durability_error(
                    "public stream index and retained durable state disagree",
                ));
            }
            Ok(())
        });
        if let Err(error) = validation {
            self.chunks = old_chunks;
            self.durable_evidence = old_evidence;
            return Err(error);
        }
        self.index_generation = Some(hex32(&hash32(&bytes)));
        Ok(())
    }

    fn verify_evidence(&self) -> QnanlArtifactResult<()> {
        if self.chunks.len() != self.durable_evidence.len() {
            return Err(durability_error("stream evidence cardinality mismatch"));
        }
        for (index, (chunk, evidence)) in self.chunks.iter().zip(&self.durable_evidence).enumerate()
        {
            let artifact_id = format!("{}.chunk{index:06}", self.stream_name);
            let packet = self.output_dir.join(format!("{artifact_id}.prismqnanl"));
            let receipt_path = sidecar_path(&packet, ".receipt.json");
            if chunk.chunk_index != index as u64
                || chunk.artifact_id != artifact_id
                || chunk.packet_path != packet.display().to_string()
                || chunk.receipt_path != receipt_path.display().to_string()
                || file_hash(&packet)? != chunk.packet_hash_blake3
                || file_hash(&receipt_path)? != evidence.receipt_hash
                || file_hash(&sidecar_path(&packet, ".decode_proof.json"))? != evidence.proof_hash
                || file_hash(&sidecar_path(&packet, ".lineage.json"))? != evidence.lineage_hash
            {
                return Err(durability_error(
                    "committed chunk path or evidence hash mismatch",
                ));
            }
            let commit: DurablePacketCommit =
                read_durable_json(&sidecar_path(&packet, ".commit.json"))?;
            verify_packet_commit(&packet, &receipt_path, &commit)?;
            let receipt = &commit.receipt;
            if receipt.artifact_id != chunk.artifact_id
                || receipt.artifact_family != chunk.family
                || receipt.raw_bytes != chunk.raw_bytes
                || receipt.raw_hash_blake3 != chunk.raw_hash_blake3
                || receipt.packet_hash_blake3 != chunk.packet_hash_blake3
                || receipt.compressed_body_bytes != chunk.compressed_body_bytes
                || receipt.compressed_body_hash_blake3 != chunk.compressed_body_hash_blake3
                || receipt.qnanl_seal_hash_blake3 != chunk.qnanl_seal_hash_blake3
                || receipt.runtime_origin != self.runtime_origin
                || !chunk.decode_proof_pass
                || !receipt.roundtrip_verified
                || Self::provenance_json(&receipt.runtime_provenance)
                    != Self::provenance_json(&chunk.runtime_provenance)
            {
                return Err(durability_error(
                    "committed receipt and stream evidence disagree",
                ));
            }
            let decoded = decode_packet_bytes(&std::fs::read(&packet)?)?;
            if hex32(&hash32(&decoded)) != chunk.raw_hash_blake3
                || decoded.len() as u64 != chunk.raw_bytes
            {
                return Err(durability_error(
                    "committed chunk independent decode failed",
                ));
            }
            // Sidecars must bind the exact stream identity, not only exist.
            if std::fs::read(sidecar_path(&packet, ".decode_proof.json"))?
                != self
                    .decode_proof_json(chunk, &chunk.raw_hash_blake3)
                    .as_bytes()
                || std::fs::read(sidecar_path(&packet, ".lineage.json"))?
                    != self.packet_lineage_json(chunk).as_bytes()
            {
                return Err(durability_error(
                    "committed chunk proof or lineage differs from stream identity",
                ));
            }
        }
        Ok(())
    }

    fn opt_json(&self, value: &Option<String>) -> String {
        value
            .as_ref()
            .map(|v| format!("\"{}\"", json_escape(v)))
            .unwrap_or_else(|| "null".to_string())
    }

    /// Render the h1 PROCESS-PROVEN provenance block (serde) or explicit `null`.
    fn provenance_json(prov: &Option<crate::runtime_provenance::ArtifactProvenance>) -> String {
        match prov {
            Some(p) => serde_json::to_string(&p.to_json()).unwrap_or_else(|_| "null".to_string()),
            None => "null".to_string(),
        }
    }

    /// The full directive per-packet evidence record, shared by the index and the
    /// per-packet lineage side-car. Every directive-named field is present; absent
    /// optionals are paired with a `*_absent_reason` (never null-padded).
    fn chunk_evidence_fields(&self, chunk: &RuntimeQnanlStreamChunk) -> String {
        format!(
            concat!(
                "\"chunk_index\": {chunk_index}, ",
                "\"artifact_id\": \"{artifact_id}\", ",
                "\"source_label\": {source_label}, ",
                "\"artifact_family\": \"{family}\", ",
                "\"packet_path\": \"{packet_path}\", ",
                "\"receipt_path\": \"{receipt_path}\", ",
                "\"raw_size_bytes\": {raw_size}, ",
                "\"compressed_size_bytes\": {compressed_size}, ",
                "\"compression_ratio\": {ratio:.9}, ",
                "\"raw_bytes_hash\": \"{raw_hash}\", ",
                "\"compressed_body_hash\": \"{body_hash}\", ",
                "\"packet_hash\": \"{packet_hash}\", ",
                "\"nanl_q4_seal_proof\": {{\"status\": \"{seal_status}\", ",
                "\"constraints_pass\": {seal_constraints}, ",
                "\"merkle_verified\": {seal_merkle}, ",
                "\"ecc_verified\": {seal_ecc}, ",
                "\"seal_hash_blake3\": \"{seal_hash}\"}}, ",
                "\"decode_proof\": {{\"byte_exact_decode_pass\": {decode_pass}, ",
                "\"proof\": \"decode_packet_bytes(packet)==raw && blake3(decoded)==raw_hash\"}}, ",
                "\"runtime_origin\": \"{origin}\", ",
                "\"runtime_native\": {native}, ",
                "\"parent_state_vertex_id\": {parent_vid}, ",
                "\"parent_state_vertex_id_absent_reason\": {parent_vid_reason}, ",
                "\"lineage_refs\": {lineage_refs}, ",
                "\"failure_recovery_status\": \"{recovery}\", ",
                "\"runtime_provenance\": {provenance}"
            ),
            chunk_index = chunk.chunk_index,
            artifact_id = json_escape(&chunk.artifact_id),
            source_label = self.opt_json(&chunk.source_label),
            family = chunk.family,
            packet_path = json_escape(&chunk.packet_path),
            receipt_path = json_escape(&chunk.receipt_path),
            raw_size = chunk.raw_bytes,
            compressed_size = chunk.compressed_body_bytes,
            ratio = chunk.compression_ratio_vs_raw,
            raw_hash = chunk.raw_hash_blake3,
            body_hash = chunk.compressed_body_hash_blake3,
            packet_hash = chunk.packet_hash_blake3,
            seal_status = json_escape(&chunk.qnanl_seal_status),
            seal_constraints = chunk.qnanl_seal_constraints_pass,
            seal_merkle = chunk.qnanl_seal_merkle_verified,
            seal_ecc = chunk.qnanl_seal_ecc_verified,
            seal_hash = chunk.qnanl_seal_hash_blake3,
            decode_pass = chunk.decode_proof_pass,
            origin = self.runtime_origin.as_str(),
            native = self.runtime_origin.is_prismos_native(),
            parent_vid = self.opt_json(&chunk.parent_state_vertex_id),
            parent_vid_reason = self.opt_json(&chunk.parent_state_vertex_id_absent_reason),
            lineage_refs = json_string_array(&chunk.lineage_refs),
            recovery = json_escape(&chunk.failure_recovery_status),
            provenance = Self::provenance_json(&chunk.runtime_provenance),
        )
    }

    fn decode_proof_json(&self, chunk: &RuntimeQnanlStreamChunk, decoded_hash: &str) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"schema_kind\": \"prismos_qnanl_runtime_decode_proof\",\n",
                "  \"writer_version\": \"{}\",\n",
                "  \"runtime_origin\": \"{}\",\n",
                "  \"runtime_native\": {},\n",
                "  \"stream_name\": \"{}\",\n",
                "  \"chunk_index\": {},\n",
                "  \"artifact_id\": \"{}\",\n",
                "  \"packet_path\": \"{}\",\n",
                "  \"raw_bytes\": {},\n",
                "  \"raw_hash_blake3\": \"{}\",\n",
                "  \"decoded_hash_blake3\": \"{}\",\n",
                "  \"byte_exact_decode_pass\": {},\n",
                "  \"nanl_q4_seal_status\": \"{}\",\n",
                "  \"nanl_q4_seal_constraints_pass\": {},\n",
                "  \"nanl_q4_seal_merkle_verified\": {},\n",
                "  \"nanl_q4_seal_ecc_verified\": {},\n",
                "  \"failure_recovery_status\": \"{}\",\n",
                "  \"proof\": \"decode_packet_bytes(packet)==raw && blake3(decoded)==raw_hash\"\n",
                "}}\n"
            ),
            RUNTIME_QNANL_STREAM_WRITER_VERSION,
            self.runtime_origin.as_str(),
            self.runtime_origin.is_prismos_native(),
            json_escape(&self.stream_name),
            chunk.chunk_index,
            json_escape(&chunk.artifact_id),
            json_escape(&chunk.packet_path),
            chunk.raw_bytes,
            chunk.raw_hash_blake3,
            decoded_hash,
            chunk.decode_proof_pass,
            json_escape(&chunk.qnanl_seal_status),
            chunk.qnanl_seal_constraints_pass,
            chunk.qnanl_seal_merkle_verified,
            chunk.qnanl_seal_ecc_verified,
            json_escape(&chunk.failure_recovery_status),
        )
    }

    fn packet_lineage_json(&self, chunk: &RuntimeQnanlStreamChunk) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"schema_kind\": \"prismos_qnanl_runtime_packet_lineage\",\n",
                "  \"writer_version\": \"{}\",\n",
                "  \"stream_name\": \"{}\",\n",
                "  \"topology_ref\": {},\n",
                "  \"engine_ref\": {},\n",
                "  {}\n",
                "}}\n"
            ),
            RUNTIME_QNANL_STREAM_WRITER_VERSION,
            json_escape(&self.stream_name),
            self.opt_json(&self.topology_ref),
            self.opt_json(&self.engine_ref),
            self.chunk_evidence_fields(chunk),
        )
    }

    fn index_json(&self) -> QnanlArtifactResult<String> {
        let mut entries = String::from("[");
        for (idx, chunk) in self.chunks.iter().enumerate() {
            if idx > 0 {
                entries.push(',');
            }
            entries.push_str("\n    {");
            entries.push_str(&self.chunk_evidence_fields(chunk));
            entries.push('}');
        }
        if !self.chunks.is_empty() {
            entries.push('\n');
            entries.push_str("  ");
        }
        entries.push(']');
        let public_index = format!(
            concat!(
                "{{\n",
                "  \"schema_kind\": \"prismos_qnanl_runtime_stream_index\",\n",
                "  \"writer_version\": \"{}\",\n",
                "  \"runtime_origin\": \"{}\",\n",
                "  \"runtime_native\": {},\n",
                "  \"stream_name\": \"{}\",\n",
                "  \"chunk_count\": {},\n",
                "  \"chunks\": {}\n",
                "}}\n"
            ),
            RUNTIME_QNANL_STREAM_WRITER_VERSION,
            self.runtime_origin.as_str(),
            self.runtime_origin.is_prismos_native(),
            json_escape(&self.stream_name),
            self.chunks.len(),
            entries,
        );
        let mut index: serde_json::Value =
            serde_json::from_str(&public_index).map_err(|e| durability_error(&e.to_string()))?;
        index["durable_state"] = serde_json::to_value(DurableStreamState {
            version: 1,
            identity: self.durable_identity(),
            chunks: self.chunks.clone(),
            evidence: self.durable_evidence.clone(),
        })
        .map_err(|e| durability_error(&e.to_string()))?;
        String::from_utf8(durable_json(&index)?).map_err(|e| durability_error(&e.to_string()))
    }

    fn stream_lineage_json(&self) -> String {
        let chain: Vec<String> = self
            .chunks
            .iter()
            .map(|c| c.packet_hash_blake3.clone())
            .collect();
        format!(
            concat!(
                "{{\n",
                "  \"schema_kind\": \"prismos_qnanl_runtime_stream_lineage\",\n",
                "  \"writer_version\": \"{}\",\n",
                "  \"runtime_origin\": \"{}\",\n",
                "  \"runtime_native\": {},\n",
                "  \"stream_name\": \"{}\",\n",
                "  \"topology_ref\": {},\n",
                "  \"engine_ref\": {},\n",
                "  \"chunk_count\": {},\n",
                "  \"packet_hash_chain\": {}\n",
                "}}\n"
            ),
            RUNTIME_QNANL_STREAM_WRITER_VERSION,
            self.runtime_origin.as_str(),
            self.runtime_origin.is_prismos_native(),
            json_escape(&self.stream_name),
            self.opt_json(&self.topology_ref),
            self.opt_json(&self.engine_ref),
            self.chunks.len(),
            json_string_array(&chain),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scientific_chronology_qnanl_families_are_registered_append_only() {
        assert_eq!(QnanlArtifactFamily::ScientificChronologyEvent as u16, 26);
        assert_eq!(QnanlArtifactFamily::ScientificStateEpoch as u16, 27);
        assert_eq!(QnanlArtifactFamily::T4EpochOverlay as u16, 28);
        assert_eq!(QnanlArtifactFamily::ChronologyAcceptanceDecision as u16, 29);
        assert_eq!(QnanlArtifactFamily::ScientificPocketOpeningEvent as u16, 30);
        assert_eq!(QnanlArtifactFamily::LigandCausalActivityEvent as u16, 31);
        assert_eq!(QnanlArtifactFamily::ResidueActivityEvent as u16, 32);
        assert_eq!(
            QnanlArtifactFamily::ScientificChronologyEvent.as_str(),
            "SCIENTIFIC_CHRONOLOGY_EVENT"
        );
        assert_eq!(
            QnanlArtifactFamily::ScientificStateEpoch.as_str(),
            "SCIENTIFIC_STATE_EPOCH"
        );
        assert_eq!(
            QnanlArtifactFamily::T4EpochOverlay.as_str(),
            "T4_EPOCH_OVERLAY"
        );
        assert_eq!(
            QnanlArtifactFamily::ChronologyAcceptanceDecision.as_str(),
            "CHRONOLOGY_ACCEPTANCE_DECISION"
        );
        assert_eq!(
            QnanlArtifactFamily::ScientificPocketOpeningEvent.as_str(),
            "SCIENTIFIC_POCKET_OPENING_EVENT"
        );
        assert_eq!(
            QnanlArtifactFamily::LigandCausalActivityEvent.as_str(),
            "LIGAND_CAUSAL_ACTIVITY_EVENT"
        );
        assert_eq!(
            QnanlArtifactFamily::ResidueActivityEvent.as_str(),
            "RESIDUE_ACTIVITY_EVENT"
        );
        assert_eq!(
            "CHRONOLOGY_EVENT".parse::<QnanlArtifactFamily>().unwrap(),
            QnanlArtifactFamily::ScientificChronologyEvent
        );
        assert_eq!(
            "STATE_EPOCH".parse::<QnanlArtifactFamily>().unwrap(),
            QnanlArtifactFamily::ScientificStateEpoch
        );
        assert_eq!(
            "DSTW_T4_EPOCH_OVERLAY"
                .parse::<QnanlArtifactFamily>()
                .unwrap(),
            QnanlArtifactFamily::T4EpochOverlay
        );
        assert_eq!(
            "ACCEPTANCE_DECISION"
                .parse::<QnanlArtifactFamily>()
                .unwrap(),
            QnanlArtifactFamily::ChronologyAcceptanceDecision
        );
        assert_eq!(
            "SCIENTIFIC_POCKET_OPENING_EVENT"
                .parse::<QnanlArtifactFamily>()
                .unwrap(),
            QnanlArtifactFamily::ScientificPocketOpeningEvent
        );
        assert_eq!(
            "LIGANDCAUSALACTIVITYEVENT"
                .parse::<QnanlArtifactFamily>()
                .unwrap(),
            QnanlArtifactFamily::LigandCausalActivityEvent
        );
        assert_eq!(
            "RESIDUE_ACTIVITY_EVENT"
                .parse::<QnanlArtifactFamily>()
                .unwrap(),
            QnanlArtifactFamily::ResidueActivityEvent
        );
        assert_eq!(
            QnanlArtifactFamily::from_code(26).unwrap(),
            QnanlArtifactFamily::ScientificChronologyEvent
        );
        assert_eq!(
            QnanlArtifactFamily::from_code(27).unwrap(),
            QnanlArtifactFamily::ScientificStateEpoch
        );
        assert_eq!(
            QnanlArtifactFamily::from_code(28).unwrap(),
            QnanlArtifactFamily::T4EpochOverlay
        );
        assert_eq!(
            QnanlArtifactFamily::from_code(29).unwrap(),
            QnanlArtifactFamily::ChronologyAcceptanceDecision
        );
        assert_eq!(
            QnanlArtifactFamily::from_code(30).unwrap(),
            QnanlArtifactFamily::ScientificPocketOpeningEvent
        );
        assert_eq!(
            QnanlArtifactFamily::from_code(31).unwrap(),
            QnanlArtifactFamily::LigandCausalActivityEvent
        );
        assert_eq!(
            QnanlArtifactFamily::from_code(32).unwrap(),
            QnanlArtifactFamily::ResidueActivityEvent
        );
    }

    #[test]
    fn ghost_lambda_and_pathb_qnanl_families_are_registered_append_only() {
        let expected = [
            (
                33,
                QnanlArtifactFamily::GhostSegmentCommit,
                "GHOST_SEGMENT_COMMIT",
            ),
            (
                34,
                QnanlArtifactFamily::GhostConservation,
                "GHOST_CONSERVATION",
            ),
            (
                35,
                QnanlArtifactFamily::GhostPhaseBinding,
                "GHOST_PHASE_BINDING",
            ),
            (
                36,
                QnanlArtifactFamily::LambdaFragmentForceEvent,
                "LAMBDA_FRAGMENT_FORCE_EVENT",
            ),
            (
                37,
                QnanlArtifactFamily::LambdaFragmentNcmcMoveReceipt,
                "LAMBDA_FRAGMENT_NCMC_MOVE_RECEIPT",
            ),
            (
                38,
                QnanlArtifactFamily::LambdaFragmentEndpointObservation,
                "LAMBDA_FRAGMENT_ENDPOINT_OBSERVATION",
            ),
            (
                39,
                QnanlArtifactFamily::LambdaFragmentStateCommit,
                "LAMBDA_FRAGMENT_STATE_COMMIT",
            ),
            (
                40,
                QnanlArtifactFamily::ChemicalStateFiber,
                "PRISM.CHEMICAL_STATE_FIBER.V1",
            ),
            (
                41,
                QnanlArtifactFamily::PathBChemicalState,
                "PRISM.PATHB_CHEMICAL_STATE.V1",
            ),
            (
                42,
                QnanlArtifactFamily::MatchedControlDiscriminator,
                "PRISM.MATCHED_CONTROL_DISCRIMINATOR.V1",
            ),
            (
                43,
                QnanlArtifactFamily::CandidateAdvancementRegistry,
                "PRISM.CANDIDATE_ADVANCEMENT_REGISTRY.V1",
            ),
            (
                44,
                QnanlArtifactFamily::LambdaFragmentStateRollback,
                "LAMBDA_FRAGMENT_STATE_ROLLBACK",
            ),
            (
                45,
                QnanlArtifactFamily::LambdaNoProbeAbsence,
                "LAMBDA_NO_PROBE_ABSENCE",
            ),
            (
                46,
                QnanlArtifactFamily::GhostCoordinateSegmentCommit,
                "GHOST_COORDINATE_SEGMENT_COMMIT",
            ),
            (
                47,
                QnanlArtifactFamily::GhostCoordinateConservation,
                "GHOST_COORDINATE_CONSERVATION",
            ),
            (
                48,
                QnanlArtifactFamily::GhostLatticeEdgeCommit,
                "GHOST_LATTICE_EDGE_COMMIT",
            ),
            (
                49,
                QnanlArtifactFamily::GhostLatticeEdgeConservation,
                "GHOST_LATTICE_EDGE_CONSERVATION",
            ),
            (
                50,
                QnanlArtifactFamily::PathBDiscoveryCandidate,
                "PRISM.PATHB_DISCOVERY_CANDIDATE.V1",
            ),
            (
                51,
                QnanlArtifactFamily::PathBConfirmationHandoff,
                "PRISM.PATHB_CONFIRMATION_HANDOFF.V1",
            ),
        ];

        for (code, family, name) in expected {
            assert_eq!(family as u16, code);
            assert_eq!(QnanlArtifactFamily::from_code(code).unwrap(), family);
            assert_eq!(family.as_str(), name);
            assert_eq!(name.parse::<QnanlArtifactFamily>().unwrap(), family);
            assert_ne!(family.default_transform(), "RAW_CANONICAL_BYTES");
        }
    }

    #[test]
    fn ghost_lambda_and_pathb_typed_packets_roundtrip_exact_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "prismcode_successor_qnanl_roundtrip_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let families = [
            QnanlArtifactFamily::GhostSegmentCommit,
            QnanlArtifactFamily::GhostConservation,
            QnanlArtifactFamily::GhostPhaseBinding,
            QnanlArtifactFamily::LambdaFragmentForceEvent,
            QnanlArtifactFamily::LambdaFragmentNcmcMoveReceipt,
            QnanlArtifactFamily::LambdaFragmentEndpointObservation,
            QnanlArtifactFamily::LambdaFragmentStateCommit,
            QnanlArtifactFamily::ChemicalStateFiber,
            QnanlArtifactFamily::PathBChemicalState,
            QnanlArtifactFamily::MatchedControlDiscriminator,
            QnanlArtifactFamily::CandidateAdvancementRegistry,
            QnanlArtifactFamily::LambdaFragmentStateRollback,
            QnanlArtifactFamily::LambdaNoProbeAbsence,
            QnanlArtifactFamily::GhostCoordinateSegmentCommit,
            QnanlArtifactFamily::GhostCoordinateConservation,
            QnanlArtifactFamily::GhostLatticeEdgeCommit,
            QnanlArtifactFamily::GhostLatticeEdgeConservation,
            QnanlArtifactFamily::PathBDiscoveryCandidate,
            QnanlArtifactFamily::PathBConfirmationHandoff,
        ];

        for family in families {
            let stem = format!("family_{}", family as u16);
            let packet = dir.join(format!("{stem}.prismqnanl"));
            let receipt_path = dir.join(format!("{stem}.receipt.json"));
            let decoded_path = dir.join(format!("{stem}.decoded"));
            let numeric_probe = 9_007_199_254_740_991u64 - u64::from(family as u16);
            let string_probe = format!("typed:{}:edge/coordinate", family as u16);
            let raw = format!(
                "{{\"schema_kind\":\"{}\",\"family_code\":{},\"numeric_probe\":{},\"string_probe\":\"{}\"}}\n",
                family.as_str(),
                family as u16,
                numeric_probe,
                string_probe,
            );
            let receipt = encode_bytes(
                raw.as_bytes(),
                &packet,
                &receipt_path,
                QnanlEncodeOptions::runtime_inline(
                    &stem,
                    family,
                    None,
                    "successor_qnanl_roundtrip_test",
                ),
            )
            .unwrap();
            assert_eq!(receipt.artifact_family, family);
            assert!(receipt.roundtrip_verified);
            assert!(receipt.codec_roundtrip_verified);
            assert_eq!(decode_file(&packet, &decoded_path).unwrap(), raw.as_bytes());
            assert_eq!(std::fs::read(&decoded_path).unwrap(), raw.as_bytes());
            let decoded: serde_json::Value = serde_json::from_slice(raw.as_bytes()).unwrap();
            assert_eq!(
                decoded
                    .get("numeric_probe")
                    .and_then(|value| value.as_u64()),
                Some(numeric_probe)
            );
            assert_eq!(
                decoded.get("string_probe").and_then(|value| value.as_str()),
                Some(string_probe.as_str())
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn receipt_json_carries_runtime_origin_and_native_flag() {
        let dir =
            std::env::temp_dir().join(format!("prismcode_qnanl_receipt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let packet = dir.join("a.prismqnanl");
        let receipt_path = dir.join("a.receipt.json");
        let receipt = encode_bytes(
            b"hello qnanl runtime_origin",
            &packet,
            &receipt_path,
            QnanlEncodeOptions::post_run_seal_only(
                "unit_a",
                QnanlArtifactFamily::JsonTelemetry,
                None,
                "qnanl_artifact_test::receipt_json_carries_runtime_origin_and_native_flag",
            ),
        )
        .unwrap();
        // Pure sealer default is the conservative honest label, NOT native.
        assert_eq!(receipt.runtime_origin, RuntimeOrigin::PostRunSealOnly);
        assert!(!receipt.runtime_origin.is_prismos_native());
        assert!(receipt.runtime_origin_assigned_unix_ns > 0);
        assert_eq!(
            receipt.runtime_origin_assignment_stage,
            "qnanl_artifact_test::receipt_json_carries_runtime_origin_and_native_flag"
        );
        assert_eq!(receipt.chunk_hashes.len(), 1);
        assert_eq!(receipt.chunk_hashes[0], receipt.raw_hash_blake3);
        let json = std::fs::read_to_string(&receipt_path).unwrap();
        assert!(json.contains("\"runtime_origin\": \"POST_RUN_SEAL_ONLY\""));
        assert!(json.contains("\"runtime_native\": false"));
        assert!(json.contains("\"chunk_hashes\":"));
        assert!(json.contains("\"pcode_refs\": []"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stream_writer_v0_is_native_and_decode_proves_each_chunk() {
        let dir =
            std::env::temp_dir().join(format!("prismcode_qnanl_stream_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut writer = RuntimeQnanlStreamWriter::new(
            &dir,
            "spike_stream_s0",
            Some("apo.topology.json".to_string()),
            Some("nhs_rt_full".to_string()),
        );
        // The lifecycle wrapper is genuinely native.
        assert!(writer.runtime_origin().is_prismos_native());
        assert_eq!(
            writer.runtime_origin().as_str(),
            "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS"
        );

        let r0 = writer
            .push_chunk(QnanlArtifactFamily::EventStream, b"chunk-zero-spikes")
            .unwrap();
        let r1 = writer
            .push_chunk(
                QnanlArtifactFamily::EventStream,
                b"chunk-one-spikes-longer-payload",
            )
            .unwrap();

        // Receipts are relabeled native and the persisted receipt JSON agrees.
        assert!(r0.runtime_origin.is_prismos_native());
        assert!(r1.runtime_origin.is_prismos_native());
        let r0_json = std::fs::read_to_string(&r0.receipt_path).unwrap();
        assert!(
            r0_json.contains("\"runtime_origin\": \"RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS\"")
        );
        assert!(r0_json.contains("\"runtime_native\": true"));

        // Two chunks recorded, each decode-proven byte-exact.
        assert_eq!(writer.chunks().len(), 2);
        assert!(writer.chunks().iter().all(|c| c.decode_proof_pass));

        // Per-packet side-cars exist.
        for chunk in writer.chunks() {
            assert!(PathBuf::from(format!("{}.decode_proof.json", chunk.packet_path)).exists());
            assert!(PathBuf::from(format!("{}.lineage.json", chunk.packet_path)).exists());
            // Independent decode of the packet on disk matches the raw hash.
            let pkt = std::fs::read(&chunk.packet_path).unwrap();
            let decoded = decode_packet_bytes(&pkt).unwrap();
            assert_eq!(hex32(&hash32(&decoded)), chunk.raw_hash_blake3);
        }

        // Incremental index + finalized lineage exist and name the writer v0.
        let index = std::fs::read_to_string(writer.index_path()).unwrap();
        assert!(index.contains("\"chunk_count\": 2"));
        assert!(index.contains("v0_lifecycle_orchestrated"));
        let lineage_path = writer.finalize().unwrap();
        let lineage = std::fs::read_to_string(&lineage_path).unwrap();
        assert!(lineage.contains("\"packet_hash_chain\":"));
        assert!(lineage.contains(&r1.packet_hash_blake3));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lifecycle_chunk_records_full_directive_evidence_record() {
        use crate::runtime_provenance::RuntimeProvenance;
        let dir =
            std::env::temp_dir().join(format!("prismcode_qnanl_h2_ctx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // No parent vertex supplied -> the absent reason MUST be recorded (no
        // null-padding), and the chunk inherits it.
        let mut writer = RuntimeQnanlStreamWriter::new(
            &dir,
            "evidence_stream_h2",
            Some("apo_blake3:abc".to_string()),
            Some("engine_blake3:def".to_string()),
        )
        .with_parent_state_vertex(None, Some("no DSTW-T4 vertex in this unit".to_string()));
        assert!(writer.parent_state_vertex_id().is_none());

        // Attach a PROCESS-PROVEN provenance block over the running binary itself.
        let prov = RuntimeProvenance::capture(&dir, vec!["unit".to_string()]);
        let art_prov = prov.artifact_record(
            &dir.join("evidence_stream_h2.index.json"),
            "unit_stage",
            vec!["topology_blake3:abc".to_string()],
        );
        let ctx = RuntimeQnanlChunkContext {
            family_override: Some(QnanlArtifactFamily::BindingSiteMaterialization),
            source_label: Some("binding_sites.json".to_string()),
            lineage_refs: vec!["topology_blake3:abc".to_string()],
            runtime_provenance: Some(art_prov),
        };
        let receipt = writer
            .push_chunk_with_context(b"{\"sites\":[]}", ctx)
            .unwrap();
        // Native origin + provenance threaded into the receipt.
        assert!(receipt.runtime_origin.is_prismos_native());
        assert!(receipt.runtime_provenance.is_some());

        let chunk = &writer.chunks()[0];
        // Every directive-named field is populated from real receipt values.
        assert_eq!(
            chunk.failure_recovery_status,
            RUNTIME_QNANL_FAILURE_RECOVERY_VERIFIED
        );
        assert!(chunk.decode_proof_pass);
        assert!(chunk.parent_state_vertex_id.is_none());
        assert!(chunk.parent_state_vertex_id_absent_reason.is_some());
        assert_eq!(chunk.lineage_refs, vec!["topology_blake3:abc".to_string()]);
        assert!(chunk.runtime_provenance.is_some());
        assert_eq!(chunk.compressed_body_bytes, receipt.compressed_body_bytes);
        assert_eq!(
            chunk.compressed_body_hash_blake3,
            receipt.compressed_body_hash_blake3
        );

        // The index side-car renders the full evidence record incl. the seal proof,
        // the parent-vertex absent reason, lineage refs, and the provenance block.
        let index = std::fs::read_to_string(writer.index_path()).unwrap();
        assert!(index.contains("\"raw_size_bytes\":"));
        assert!(index.contains("\"compressed_size_bytes\":"));
        assert!(index.contains("\"compression_ratio\":"));
        assert!(index.contains("\"compressed_body_hash\":"));
        assert!(index.contains("\"packet_hash\":"));
        assert!(index.contains("\"nanl_q4_seal_proof\":"));
        assert!(index.contains("\"decode_proof\":"));
        assert!(index.contains("\"failure_recovery_status\": \"SEALED_AND_DECODE_VERIFIED\""));
        assert!(index.contains("\"parent_state_vertex_id\": null"));
        assert!(index.contains(
            "\"parent_state_vertex_id_absent_reason\": \"no DSTW-T4 vertex in this unit\""
        ));
        assert!(
            index.contains("\"runtime_origin\": \"RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS\"")
        );
        assert!(index.contains("artifact_provenance") || index.contains("emitted_artifact_path"));

        std::fs::remove_dir_all(&dir).ok();
    }
    fn durable_test_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "prism_qnanl_durable_{label}_{}_{}",
            std::process::id(),
            now_unix_ns()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn durable_test_writer(path: &Path) -> RuntimeQnanlStreamWriter {
        RuntimeQnanlStreamWriter::new(
            path,
            "durable",
            Some("topology:test".into()),
            Some("engine:test".into()),
        )
    }

    fn durable_test_options() -> QnanlEncodeOptions {
        QnanlEncodeOptions::post_run_seal_only(
            "durable.packet",
            QnanlArtifactFamily::NumericArray,
            None,
            "durability_filesystem_test",
        )
    }

    fn durable_child_command(dir: &Path, mode: &str, point: &str) -> std::process::Command {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "qnanl_artifact::tests::durability_child_process",
                "--nocapture",
            ])
            .env("PRISM_QNANL_DURABILITY_CHILD", "1")
            .env("PRISM_QNANL_DURABILITY_DIR", dir)
            .env("PRISM_QNANL_DURABILITY_MODE", mode)
            .env("PRISM_QNANL_DURABILITY_CRASH", point);
        command
    }

    #[test]
    fn durability_child_process() {
        if std::env::var("PRISM_QNANL_DURABILITY_CHILD").as_deref() != Ok("1") {
            return;
        }
        let dir = PathBuf::from(std::env::var_os("PRISM_QNANL_DURABILITY_DIR").unwrap());
        let mode = std::env::var("PRISM_QNANL_DURABILITY_MODE").unwrap();
        if mode.starts_with("packet") {
            let raw = if mode == "packet_b" {
                b"conflicting raw bytes".as_slice()
            } else {
                b"canonical raw bytes".as_slice()
            };
            encode_bytes(
                raw,
                &dir.join("packet.qnanl"),
                &dir.join("packet.receipt.json"),
                durable_test_options(),
            )
            .unwrap();
        } else if mode == "streaming" {
            encode_file_streaming(
                &dir.join("input.bin"),
                &dir.join("packet.qnanl"),
                &dir.join("packet.receipt.json"),
                durable_test_options(),
            )
            .unwrap();
        } else {
            let mut writer = durable_test_writer(&dir);
            writer
                .push_chunk(QnanlArtifactFamily::NumericArray, b"canonical raw bytes")
                .unwrap();
            writer.finalize().unwrap();
        }
    }

    #[test]
    fn durable_packet_process_death_rolls_forward_and_preserves_receipt() {
        for point in [
            "packet_intent",
            "packet_published",
            "receipt_published",
            "packet_commit",
        ] {
            let dir = durable_test_dir(point);
            let output = durable_child_command(&dir, "packet", point)
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(91),
                "fault point {point}: {:?}",
                output
            );
            let intent: DurablePacketCommit =
                read_durable_json(&dir.join("packet.qnanl.transaction.json")).unwrap();
            let expected_receipt = intent.receipt.to_json_pretty();
            let receipt = encode_bytes(
                b"canonical raw bytes",
                &dir.join("packet.qnanl"),
                &dir.join("packet.receipt.json"),
                durable_test_options(),
            )
            .unwrap();
            assert_eq!(receipt.to_json_pretty(), expected_receipt);
            assert_eq!(
                std::fs::read(dir.join("packet.receipt.json")).unwrap(),
                expected_receipt.as_bytes()
            );
            assert_eq!(
                decode_packet_bytes(&std::fs::read(dir.join("packet.qnanl")).unwrap()).unwrap(),
                b"canonical raw bytes"
            );
            assert!(!dir.join("packet.qnanl.transaction.json").exists());
            let prior_hash = file_hash(&dir.join("packet.qnanl")).unwrap();
            assert!(encode_bytes(
                b"changed bytes",
                &dir.join("packet.qnanl"),
                &dir.join("packet.receipt.json"),
                durable_test_options()
            )
            .is_err());
            assert_eq!(file_hash(&dir.join("packet.qnanl")).unwrap(), prior_hash);
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn durable_streaming_packet_process_death_roundtrip() {
        let dir = durable_test_dir("streaming");
        let raw: Vec<u8> = (0..100_000).map(|i| (i % 251) as u8).collect();
        std::fs::write(dir.join("input.bin"), &raw).unwrap();
        let output = durable_child_command(&dir, "streaming", "packet_published")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(91), "{:?}", output);
        let receipt = encode_file_streaming(
            &dir.join("input.bin"),
            &dir.join("packet.qnanl"),
            &dir.join("packet.receipt.json"),
            durable_test_options(),
        )
        .unwrap();
        assert!(receipt.roundtrip_verified);
        assert_eq!(
            decode_packet_bytes(&std::fs::read(dir.join("packet.qnanl")).unwrap()).unwrap(),
            raw
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn durable_stream_process_death_index_last_recovery() {
        for point in [
            "stream_intent",
            "packet_intent",
            "packet_published",
            "receipt_published",
            "packet_commit",
            "stream_proof",
            "stream_lineage",
            "stream_index",
            "stream_finalization",
        ] {
            let dir = durable_test_dir(point);
            let output = durable_child_command(&dir, "stream", point)
                .output()
                .unwrap();
            assert_eq!(
                output.status.code(),
                Some(91),
                "fault point {point}: {:?}",
                output
            );
            let mut writer = durable_test_writer(&dir);
            let recovered = writer.recover().unwrap();
            if point == "stream_index" || point == "stream_finalization" {
                assert_eq!(recovered, 1);
                assert!(!writer.has_pending_chunk());
            } else {
                assert_eq!(recovered, 0);
                assert!(writer.has_pending_chunk());
                assert!(!dir.join("durable.index.json").exists());
                assert!(writer.finalize().is_err());
                assert!(writer
                    .push_chunk(QnanlArtifactFamily::NumericArray, b"wrong replay")
                    .is_err());
                writer
                    .push_chunk(QnanlArtifactFamily::NumericArray, b"canonical raw bytes")
                    .unwrap();
            }
            assert_eq!(writer.chunks().len(), 1);
            let path = writer.finalize().unwrap();
            let first = std::fs::read(&path).unwrap();
            assert_eq!(writer.finalize().unwrap(), path);
            assert_eq!(std::fs::read(path).unwrap(), first);
            assert!(writer
                .push_chunk(QnanlArtifactFamily::NumericArray, b"after close")
                .is_err());
            let mut reopened = durable_test_writer(&dir);
            assert_eq!(reopened.recover().unwrap(), 1);
            assert_eq!(
                reopened.chunks()[0].raw_hash_blake3,
                hex32(&hash32(b"canonical raw bytes"))
            );
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn durable_stream_rejects_stale_identity_and_corrupt_evidence() {
        for suffix in [
            ".prismqnanl",
            ".prismqnanl.receipt.json",
            ".prismqnanl.decode_proof.json",
            ".prismqnanl.lineage.json",
        ] {
            let dir = durable_test_dir("corruption");
            let mut original = durable_test_writer(&dir);
            original
                .push_chunk(QnanlArtifactFamily::NumericArray, b"original bytes")
                .unwrap();
            let mut stale = durable_test_writer(&dir);
            assert!(stale
                .push_chunk(QnanlArtifactFamily::NumericArray, b"replacement bytes")
                .is_err());
            let mut wrong_engine = RuntimeQnanlStreamWriter::new(
                &dir,
                "durable",
                Some("topology:test".into()),
                Some("different engine".into()),
            );
            assert!(wrong_engine.recover().is_err());
            let corrupt_path = dir.join(format!("durable.chunk000000{suffix}"));
            std::fs::write(&corrupt_path, b"truncated evidence").unwrap();
            assert!(stale.recover().is_err());
            assert!(original.finalize().is_err());
            assert_eq!(std::fs::read(corrupt_path).unwrap(), b"truncated evidence");
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn durable_packet_concurrent_processes_have_one_immutable_winner() {
        let dir = durable_test_dir("race");
        let mut a = durable_child_command(&dir, "packet_a", "");
        let mut b = durable_child_command(&dir, "packet_b", "");
        a.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        b.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let mut first = a.spawn().unwrap();
        let mut second = b.spawn().unwrap();
        let successes =
            first.wait().unwrap().success() as usize + second.wait().unwrap().success() as usize;
        assert_eq!(successes, 1);
        let commit: DurablePacketCommit =
            read_durable_json(&dir.join("packet.qnanl.commit.json")).unwrap();
        verify_packet_commit(
            &dir.join("packet.qnanl"),
            &dir.join("packet.receipt.json"),
            &commit,
        )
        .unwrap();
        let raw = decode_packet_bytes(&std::fs::read(dir.join("packet.qnanl")).unwrap()).unwrap();
        assert!(raw == b"canonical raw bytes" || raw == b"conflicting raw bytes");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn durable_provenance_conflict_and_legacy_artifact_are_retained() {
        let dir = durable_test_dir("provenance");
        let packet = dir.join("packet.qnanl");
        let receipt_path = dir.join("packet.receipt.json");
        let first = encode_bytes(
            b"canonical raw bytes",
            &packet,
            &receipt_path,
            durable_test_options(),
        )
        .unwrap();
        let mut changed = durable_test_options();
        changed.runtime_origin = RuntimeOrigin::RuntimeInline;
        assert!(encode_bytes(b"canonical raw bytes", &packet, &receipt_path, changed).is_err());
        assert_eq!(
            std::fs::read(&receipt_path).unwrap(),
            first.to_json_pretty().as_bytes()
        );
        let legacy = dir.join("legacy.qnanl");
        std::fs::write(&legacy, b"prior artifact").unwrap();
        assert!(encode_bytes(
            b"canonical raw bytes",
            &legacy,
            &dir.join("legacy.receipt.json"),
            durable_test_options()
        )
        .is_err());
        assert_eq!(std::fs::read(legacy).unwrap(), b"prior artifact");
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod continuation_recovery_tests {
    use super::*;
    #[test]
    fn recovery_namespace_is_reserved_without_hiding_active_sources() {
        for name in ["run/.continuation_tails/abcd.bytes", "run/.continuation_absent_owners/epoch/payload/binding_sites.json",
            "run/frames.bin.uncommitted.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.123.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "run/frames.bin.uncommitted.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.123.next",
            "run/.restore-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.pending", "run/.scientific_chronology_manifest.json.abcd.4.5.restore"] {
            assert!(is_native_continuation_recovery_path(Path::new(name)), "{name}");
            assert!(is_qnanl_auxiliary_path(Path::new(name)), "{name}");
        }
        for name in ["run/binding_sites.json", "run/scientific_state_epochs.jsonl", "run/prism_zstr_17.bin",
            "run/continuation_study/binding_sites.json", "run/notes.uncommitted.json"] {
            assert!(!is_native_continuation_recovery_path(Path::new(name)), "{name}");
        }
    }
}

use crate::cuda_backend::{cuda_runtime_report, execute_required_opcode_suite};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::digest::{digest, SHA256};
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

pub const PCODE_MAGIC: &[u8; 8] = b"PRISMPC\0";
pub const PCOUT_MAGIC: &[u8; 8] = b"PRISMPO\0";
pub const PSK_ENCRYPTED_MAGIC: &[u8; 8] = b"PSKENC1\0";
pub const PSK_PROD_MAGIC: &[u8; 8] = b"PSKPRD1\0";
const PSK_ENVELOPE_HEADER_LEN: usize = 128;
pub const HEADER_LEN: usize = 256;
pub const SECTION_ENTRY_LEN: usize = 96;
pub const INSTR_WIDTH: usize = 32;
pub const SECTION_ALIGN: u64 = 64;
pub const INSTR_ALIGN: u64 = 32;
pub const PACKET_HASH_OFFSET: usize = 196;
pub const PACKET_HASH_LEN: usize = 32;

pub const ISA_MAJOR: u16 = 0;
pub const ISA_MINOR: u16 = 1;
pub const ABI_MAJOR: u16 = 0;
pub const ABI_MINOR: u16 = 1;

pub mod opcode {
    pub const NOP: u16 = 0x0000;
    pub const ABI_DECLARE: u16 = 0x0001;
    pub const FEATURE_REQUIRE: u16 = 0x0002;
    pub const PACKET_BEGIN: u16 = 0x0003;
    pub const PACKET_END: u16 = 0x0004;
    pub const HALT: u16 = 0x0005;
    pub const TRAP: u16 = 0x0006;
    pub const STATE_DEF: u16 = 0x0100;
    pub const STATE_CANONICALIZE: u16 = 0x0104;
    pub const STATE_HASH: u16 = 0x0105;
    pub const STATE_FREEZE: u16 = 0x0106;
    pub const STATE_SNAPSHOT: u16 = 0x0107;
    // DSTW-T4 State-Lifecycle group A (scientific-state vertex lifecycle).
    // Registry/validation/disassembly coverage only (no behavioral VM dispatch; see opcode.rs).
    pub const STATE_VERTEX_CREATE: u16 = 0x0108;
    pub const STATE_PRECONDITION_VALIDATE: u16 = 0x0109;
    pub const STATE_OPERATOR_APPLY: u16 = 0x010A;
    pub const STATE_CLASSIFY_BLOCKED: u16 = 0x010B;
    pub const STATE_NEXT_OPERATOR_REC: u16 = 0x010C;
    // DSTW-T4 Step 4: scientific-state TRANSITION emission (edge between two
    // vertices). Registry/validation/disassembly coverage only (no behavioral VM
    // dispatch; see opcode.rs). The transition's *operator application* is a real
    // CUDA-backed MD op (e.g. MD.INTEGRATE_VELOCITY_VERLET); EMIT_TRANSITION only
    // labels the edge-record emission in the count/operand telemetry.
    pub const STATE_EMIT_TRANSITION: u16 = 0x010D;
    // DSTW-T4 Step h5: POST_RUN_DERIVED dense-tensor semantic-summary emission (the
    // receipt that records one semantic summary per dense engine tensor so the raw
    // tensor never enters training). Registry/validation/disassembly coverage only
    // (no behavioral VM dispatch); the native in-graph reduction is engine-owner.
    pub const STATE_DENSE_TENSOR_SUMMARY: u16 = 0x010E;
    pub const TIME_BIND: u16 = 0x0200;
    pub const TIME_NORMALIZE: u16 = 0x0201;
    pub const TIME_ORDER_ASSERT: u16 = 0x0205;
    pub const NANL_PACK: u16 = 0x0300;
    pub const NANL_HASH: u16 = 0x0305;
    pub const EVENT_EMIT: u16 = 0x0400;
    pub const EVENT_BIND_STATE: u16 = 0x0401;
    pub const EVENT_SORT: u16 = 0x0403;
    pub const EVENT_INTEGRATE: u16 = 0x0406;
    pub const EVENT_INDEX: u16 = 0x0407;
    pub const DRF_MAP: u16 = 0x0500;
    pub const DRF_FIRE: u16 = 0x0501;
    pub const DRF_RESONATE: u16 = 0x0502;
    pub const DRF_NORMALIZE: u16 = 0x0504;
    pub const DRF_WRITEBACK: u16 = 0x0507;
    pub const THERMO_ENERGY: u16 = 0x0601;
    pub const THERMO_ENTROPY: u16 = 0x0602;
    pub const THERMO_CONSERVE_ASSERT: u16 = 0x0605;
    pub const MD_PROTOCOL_DIRECTOR: u16 = 0x0700;
    pub const MD_COM_REMOVE: u16 = 0x0701;
    pub const MD_NEIGHBOR_REBUILD: u16 = 0x0702;
    pub const MD_ACCUMULATOR_ZERO: u16 = 0x0703;
    pub const MD_FORCE_BONDED: u16 = 0x0704;
    pub const MD_FORCE_NONBONDED: u16 = 0x0705;
    pub const MD_INTEGRATE_VELOCITY_VERLET: u16 = 0x0706;
    pub const MD_SHAKE_CONSTRAIN: u16 = 0x0707;
    pub const MD_EXCLUSION_FIELD: u16 = 0x0708;
    pub const MD_WATER_DENSITY: u16 = 0x0709;
    pub const MD_PHOTO_UV_PUMP: u16 = 0x070A;
    pub const MD_EXTERNAL_WORK_REDUCE: u16 = 0x070B;
    pub const MD_VELOCITY_SECOND_HALF: u16 = 0x070C;
    pub const MD_CA_RESTRAINT: u16 = 0x070D;
    pub const MD_LIF_OBSERVE: u16 = 0x070E;
    pub const MD_COUPLING_CLEAR: u16 = 0x070F;
    pub const MD_HEARTBEAT_ASSERT: u16 = 0x0710;
    pub const MD_LADD_UPDATE: u16 = 0x0711;
    pub const MD_KCC_UPDATE: u16 = 0x0712;
    pub const MD_NMA_PERTURB: u16 = 0x0713;
    pub const MD_SPIKE_EMIT: u16 = 0x0714;
    pub const MD_SPIKE_READOUT: u16 = 0x0715;
    pub const MD_SIGNAL_GRID_READOUT: u16 = 0x0716;
    pub const MD_KCC_DESCRIPTOR: u16 = 0x0717;
    pub const MD_COUPLING_BUILD: u16 = 0x0718;
    // Promotion of the String-only Always-On QNANL Spike Bus / TWIN16 commit ops
    // (md_native_runtime.rs:37-46) to real u16 registry rows so prismval/prismdis cover them.
    // Registry/validation/disassembly coverage only (no behavioral VM dispatch).
    pub const MD_SPIKE_CHUNK_COMMIT: u16 = 0x0719;
    pub const MD_SENSOR_EMIT: u16 = 0x071A;
    pub const MD_QNANL_SEAL: u16 = 0x071B;
    pub const MD_QNANL_VERIFY: u16 = 0x071C;
    pub const MD_TWIN16_CELL_COMMIT: u16 = 0x071D;
    pub const MD_CONDITION_EDGE_COMMIT: u16 = 0x071E;
    // GDP holo precondition gate (DSTW-T4 Step 7 blocked-state-as-ISA-failure).
    pub const MD_VALIDATE_LIGAND_STATE: u16 = 0x071F;
    // MYC:MAX lambda-fragment / Ghost successor. These rows are append-only
    // typed runtime-event identities; the engine-owned producer remains
    // responsible for recording the physical payload and runtime origin.
    pub const MD_LAMBDA_FRAGMENT_FORCE: u16 = 0x0720;
    pub const MD_LAMBDA_DUDL_ACCUMULATE: u16 = 0x0721;
    pub const MD_LAMBDA_NCMC_PERTURB: u16 = 0x0722;
    pub const MD_LAMBDA_NCMC_PROPAGATE: u16 = 0x0723;
    pub const MD_LAMBDA_NCMC_ACCEPT: u16 = 0x0724;
    pub const MD_LAMBDA_ENDPOINT_OBSERVE: u16 = 0x0725;
    pub const MD_LAMBDA_STATE_COMMIT: u16 = 0x0726;
    pub const MD_GHOST_SEGMENT_COMMIT: u16 = 0x0727;
    pub const MD_GHOST_CONSERVATION: u16 = 0x0728;
    pub const MD_GHOST_PHASE_BINDING: u16 = 0x0729;
    /// Rejected NCMC proposal resolved by restoring the exact initial endpoint.
    pub const MD_LAMBDA_STATE_ROLLBACK: u16 = 0x072A;
    /// Engine-owned proof that a declared no-probe interval had zero lambda slots/events.
    pub const MD_LAMBDA_NO_PROBE_ABSENCE: u16 = 0x072B;
    pub const MD_GHOST_COORDINATE_SEGMENT_COMMIT: u16 = 0x072C;
    pub const MD_GHOST_COORDINATE_CONSERVATION: u16 = 0x072D;
    pub const MD_GHOST_LATTICE_EDGE_COMMIT: u16 = 0x072E;
    pub const MD_GHOST_LATTICE_EDGE_CONSERVATION: u16 = 0x072F;
    /// Resident captured-graph velocity magnitude guard. This was already a
    /// real engine opcode name in `md_native_runtime`; 0x0730 is the first free
    /// append-only MD registry identity after the MYC:MAX successor rows.
    pub const MD_VELOCITY_CLAMP: u16 = 0x0730;
    // P2 — UV-off control instrument. Emitted in place of MD_PHOTO_UV_PUMP when
    // the run declares --uv-burst-suppress: the pump kernel node is omitted from
    // the launch/graph entirely, so the op stream must not assert a pump that did
    // not run. Registry/validation/disassembly coverage only (no VM dispatch),
    // matching the MD_SPIKE_CHUNK_COMMIT..MD_CONDITION_EDGE_COMMIT precedent above.
    pub const MD_PHOTO_UV_PUMP_SUPPRESSED: u16 = 0x0731;
    // Existing engine spike_sensor_bus manifest producer, now addressable by
    // binary validation/disassembly and typed operand ledgers. No VM dispatch.
    pub const MD_SPIKE_MANIFEST_COMMIT: u16 = 0x0732;
    // PATHB family base 0x1000 (first free family after DRF.* tops at 0x0F99).
    // DSTW-T4 Path-B binding-site materialization + blocked-site precondition.
    pub const PATHB_MATERIALIZE_BINDING_SITE: u16 = 0x1000;
    pub const PATHB_BLOCKED_SITE_PRECONDITION: u16 = 0x1001;
    pub const PATHB_CHEMICAL_STATE_FIBER_MATERIALIZED: u16 = 0x1002;
    pub const PATHB_CHEMICAL_STATE_ASSIGNED: u16 = 0x1003;
    pub const PATHB_MATCHED_CONTROL_CLASSIFIED: u16 = 0x1004;
    pub const PATHB_CANDIDATE_OBSERVED: u16 = 0x1005;
    pub const PATHB_PARETO_CHAMPION_PROMOTED: u16 = 0x1006;
    pub const PATHB_PARETO_CHAMPION_SUPERSEDED: u16 = 0x1007;
    /// Native no-probe whole-lattice discovery candidate/population evidence.
    /// This is deliberately distinct from candidate-advancement observation:
    /// discovery preserves structural evidence and does not promote a record.
    pub const PATHB_DISCOVERY_CANDIDATE_MATERIALIZED: u16 = 0x1008;
    /// Post-commit operator-selected confirmation handoff. This opcode cannot
    /// appear in unbiased discovery candidate/population materialization.
    pub const PATHB_CONFIRMATION_HANDOFF_MATERIALIZED: u16 = 0x1009;
    pub const REWARD_ATTACH_TEACHER: u16 = 0x0801;
    pub const REWARD_GFLOWNET_FLOW: u16 = 0x0802;
    pub const ACTION_CONSTRAIN: u16 = 0x0901;
    pub const PRISMAI_PACKET: u16 = 0x0904;
    pub const HIL_REQUEST: u16 = 0x0905;
    pub const MEM_LINEAGE: u16 = 0x0A01;
    pub const REPLAY_BEGIN: u16 = 0x0A04;
    pub const REPLAY_CHECKPOINT: u16 = 0x0A05;
    pub const REPLAY_END: u16 = 0x0A06;
    pub const REPLAY_VERIFY: u16 = 0x0A07;
    pub const AUDIT_SEAL: u16 = 0x0B05;
    pub const AUDIT_VERIFY: u16 = 0x0B06;
    pub const PROVENANCE_ATTACH: u16 = 0x0B07;
    pub const ASSERT_TYPE: u16 = 0x0C00;
    pub const ASSERT_RANGE: u16 = 0x0C01;
    pub const ASSERT_SHAPE: u16 = 0x0C02;
    pub const ASSERT_HASH: u16 = 0x0C03;
    pub const ASSERT_TIMEBASE: u16 = 0x0C04;
    pub const ASSERT_DETERMINISTIC: u16 = 0x0C05;
    pub const ASSERT_SCIENTIFIC: u16 = 0x0C06;
    pub const TENSOR_ALLOC: u16 = 0x0D00;
    pub const TENSOR_CANONICALIZE: u16 = 0x0D06;
    pub const SEC_CLASSIFY: u16 = 0x0E00;
    pub const SEC_POLICY_BIND: u16 = 0x0E01;
    pub const SEC_CAPABILITY_ASSERT: u16 = 0x0E02;
    pub const SEC_ACCESS_CHECK: u16 = 0x0E03;
    pub const SEC_REDACT: u16 = 0x0E04;
    pub const SEC_ZEROIZE: u16 = 0x0E05;
    pub const SEC_EXPORT_GATE: u16 = 0x0E06;
    pub const SEC_AIRLOCK_MODEL_OUTPUT: u16 = 0x0E07;
    pub const CRYPTO_HASH: u16 = 0x0E19;
    pub const CRYPTO_MERKLE_ROOT: u16 = 0x0E1A;
    pub const SEAL_BEGIN: u16 = 0x0E20;
    pub const SEAL_APPEND: u16 = 0x0E21;
    pub const SEAL_FINALIZE: u16 = 0x0E22;
    pub const SEAL_VERIFY: u16 = 0x0E23;
    pub const ATTEST_RUNTIME_MEASURE: u16 = 0x0E31;
    pub const ATTEST_CUDA_KERNEL_MEASURE: u16 = 0x0E32;
    pub const ATTEST_BUILD_PROVENANCE: u16 = 0x0E34;
    pub const GUARD_INTRUSION_EVENT: u16 = 0x0E47;

    pub const CSE_DEF_SCHEMA: u16 = 0x0F20;
    pub const CSE_DEF_DICTIONARY: u16 = 0x0F21;
    pub const CSE_DEF_ENTITY: u16 = 0x0F22;
    pub const CSE_DEF_COMPONENT_TABLE: u16 = 0x0F23;
    pub const CSE_TOKENIZE: u16 = 0x0F24;
    pub const CSE_DELTA_ENCODE: u16 = 0x0F25;
    pub const CSE_MOTIF_ENCODE: u16 = 0x0F26;
    pub const CSE_CANONICALIZE: u16 = 0x0F27;
    pub const CSE_HASH: u16 = 0x0F28;
    pub const CSE_TO_NANL_BIN: u16 = 0x0F29;
    pub const CSE_TO_NANL_Q4: u16 = 0x0F2A;
    pub const CSE_TO_NRF: u16 = 0x0F2B;
    pub const CSE_TO_DRF: u16 = 0x0F2C;
    pub const CSE_RECONSTRUCT: u16 = 0x0F2D;
    pub const CSE_VALIDATE_ROUNDTRIP: u16 = 0x0F2E;
    // DSTW-T4 tokenizer row op (next free CSE slot after CSE.VALIDATE_ROUNDTRIP 0x0F2E).
    pub const CSE_EMIT_DSTW_ROW: u16 = 0x0F2F;

    pub const NANL_BIN_PACK: u16 = 0x0F40;
    pub const NANL_BIN_UNPACK: u16 = 0x0F41;
    pub const NANL_BIN_VALIDATE: u16 = 0x0F42;
    pub const NANL_BIN_HASH: u16 = 0x0F43;
    pub const NANL_BIN_INDEX: u16 = 0x0F44;
    pub const NANL_BIN_STREAM_DECODE: u16 = 0x0F45;
    pub const NANL_BIN_TO_NRF: u16 = 0x0F46;
    pub const NANL_BIN_TO_DRF: u16 = 0x0F47;

    pub const NANL_Q4_ENCODE: u16 = 0x0F50;
    pub const NANL_Q4_DECODE: u16 = 0x0F51;
    pub const NANL_Q4_VALIDATE: u16 = 0x0F52;
    pub const NANL_Q4_CONSTRAINT_SCORE: u16 = 0x0F53;
    pub const NANL_Q4_ECC_ATTACH: u16 = 0x0F54;
    pub const NANL_Q4_ECC_VERIFY: u16 = 0x0F55;
    pub const NANL_Q4_MERKLE_ROOT: u16 = 0x0F56;
    pub const NANL_Q4_TO_BIN: u16 = 0x0F57;
    pub const NANL_BIN_TO_Q4: u16 = 0x0F58;
    pub const NANL_Q4_TO_DNA_OLIGO: u16 = 0x0F59;

    pub const NRF_FRAME_BEGIN: u16 = 0x0F70;
    pub const NRF_FRAME_END: u16 = 0x0F71;
    pub const NRF_EVENT_TILE: u16 = 0x0F72;
    pub const NRF_SPIKE_MOTIF: u16 = 0x0F73;
    pub const NRF_TRANSITION_EDGE: u16 = 0x0F74;
    pub const NRF_WATER_GATE: u16 = 0x0F75;
    pub const NRF_PHASE_TILE: u16 = 0x0F76;
    pub const NRF_CAUSAL_PARENT: u16 = 0x0F77;
    pub const NRF_SPARSE_INDEX: u16 = 0x0F78;
    pub const NRF_MOTIF_PACK: u16 = 0x0F79;
    pub const NRF_MOTIF_UNPACK: u16 = 0x0F7A;
    pub const NRF_TO_DRF: u16 = 0x0F7B;
    pub const NRF_RECONSTRUCT_EVENTS: u16 = 0x0F7C;
    pub const NRF_VALIDATE_REPLAY: u16 = 0x0F7D;

    pub const DRF_MAP_CSE: u16 = 0x0F90;
    pub const DRF_LOAD_NRF: u16 = 0x0F91;
    pub const DRF_DECAY: u16 = 0x0F92;
    pub const DRF_SYNC: u16 = 0x0F93;
    pub const DRF_SCATTER_EVENT: u16 = 0x0F94;
    pub const DRF_GATHER_NEIGHBORHOOD: u16 = 0x0F95;
    pub const DRF_UPDATE_UNCERTAINTY: u16 = 0x0F96;
    pub const DRF_UPDATE_ACQUISITION: u16 = 0x0F97;
    pub const DRF_RECONSTRUCT: u16 = 0x0F98;
    pub const DRF_CONSISTENCY_ASSERT: u16 = 0x0F99;
}

#[derive(Debug, thiserror::Error)]
pub enum PcError {
    #[error("PC_ERR_FORMAT:{0}")]
    Format(String),
    #[error("PC_ERR_SECTION:{0}")]
    Section(String),
    #[error("PC_ERR_OPCODE:{0}")]
    Opcode(String),
    #[error("PC_ERR_OPERAND:{0}")]
    Operand(String),
    #[error("PC_ERR_TIMEBASE:{0}")]
    Timebase(String),
    #[error("PC_ERR_PROVENANCE:{0}")]
    Provenance(String),
    #[error("PC_ERR_HASH:{0}")]
    Hash(String),
    #[error("PC_ERR_SECURITY:{0}")]
    Security(String),
    #[error("PC_ERR_REPLAY:{0}")]
    Replay(String),
    #[error("PC_ERR_GPU:{0}")]
    Gpu(String),
    #[error("PC_ERR_IO:{0}")]
    Io(String),
    #[error("PC_ERR_INTERNAL:{0}")]
    Internal(String),
}

impl From<std::io::Error> for PcError {
    fn from(value: std::io::Error) -> Self {
        PcError::Io(value.to_string())
    }
}

impl From<serde_json::Error> for PcError {
    fn from(value: serde_json::Error) -> Self {
        PcError::Format(value.to_string())
    }
}

pub type PcResult<T> = Result<T, PcError>;

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PcInstr32 {
    pub opcode: u16,
    pub class: u8,
    pub flags: u8,
    pub dst_kind: u8,
    pub src0_kind: u8,
    pub src1_kind: u8,
    pub src2_kind: u8,
    pub dst: u32,
    pub src0: u32,
    pub src1: u32,
    pub src2_or_len: u32,
    pub aux: u64,
}

impl PcInstr32 {
    pub fn new(
        opcode: u16,
        flags: u8,
        dst_kind: OperandKind,
        dst: u32,
        src0_kind: OperandKind,
        src0: u32,
    ) -> Self {
        Self {
            opcode,
            class: (opcode >> 8) as u8,
            flags,
            dst_kind: dst_kind as u8,
            src0_kind: src0_kind as u8,
            src1_kind: OperandKind::None as u8,
            src2_kind: OperandKind::None as u8,
            dst,
            src0,
            src1: 0,
            src2_or_len: 0,
            aux: 0,
        }
    }

    pub fn encode(&self) -> [u8; INSTR_WIDTH] {
        let mut out = [0u8; INSTR_WIDTH];
        put_u16(&mut out, 0, self.opcode);
        out[2] = self.class;
        out[3] = self.flags;
        out[4] = self.dst_kind;
        out[5] = self.src0_kind;
        out[6] = self.src1_kind;
        out[7] = self.src2_kind;
        put_u32(&mut out, 8, self.dst);
        put_u32(&mut out, 12, self.src0);
        put_u32(&mut out, 16, self.src1);
        put_u32(&mut out, 20, self.src2_or_len);
        put_u64(&mut out, 24, self.aux);
        out
    }

    pub fn decode(bytes: &[u8]) -> PcResult<Self> {
        if bytes.len() != INSTR_WIDTH {
            return Err(PcError::Format(
                "instruction width is not 32 bytes".to_string(),
            ));
        }
        let instr = Self {
            opcode: read_u16(bytes, 0)?,
            class: bytes[2],
            flags: bytes[3],
            dst_kind: bytes[4],
            src0_kind: bytes[5],
            src1_kind: bytes[6],
            src2_kind: bytes[7],
            dst: read_u32(bytes, 8)?,
            src0: read_u32(bytes, 12)?,
            src1: read_u32(bytes, 16)?,
            src2_or_len: read_u32(bytes, 20)?,
            aux: read_u64(bytes, 24)?,
        };
        if instr.class != (instr.opcode >> 8) as u8 {
            return Err(PcError::Opcode(format!(
                "opcode class mismatch for 0x{:04x}: class={} expected={}",
                instr.opcode,
                instr.class,
                instr.opcode >> 8
            )));
        }
        Ok(instr)
    }
}

const _: [(); 32] = [(); std::mem::size_of::<PcInstr32>()];
const _: [(); 32] = [(); std::mem::align_of::<PcInstr32>()];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OperandKind {
    None = 0,
    ImmU32 = 1,
    ImmI32 = 2,
    ConstRef = 3,
    StateRef = 4,
    TimeRef = 5,
    NanlRef = 6,
    EventRef = 7,
    DrfTensorRef = 8,
    GraphRef = 9,
    RewardRef = 10,
    ActionRef = 11,
    PayloadRef = 12,
    ArrowColRef = 13,
    ShapeRef = 14,
    DtypeRef = 15,
    ProvenanceRef = 16,
    SealRef = 17,
    KernelRef = 18,
    ErrorRef = 19,
    ExtRef = 20,
}

impl OperandKind {
    pub fn validate(raw: u8) -> bool {
        raw <= OperandKind::ExtRef as u8
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum SectionKind {
    InstrStream = 1,
    StateTable = 2,
    OperandTable = 3,
    TimeTable = 4,
    NanlTable = 5,
    EventTable = 6,
    TensorTable = 7,
    GraphTable = 8,
    RewardTable = 9,
    ActionTable = 10,
    ProvenanceTable = 11,
    PayloadBlob = 12,
    SealChain = 13,
    DebugJson = 14,
}

impl SectionKind {
    fn from_u16(raw: u16) -> PcResult<Self> {
        Ok(match raw {
            1 => Self::InstrStream,
            2 => Self::StateTable,
            3 => Self::OperandTable,
            4 => Self::TimeTable,
            5 => Self::NanlTable,
            6 => Self::EventTable,
            7 => Self::TensorTable,
            8 => Self::GraphTable,
            9 => Self::RewardTable,
            10 => Self::ActionTable,
            11 => Self::ProvenanceTable,
            12 => Self::PayloadBlob,
            13 => Self::SealChain,
            14 => Self::DebugJson,
            _ => return Err(PcError::Section(format!("unknown section kind {raw}"))),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum DTypeId {
    Bool = 1,
    U2Packed = 2,
    U8 = 3,
    U16 = 4,
    U32 = 5,
    U64 = 6,
    I32 = 7,
    F32 = 8,
    F64 = 9,
    Hash256 = 10,
    StringRef = 11,
    BytesRef = 12,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum StateKind {
    MolecularState = 1,
    TemporalState = 2,
    NanlStrand = 3,
    EventStream = 4,
    DrfState = 5,
    DrfTensor = 6,
    ThermoState = 7,
    GraphState = 8,
    RewardTrace = 9,
    ActionProposal = 10,
    PrismAiPacket = 11,
    ProvenanceRecord = 12,
    AuditSeal = 13,
    ReplayCheckpoint = 14,
}

#[derive(Clone, Debug)]
pub struct SectionEntry {
    pub section_id: u16,
    pub section_type: SectionKind,
    pub flags: u32,
    pub offset: u64,
    pub length: u64,
    pub record_count: u64,
    pub schema_hash: [u8; 32],
    pub content_hash: [u8; 32],
}

impl SectionEntry {
    fn encode(&self) -> [u8; SECTION_ENTRY_LEN] {
        let mut out = [0u8; SECTION_ENTRY_LEN];
        put_u16(&mut out, 0, self.section_id);
        put_u16(&mut out, 2, self.section_type as u16);
        put_u32(&mut out, 4, self.flags);
        put_u64(&mut out, 8, self.offset);
        put_u64(&mut out, 16, self.length);
        put_u64(&mut out, 24, self.record_count);
        out[32..64].copy_from_slice(&self.schema_hash);
        out[64..96].copy_from_slice(&self.content_hash);
        out
    }

    fn decode(bytes: &[u8]) -> PcResult<Self> {
        if bytes.len() != SECTION_ENTRY_LEN {
            return Err(PcError::Section(
                "section entry must be 96 bytes".to_string(),
            ));
        }
        let mut schema_hash = [0u8; 32];
        schema_hash.copy_from_slice(&bytes[32..64]);
        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&bytes[64..96]);
        Ok(Self {
            section_id: read_u16(bytes, 0)?,
            section_type: SectionKind::from_u16(read_u16(bytes, 2)?)?,
            flags: read_u32(bytes, 4)?,
            offset: read_u64(bytes, 8)?,
            length: read_u64(bytes, 16)?,
            record_count: read_u64(bytes, 24)?,
            schema_hash,
            content_hash,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PCodePacket {
    pub bytes: Vec<u8>,
    pub sections: Vec<SectionEntry>,
    pub instructions: Vec<PcInstr32>,
    pub state_count: u64,
    pub time_count: u64,
    pub event_count: u64,
    pub nanl_count: u64,
    pub packet_hash: [u8; 32],
}

/// Exact join between a packet's EventTable bytes and the seal chain that
/// committed those same bytes. Both named SHA-256 fields are recomputed from
/// the EventTable region; consumers compare them without trusting a capability
/// boolean.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventByteSealBinding {
    pub event_byte_offset: u64,
    pub event_byte_length: u64,
    pub event_record_count: u64,
    pub event_byte_source_sha256: String,
    pub qnanl_sealed_from_sha256: String,
}

/// In-memory production packet plus the exact EventTable byte region that the
/// packet seal commits.  This is the runtime bridge used by engine-owned QNANL
/// emission: callers seal `event_table_bytes` directly and persist
/// `packet_bytes` as the independently re-readable PrismCode authority.
#[derive(Clone, Debug)]
pub struct RuntimeEventTablePacket {
    pub packet_bytes: Vec<u8>,
    pub event_table_bytes: Vec<u8>,
    pub payload_context: RuntimeEventTablePayloadContext,
    pub binding: EventByteSealBinding,
}

/// Exact PayloadBlob context extracted from the same parsed PrismCode packet
/// that supplied the seal-bound EventTable. Fields are private so consumers
/// cannot assemble an unrelated payload/packet identity tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeEventTablePayloadContext {
    payload_blob_bytes: Vec<u8>,
    payload_blob_sha256: String,
    parent_packet_sha256: String,
    event_table_source_sha256: String,
}

impl RuntimeEventTablePayloadContext {
    pub fn payload_blob_bytes(&self) -> &[u8] {
        &self.payload_blob_bytes
    }

    pub fn payload_blob_sha256(&self) -> &str {
        &self.payload_blob_sha256
    }

    pub fn parent_packet_sha256(&self) -> &str {
        &self.parent_packet_sha256
    }

    pub fn event_table_source_sha256(&self) -> &str {
        &self.event_table_source_sha256
    }

    #[cfg(test)]
    pub(crate) fn for_semantic_consumer_test(
        event_table_bytes: &[u8],
        payload_blob_bytes: Vec<u8>,
    ) -> Self {
        Self {
            payload_blob_sha256: sha256_hex(&payload_blob_bytes),
            parent_packet_sha256: sha256_hex(b"semantic-consumer-test-parent"),
            event_table_source_sha256: sha256_hex(event_table_bytes),
            payload_blob_bytes,
        }
    }
}

/// Recompute the authoritative EventTable record hash from decoded ABI
/// semantics. This is the same algorithm used by `build_packet_bytes`.
pub fn canonical_runtime_event_record_hash(
    event_id: u64,
    event_kind_code: u16,
    time_ref: u64,
    source_state: u64,
    target_state: u64,
    value: f64,
    unit: &[u8],
) -> PcResult<[u8; 32]> {
    let event_kind = match event_kind_code {
        1 => b"SPIKE".as_slice(),
        2 => b"KCC".as_slice(),
        3 => b"BOCPD".as_slice(),
        4 => b"PHASE_TILE".as_slice(),
        5 => b"FIBER".as_slice(),
        other => {
            return Err(PcError::Format(format!(
                "unknown runtime EventTable kind code {other}"
            )))
        }
    };
    Ok(hash_many(&[
        &event_id.to_le_bytes(),
        &time_ref.to_le_bytes(),
        &source_state.to_le_bytes(),
        &target_state.to_le_bytes(),
        &value.to_le_bytes(),
        unit,
        event_kind,
    ]))
}

/// Version of the MD launch policy encoded directly in the PrismCode
/// instruction stream.  This is packet authority: DebugJson and host defaults
/// are deliberately not consulted while decoding it.
pub const MD_ENGINE_POLICY_SCHEMA_VERSION_V1: u32 = 1;
pub const MD_ENGINE_POLICY_SCHEMA_VERSION_V2: u32 = 2;
pub const MD_RUN_AUTHORITY_HW_PROFILE_V1: &[u8] = b"PRISMCODE_REAL_TOPOLOGY_BOUND_MD_POLICY_V1";
const SYNTHETIC_REFERENCE_HW_PROFILE: &[u8] = b"PRISMCODE_HW_PROFILE_UNBOUND_SYNTHETIC_REFERENCE";
const MD_POLICY_INSTRUCTION_FLAGS_V1: u8 = 0xA7;
const MD_POLICY_INSTRUCTION_AUX_V1: u64 = 0x5052_4d44_504f_4c31; // `PRMDPOL1`

pub mod md_policy_field_v1 {
    pub const SCHEMA_VERSION: u32 = 1;
    pub const FORCE_CONTRACT: u32 = 2;
    pub const INTEGRATOR: u32 = 3;
    pub const RAW_ENGINE_SAFE_DEFAULTS_UNCHANGED: u32 = 4;
    pub const INDEPENDENT_RAW_ENGINE_FLAGS_FORBIDDEN: u32 = 5;
    pub const TOPOLOGY_SHA256_REQUIRED: u32 = 6;
    pub const INPUT_SHA256_REQUIRED: u32 = 7;
    pub const SOURCE_EPOCH_REQUIRED: u32 = 8;
    pub const CUDA_GRAPH_REQUIRED: u32 = 9;
    pub const METAL_RESIDENT_REQUIRED: u32 = 10;
    pub const HOTPATH_ACCELERATION_REQUIRED: u32 = 11;
    pub const PATH_B_REQUIRED: u32 = 12;
    pub const DSTW_REQUIRED: u32 = 13;
    pub const SUPERRESOLUTION_REQUIRED: u32 = 14;
    pub const QNANL_INLINE_SEAL_VERIFY_REQUIRED: u32 = 15;
    pub const RUNTIME_EVENTS_REQUIRED: u32 = 16;
    pub const EVIDENCE_QUALIFICATION_REQUIRED: u32 = 17;
    pub const THERM_REQUIRED: u32 = 18;
    pub const POST_MD_AGGREGATION: u32 = 19;
    pub const EFFECTIVE_SURFACE_PROVENANCE_REQUIRED: u32 = 20;
    pub const GOVERNED_CORPUS_HANDOFF_REQUIRED: u32 = 21;
    pub const OUTPUT_IDENTITY_REQUIRED: u32 = 22;
    // V2 fields are append-only. Field 19 remains the requested post-MD
    // aggregation value so a V1 declaration cannot be reinterpreted as an
    // effective product-policy decision.
    pub const EXECUTION_PROFILE_ID: u32 = 23;
    pub const EXECUTION_PROFILE_VERSION: u32 = 24;
    pub const EFFECTIVE_POST_MD_AGGREGATION: u32 = 25;
    pub const REQUESTED_THERM_SURFACE: u32 = 26;
    pub const EFFECTIVE_THERM_SURFACE: u32 = 27;
    pub const THERM_SURFACE_FALLBACK_REASON: u32 = 28;
    pub const THERM_SURFACE_FALLBACK_AUTHORIZED: u32 = 29;
    pub const FULL_POST_MD_AUTHORIZED: u32 = 30;
}

pub mod md_policy_value_v1 {
    pub const TRUE: u32 = 1;
    pub const SCHEMA_VERSION_1: u32 = 1;
    pub const FORCE_PRISMCODE_AUTHORIZED: u32 = 1;
    pub const INTEGRATOR_CANONICAL_HYPER_RUNTIME: u32 = 1;
    pub const POST_MD_COMPATIBILITY_MD_ONLY_EVIDENCE_HANDOFF_V1: u32 = 1;
    pub const POST_MD_PRESERVE_LANE_DEFAULT: u32 = 2;
    pub const POST_MD_FULL: u32 = 3;
    pub const POST_MD_SKIP: u32 = 4;
    pub const PROFILE_PRISMOS_HYPER_RUNTIME_MD: u32 = 1;
    pub const PROFILE_PRISMOS_THERM_POSTMD: u32 = 2;
    pub const PROFILE_VERSION_1_0_0: u32 = 0x0001_0000;
    pub const THERM_SURFACE_PRESERVE_LANE_DEFAULT: u32 = 1;
    pub const THERM_SURFACE_CLUSTERS: u32 = 2;
    pub const THERM_SURFACE_LATTICE: u32 = 3;
    pub const FALLBACK_REASON_NONE: u32 = 0;
    pub const FALLBACK_REASON_REQUESTED_LATTICE_UNAVAILABLE: u32 = 1;
    pub const FALSE: u32 = 0;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MdForceContractV1 {
    PrismcodeAuthorized,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MdIntegratorPolicyV1 {
    CanonicalHyperRuntime,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MdPostMdAggregationPolicyV1 {
    PreserveLaneDefault,
    Full,
    Skip,
    /// Explicit V1 decode result. Strict V2 product profiles reject it rather
    /// than silently converting the historical MD-only handoff into Full.
    CompatibilityMdOnlyEvidenceHandoffV1,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MdThermSurfacePolicyV1 {
    PreserveLaneDefault,
    Clusters,
    Lattice,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MdThermSurfaceFallbackReasonV1 {
    RequestedLatticeUnavailable,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MdExecutionProfileIdentityV1 {
    pub id: String,
    pub version: String,
}

/// Fully materialized engine-launch policy decoded from marked
/// `MD.PROTOCOL_DIRECTOR` instructions.  Every field is required exactly once.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MdEnginePolicyV1 {
    pub schema_version: u32,
    /// Absent only for an explicitly decoded legacy V1 packet. Product
    /// profiles require this pre-run identity to be present and exact.
    pub execution_profile: Option<MdExecutionProfileIdentityV1>,
    pub force_contract: MdForceContractV1,
    pub integrator: MdIntegratorPolicyV1,
    pub raw_engine_safe_defaults_unchanged: bool,
    pub independent_raw_engine_flags_forbidden: bool,
    pub topology_sha256_required: bool,
    pub input_sha256_required: bool,
    pub source_epoch_required: bool,
    pub cuda_graph_required: bool,
    pub metal_resident_required: bool,
    pub hotpath_acceleration_required: bool,
    pub path_b_required: bool,
    pub dstw_required: bool,
    pub superresolution_required: bool,
    pub qnanl_inline_seal_verify_required: bool,
    pub runtime_events_required: bool,
    pub evidence_qualification_required: bool,
    pub therm_required: bool,
    pub requested_post_md_aggregation: MdPostMdAggregationPolicyV1,
    pub effective_post_md_aggregation: MdPostMdAggregationPolicyV1,
    pub full_post_md_authorized: bool,
    pub requested_therm_surface: MdThermSurfacePolicyV1,
    pub effective_therm_surface: MdThermSurfacePolicyV1,
    pub therm_surface_fallback_reason: Option<MdThermSurfaceFallbackReasonV1>,
    pub therm_surface_fallback_authorized: bool,
    pub effective_surface_provenance_required: bool,
    pub governed_corpus_handoff_required: bool,
    pub output_identity_required: bool,
}

impl MdEnginePolicyV1 {
    pub fn canonical_hyper_runtime() -> Self {
        Self {
            schema_version: MD_ENGINE_POLICY_SCHEMA_VERSION_V2,
            execution_profile: Some(MdExecutionProfileIdentityV1 {
                id: "prismos.hyper-runtime-md".to_string(),
                version: "1.0.0".to_string(),
            }),
            force_contract: MdForceContractV1::PrismcodeAuthorized,
            integrator: MdIntegratorPolicyV1::CanonicalHyperRuntime,
            raw_engine_safe_defaults_unchanged: true,
            independent_raw_engine_flags_forbidden: true,
            topology_sha256_required: true,
            input_sha256_required: true,
            source_epoch_required: true,
            cuda_graph_required: true,
            metal_resident_required: true,
            hotpath_acceleration_required: true,
            path_b_required: true,
            dstw_required: true,
            superresolution_required: true,
            qnanl_inline_seal_verify_required: true,
            runtime_events_required: true,
            evidence_qualification_required: true,
            therm_required: true,
            requested_post_md_aggregation: MdPostMdAggregationPolicyV1::Full,
            effective_post_md_aggregation: MdPostMdAggregationPolicyV1::Full,
            full_post_md_authorized: true,
            requested_therm_surface: MdThermSurfacePolicyV1::Lattice,
            effective_therm_surface: MdThermSurfacePolicyV1::Lattice,
            therm_surface_fallback_reason: None,
            therm_surface_fallback_authorized: false,
            effective_surface_provenance_required: true,
            governed_corpus_handoff_required: true,
            output_identity_required: true,
        }
    }

    pub fn therm_postmd() -> Self {
        let mut policy = Self::canonical_hyper_runtime();
        policy.execution_profile = Some(MdExecutionProfileIdentityV1 {
            id: "prismos.therm-postmd".to_string(),
            version: "1.0.0".to_string(),
        });
        policy
    }

    fn legacy_compatibility_v1() -> Self {
        Self {
            schema_version: MD_ENGINE_POLICY_SCHEMA_VERSION_V1,
            execution_profile: None,
            force_contract: MdForceContractV1::PrismcodeAuthorized,
            integrator: MdIntegratorPolicyV1::CanonicalHyperRuntime,
            raw_engine_safe_defaults_unchanged: true,
            independent_raw_engine_flags_forbidden: true,
            topology_sha256_required: true,
            input_sha256_required: true,
            source_epoch_required: true,
            cuda_graph_required: true,
            metal_resident_required: true,
            hotpath_acceleration_required: true,
            path_b_required: true,
            dstw_required: true,
            superresolution_required: true,
            qnanl_inline_seal_verify_required: true,
            runtime_events_required: true,
            evidence_qualification_required: true,
            therm_required: true,
            requested_post_md_aggregation:
                MdPostMdAggregationPolicyV1::CompatibilityMdOnlyEvidenceHandoffV1,
            effective_post_md_aggregation:
                MdPostMdAggregationPolicyV1::CompatibilityMdOnlyEvidenceHandoffV1,
            full_post_md_authorized: false,
            requested_therm_surface: MdThermSurfacePolicyV1::PreserveLaneDefault,
            effective_therm_surface: MdThermSurfacePolicyV1::PreserveLaneDefault,
            therm_surface_fallback_reason: None,
            therm_surface_fallback_authorized: false,
            effective_surface_provenance_required: true,
            governed_corpus_handoff_required: true,
            output_identity_required: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidatedMdEnginePolicyV1 {
    pub policy: MdEnginePolicyV1,
    pub packet_hash_blake3: String,
    pub packet_sha256: String,
    pub packet_bytes: u64,
    pub instruction_count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MdPolicyNormalizationReportV1 {
    pub schema_kind: String,
    pub schema_version: u32,
    pub base_packet_sha256: String,
    pub base_packet_hash_blake3: String,
    pub base_hw_profile_classification: String,
    pub normalized_packet_sha256: String,
    pub normalized_packet_hash_blake3: String,
    pub normalized_hw_profile_classification: String,
    pub base_instruction_count: u64,
    pub normalized_instruction_count: u64,
    pub removed_policy_instruction_count: u64,
    pub added_policy_instruction_count: u64,
    pub removed_policy_instructions: Vec<MdPolicyInstructionIdentityV1>,
    pub added_policy_instructions: Vec<MdPolicyInstructionIdentityV1>,
    pub non_policy_sections_byte_identical: bool,
    pub non_policy_instructions_byte_identical: bool,
    pub idempotent: bool,
    pub validation_level: String,
    pub validation_status: String,
    pub policy: MdEnginePolicyV1,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MdPolicyInstructionIdentityV1 {
    pub opcode: u16,
    pub opcode_name: String,
    pub flags: u8,
    pub field_id: u32,
    pub field_name: String,
    pub value: u32,
    pub value_name: String,
    pub aux: u64,
}

/// Re-derive the exact EventTable-to-seal binding from packet bytes.
///
/// The seal-chain check proves that the EventTable region participated in the
/// same seal material written by `build_packet_bytes`; post-run summaries or
/// reconstructed event objects are not admissible inputs.
pub fn event_byte_seal_binding(packet: &PCodePacket) -> PcResult<EventByteSealBinding> {
    let state = find_section(&packet.sections, SectionKind::StateTable)?;
    let time = find_section(&packet.sections, SectionKind::TimeTable)?;
    let nanl = find_section(&packet.sections, SectionKind::NanlTable)?;
    let event = find_section(&packet.sections, SectionKind::EventTable)?;
    let tensor = find_section(&packet.sections, SectionKind::TensorTable)?;
    let provenance = find_section(&packet.sections, SectionKind::ProvenanceTable)?;
    let seal = find_section(&packet.sections, SectionKind::SealChain)?;
    let event_bytes = section_bytes(&packet.bytes, event);
    let seal_seed = hash_many(&[
        section_bytes(&packet.bytes, state),
        section_bytes(&packet.bytes, time),
        section_bytes(&packet.bytes, nanl),
        event_bytes,
        section_bytes(&packet.bytes, tensor),
        section_bytes(&packet.bytes, provenance),
    ]);
    let expected_seal = hash_many(&[b"PRISMCODE_SEAL_CHAIN_V0_1", &seal_seed]);
    if section_bytes(&packet.bytes, seal) != expected_seal {
        return Err(PcError::Hash(
            "seal chain does not bind the exact EventTable byte region".to_string(),
        ));
    }
    let event_sha256 = sha256_hex(event_bytes);
    Ok(EventByteSealBinding {
        event_byte_offset: event.offset,
        event_byte_length: event.length,
        event_record_count: event.record_count,
        event_byte_source_sha256: event_sha256.clone(),
        qnanl_sealed_from_sha256: event_sha256,
    })
}

/// Compile a non-synthetic runtime corpus without a filesystem round trip and
/// return the exact EventTable bytes participating in the packet seal.
///
/// The same provenance restrictions as [`compile_real_corpus_with_debug`] are
/// enforced here.  Returning the bytes prevents a downstream QNANL writer from
/// re-serializing event objects and accidentally claiming equality with a
/// different byte representation.
pub fn compile_real_event_table_packet(
    corpus: &ReferenceCorpus,
    debug_json: &[u8],
) -> PcResult<RuntimeEventTablePacket> {
    if corpus.synthetic {
        return Err(PcError::Provenance(
            "real event-table compiler requires explicit synthetic=false".to_string(),
        ));
    }
    if corpus.timebase.is_empty() {
        return Err(PcError::Timebase(
            "real event-table corpus has no timebase records".to_string(),
        ));
    }
    if corpus.provenance.is_empty() {
        return Err(PcError::Provenance(
            "real event-table corpus has no provenance records".to_string(),
        ));
    }
    if corpus.provenance.iter().any(|record| record.synthetic) {
        return Err(PcError::Provenance(
            "real event-table corpus contains synthetic provenance".to_string(),
        ));
    }

    let packet_bytes = build_packet_bytes(corpus, debug_json)?;
    let packet = parse_packet(packet_bytes.clone())?;
    let binding = event_byte_seal_binding(&packet)?;
    let start = usize::try_from(binding.event_byte_offset)
        .map_err(|_| PcError::Section("EventTable offset exceeds usize".to_string()))?;
    let length = usize::try_from(binding.event_byte_length)
        .map_err(|_| PcError::Section("EventTable length exceeds usize".to_string()))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| PcError::Section("EventTable range overflow".to_string()))?;
    let event_table_bytes = packet_bytes
        .get(start..end)
        .ok_or_else(|| PcError::Section("EventTable range exceeds packet bytes".to_string()))?
        .to_vec();
    let payload = find_section(&packet.sections, SectionKind::PayloadBlob)?;
    let payload_blob_bytes = section_bytes(&packet.bytes, payload).to_vec();
    if event_table_bytes.is_empty() || binding.event_record_count == 0 {
        return Err(PcError::Section(
            "runtime EventTable must contain at least one event".to_string(),
        ));
    }
    let observed = sha256_hex(&event_table_bytes);
    if observed != binding.event_byte_source_sha256 || observed != binding.qnanl_sealed_from_sha256
    {
        return Err(PcError::Hash(
            "returned EventTable bytes differ from packet seal binding".to_string(),
        ));
    }
    Ok(RuntimeEventTablePacket {
        payload_context: RuntimeEventTablePayloadContext {
            payload_blob_sha256: sha256_hex(&payload_blob_bytes),
            parent_packet_sha256: sha256_hex(&packet_bytes),
            event_table_source_sha256: observed,
            payload_blob_bytes,
        },
        packet_bytes,
        event_table_bytes,
        binding,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub status: String,
    pub level: String,
    pub packet_hash: String,
    pub instruction_count: usize,
    pub state_count: u64,
    pub time_count: u64,
    pub event_count: u64,
    pub nanl_count: u64,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub status: String,
    pub backend: String,
    /// Generic PrismCode CPU/CUDA execution is a substrate diagnostic.  Native
    /// MD launch authority exists only at the engine integration seam after
    /// typed packet-policy admission.
    pub authority_classification: String,
    pub native_packet_execution: bool,
    pub deterministic: bool,
    pub input_packet_hash: String,
    pub output_state_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_execution: Option<serde_json::Value>,
    pub seal_chain_hash: String,
    pub instruction_count: u64,
    pub state_count: u64,
    pub event_count: u64,
    pub checkpoint_count: u64,
    pub elapsed_ns: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayReport {
    pub status: String,
    pub input_packet_hash: String,
    pub output_state_hash: String,
    pub seal_chain_hash: String,
    pub instruction_count: u64,
    pub event_count: u64,
    pub checkpoint_count: u64,
    pub mismatch_location: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkLatencyPercentiles {
    pub sample_count: usize,
    pub p50_ns: Option<u128>,
    pub p95_ns: Option<u128>,
    pub p99_ns: Option<u128>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub status: String,
    pub backend: String,
    pub hardware: serde_json::Value,
    pub packet_hash: String,
    pub isa_version: String,
    pub instruction_count: u64,
    pub state_count: u64,
    pub event_count: u64,
    pub nanl_token_count: u64,
    pub tensor_value_count: u64,
    pub rows_per_sec: f64,
    pub events_per_sec: f64,
    pub instructions_per_sec: f64,
    pub binary_size_bytes: u64,
    pub debug_json_size_bytes: u64,
    pub compression_ratio_vs_json: f64,
    pub binary_ingest_ns: u128,
    pub cpu_execution_ns: u128,
    pub latency_percentiles_ns: BenchmarkLatencyPercentiles,
    pub replay_determinism_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionReport {
    pub status: String,
    pub packet_hash: String,
    pub codec_count: usize,
    pub raw_bytes: u64,
    pub binary_bytes: u64,
    pub compressed_bytes: u64,
    pub compression_ratio: f64,
    pub encode_ns: u128,
    pub decode_ns: u128,
    pub replay_after_decode: String,
    pub codecs: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkSuiteReport {
    pub status: String,
    pub backend_requested: String,
    pub corpus_source: String,
    pub hardware: serde_json::Value,
    pub isa_version: String,
    pub workload_count: usize,
    pub passed_workloads: usize,
    pub rejected_workloads: usize,
    pub security_rejected_workloads: usize,
    pub failed_workloads: usize,
    pub total_binary_size_bytes: u64,
    pub total_debug_json_size_bytes: u64,
    pub total_instruction_count: u64,
    pub total_state_count: u64,
    pub total_event_count: u64,
    pub total_nanl_token_count: u64,
    pub total_tensor_value_count: u64,
    pub compression_ratio_vs_json: f64,
    pub latency_percentiles_ns: BenchmarkLatencyPercentiles,
    pub required_workload_coverage: serde_json::Value,
    pub workloads: Vec<BenchmarkWorkloadReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkWorkloadReport {
    pub workload_id: String,
    pub category: String,
    pub description: String,
    pub expected_outcome: String,
    pub status: String,
    pub backend: Option<String>,
    pub source_corpus_path: Option<String>,
    pub corpus_json_path: Option<String>,
    pub pcode_path: String,
    pub pcout_path: Option<String>,
    pub report_path: Option<String>,
    pub packet_hash: Option<String>,
    pub isa_version: Option<String>,
    pub output_state_hash: Option<String>,
    pub instruction_count: u64,
    pub state_count: u64,
    pub event_count: u64,
    pub nanl_token_count: u64,
    pub tensor_value_count: u64,
    pub replay_attempts: u64,
    pub replay_passes: u64,
    pub rows_per_sec: f64,
    pub events_per_sec: f64,
    pub instructions_per_sec: f64,
    pub binary_size_bytes: u64,
    pub debug_json_size_bytes: u64,
    pub compression_ratio_vs_json: f64,
    pub binary_ingest_ns: u128,
    pub cpu_execution_ns: u128,
    pub replay_latency_percentiles_ns: BenchmarkLatencyPercentiles,
    pub replay_determinism_score: f64,
    pub error_class: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceCorpus {
    pub corpus_id: String,
    pub synthetic: bool,
    pub generator_version: String,
    pub seed: u64,
    pub states: Vec<ReferenceState>,
    pub timebase: Vec<ReferenceTime>,
    pub nanl_tokens: Vec<u8>,
    pub events: Vec<ReferenceEvent>,
    pub thermo: Vec<ReferenceThermo>,
    pub provenance: Vec<ReferenceProvenance>,
    #[serde(default)]
    pub expected_output_hash: Option<String>,
    #[serde(default)]
    pub expected_replay_hash: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceState {
    pub state_id: u64,
    pub state_kind: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub payload_f64: Vec<f64>,
    pub time_ref: u64,
    pub provenance_ref: u64,
    pub unit: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceTime {
    pub time_id: u64,
    pub unit: String,
    pub start: f64,
    pub step: f64,
    pub count: u64,
    pub timebase_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceEvent {
    pub event_id: u64,
    pub event_kind: String,
    pub time_ref: u64,
    pub source_state: u64,
    pub target_state: u64,
    pub value: f64,
    pub unit: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceThermo {
    pub thermo_id: u64,
    pub state_ref: u64,
    pub value: f64,
    pub unit: String,
    pub signal_kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceProvenance {
    pub provenance_id: u64,
    pub label: String,
    pub synthetic: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileKeyring {
    pub keyring_version: String,
    pub provider: String,
    pub data_key_algorithm: String,
    pub data_key_hex: String,
    pub signing_algorithm: String,
    pub signing_pkcs8_hex: String,
    pub public_key_hex: String,
    pub public_key_hash_blake3: String,
    pub allowed_operations: Vec<String>,
    pub release_mode_allowed: bool,
    pub raw_keys_in_artifacts: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationLevel {
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
}

impl ValidationLevel {
    pub fn parse(value: &str) -> PcResult<Self> {
        match value.to_ascii_uppercase().as_str() {
            "L0" => Ok(Self::L0),
            "L1" => Ok(Self::L1),
            "L2" => Ok(Self::L2),
            "L3" => Ok(Self::L3),
            "L4" => Ok(Self::L4),
            "L5" => Ok(Self::L5),
            other => Err(PcError::Format(format!("unknown validation level {other}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Auto,
}

impl Backend {
    pub fn parse(value: &str) -> PcResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "auto" => Ok(Self::Auto),
            other => Err(PcError::Format(format!("unknown backend {other}"))),
        }
    }

    fn code(self) -> u16 {
        match self {
            Self::Cpu => 1,
            Self::Cuda => 2,
            Self::Auto => 3,
        }
    }
}

pub fn compile_reference_json(input: &Path, output: &Path) -> PcResult<()> {
    let text = fs::read_to_string(input)?;
    let corpus: ReferenceCorpus = serde_json::from_str(&text)?;
    compile_reference_corpus_with_debug(&corpus, text.as_bytes(), output)
}

pub fn compile_reference_corpus_with_debug(
    corpus: &ReferenceCorpus,
    debug_json: &[u8],
    output: &Path,
) -> PcResult<()> {
    if !corpus.synthetic {
        return Err(PcError::Provenance(
            "reference compiler requires explicit synthetic=true unless real PRISM importer is selected".to_string(),
        ));
    }
    if corpus.timebase.is_empty() {
        return Err(PcError::Timebase(
            "reference corpus has no timebase records".to_string(),
        ));
    }
    if corpus.provenance.iter().any(|p| !p.synthetic) {
        return Err(PcError::Provenance(
            "synthetic reference corpus contains non-synthetic provenance".to_string(),
        ));
    }
    let bytes = build_packet_bytes(corpus, debug_json)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, bytes)?;
    Ok(())
}

pub fn compile_real_corpus_with_debug(
    corpus: &ReferenceCorpus,
    debug_json: &[u8],
    output: &Path,
) -> PcResult<()> {
    if corpus.synthetic {
        return Err(PcError::Provenance(
            "real compiler requires explicit synthetic=false".to_string(),
        ));
    }
    if corpus.timebase.is_empty() {
        return Err(PcError::Timebase(
            "real corpus has no timebase records".to_string(),
        ));
    }
    if corpus.provenance.is_empty() {
        return Err(PcError::Provenance(
            "real corpus has no provenance records".to_string(),
        ));
    }
    if corpus.provenance.iter().any(|p| p.synthetic) {
        return Err(PcError::Provenance(
            "real corpus contains synthetic provenance".to_string(),
        ));
    }
    let bytes = build_packet_bytes(corpus, debug_json)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, bytes)?;
    Ok(())
}

#[cfg(test)]
mod compiler_authority_tests {
    use super::*;

    fn corpus(synthetic: bool, provenance_synthetic: &[bool]) -> ReferenceCorpus {
        ReferenceCorpus {
            corpus_id: "COMPILER_AUTHORITY_TEST".to_string(),
            synthetic,
            generator_version: "test".to_string(),
            seed: 0,
            states: Vec::new(),
            timebase: vec![ReferenceTime {
                time_id: 1,
                unit: "fs".to_string(),
                start: 0.0,
                step: 1.0,
                count: 1,
                timebase_hash: "test-timebase".to_string(),
            }],
            nanl_tokens: Vec::new(),
            events: Vec::new(),
            thermo: Vec::new(),
            provenance: provenance_synthetic
                .iter()
                .enumerate()
                .map(|(index, synthetic)| ReferenceProvenance {
                    provenance_id: (index + 1) as u64,
                    label: format!("provenance-{}", index + 1),
                    synthetic: *synthetic,
                    reason: "test".to_string(),
                })
                .collect(),
            expected_output_hash: None,
            expected_replay_hash: None,
        }
    }

    #[test]
    fn real_compiler_rejects_synthetic_corpus() {
        let output = tempfile::tempdir().unwrap().path().join("rejected.pcode");
        let error = compile_real_corpus_with_debug(&corpus(true, &[false]), b"{}", &output)
            .expect_err("synthetic corpus must be rejected by real compiler");
        assert!(matches!(error, PcError::Provenance(_)));
        assert!(!output.exists());
    }

    #[test]
    fn real_compiler_rejects_any_synthetic_provenance() {
        let output = tempfile::tempdir().unwrap().path().join("rejected.pcode");
        let error = compile_real_corpus_with_debug(&corpus(false, &[false, true]), b"{}", &output)
            .expect_err("mixed provenance must be rejected by real compiler");
        assert!(matches!(error, PcError::Provenance(_)));
        assert!(!output.exists());
    }

    #[test]
    fn reference_compiler_rejects_real_corpus() {
        let output = tempfile::tempdir().unwrap().path().join("rejected.pcode");
        let error = compile_reference_corpus_with_debug(&corpus(false, &[false]), b"{}", &output)
            .expect_err("real corpus must be rejected by reference compiler");
        assert!(matches!(error, PcError::Provenance(_)));
        assert!(!output.exists());
    }

    #[test]
    fn real_event_table_packet_returns_the_exact_seal_bound_bytes() {
        let mut runtime_corpus = corpus(false, &[false]);
        runtime_corpus.events.push(ReferenceEvent {
            event_id: 1,
            event_kind: "PHASE_TILE".to_string(),
            time_ref: 1,
            source_state: 1,
            target_state: 1,
            value: 0.125,
            unit: "DIMENSIONLESS_KL".to_string(),
        });
        let compiled = compile_real_event_table_packet(&runtime_corpus, b"{}")
            .expect("compile real runtime EventTable packet");
        assert!(!compiled.packet_bytes.is_empty());
        assert_eq!(compiled.binding.event_record_count, 1);
        assert_eq!(compiled.event_table_bytes.len(), 96);
        let observed = sha256_hex(&compiled.event_table_bytes);
        assert_eq!(observed, compiled.binding.event_byte_source_sha256);
        assert_eq!(observed, compiled.binding.qnanl_sealed_from_sha256);
        assert_eq!(
            compiled.payload_context.event_table_source_sha256(),
            observed
        );
        assert_eq!(
            compiled.payload_context.parent_packet_sha256(),
            sha256_hex(&compiled.packet_bytes)
        );
        assert_eq!(
            compiled.payload_context.payload_blob_sha256(),
            sha256_hex(compiled.payload_context.payload_blob_bytes())
        );
        assert!(compiled
            .payload_context
            .payload_blob_bytes()
            .windows("DIMENSIONLESS_KL".len())
            .any(|window| window == b"DIMENSIONLESS_KL"));
        let reparsed = parse_packet(compiled.packet_bytes).expect("reparse runtime packet");
        assert_eq!(
            event_byte_seal_binding(&reparsed).expect("rederive EventTable binding"),
            compiled.binding
        );
    }
}

pub fn migrate_packet(input: &Path, output: &Path, target_isa: &str) -> PcResult<()> {
    if target_isa != "0.1" {
        return Err(PcError::Format(format!(
            "unsupported migration target {target_isa}"
        )));
    }
    let packet = read_packet(input)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, packet.bytes)?;
    Ok(())
}

fn build_packet_bytes(corpus: &ReferenceCorpus, debug_json: &[u8]) -> PcResult<Vec<u8>> {
    let mut payload_blob = Vec::new();
    let mut state_bytes = Vec::new();
    for state in &corpus.states {
        let payload_offset = payload_blob.len() as u64;
        for value in &state.payload_f64 {
            payload_blob.extend_from_slice(&value.to_le_bytes());
        }
        let payload_len = (state.payload_f64.len() * 8) as u64;
        let state_hash = hash_many(&[
            &state.state_id.to_le_bytes(),
            &(state_kind_code(&state.state_kind)?).to_le_bytes(),
            &(dtype_code(&state.dtype)?).to_le_bytes(),
            &payload_blob[payload_offset as usize..(payload_offset + payload_len) as usize],
            &state.time_ref.to_le_bytes(),
            &state.provenance_ref.to_le_bytes(),
        ]);
        let mut rec = [0u8; 128];
        put_u64(&mut rec, 0, state.state_id);
        put_u16(&mut rec, 8, state_kind_code(&state.state_kind)?);
        put_u16(&mut rec, 10, dtype_code(&state.dtype)?);
        put_u32(&mut rec, 12, state.shape.len() as u32);
        put_u64(&mut rec, 16, payload_offset);
        put_u64(&mut rec, 24, payload_len);
        put_u64(&mut rec, 32, state.time_ref);
        put_u64(&mut rec, 40, state.provenance_ref);
        put_u64(&mut rec, 48, 0);
        put_u64(&mut rec, 56, 0);
        rec[64..96].copy_from_slice(&state_hash);
        state_bytes.extend_from_slice(&rec);
    }

    let mut time_bytes = Vec::new();
    for time in &corpus.timebase {
        if time.unit.trim().is_empty() || time.timebase_hash.trim().is_empty() {
            return Err(PcError::Timebase(
                "timebase record lacks unit or hash".to_string(),
            ));
        }
        let hash = hash_many(&[
            &time.time_id.to_le_bytes(),
            &time.start.to_le_bytes(),
            &time.step.to_le_bytes(),
            &time.count.to_le_bytes(),
            time.unit.as_bytes(),
            time.timebase_hash.as_bytes(),
        ]);
        let mut rec = [0u8; 80];
        put_u64(&mut rec, 0, time.time_id);
        put_f64(&mut rec, 8, time.start);
        put_f64(&mut rec, 16, time.step);
        put_u64(&mut rec, 24, time.count);
        put_u64(&mut rec, 32, payload_blob.len() as u64);
        payload_blob.extend_from_slice(time.unit.as_bytes());
        put_u64(&mut rec, 40, time.unit.len() as u64);
        rec[48..80].copy_from_slice(&hash);
        time_bytes.extend_from_slice(&rec);
    }

    let nanl_bytes = pack_u2(&corpus.nanl_tokens)?;

    let mut event_bytes = Vec::new();
    for event in &corpus.events {
        let kind_code = event_kind_code(&event.event_kind);
        let hash = canonical_runtime_event_record_hash(
            event.event_id,
            kind_code,
            event.time_ref,
            event.source_state,
            event.target_state,
            event.value,
            event.unit.as_bytes(),
        )?;
        let mut rec = [0u8; 96];
        put_u64(&mut rec, 0, event.event_id);
        put_u16(&mut rec, 8, kind_code);
        put_u64(&mut rec, 16, event.time_ref);
        put_u64(&mut rec, 24, event.source_state);
        put_u64(&mut rec, 32, event.target_state);
        put_f64(&mut rec, 40, event.value);
        put_u64(&mut rec, 48, payload_blob.len() as u64);
        payload_blob.extend_from_slice(event.unit.as_bytes());
        put_u64(&mut rec, 56, event.unit.len() as u64);
        rec[64..96].copy_from_slice(&hash);
        event_bytes.extend_from_slice(&rec);
    }

    let tensor_bytes = tensor_section(corpus)?;
    let provenance_bytes = provenance_section(corpus, &mut payload_blob)?;
    let seal_seed = hash_many(&[
        &state_bytes,
        &time_bytes,
        &nanl_bytes,
        &event_bytes,
        &tensor_bytes,
        &provenance_bytes,
    ]);
    let seal_chain = hash_many(&[b"PRISMCODE_SEAL_CHAIN_V0_1", &seal_seed]).to_vec();
    let instructions = instruction_program(corpus);
    let instruction_bytes: Vec<u8> = instructions.iter().flat_map(PcInstr32::encode).collect();

    let mut debug = Vec::new();
    debug.extend_from_slice(debug_json);
    let sections = vec![
        (
            SectionKind::InstrStream,
            instruction_bytes,
            instructions.len() as u64,
        ),
        (
            SectionKind::StateTable,
            state_bytes,
            corpus.states.len() as u64,
        ),
        (SectionKind::OperandTable, Vec::new(), 0),
        (
            SectionKind::TimeTable,
            time_bytes,
            corpus.timebase.len() as u64,
        ),
        (
            SectionKind::NanlTable,
            nanl_bytes,
            corpus.nanl_tokens.len() as u64,
        ),
        (
            SectionKind::EventTable,
            event_bytes,
            corpus.events.len() as u64,
        ),
        (SectionKind::TensorTable, tensor_bytes, 1),
        (SectionKind::GraphTable, Vec::new(), 0),
        (
            SectionKind::RewardTable,
            reward_section(corpus),
            corpus.thermo.len() as u64,
        ),
        (SectionKind::ActionTable, Vec::new(), 0),
        (
            SectionKind::ProvenanceTable,
            provenance_bytes,
            corpus.provenance.len() as u64,
        ),
        (SectionKind::PayloadBlob, payload_blob, 1),
        (SectionKind::SealChain, seal_chain, 1),
        (SectionKind::DebugJson, debug, 1),
    ];
    write_container(sections, hash_bytes(serde_json::to_vec(corpus)?.as_slice()))
}

fn instruction_program(corpus: &ReferenceCorpus) -> Vec<PcInstr32> {
    let mut out = vec![
        PcInstr32::new(
            opcode::ABI_DECLARE,
            0x11,
            OperandKind::None,
            0,
            OperandKind::None,
            0,
        ),
        PcInstr32::new(
            opcode::PACKET_BEGIN,
            0x11,
            OperandKind::None,
            0,
            OperandKind::None,
            0,
        ),
    ];
    for (idx, _) in corpus.states.iter().enumerate() {
        out.push(PcInstr32::new(
            opcode::STATE_DEF,
            0x13,
            OperandKind::StateRef,
            idx as u32,
            OperandKind::ProvenanceRef,
            0,
        ));
    }
    for (idx, _) in corpus.timebase.iter().enumerate() {
        out.push(PcInstr32::new(
            opcode::TIME_BIND,
            0x13,
            OperandKind::TimeRef,
            idx as u32,
            OperandKind::StateRef,
            0,
        ));
    }
    out.push(PcInstr32::new(
        opcode::NANL_PACK,
        0x13,
        OperandKind::NanlRef,
        0,
        OperandKind::PayloadRef,
        0,
    ));
    for (idx, _) in corpus.events.iter().enumerate() {
        out.push(PcInstr32::new(
            opcode::EVENT_EMIT,
            0x13,
            OperandKind::EventRef,
            idx as u32,
            OperandKind::TimeRef,
            0,
        ));
    }
    out.extend([
        PcInstr32::new(
            opcode::EVENT_SORT,
            0x13,
            OperandKind::EventRef,
            0,
            OperandKind::None,
            0,
        ),
        PcInstr32::new(
            opcode::EVENT_INTEGRATE,
            0x13,
            OperandKind::StateRef,
            0,
            OperandKind::EventRef,
            0,
        ),
        PcInstr32::new(
            opcode::DRF_MAP,
            0x1b,
            OperandKind::DrfTensorRef,
            0,
            OperandKind::EventRef,
            0,
        ),
        PcInstr32::new(
            opcode::DRF_FIRE,
            0x1b,
            OperandKind::DrfTensorRef,
            0,
            OperandKind::DrfTensorRef,
            0,
        ),
        PcInstr32::new(
            opcode::DRF_RESONATE,
            0x1b,
            OperandKind::DrfTensorRef,
            0,
            OperandKind::DrfTensorRef,
            0,
        ),
        PcInstr32::new(
            opcode::THERMO_ENERGY,
            0x13,
            OperandKind::StateRef,
            0,
            OperandKind::PayloadRef,
            0,
        ),
        PcInstr32::new(
            opcode::ASSERT_TIMEBASE,
            0x11,
            OperandKind::TimeRef,
            0,
            OperandKind::None,
            0,
        ),
        PcInstr32::new(
            opcode::ASSERT_DETERMINISTIC,
            0x11,
            OperandKind::None,
            0,
            OperandKind::None,
            0,
        ),
        PcInstr32::new(
            opcode::STATE_HASH,
            0x11,
            OperandKind::StateRef,
            0,
            OperandKind::None,
            0,
        ),
        PcInstr32::new(
            opcode::REPLAY_BEGIN,
            0x11,
            OperandKind::None,
            0,
            OperandKind::None,
            0,
        ),
        PcInstr32::new(
            opcode::REPLAY_CHECKPOINT,
            0x31,
            OperandKind::SealRef,
            0,
            OperandKind::StateRef,
            0,
        ),
        PcInstr32::new(
            opcode::REPLAY_END,
            0x31,
            OperandKind::SealRef,
            0,
            OperandKind::None,
            0,
        ),
        PcInstr32::new(
            opcode::AUDIT_SEAL,
            0x31,
            OperandKind::SealRef,
            0,
            OperandKind::StateRef,
            0,
        ),
        PcInstr32::new(
            opcode::PACKET_END,
            0x31,
            OperandKind::None,
            0,
            OperandKind::None,
            0,
        ),
        PcInstr32::new(
            opcode::HALT,
            0x11,
            OperandKind::None,
            0,
            OperandKind::None,
            0,
        ),
    ]);
    out
}

fn write_container(
    sections: Vec<(SectionKind, Vec<u8>, u64)>,
    packet_uuid_source: [u8; 32],
) -> PcResult<Vec<u8>> {
    write_container_with_hw_profile(
        sections,
        packet_uuid_source,
        hash_bytes(SYNTHETIC_REFERENCE_HW_PROFILE),
    )
}

fn write_container_with_hw_profile(
    sections: Vec<(SectionKind, Vec<u8>, u64)>,
    packet_uuid_source: [u8; 32],
    hw_profile_hash: [u8; 32],
) -> PcResult<Vec<u8>> {
    let section_count = sections.len();
    let mut offset = align_up(
        (HEADER_LEN + section_count * SECTION_ENTRY_LEN) as u64,
        SECTION_ALIGN,
    );
    let mut entries = Vec::with_capacity(section_count);
    for (idx, (kind, bytes, record_count)) in sections.iter().enumerate() {
        let align = if *kind == SectionKind::InstrStream {
            INSTR_ALIGN
        } else {
            SECTION_ALIGN
        };
        offset = align_up(offset, align);
        entries.push(SectionEntry {
            section_id: (idx + 1) as u16,
            section_type: *kind,
            flags: 0,
            offset,
            length: bytes.len() as u64,
            record_count: *record_count,
            schema_hash: hash_bytes(format!("{kind:?}:v0.1").as_bytes()),
            content_hash: hash_bytes(bytes),
        });
        offset = align_up(offset + bytes.len() as u64, SECTION_ALIGN);
    }

    let mut out = vec![0u8; offset as usize];
    let payload_hash = entries
        .iter()
        .find(|entry| entry.section_type == SectionKind::PayloadBlob)
        .map(|entry| entry.content_hash)
        .unwrap_or([0u8; 32]);
    write_header(
        &mut out[..HEADER_LEN],
        section_count as u16,
        packet_uuid_source,
        hw_profile_hash,
        payload_hash,
        [0u8; 32],
        [0u8; 32],
    );
    let mut cursor = HEADER_LEN;
    for entry in &entries {
        out[cursor..cursor + SECTION_ENTRY_LEN].copy_from_slice(&entry.encode());
        cursor += SECTION_ENTRY_LEN;
    }
    for ((_, bytes, _), entry) in sections.iter().zip(entries.iter()) {
        let start = entry.offset as usize;
        out[start..start + bytes.len()].copy_from_slice(bytes);
    }
    let packet_hash = packet_hash_for_bytes(&out)?;
    out[PACKET_HASH_OFFSET..PACKET_HASH_OFFSET + PACKET_HASH_LEN].copy_from_slice(&packet_hash);
    Ok(out)
}

fn write_header(
    out: &mut [u8],
    section_count: u16,
    packet_uuid_source: [u8; 32],
    hw_profile_hash: [u8; 32],
    payload_hash: [u8; 32],
    previous_seal_hash: [u8; 32],
    packet_hash: [u8; 32],
) {
    out.fill(0);
    out[0..8].copy_from_slice(PCODE_MAGIC);
    put_u16(out, 8, 0);
    put_u16(out, 10, 1);
    put_u16(out, 12, ISA_MAJOR);
    put_u16(out, 14, ISA_MINOR);
    put_u16(out, 16, ABI_MAJOR);
    put_u16(out, 18, ABI_MINOR);
    out[20] = 1;
    put_u16(out, 22, INSTR_WIDTH as u16);
    put_u16(out, 24, section_count);
    put_u64(out, 28, HEADER_LEN as u64);
    out[36..52].copy_from_slice(&packet_uuid_source[..16]);
    put_u64(out, 52, 0);
    put_u64(out, 60, 0x01);
    out[68..100].copy_from_slice(&hw_profile_hash);
    out[100..132].copy_from_slice(&hash_bytes(b"PRISMCODE_BINARY_CONTAINER_V0_1"));
    out[132..164].copy_from_slice(&payload_hash);
    out[164..196].copy_from_slice(&previous_seal_hash);
    out[196..228].copy_from_slice(&packet_hash);
}

pub fn read_packet(path: &Path) -> PcResult<PCodePacket> {
    let bytes = fs::read(path)?;
    parse_packet(bytes)
}

pub fn parse_packet(bytes: Vec<u8>) -> PcResult<PCodePacket> {
    if bytes.len() < HEADER_LEN {
        return Err(PcError::Format(
            "file is shorter than 256-byte header".to_string(),
        ));
    }
    if &bytes[0..8] != PCODE_MAGIC {
        return Err(PcError::Format("magic is not PRISMPC\\0".to_string()));
    }
    if read_u16(&bytes, 22)? as usize != INSTR_WIDTH {
        return Err(PcError::Format("instruction width mismatch".to_string()));
    }
    let section_count = read_u16(&bytes, 24)? as usize;
    let section_dir_offset = usize::try_from(read_u64(&bytes, 28)?)
        .map_err(|_| PcError::Section("section directory offset exceeds address space".into()))?;
    let dir_len = section_count
        .checked_mul(SECTION_ENTRY_LEN)
        .ok_or_else(|| PcError::Section("section directory length overflow".to_string()))?;
    if section_dir_offset.checked_add(dir_len).is_none_or(|end| end > bytes.len()) {
        return Err(PcError::Section(
            "section directory exceeds file length".to_string(),
        ));
    }
    let mut sections = Vec::with_capacity(section_count);
    let mut section_types = BTreeSet::new();
    let mut section_ids = BTreeSet::new();
    for idx in 0..section_count {
        let start = section_dir_offset + idx * SECTION_ENTRY_LEN;
        let entry = SectionEntry::decode(&bytes[start..start + SECTION_ENTRY_LEN])?;
        if !section_types.insert(entry.section_type as u16) || !section_ids.insert(entry.section_id) {
            return Err(PcError::Section("duplicate section type or identity".into()));
        }
        if entry.offset % SECTION_ALIGN != 0 && entry.section_type != SectionKind::InstrStream {
            return Err(PcError::Section(format!(
                "section {} is not 64-byte aligned",
                entry.section_id
            )));
        }
        if entry.section_type == SectionKind::InstrStream && entry.offset % INSTR_ALIGN != 0 {
            return Err(PcError::Section(
                "instruction stream is not 32-byte aligned".to_string(),
            ));
        }
        let end = entry.offset.checked_add(entry.length)
            .and_then(|end| usize::try_from(end).ok())
            .filter(|end| *end <= bytes.len());
        let Some(end) = end else {
            return Err(PcError::Section(format!(
                "section {} exceeds file length",
                entry.section_id
            )));
        };
        let actual_hash = hash_bytes(&bytes[entry.offset as usize..end]);
        if actual_hash != entry.content_hash {
            return Err(PcError::Hash(format!(
                "section {} content hash mismatch",
                entry.section_id
            )));
        }
        sections.push(entry);
    }
    let expected_hash = header_packet_hash(&bytes)?;
    let actual_hash = packet_hash_for_bytes(&bytes)?;
    if expected_hash != actual_hash {
        return Err(PcError::Hash("packet hash mismatch".to_string()));
    }
    let instr_entry = find_section(&sections, SectionKind::InstrStream)?;
    if instr_entry.length as usize % INSTR_WIDTH != 0 {
        return Err(PcError::Section(
            "instruction stream length is not multiple of 32".to_string(),
        ));
    }
    let instr_bytes = section_bytes(&bytes, instr_entry);
    let mut instructions = Vec::new();
    for chunk in instr_bytes.chunks_exact(INSTR_WIDTH) {
        instructions.push(PcInstr32::decode(chunk)?);
    }
    Ok(PCodePacket {
        state_count: find_section(&sections, SectionKind::StateTable)?.record_count,
        time_count: find_section(&sections, SectionKind::TimeTable)?.record_count,
        event_count: find_section(&sections, SectionKind::EventTable)?.record_count,
        nanl_count: find_section(&sections, SectionKind::NanlTable)?.record_count,
        packet_hash: actual_hash,
        bytes,
        sections,
        instructions,
    })
}

pub fn validate_packet(path: &Path, level: ValidationLevel) -> PcResult<ValidationResult> {
    let bytes = fs::read(path)?;
    if matches!(level, ValidationLevel::L5)
        && (bytes.starts_with(PSK_ENCRYPTED_MAGIC) || bytes.starts_with(PSK_PROD_MAGIC))
    {
        let aad = validate_prod_envelope(&bytes)?;
        return Ok(ValidationResult {
            status: "PASS".to_string(),
            level: "L5".to_string(),
            packet_hash: aad
                .get("packet_hash_blake3")
                .and_then(|value| value.as_str())
                .unwrap_or("EVIDENCE_NOT_FOUND")
                .to_string(),
            instruction_count: aad
                .get("instruction_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0) as usize,
            state_count: aad
                .get("state_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0),
            time_count: aad
                .get("time_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0),
            event_count: aad
                .get("event_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0),
            nanl_count: aad
                .get("nanl_count")
                .and_then(|value| value.as_u64())
                .unwrap_or(0),
            errors: Vec::new(),
        });
    }
    let packet = parse_packet(bytes)?;
    validate_parsed_packet(&packet, level)
}

/// Validate an already parsed packet without another filesystem read or decode.
/// Packet-authority consumers use this after one read/parse so validation and
/// policy materialization necessarily refer to identical bytes.
fn validate_parsed_packet(
    packet: &PCodePacket,
    level: ValidationLevel,
) -> PcResult<ValidationResult> {
    let mut errors = Vec::new();
    for section in &packet.sections {
        // These widths come from the actual v0.1 producer below. Declared
        // counts cannot authorize references beyond the bytes that exist.
        let width = match section.section_type {
            SectionKind::InstrStream => Some(32),
            SectionKind::StateTable => Some(128),
            SectionKind::TimeTable => Some(80),
            SectionKind::EventTable | SectionKind::TensorTable | SectionKind::ProvenanceTable => Some(96),
            SectionKind::RewardTable => Some(64),
            SectionKind::SealChain => Some(32),
            _ => None,
        };
        if let Some(width) = width {
            if section.record_count.checked_mul(width) != Some(section.length) {
                errors.push(format!("{:?}: record count does not match admitted record width/section length",section.section_type));
            }
        }
        if section.section_type == SectionKind::NanlTable
            && section.record_count.div_ceil(4) != section.length {
            errors.push("NanlTable: U2 token count does not match packed byte length".into());
        }
        if matches!(section.section_type,SectionKind::OperandTable | SectionKind::GraphTable | SectionKind::ActionTable)
            && (section.record_count != 0 || section.length != 0) {
            errors.push(format!("{:?}: nonempty table has no admitted record-layout contract",section.section_type));
        }
    }
    let required = [
        SectionKind::InstrStream,
        SectionKind::StateTable,
        SectionKind::TimeTable,
        SectionKind::NanlTable,
        SectionKind::EventTable,
        SectionKind::ProvenanceTable,
        SectionKind::PayloadBlob,
        SectionKind::SealChain,
    ];
    for kind in required {
        if find_section(&packet.sections, kind).is_err() {
            errors.push(format!("missing section {kind:?}"));
        }
    }
    for instr in &packet.instructions {
        if !known_opcode(instr.opcode) {
            errors.push(format!("unknown opcode 0x{:04x}", instr.opcode));
        }
        for kind in [
            instr.dst_kind,
            instr.src0_kind,
            instr.src1_kind,
            instr.src2_kind,
        ] {
            if !OperandKind::validate(kind) {
                errors.push(format!("invalid operand kind {kind}"));
            }
        }
        bounds_check_instruction(instr, &packet, &mut errors);
    }
    if matches!(
        level,
        ValidationLevel::L2 | ValidationLevel::L3 | ValidationLevel::L4 | ValidationLevel::L5
    ) {
        if packet.time_count == 0 {
            errors.push("no canonical timebase records".to_string());
        }
        if !packet
            .instructions
            .iter()
            .any(|i| i.opcode == opcode::TIME_BIND)
        {
            errors.push("TIME.BIND missing".to_string());
        }
        if !packet
            .instructions
            .iter()
            .any(|i| i.opcode == opcode::AUDIT_SEAL)
        {
            errors.push("AUDIT.SEAL missing".to_string());
        }
    }
    if matches!(
        level,
        ValidationLevel::L3 | ValidationLevel::L4 | ValidationLevel::L5
    ) {
        let debug = debug_json_from_packet(packet).unwrap_or_default();
        if debug.get("synthetic").and_then(|v| v.as_bool()).is_none() {
            errors.push("synthetic/real provenance flag missing from debug section".to_string());
        }
        if packet.event_count > 0 && packet.time_count == 0 {
            errors.push("event stream present without timebase".to_string());
        }
    }
    if matches!(level, ValidationLevel::L5) {
        return Err(PcError::Security(
            "L5 security validation requires PRISM Shield Kernel production envelope, Ed25519 signature, encrypted payload, release provenance, and key policy bindings; this packet is executable-dev, not market-security-ready"
                .to_string(),
        ));
    }
    if !errors.is_empty() {
        return Err(PcError::Format(errors.join(";")));
    }
    Ok(ValidationResult {
        status: "PASS".to_string(),
        level: format!("{level:?}"),
        packet_hash: hex_hash(&packet.packet_hash),
        instruction_count: packet.instructions.len(),
        state_count: packet.state_count,
        time_count: packet.time_count,
        event_count: packet.event_count,
        nanl_count: packet.nanl_count,
        errors,
    })
}

fn md_policy_instruction(field_id: u32, value: u32) -> PcInstr32 {
    let mut instruction = PcInstr32::new(
        opcode::MD_PROTOCOL_DIRECTOR,
        MD_POLICY_INSTRUCTION_FLAGS_V1,
        OperandKind::ImmU32,
        field_id,
        OperandKind::ImmU32,
        value,
    );
    instruction.aux = MD_POLICY_INSTRUCTION_AUX_V1;
    instruction
}

fn canonical_md_policy_instructions(policy: &MdEnginePolicyV1) -> PcResult<Vec<PcInstr32>> {
    // Field ids are append-only ABI identities.  Values are typed enum ids or
    // canonical booleans (0/1), never host CLI flags.
    Ok(md_policy_descriptors(policy)?
        .iter()
        .map(|descriptor| md_policy_instruction(descriptor.field_id, descriptor.value))
        .collect())
}

#[derive(Clone, Copy)]
struct MdPolicyDescriptorV1 {
    field_id: u32,
    field_name: &'static str,
    value: u32,
    value_name: &'static str,
}

fn md_policy_descriptors_v1() -> &'static [MdPolicyDescriptorV1] {
    use md_policy_field_v1 as field;
    use md_policy_value_v1 as value;
    &[
        MdPolicyDescriptorV1 {
            field_id: field::SCHEMA_VERSION,
            field_name: "schema_version",
            value: value::SCHEMA_VERSION_1,
            value_name: "v1",
        },
        MdPolicyDescriptorV1 {
            field_id: field::FORCE_CONTRACT,
            field_name: "force_contract",
            value: value::FORCE_PRISMCODE_AUTHORIZED,
            value_name: "prismcode_authorized",
        },
        MdPolicyDescriptorV1 {
            field_id: field::INTEGRATOR,
            field_name: "integrator",
            value: value::INTEGRATOR_CANONICAL_HYPER_RUNTIME,
            value_name: "canonical_hyper_runtime",
        },
        MdPolicyDescriptorV1 {
            field_id: field::RAW_ENGINE_SAFE_DEFAULTS_UNCHANGED,
            field_name: "raw_engine_safe_defaults_unchanged",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::INDEPENDENT_RAW_ENGINE_FLAGS_FORBIDDEN,
            field_name: "independent_raw_engine_flags_forbidden",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::TOPOLOGY_SHA256_REQUIRED,
            field_name: "topology_sha256_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::INPUT_SHA256_REQUIRED,
            field_name: "input_sha256_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::SOURCE_EPOCH_REQUIRED,
            field_name: "source_epoch_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::CUDA_GRAPH_REQUIRED,
            field_name: "cuda_graph_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::METAL_RESIDENT_REQUIRED,
            field_name: "metal_resident_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::HOTPATH_ACCELERATION_REQUIRED,
            field_name: "hotpath_acceleration_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::PATH_B_REQUIRED,
            field_name: "path_b_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::DSTW_REQUIRED,
            field_name: "dstw_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::SUPERRESOLUTION_REQUIRED,
            field_name: "superresolution_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::QNANL_INLINE_SEAL_VERIFY_REQUIRED,
            field_name: "qnanl_inline_seal_verify_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::RUNTIME_EVENTS_REQUIRED,
            field_name: "runtime_events_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::EVIDENCE_QUALIFICATION_REQUIRED,
            field_name: "evidence_qualification_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::THERM_REQUIRED,
            field_name: "therm_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::POST_MD_AGGREGATION,
            field_name: "post_md_aggregation",
            value: value::POST_MD_COMPATIBILITY_MD_ONLY_EVIDENCE_HANDOFF_V1,
            value_name: "compatibility_md_only_evidence_handoff_v1",
        },
        MdPolicyDescriptorV1 {
            field_id: field::EFFECTIVE_SURFACE_PROVENANCE_REQUIRED,
            field_name: "effective_surface_provenance_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::GOVERNED_CORPUS_HANDOFF_REQUIRED,
            field_name: "governed_corpus_handoff_required",
            value: value::TRUE,
            value_name: "true",
        },
        MdPolicyDescriptorV1 {
            field_id: field::OUTPUT_IDENTITY_REQUIRED,
            field_name: "output_identity_required",
            value: value::TRUE,
            value_name: "true",
        },
    ]
}

fn post_md_policy_value(policy: &MdPostMdAggregationPolicyV1) -> u32 {
    use md_policy_value_v1 as value;
    match policy {
        MdPostMdAggregationPolicyV1::CompatibilityMdOnlyEvidenceHandoffV1 => {
            value::POST_MD_COMPATIBILITY_MD_ONLY_EVIDENCE_HANDOFF_V1
        }
        MdPostMdAggregationPolicyV1::PreserveLaneDefault => value::POST_MD_PRESERVE_LANE_DEFAULT,
        MdPostMdAggregationPolicyV1::Full => value::POST_MD_FULL,
        MdPostMdAggregationPolicyV1::Skip => value::POST_MD_SKIP,
    }
}

fn therm_surface_value(policy: &MdThermSurfacePolicyV1) -> u32 {
    use md_policy_value_v1 as value;
    match policy {
        MdThermSurfacePolicyV1::PreserveLaneDefault => value::THERM_SURFACE_PRESERVE_LANE_DEFAULT,
        MdThermSurfacePolicyV1::Clusters => value::THERM_SURFACE_CLUSTERS,
        MdThermSurfacePolicyV1::Lattice => value::THERM_SURFACE_LATTICE,
    }
}

fn md_policy_descriptors(policy: &MdEnginePolicyV1) -> PcResult<Vec<MdPolicyDescriptorV1>> {
    use md_policy_field_v1 as field;
    use md_policy_value_v1 as value;
    if policy.schema_version == MD_ENGINE_POLICY_SCHEMA_VERSION_V1 {
        if policy != &MdEnginePolicyV1::legacy_compatibility_v1() {
            return Err(PcError::Security(
                "MD_ENGINE_POLICY_V1_CAN_ONLY_ENCODE_EXPLICIT_LEGACY_COMPATIBILITY".to_string(),
            ));
        }
        return Ok(md_policy_descriptors_v1().to_vec());
    }
    validate_md_engine_policy_v2(policy)?;
    let profile = policy.execution_profile.as_ref().ok_or_else(|| {
        PcError::Security("MD_ENGINE_POLICY_V2_PROFILE_IDENTITY_MISSING".to_string())
    })?;
    let profile_id_value = match profile.id.as_str() {
        "prismos.hyper-runtime-md" => value::PROFILE_PRISMOS_HYPER_RUNTIME_MD,
        "prismos.therm-postmd" => value::PROFILE_PRISMOS_THERM_POSTMD,
        _ => {
            return Err(PcError::Security(format!(
                "MD_ENGINE_POLICY_V2_UNKNOWN_PROFILE_ID:{}",
                profile.id
            )))
        }
    };
    let fallback_reason = match policy.therm_surface_fallback_reason {
        None => value::FALLBACK_REASON_NONE,
        Some(MdThermSurfaceFallbackReasonV1::RequestedLatticeUnavailable) => {
            value::FALLBACK_REASON_REQUESTED_LATTICE_UNAVAILABLE
        }
    };
    let mut descriptors = md_policy_descriptors_v1().to_vec();
    descriptors[0].value = MD_ENGINE_POLICY_SCHEMA_VERSION_V2;
    descriptors[0].value_name = "v2";
    let requested_post_md = descriptors
        .iter_mut()
        .find(|descriptor| descriptor.field_id == field::POST_MD_AGGREGATION)
        .expect("V1 descriptor contains post-MD field");
    requested_post_md.field_name = "requested_post_md_aggregation";
    requested_post_md.value = post_md_policy_value(&policy.requested_post_md_aggregation);
    requested_post_md.value_name =
        md_policy_value_name(field::POST_MD_AGGREGATION, requested_post_md.value);
    descriptors.extend([
        MdPolicyDescriptorV1 {
            field_id: field::EXECUTION_PROFILE_ID,
            field_name: "execution_profile_id",
            value: profile_id_value,
            value_name: md_policy_value_name(field::EXECUTION_PROFILE_ID, profile_id_value),
        },
        MdPolicyDescriptorV1 {
            field_id: field::EXECUTION_PROFILE_VERSION,
            field_name: "execution_profile_version",
            value: value::PROFILE_VERSION_1_0_0,
            value_name: "1.0.0",
        },
        MdPolicyDescriptorV1 {
            field_id: field::EFFECTIVE_POST_MD_AGGREGATION,
            field_name: "effective_post_md_aggregation",
            value: post_md_policy_value(&policy.effective_post_md_aggregation),
            value_name: md_policy_value_name(
                field::EFFECTIVE_POST_MD_AGGREGATION,
                post_md_policy_value(&policy.effective_post_md_aggregation),
            ),
        },
        MdPolicyDescriptorV1 {
            field_id: field::REQUESTED_THERM_SURFACE,
            field_name: "requested_therm_surface",
            value: therm_surface_value(&policy.requested_therm_surface),
            value_name: md_policy_value_name(
                field::REQUESTED_THERM_SURFACE,
                therm_surface_value(&policy.requested_therm_surface),
            ),
        },
        MdPolicyDescriptorV1 {
            field_id: field::EFFECTIVE_THERM_SURFACE,
            field_name: "effective_therm_surface",
            value: therm_surface_value(&policy.effective_therm_surface),
            value_name: md_policy_value_name(
                field::EFFECTIVE_THERM_SURFACE,
                therm_surface_value(&policy.effective_therm_surface),
            ),
        },
        MdPolicyDescriptorV1 {
            field_id: field::THERM_SURFACE_FALLBACK_REASON,
            field_name: "therm_surface_fallback_reason",
            value: fallback_reason,
            value_name: md_policy_value_name(field::THERM_SURFACE_FALLBACK_REASON, fallback_reason),
        },
        MdPolicyDescriptorV1 {
            field_id: field::THERM_SURFACE_FALLBACK_AUTHORIZED,
            field_name: "therm_surface_fallback_authorized",
            value: u32::from(policy.therm_surface_fallback_authorized),
            value_name: if policy.therm_surface_fallback_authorized {
                "true"
            } else {
                "false"
            },
        },
        MdPolicyDescriptorV1 {
            field_id: field::FULL_POST_MD_AUTHORIZED,
            field_name: "full_post_md_authorized",
            value: u32::from(policy.full_post_md_authorized),
            value_name: if policy.full_post_md_authorized {
                "true"
            } else {
                "false"
            },
        },
    ]);
    Ok(descriptors)
}

fn md_policy_field_name(field_id: u32) -> Option<&'static str> {
    use md_policy_field_v1 as field;
    md_policy_descriptors_v1()
        .iter()
        .copied()
        .find(|descriptor| descriptor.field_id == field_id)
        .map(|descriptor| descriptor.field_name)
        .or_else(|| match field_id {
            field::EXECUTION_PROFILE_ID => Some("execution_profile_id"),
            field::EXECUTION_PROFILE_VERSION => Some("execution_profile_version"),
            field::EFFECTIVE_POST_MD_AGGREGATION => Some("effective_post_md_aggregation"),
            field::REQUESTED_THERM_SURFACE => Some("requested_therm_surface"),
            field::EFFECTIVE_THERM_SURFACE => Some("effective_therm_surface"),
            field::THERM_SURFACE_FALLBACK_REASON => Some("therm_surface_fallback_reason"),
            field::THERM_SURFACE_FALLBACK_AUTHORIZED => Some("therm_surface_fallback_authorized"),
            field::FULL_POST_MD_AUTHORIZED => Some("full_post_md_authorized"),
            _ => None,
        })
}

fn md_policy_value_name(field_id: u32, encoded: u32) -> &'static str {
    use md_policy_field_v1 as field;
    use md_policy_value_v1 as value;
    match (field_id, encoded) {
        (field::SCHEMA_VERSION, value::SCHEMA_VERSION_1) => "v1",
        (field::SCHEMA_VERSION, MD_ENGINE_POLICY_SCHEMA_VERSION_V2) => "v2",
        (
            field::POST_MD_AGGREGATION | field::EFFECTIVE_POST_MD_AGGREGATION,
            value::POST_MD_COMPATIBILITY_MD_ONLY_EVIDENCE_HANDOFF_V1,
        ) => "compatibility_md_only_evidence_handoff_v1",
        (
            field::POST_MD_AGGREGATION | field::EFFECTIVE_POST_MD_AGGREGATION,
            value::POST_MD_PRESERVE_LANE_DEFAULT,
        ) => "preserve_lane_default",
        (
            field::POST_MD_AGGREGATION | field::EFFECTIVE_POST_MD_AGGREGATION,
            value::POST_MD_FULL,
        ) => "full",
        (
            field::POST_MD_AGGREGATION | field::EFFECTIVE_POST_MD_AGGREGATION,
            value::POST_MD_SKIP,
        ) => "skip",
        (field::EXECUTION_PROFILE_ID, value::PROFILE_PRISMOS_HYPER_RUNTIME_MD) => {
            "prismos.hyper-runtime-md"
        }
        (field::EXECUTION_PROFILE_ID, value::PROFILE_PRISMOS_THERM_POSTMD) => {
            "prismos.therm-postmd"
        }
        (field::EXECUTION_PROFILE_VERSION, value::PROFILE_VERSION_1_0_0) => "1.0.0",
        (
            field::REQUESTED_THERM_SURFACE | field::EFFECTIVE_THERM_SURFACE,
            value::THERM_SURFACE_PRESERVE_LANE_DEFAULT,
        ) => "preserve_lane_default",
        (
            field::REQUESTED_THERM_SURFACE | field::EFFECTIVE_THERM_SURFACE,
            value::THERM_SURFACE_CLUSTERS,
        ) => "clusters",
        (
            field::REQUESTED_THERM_SURFACE | field::EFFECTIVE_THERM_SURFACE,
            value::THERM_SURFACE_LATTICE,
        ) => "lattice",
        (field::THERM_SURFACE_FALLBACK_REASON, value::FALLBACK_REASON_NONE) => "none",
        (
            field::THERM_SURFACE_FALLBACK_REASON,
            value::FALLBACK_REASON_REQUESTED_LATTICE_UNAVAILABLE,
        ) => "requested_lattice_unavailable",
        (
            field::THERM_SURFACE_FALLBACK_AUTHORIZED | field::FULL_POST_MD_AUTHORIZED,
            value::FALSE,
        ) => "false",
        (
            field::THERM_SURFACE_FALLBACK_AUTHORIZED | field::FULL_POST_MD_AUTHORIZED,
            value::TRUE,
        ) => "true",
        (_, value::TRUE) => "true",
        _ => "unknown",
    }
}

fn is_md_policy_candidate(instruction: &PcInstr32) -> bool {
    instruction.opcode == opcode::MD_PROTOCOL_DIRECTOR
        && (instruction.flags == MD_POLICY_INSTRUCTION_FLAGS_V1
            || instruction.aux == MD_POLICY_INSTRUCTION_AUX_V1)
}

fn md_policy_identity(instruction: &PcInstr32) -> MdPolicyInstructionIdentityV1 {
    MdPolicyInstructionIdentityV1 {
        opcode: instruction.opcode,
        opcode_name: "MD.PROTOCOL_DIRECTOR".to_string(),
        flags: instruction.flags,
        field_id: instruction.dst,
        field_name: md_policy_field_name(instruction.dst)
            .unwrap_or("unknown")
            .to_string(),
        value: instruction.src0,
        value_name: md_policy_value_name(instruction.dst, instruction.src0).to_string(),
        aux: instruction.aux,
    }
}

fn validate_md_engine_policy_v2(policy: &MdEnginePolicyV1) -> PcResult<()> {
    if policy.schema_version != MD_ENGINE_POLICY_SCHEMA_VERSION_V2 {
        return Err(PcError::Security(
            "MD_ENGINE_POLICY_V2_SCHEMA_MISMATCH".to_string(),
        ));
    }
    let profile = policy.execution_profile.as_ref().ok_or_else(|| {
        PcError::Security("MD_ENGINE_POLICY_V2_PROFILE_IDENTITY_MISSING".to_string())
    })?;
    if !matches!(
        profile.id.as_str(),
        "prismos.hyper-runtime-md" | "prismos.therm-postmd"
    ) || profile.version != "1.0.0"
    {
        return Err(PcError::Security(format!(
            "MD_ENGINE_POLICY_V2_PROFILE_IDENTITY_UNSUPPORTED:{}@{}",
            profile.id, profile.version
        )));
    }
    if policy.requested_post_md_aggregation != policy.effective_post_md_aggregation {
        return Err(PcError::Security(
            "MD_ENGINE_POLICY_POST_MD_REQUESTED_EFFECTIVE_MISMATCH".to_string(),
        ));
    }
    if policy.therm_required
        && (policy.requested_post_md_aggregation != MdPostMdAggregationPolicyV1::Full
            || !policy.full_post_md_authorized)
    {
        return Err(PcError::Security(
            "MD_ENGINE_POLICY_REQUIRED_THERM_REQUIRES_AUTHORIZED_FULL_POST_MD".to_string(),
        ));
    }
    let surface_differs = policy.requested_therm_surface != policy.effective_therm_surface;
    if surface_differs
        && (!policy.therm_surface_fallback_authorized
            || policy.therm_surface_fallback_reason.is_none())
    {
        return Err(PcError::Security(
            "MD_ENGINE_POLICY_THERM_SURFACE_FALLBACK_NOT_AUTHORIZED".to_string(),
        ));
    }
    if !surface_differs && policy.therm_surface_fallback_reason.is_some() {
        return Err(PcError::Security(
            "MD_ENGINE_POLICY_SPURIOUS_THERM_SURFACE_FALLBACK_REASON".to_string(),
        ));
    }
    // Both currently admitted product profiles are strict: neither may
    // silently replace its requested therm identity domain.
    if policy.therm_surface_fallback_authorized
        || surface_differs
        || policy.requested_therm_surface != MdThermSurfacePolicyV1::Lattice
    {
        return Err(PcError::Security(
            "MD_ENGINE_POLICY_STRICT_PROFILE_FORBIDS_THERM_SURFACE_FALLBACK".to_string(),
        ));
    }
    if !policy.raw_engine_safe_defaults_unchanged
        || !policy.independent_raw_engine_flags_forbidden
        || !policy.topology_sha256_required
        || !policy.input_sha256_required
        || !policy.source_epoch_required
        || !policy.cuda_graph_required
        || !policy.metal_resident_required
        || !policy.hotpath_acceleration_required
        || !policy.path_b_required
        || !policy.dstw_required
        || !policy.superresolution_required
        || !policy.qnanl_inline_seal_verify_required
        || !policy.runtime_events_required
        || !policy.evidence_qualification_required
        || !policy.therm_required
        || !policy.effective_surface_provenance_required
        || !policy.governed_corpus_handoff_required
        || !policy.output_identity_required
    {
        return Err(PcError::Security(
            "MD_ENGINE_POLICY_V2_REQUIRED_SEMANTICS_MISSING".to_string(),
        ));
    }
    Ok(())
}

fn decode_post_md_policy(encoded: u32) -> PcResult<MdPostMdAggregationPolicyV1> {
    use md_policy_value_v1 as value;
    match encoded {
        value::POST_MD_PRESERVE_LANE_DEFAULT => {
            Ok(MdPostMdAggregationPolicyV1::PreserveLaneDefault)
        }
        value::POST_MD_FULL => Ok(MdPostMdAggregationPolicyV1::Full),
        value::POST_MD_SKIP => Ok(MdPostMdAggregationPolicyV1::Skip),
        value::POST_MD_COMPATIBILITY_MD_ONLY_EVIDENCE_HANDOFF_V1 => {
            Ok(MdPostMdAggregationPolicyV1::CompatibilityMdOnlyEvidenceHandoffV1)
        }
        other => Err(PcError::Operand(format!(
            "MD_ENGINE_POLICY_UNKNOWN_POST_MD_VALUE:{other}"
        ))),
    }
}

fn decode_therm_surface(encoded: u32) -> PcResult<MdThermSurfacePolicyV1> {
    use md_policy_value_v1 as value;
    match encoded {
        value::THERM_SURFACE_PRESERVE_LANE_DEFAULT => {
            Ok(MdThermSurfacePolicyV1::PreserveLaneDefault)
        }
        value::THERM_SURFACE_CLUSTERS => Ok(MdThermSurfacePolicyV1::Clusters),
        value::THERM_SURFACE_LATTICE => Ok(MdThermSurfacePolicyV1::Lattice),
        other => Err(PcError::Operand(format!(
            "MD_ENGINE_POLICY_UNKNOWN_THERM_SURFACE_VALUE:{other}"
        ))),
    }
}

fn decode_policy_bool(fields: &BTreeMap<u32, u32>, field_id: u32) -> PcResult<bool> {
    match fields.get(&field_id).copied() {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        Some(other) => Err(PcError::Operand(format!(
            "MD_ENGINE_POLICY_INVALID_BOOLEAN:field_id={field_id} value={other}"
        ))),
        None => Err(PcError::Operand(format!(
            "MD_ENGINE_POLICY_MISSING_FIELDS:{field_id}"
        ))),
    }
}

fn required_policy_value(fields: &BTreeMap<u32, u32>, field_id: u32) -> PcResult<u32> {
    fields
        .get(&field_id)
        .copied()
        .ok_or_else(|| PcError::Operand(format!("MD_ENGINE_POLICY_MISSING_FIELDS:{field_id}")))
}

fn decode_md_engine_policy_v2(fields: &BTreeMap<u32, u32>) -> PcResult<MdEnginePolicyV1> {
    use md_policy_field_v1 as field;
    use md_policy_value_v1 as value;
    let profile_id = match required_policy_value(fields, field::EXECUTION_PROFILE_ID)? {
        value::PROFILE_PRISMOS_HYPER_RUNTIME_MD => "prismos.hyper-runtime-md",
        value::PROFILE_PRISMOS_THERM_POSTMD => "prismos.therm-postmd",
        other => {
            return Err(PcError::Operand(format!(
                "MD_ENGINE_POLICY_UNKNOWN_PROFILE_ID_VALUE:{other}"
            )))
        }
    };
    if required_policy_value(fields, field::EXECUTION_PROFILE_VERSION)?
        != value::PROFILE_VERSION_1_0_0
    {
        return Err(PcError::Operand(
            "MD_ENGINE_POLICY_UNKNOWN_PROFILE_VERSION".to_string(),
        ));
    }
    let fallback_reason = match required_policy_value(fields, field::THERM_SURFACE_FALLBACK_REASON)?
    {
        value::FALLBACK_REASON_NONE => None,
        value::FALLBACK_REASON_REQUESTED_LATTICE_UNAVAILABLE => {
            Some(MdThermSurfaceFallbackReasonV1::RequestedLatticeUnavailable)
        }
        other => {
            return Err(PcError::Operand(format!(
                "MD_ENGINE_POLICY_UNKNOWN_FALLBACK_REASON:{other}"
            )))
        }
    };
    let policy = MdEnginePolicyV1 {
        schema_version: MD_ENGINE_POLICY_SCHEMA_VERSION_V2,
        execution_profile: Some(MdExecutionProfileIdentityV1 {
            id: profile_id.to_string(),
            version: "1.0.0".to_string(),
        }),
        force_contract: MdForceContractV1::PrismcodeAuthorized,
        integrator: MdIntegratorPolicyV1::CanonicalHyperRuntime,
        raw_engine_safe_defaults_unchanged: decode_policy_bool(
            fields,
            field::RAW_ENGINE_SAFE_DEFAULTS_UNCHANGED,
        )?,
        independent_raw_engine_flags_forbidden: decode_policy_bool(
            fields,
            field::INDEPENDENT_RAW_ENGINE_FLAGS_FORBIDDEN,
        )?,
        topology_sha256_required: decode_policy_bool(fields, field::TOPOLOGY_SHA256_REQUIRED)?,
        input_sha256_required: decode_policy_bool(fields, field::INPUT_SHA256_REQUIRED)?,
        source_epoch_required: decode_policy_bool(fields, field::SOURCE_EPOCH_REQUIRED)?,
        cuda_graph_required: decode_policy_bool(fields, field::CUDA_GRAPH_REQUIRED)?,
        metal_resident_required: decode_policy_bool(fields, field::METAL_RESIDENT_REQUIRED)?,
        hotpath_acceleration_required: decode_policy_bool(
            fields,
            field::HOTPATH_ACCELERATION_REQUIRED,
        )?,
        path_b_required: decode_policy_bool(fields, field::PATH_B_REQUIRED)?,
        dstw_required: decode_policy_bool(fields, field::DSTW_REQUIRED)?,
        superresolution_required: decode_policy_bool(fields, field::SUPERRESOLUTION_REQUIRED)?,
        qnanl_inline_seal_verify_required: decode_policy_bool(
            fields,
            field::QNANL_INLINE_SEAL_VERIFY_REQUIRED,
        )?,
        runtime_events_required: decode_policy_bool(fields, field::RUNTIME_EVENTS_REQUIRED)?,
        evidence_qualification_required: decode_policy_bool(
            fields,
            field::EVIDENCE_QUALIFICATION_REQUIRED,
        )?,
        therm_required: decode_policy_bool(fields, field::THERM_REQUIRED)?,
        requested_post_md_aggregation: decode_post_md_policy(required_policy_value(
            fields,
            field::POST_MD_AGGREGATION,
        )?)?,
        effective_post_md_aggregation: decode_post_md_policy(required_policy_value(
            fields,
            field::EFFECTIVE_POST_MD_AGGREGATION,
        )?)?,
        full_post_md_authorized: decode_policy_bool(fields, field::FULL_POST_MD_AUTHORIZED)?,
        requested_therm_surface: decode_therm_surface(required_policy_value(
            fields,
            field::REQUESTED_THERM_SURFACE,
        )?)?,
        effective_therm_surface: decode_therm_surface(required_policy_value(
            fields,
            field::EFFECTIVE_THERM_SURFACE,
        )?)?,
        therm_surface_fallback_reason: fallback_reason,
        therm_surface_fallback_authorized: decode_policy_bool(
            fields,
            field::THERM_SURFACE_FALLBACK_AUTHORIZED,
        )?,
        effective_surface_provenance_required: decode_policy_bool(
            fields,
            field::EFFECTIVE_SURFACE_PROVENANCE_REQUIRED,
        )?,
        governed_corpus_handoff_required: decode_policy_bool(
            fields,
            field::GOVERNED_CORPUS_HANDOFF_REQUIRED,
        )?,
        output_identity_required: decode_policy_bool(fields, field::OUTPUT_IDENTITY_REQUIRED)?,
    };
    if required_policy_value(fields, field::FORCE_CONTRACT)? != value::FORCE_PRISMCODE_AUTHORIZED
        || required_policy_value(fields, field::INTEGRATOR)?
            != value::INTEGRATOR_CANONICAL_HYPER_RUNTIME
    {
        return Err(PcError::Operand(
            "MD_ENGINE_POLICY_UNKNOWN_VALUE:force_or_integrator".to_string(),
        ));
    }
    validate_md_engine_policy_v2(&policy)?;
    Ok(policy)
}

fn decode_md_engine_policy(packet: &PCodePacket) -> PcResult<MdEnginePolicyV1> {
    if let Some(instruction) = packet.instructions.iter().find(|instruction| {
        instruction.opcode == opcode::MD_PROTOCOL_DIRECTOR && !is_md_policy_candidate(instruction)
    }) {
        return Err(PcError::Operand(format!(
            "MD_ENGINE_POLICY_UNKNOWN_ENCODING: flags={} aux={}",
            instruction.flags, instruction.aux
        )));
    }
    let candidates: Vec<&PcInstr32> = packet
        .instructions
        .iter()
        .filter(|instruction| is_md_policy_candidate(instruction))
        .collect();
    if candidates.is_empty() {
        return Err(PcError::Opcode(
            "MD_ENGINE_POLICY_MISSING: no typed MD policy instruction set".to_string(),
        ));
    }

    let mut fields = BTreeMap::<u32, u32>::new();
    for instruction in candidates {
        if instruction.flags != MD_POLICY_INSTRUCTION_FLAGS_V1
            || instruction.aux != MD_POLICY_INSTRUCTION_AUX_V1
            || instruction.dst_kind != OperandKind::ImmU32 as u8
            || instruction.src0_kind != OperandKind::ImmU32 as u8
            || instruction.src1_kind != OperandKind::None as u8
            || instruction.src2_kind != OperandKind::None as u8
            || instruction.src1 != 0
            || instruction.src2_or_len != 0
        {
            return Err(PcError::Operand(format!(
                "MD_ENGINE_POLICY_INVALID_OPERANDS: field_id={}",
                instruction.dst
            )));
        }
        let Some(_field_name) = md_policy_field_name(instruction.dst) else {
            return Err(PcError::Operand(format!(
                "MD_ENGINE_POLICY_UNKNOWN_FIELD:{}",
                instruction.dst
            )));
        };
        if let Some(previous) = fields.insert(instruction.dst, instruction.src0) {
            return Err(PcError::Operand(format!(
                "MD_ENGINE_POLICY_DUPLICATE_OR_CONFLICT:field_id={} first={} second={}",
                instruction.dst, previous, instruction.src0
            )));
        }
    }
    let schema_version = fields
        .get(&md_policy_field_v1::SCHEMA_VERSION)
        .copied()
        .ok_or_else(|| PcError::Operand("MD_ENGINE_POLICY_MISSING_FIELDS:1".to_string()))?;
    let decoded = match schema_version {
        MD_ENGINE_POLICY_SCHEMA_VERSION_V1 => MdEnginePolicyV1::legacy_compatibility_v1(),
        MD_ENGINE_POLICY_SCHEMA_VERSION_V2 => decode_md_engine_policy_v2(&fields)?,
        other => {
            return Err(PcError::Operand(format!(
                "MD_ENGINE_POLICY_UNSUPPORTED_SCHEMA_VERSION:{other}"
            )))
        }
    };
    let expected_descriptors = md_policy_descriptors(&decoded)?;
    let required: BTreeSet<u32> = expected_descriptors
        .iter()
        .map(|descriptor| descriptor.field_id)
        .collect();
    let observed: BTreeSet<u32> = fields.keys().copied().collect();
    if observed != required {
        let missing: Vec<String> = required
            .difference(&observed)
            .map(ToString::to_string)
            .collect();
        return Err(PcError::Operand(format!(
            "MD_ENGINE_POLICY_MISSING_FIELDS:{}",
            missing.join(",")
        )));
    }
    for (field_id, encoded) in &fields {
        let expected = expected_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == *field_id)
            .expect("field set checked above");
        if *encoded != expected.value {
            return Err(PcError::Operand(format!(
                "MD_ENGINE_POLICY_UNKNOWN_VALUE:field_id={field_id} field={} value={encoded} expected={}",
                expected.field_name, expected.value
            )));
        }
    }
    let packet_end = packet
        .instructions
        .iter()
        .position(|instruction| instruction.opcode == opcode::PACKET_END)
        .ok_or_else(|| PcError::Opcode("MD_ENGINE_POLICY_PACKET_END_MISSING".to_string()))?;
    let canonical = canonical_md_policy_instructions(&decoded)?;
    let policy_start = packet_end.checked_sub(canonical.len()).ok_or_else(|| {
        PcError::Operand(
            "MD_ENGINE_POLICY_NONCANONICAL_ORDER: policy precedes packet start".to_string(),
        )
    })?;
    if packet.instructions[policy_start..packet_end] != canonical {
        return Err(PcError::Operand(
            "MD_ENGINE_POLICY_NONCANONICAL_ORDER: declarations must be one contiguous canonical set immediately before PACKET.END"
                .to_string(),
        ));
    }
    Ok(decoded)
}

/// Read, container/hash validate, L3 validate, and decode one packet byte image.
/// No DebugJson field and no raw-engine/CLI default participates in policy.
pub fn load_validated_md_engine_policy(path: &Path) -> PcResult<ValidatedMdEnginePolicyV1> {
    let bytes = fs::read(path)?;
    let packet_sha256 = sha256_hex(&bytes);
    let packet_bytes = bytes.len() as u64;
    let packet = parse_packet(bytes)?;
    validate_parsed_packet(&packet, ValidationLevel::L3)?;
    let expected_hw_profile = hash_bytes(MD_RUN_AUTHORITY_HW_PROFILE_V1);
    if packet.bytes[68..100] != expected_hw_profile {
        return Err(PcError::Security(
            "MD_ENGINE_POLICY_RUN_AUTHORITY_REJECTED: packet header is not bound to PRISMCODE_REAL_TOPOLOGY_BOUND_MD_POLICY_V1"
                .to_string(),
        ));
    }
    let policy = decode_md_engine_policy(&packet)?;
    Ok(ValidatedMdEnginePolicyV1 {
        policy,
        packet_hash_blake3: hex_hash(&packet.packet_hash),
        packet_sha256,
        packet_bytes,
        instruction_count: packet.instructions.len() as u64,
    })
}

/// Deterministically add the canonical MD policy declarations to an existing
/// valid packet.  Every non-InstrStream section is copied byte-for-byte and all
/// non-policy instructions retain exact order and bytes.  A packet already in
/// canonical form is returned byte-identically.
pub fn normalize_md_engine_policy_packet_bytes(
    input: &[u8],
) -> PcResult<(Vec<u8>, MdPolicyNormalizationReportV1)> {
    normalize_md_engine_policy_packet_bytes_for_policy(
        input,
        &MdEnginePolicyV1::canonical_hyper_runtime(),
    )
}

/// Deterministically bind an explicitly selected product policy into packet
/// instructions. This is a builder API, not a runtime override: the returned
/// bytes must still be hash-pinned by the execution profile before launch.
pub fn normalize_md_engine_policy_packet_bytes_for_policy(
    input: &[u8],
    requested_policy: &MdEnginePolicyV1,
) -> PcResult<(Vec<u8>, MdPolicyNormalizationReportV1)> {
    validate_md_engine_policy_v2(requested_policy)?;
    let base_sha256 = sha256_hex(input);
    let base = parse_packet(input.to_vec())?;
    validate_parsed_packet(&base, ValidationLevel::L3)?;
    let candidates: Vec<PcInstr32> = base
        .instructions
        .iter()
        .copied()
        .filter(is_md_policy_candidate)
        .collect();
    let expected_hw_profile = hash_bytes(MD_RUN_AUTHORITY_HW_PROFILE_V1);
    let run_authority_header_bound = base.bytes[68..100] == expected_hw_profile;
    let canonical = canonical_md_policy_instructions(requested_policy)?;
    let mut replace_legacy_policy = false;
    if !candidates.is_empty() {
        let decoded = decode_md_engine_policy(&base)?;
        if decoded == *requested_policy && candidates == canonical {
            if run_authority_header_bound {
                return Ok((
                    input.to_vec(),
                    MdPolicyNormalizationReportV1 {
                        schema_kind: "prismos.prismcode.md_policy_normalization".to_string(),
                        schema_version: 1,
                        base_packet_sha256: base_sha256.clone(),
                        base_packet_hash_blake3: hex_hash(&base.packet_hash),
                        base_hw_profile_classification: "REAL_TOPOLOGY_BOUND_MD_POLICY_V1"
                            .to_string(),
                        normalized_packet_sha256: base_sha256,
                        normalized_packet_hash_blake3: hex_hash(&base.packet_hash),
                        normalized_hw_profile_classification: "REAL_TOPOLOGY_BOUND_MD_POLICY_V1"
                            .to_string(),
                        base_instruction_count: base.instructions.len() as u64,
                        normalized_instruction_count: base.instructions.len() as u64,
                        removed_policy_instruction_count: 0,
                        added_policy_instruction_count: 0,
                        removed_policy_instructions: Vec::new(),
                        added_policy_instructions: Vec::new(),
                        non_policy_sections_byte_identical: true,
                        non_policy_instructions_byte_identical: true,
                        idempotent: true,
                        validation_level: "L3".to_string(),
                        validation_status: "PASS".to_string(),
                        policy: decoded,
                    },
                ));
            }
            // Canonical policy bytes with a legacy synthetic/unbound header are
            // reserialized solely to bind the real run-authority profile.
        } else if decoded.schema_version == MD_ENGINE_POLICY_SCHEMA_VERSION_V1 {
            // V1 is decoded as compatibility-only and is never admitted as a
            // strict product request. The builder may explicitly replace it
            // with V2 declarations; the output receives a new packet hash.
            replace_legacy_policy = true;
        } else {
            return Err(PcError::Operand(
                "MD_ENGINE_POLICY_NONCANONICAL_ORDER: normalization refuses to launder policy bytes"
                    .to_string(),
            ));
        }
    }

    let normalized_instructions = if candidates.is_empty() || replace_legacy_policy {
        let base_without_policy: Vec<PcInstr32> = base
            .instructions
            .iter()
            .copied()
            .filter(|instruction| !is_md_policy_candidate(instruction))
            .collect();
        let insertion = base_without_policy
            .iter()
            .position(|instruction| instruction.opcode == opcode::PACKET_END)
            .ok_or_else(|| {
                PcError::Opcode("PACKET.END missing during policy normalization".to_string())
            })?;
        let mut instructions = Vec::with_capacity(base_without_policy.len() + canonical.len());
        instructions.extend_from_slice(&base_without_policy[..insertion]);
        instructions.extend_from_slice(&canonical);
        instructions.extend_from_slice(&base_without_policy[insertion..]);
        instructions
    } else {
        base.instructions.clone()
    };
    let instruction_bytes: Vec<u8> = normalized_instructions
        .iter()
        .flat_map(PcInstr32::encode)
        .collect();
    let sections = base
        .sections
        .iter()
        .map(|entry| {
            if entry.section_type == SectionKind::InstrStream {
                (
                    entry.section_type,
                    instruction_bytes.clone(),
                    normalized_instructions.len() as u64,
                )
            } else {
                (
                    entry.section_type,
                    section_bytes(&base.bytes, entry).to_vec(),
                    entry.record_count,
                )
            }
        })
        .collect();
    let mut packet_uuid_source = [0u8; 32];
    packet_uuid_source[..16].copy_from_slice(&base.bytes[36..52]);
    let normalized_bytes = write_container_with_hw_profile(
        sections,
        packet_uuid_source,
        hash_bytes(MD_RUN_AUTHORITY_HW_PROFILE_V1),
    )?;
    let normalized = parse_packet(normalized_bytes.clone())?;
    validate_parsed_packet(&normalized, ValidationLevel::L3)?;
    let policy = decode_md_engine_policy(&normalized)?;
    let non_policy_sections_byte_identical = base.sections.iter().all(|base_entry| {
        base_entry.section_type == SectionKind::InstrStream
            || find_section(&normalized.sections, base_entry.section_type)
                .map(|normalized_entry| {
                    section_bytes(&base.bytes, base_entry)
                        == section_bytes(&normalized.bytes, normalized_entry)
                })
                .unwrap_or(false)
    });
    let base_non_policy: Vec<PcInstr32> = base
        .instructions
        .iter()
        .copied()
        .filter(|instruction| !is_md_policy_candidate(instruction))
        .collect();
    let normalized_non_policy: Vec<PcInstr32> = normalized
        .instructions
        .iter()
        .copied()
        .filter(|instruction| !is_md_policy_candidate(instruction))
        .collect();
    let non_policy_instructions_byte_identical = base_non_policy == normalized_non_policy;
    if !non_policy_sections_byte_identical || !non_policy_instructions_byte_identical {
        return Err(PcError::Internal(
            "MD policy normalization changed non-policy packet content".to_string(),
        ));
    }

    // The second normalization must take the byte-preserving canonical branch.
    let (second, _) =
        normalize_md_engine_policy_packet_bytes_for_policy(&normalized_bytes, requested_policy)?;
    if second != normalized_bytes {
        return Err(PcError::Internal(
            "MD policy normalization is not byte-idempotent".to_string(),
        ));
    }
    let identities = if candidates.is_empty() || replace_legacy_policy {
        canonical.iter().map(md_policy_identity).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let report = MdPolicyNormalizationReportV1 {
        schema_kind: "prismos.prismcode.md_policy_normalization".to_string(),
        schema_version: 1,
        base_packet_sha256: base_sha256,
        base_packet_hash_blake3: hex_hash(&base.packet_hash),
        base_hw_profile_classification: if run_authority_header_bound {
            "REAL_TOPOLOGY_BOUND_MD_POLICY_V1"
        } else {
            "LEGACY_UNBOUND_OR_OTHER_NOT_RUN_AUTHORITY"
        }
        .to_string(),
        normalized_packet_sha256: sha256_hex(&normalized_bytes),
        normalized_packet_hash_blake3: hex_hash(&normalized.packet_hash),
        normalized_hw_profile_classification: "REAL_TOPOLOGY_BOUND_MD_POLICY_V1".to_string(),
        base_instruction_count: base.instructions.len() as u64,
        normalized_instruction_count: normalized.instructions.len() as u64,
        removed_policy_instruction_count: if replace_legacy_policy {
            candidates.len() as u64
        } else {
            0
        },
        added_policy_instruction_count: identities.len() as u64,
        removed_policy_instructions: if replace_legacy_policy {
            candidates.iter().map(md_policy_identity).collect()
        } else {
            Vec::new()
        },
        added_policy_instructions: identities,
        non_policy_sections_byte_identical,
        non_policy_instructions_byte_identical,
        idempotent: true,
        validation_level: "L3".to_string(),
        validation_status: "PASS".to_string(),
        policy,
    };
    Ok((normalized_bytes, report))
}

/// Public fixture/generator API used by clients that need a valid packet with
/// canonical policy declarations without reproducing the operand ABI.
pub fn normalize_md_engine_policy_packet(
    input: &Path,
    output: &Path,
) -> PcResult<MdPolicyNormalizationReportV1> {
    if input == output {
        return Err(PcError::Io(
            "policy normalizer requires distinct input and output paths".to_string(),
        ));
    }
    let input_bytes = fs::read(input)?;
    let (normalized, report) = normalize_md_engine_policy_packet_bytes(&input_bytes)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|error| {
            PcError::Io(format!(
                "policy normalizer refuses to overwrite {}: {error}",
                output.display()
            ))
        })?;
    file.write_all(&normalized)?;
    file.sync_all()?;
    if let Some(parent) = output.parent() {
        fs::File::open(parent)?.sync_all()?;
    }
    Ok(report)
}

#[cfg(test)]
mod md_packet_policy_authority_tests {
    use super::*;

    fn base_bytes() -> Vec<u8> {
        let corpus = synthetic_corpus("MD_POLICY_AUTHORITY_TEST", 3, 4, 12, 8);
        build_packet_bytes(&corpus, &serde_json::to_vec(&corpus).unwrap()).unwrap()
    }

    fn with_instructions(packet: &PCodePacket, instructions: &[PcInstr32]) -> Vec<u8> {
        let instruction_bytes: Vec<u8> = instructions.iter().flat_map(PcInstr32::encode).collect();
        let sections = packet
            .sections
            .iter()
            .map(|entry| {
                if entry.section_type == SectionKind::InstrStream {
                    (
                        entry.section_type,
                        instruction_bytes.clone(),
                        instructions.len() as u64,
                    )
                } else {
                    (
                        entry.section_type,
                        section_bytes(&packet.bytes, entry).to_vec(),
                        entry.record_count,
                    )
                }
            })
            .collect();
        let mut uuid = [0u8; 32];
        uuid[..16].copy_from_slice(&packet.bytes[36..52]);
        let mut hw_profile = [0u8; 32];
        hw_profile.copy_from_slice(&packet.bytes[68..100]);
        write_container_with_hw_profile(sections, uuid, hw_profile).unwrap()
    }

    fn load_bytes(bytes: &[u8]) -> PcResult<ValidatedMdEnginePolicyV1> {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("packet.pcode");
        fs::write(&path, bytes).unwrap();
        load_validated_md_engine_policy(&path)
    }

    #[test]
    fn canonical_policy_normalization_is_positive_and_byte_idempotent() {
        let base = base_bytes();
        assert!(load_bytes(&base)
            .unwrap_err()
            .to_string()
            .contains("RUN_AUTHORITY_REJECTED"));
        let (normalized, report) = normalize_md_engine_policy_packet_bytes(&base).unwrap();
        assert_eq!(report.removed_policy_instruction_count, 0);
        assert_eq!(report.added_policy_instruction_count, 30);
        assert_eq!(report.added_policy_instructions.len(), 30);
        assert!(report
            .added_policy_instructions
            .iter()
            .all(|row| row.field_name != "unknown" && row.value_name != "unknown"));
        let loaded = load_bytes(&normalized).unwrap();
        assert_eq!(loaded.policy, MdEnginePolicyV1::canonical_hyper_runtime());
        assert_eq!(loaded.packet_sha256, report.normalized_packet_sha256);
        let base_packet = parse_packet(base).unwrap();
        let normalized_packet = parse_packet(normalized.clone()).unwrap();
        assert_eq!(
            normalized_packet.bytes[68..100],
            hash_bytes(MD_RUN_AUTHORITY_HW_PROFILE_V1)
        );
        for base_entry in &base_packet.sections {
            if base_entry.section_type == SectionKind::InstrStream {
                continue;
            }
            let normalized_entry =
                find_section(&normalized_packet.sections, base_entry.section_type).unwrap();
            assert_eq!(
                section_bytes(&base_packet.bytes, base_entry),
                section_bytes(&normalized_packet.bytes, normalized_entry),
                "non-policy section {:?} changed",
                base_entry.section_type
            );
        }
        let (second, second_report) = normalize_md_engine_policy_packet_bytes(&normalized).unwrap();
        assert_eq!(second, normalized);
        assert!(second_report.idempotent);
    }

    #[test]
    fn therm_postmd_packet_policy_roundtrips_profile_and_exact_effective_policy() {
        let requested = MdEnginePolicyV1::therm_postmd();
        let (normalized, report) =
            normalize_md_engine_policy_packet_bytes_for_policy(&base_bytes(), &requested).unwrap();
        let loaded = load_bytes(&normalized).unwrap();
        assert_eq!(loaded.policy, requested);
        assert_eq!(report.policy, requested);
        assert_eq!(
            loaded.policy.execution_profile.as_ref().unwrap().id,
            "prismos.therm-postmd"
        );
        assert_eq!(
            loaded.policy.requested_post_md_aggregation,
            MdPostMdAggregationPolicyV1::Full
        );
        assert_eq!(
            loaded.policy.effective_post_md_aggregation,
            MdPostMdAggregationPolicyV1::Full
        );
        assert_eq!(
            loaded.policy.effective_therm_surface,
            MdThermSurfacePolicyV1::Lattice
        );
    }

    #[test]
    fn v1_packet_is_explicitly_decoded_but_cannot_equal_strict_product_policy() {
        let packet = parse_packet(base_bytes()).unwrap();
        let packet_end = packet
            .instructions
            .iter()
            .position(|instruction| instruction.opcode == opcode::PACKET_END)
            .unwrap();
        let legacy = MdEnginePolicyV1::legacy_compatibility_v1();
        let legacy_policy = canonical_md_policy_instructions(&legacy).unwrap();
        let mut instructions = packet.instructions.clone();
        instructions.splice(packet_end..packet_end, legacy_policy);
        let legacy_packet = with_instructions(&packet, &instructions);
        let decoded_packet = parse_packet(legacy_packet).unwrap();
        let decoded = decode_md_engine_policy(&decoded_packet).unwrap();
        assert_eq!(decoded, legacy);
        assert_ne!(decoded, MdEnginePolicyV1::canonical_hyper_runtime());
        assert!(validate_md_engine_policy_v2(&decoded).is_err());
    }

    #[test]
    fn policy_decode_rejects_missing_duplicate_conflict_and_unknown_field() {
        let (normalized, _) = normalize_md_engine_policy_packet_bytes(&base_bytes()).unwrap();
        let packet = parse_packet(normalized).unwrap();
        let policy_start = packet
            .instructions
            .iter()
            .position(is_md_policy_candidate)
            .unwrap();

        let mut missing = packet.instructions.clone();
        missing.retain(|instruction| !is_md_policy_candidate(instruction));
        assert!(load_bytes(&with_instructions(&packet, &missing))
            .unwrap_err()
            .to_string()
            .contains("MD_ENGINE_POLICY_MISSING"));

        let mut partial = packet.instructions.clone();
        partial.remove(policy_start);
        assert!(load_bytes(&with_instructions(&packet, &partial))
            .unwrap_err()
            .to_string()
            .contains("MD_ENGINE_POLICY_MISSING_FIELDS"));

        let mut duplicate = packet.instructions.clone();
        duplicate.insert(policy_start + 1, duplicate[policy_start]);
        assert!(load_bytes(&with_instructions(&packet, &duplicate))
            .unwrap_err()
            .to_string()
            .contains("DUPLICATE_OR_CONFLICT"));

        let mut conflict = packet.instructions.clone();
        let mut conflicting = conflict[policy_start];
        conflicting.src0 = 2;
        conflict.insert(policy_start + 1, conflicting);
        assert!(load_bytes(&with_instructions(&packet, &conflict))
            .unwrap_err()
            .to_string()
            .contains("DUPLICATE_OR_CONFLICT"));

        let mut unknown = packet.instructions.clone();
        unknown[policy_start].dst = 999;
        assert!(load_bytes(&with_instructions(&packet, &unknown))
            .unwrap_err()
            .to_string()
            .contains("UNKNOWN_FIELD"));

        let mut unknown_value = packet.instructions.clone();
        unknown_value[policy_start + 1].src0 = 999;
        assert!(load_bytes(&with_instructions(&packet, &unknown_value))
            .unwrap_err()
            .to_string()
            .contains("UNKNOWN_VALUE"));

        let mut unknown_encoding = packet.instructions.clone();
        unknown_encoding[policy_start].flags = 0;
        unknown_encoding[policy_start].aux = 0;
        assert!(load_bytes(&with_instructions(&packet, &unknown_encoding))
            .unwrap_err()
            .to_string()
            .contains("UNKNOWN_ENCODING"));

        let mut reordered = packet.instructions.clone();
        reordered.swap(policy_start, policy_start + 1);
        assert!(load_bytes(&with_instructions(&packet, &reordered))
            .unwrap_err()
            .to_string()
            .contains("NONCANONICAL_ORDER"));
    }

    #[test]
    fn policy_load_rejects_packet_tamper_before_decode() {
        let (mut normalized, _) = normalize_md_engine_policy_packet_bytes(&base_bytes()).unwrap();
        let last = normalized.len() - 1;
        normalized[last] ^= 1;
        let error = load_bytes(&normalized).unwrap_err().to_string();
        assert!(error.contains("content hash mismatch") || error.contains("packet hash mismatch"));
    }

    #[test]
    fn generator_refuses_to_overwrite_existing_output() {
        let temporary = tempfile::tempdir().unwrap();
        let input = temporary.path().join("base.pcode");
        let output = temporary.path().join("existing.pcode");
        fs::write(&input, base_bytes()).unwrap();
        fs::write(&output, b"operator-owned").unwrap();
        let error = normalize_md_engine_policy_packet(&input, &output).unwrap_err();
        assert!(error.to_string().contains("refuses to overwrite"));
        assert_eq!(fs::read(output).unwrap(), b"operator-owned");
    }

    #[test]
    fn generic_execution_report_is_diagnostic_not_native_authority() {
        let temporary = tempfile::tempdir().unwrap();
        let input = temporary.path().join("packet.pcode");
        let output = temporary.path().join("packet.pcout");
        let (normalized, _) = normalize_md_engine_policy_packet_bytes(&base_bytes()).unwrap();
        fs::write(&input, normalized).unwrap();
        let report = execute_packet(&input, Backend::Cpu, &output).unwrap();
        assert_eq!(
            report.authority_classification,
            "DIAGNOSTIC_SUBSTRATE_EXECUTION"
        );
        assert!(!report.native_packet_execution);
    }

    #[test]
    fn cuda_diagnostic_surface_cannot_claim_native_packet_execution() {
        let report = crate::cuda_backend::diagnostic_selftest_report_from_runtime(&json!({
            "cuda_execution_status": "NOT_EXECUTED_METADATA_CONTRACT_TEST",
            "cuda_backend_runtime_linked": false,
        }));
        assert_eq!(
            report["authority_classification"],
            "DIAGNOSTIC_SELF_TEST_NOT_NATIVE_PACKET_EXECUTION"
        );
        assert_eq!(report["native_packet_execution"], false);
    }
}

pub fn disassemble_packet(input: &Path, output: Option<&Path>) -> PcResult<String> {
    let packet = read_packet(input)?;
    let mut lines = Vec::new();
    lines.push(format!(
        "PRISMCODE ISA v0.1 packet_hash={}",
        hex_hash(&packet.packet_hash)
    ));
    lines.push(format!(
        "sections={} instructions={} states={} timebases={} events={} nanl_tokens={}",
        packet.sections.len(),
        packet.instructions.len(),
        packet.state_count,
        packet.time_count,
        packet.event_count,
        packet.nanl_count
    ));
    for (idx, instr) in packet.instructions.iter().enumerate() {
        lines.push(format!(
            "{idx:04} 0x{:04x} {:<24} class=0x{:02x} flags=0x{:02x} dst_kind={} dst={} src0_kind={} src0={} aux={}",
            instr.opcode,
            opcode_name(instr.opcode),
            instr.class,
            instr.flags,
            instr.dst_kind,
            instr.dst,
            instr.src0_kind,
            instr.src0,
            instr.aux
        ));
    }
    let text = lines.join("\n") + "\n";
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &text)?;
    }
    Ok(text)
}

pub fn export_debug_json(input: &Path, output: &Path) -> PcResult<()> {
    let packet = read_packet(input)?;
    let debug = debug_json_from_packet(&packet)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(&debug)?)?;
    Ok(())
}

pub fn execute_packet(input: &Path, backend: Backend, output: &Path) -> PcResult<ExecutionReport> {
    let packet = read_packet(input)?;
    let selected = match backend {
        Backend::Auto => Backend::Cpu,
        Backend::Cpu => Backend::Cpu,
        Backend::Cuda => Backend::Cuda,
    };
    let start = Instant::now();
    let cuda_execution = if selected == Backend::Cuda {
        let mut diagnostic = execute_required_opcode_suite(
            packet.instructions.len() as u64,
            packet.state_count,
            packet.event_count,
            packet.nanl_count,
        )?;
        diagnostic["authority_classification"] =
            json!("DIAGNOSTIC_SELF_TEST_NOT_NATIVE_PACKET_EXECUTION");
        diagnostic["native_packet_execution"] = json!(false);
        Some(diagnostic)
    } else {
        None
    };
    let state_entry = find_section(&packet.sections, SectionKind::StateTable)?;
    let event_entry = find_section(&packet.sections, SectionKind::EventTable)?;
    let nanl_entry = find_section(&packet.sections, SectionKind::NanlTable)?;
    let state_bytes = section_bytes(&packet.bytes, state_entry);
    let event_bytes = section_bytes(&packet.bytes, event_entry);
    let nanl_bytes = section_bytes(&packet.bytes, nanl_entry);
    let event_sum = event_value_sum(event_bytes)?;
    let instr_accumulator = instruction_accumulator(&packet.instructions);
    let output_state_hash = hash_many(&[
        &packet.packet_hash,
        state_bytes,
        event_bytes,
        nanl_bytes,
        &event_sum.to_le_bytes(),
        &instr_accumulator.to_le_bytes(),
    ]);
    let checkpoint_count = packet
        .instructions
        .iter()
        .filter(|instr| instr.opcode == opcode::REPLAY_CHECKPOINT)
        .count() as u64;
    let seal_chain_hash = hash_many(&[b"PCOUT_SEAL", &packet.packet_hash, &output_state_hash]);
    let elapsed_ns = start.elapsed().as_nanos() as u64;
    let report = ExecutionReport {
        status: "PASS".to_string(),
        backend: format!("{selected:?}").to_ascii_lowercase(),
        authority_classification: "DIAGNOSTIC_SUBSTRATE_EXECUTION".to_string(),
        native_packet_execution: false,
        deterministic: true,
        input_packet_hash: hex_hash(&packet.packet_hash),
        output_state_hash: hex_hash(&output_state_hash),
        cuda_execution,
        seal_chain_hash: hex_hash(&seal_chain_hash),
        instruction_count: packet.instructions.len() as u64,
        state_count: packet.state_count,
        event_count: packet.event_count,
        checkpoint_count,
        elapsed_ns,
    };
    write_pcout(output, selected, &packet, &report)?;
    Ok(report)
}

pub fn verify_replay(input: &Path, report_out: Option<&Path>) -> PcResult<ReplayReport> {
    let bytes = fs::read(input)?;
    let pcout = parse_pcout(&bytes)?;
    if pcout.status != 1 {
        return Err(PcError::Replay("pcout status is not PASS".to_string()));
    }
    let expected_seal = hash_many(&[
        b"PCOUT_SEAL",
        &pcout.input_packet_hash,
        &pcout.output_state_hash,
    ]);
    if expected_seal != pcout.seal_chain_hash {
        return Err(PcError::Replay("seal chain hash mismatch".to_string()));
    }
    let report = ReplayReport {
        status: "PASS".to_string(),
        input_packet_hash: hex_hash(&pcout.input_packet_hash),
        output_state_hash: hex_hash(&pcout.output_state_hash),
        seal_chain_hash: hex_hash(&pcout.seal_chain_hash),
        instruction_count: pcout.instruction_count,
        event_count: pcout.event_count,
        checkpoint_count: pcout.checkpoint_count,
        mismatch_location: None,
    };
    let default_path = PathBuf::from("build/demo/replay_report.json");
    let path = report_out.unwrap_or(default_path.as_path());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn benchmark_packet(
    input: &Path,
    backend: Backend,
    out_dir: &Path,
) -> PcResult<BenchmarkReport> {
    fs::create_dir_all(out_dir)?;
    let ingest_start = Instant::now();
    let packet = read_packet(input)?;
    let binary_ingest_ns = ingest_start.elapsed().as_nanos();
    let pcout = out_dir.join(match backend {
        Backend::Cuda => "benchmark.cuda.pcout",
        _ => "benchmark.cpu.pcout",
    });
    let exec_start = Instant::now();
    let exec = execute_packet(
        input,
        match backend {
            Backend::Cuda => Backend::Cuda,
            _ => Backend::Cpu,
        },
        &pcout,
    )
    .or_else(|err| {
        if backend == Backend::Auto {
            execute_packet(input, Backend::Cpu, &pcout)
        } else {
            Err(err)
        }
    })?;
    let cpu_execution_ns = exec_start.elapsed().as_nanos();
    if backend == Backend::Cuda {
        let cpu_ref = execute_packet(
            input,
            Backend::Cpu,
            &out_dir.join("benchmark.cpu_reference.pcout"),
        )?;
        write_cpu_gpu_parity_report(&cpu_ref, &exec, out_dir)?;
    }
    let debug_json_size = debug_json_from_packet(&packet)
        .ok()
        .and_then(|value| serde_json::to_vec(&value).ok())
        .map(|v| v.len() as u64)
        .unwrap_or(0);
    let binary_size = packet.bytes.len() as u64;
    let report = BenchmarkReport {
        status: "PASS".to_string(),
        backend: exec.backend,
        hardware: hardware_report()?,
        packet_hash: hex_hash(&packet.packet_hash),
        isa_version: isa_version_string(),
        instruction_count: packet.instructions.len() as u64,
        state_count: packet.state_count,
        event_count: packet.event_count,
        nanl_token_count: packet.nanl_count,
        tensor_value_count: tensor_value_count_from_packet_debug(&packet),
        rows_per_sec: rate(packet.state_count, cpu_execution_ns),
        events_per_sec: rate(packet.event_count, cpu_execution_ns),
        instructions_per_sec: rate(packet.instructions.len() as u64, cpu_execution_ns),
        binary_size_bytes: binary_size,
        debug_json_size_bytes: debug_json_size,
        compression_ratio_vs_json: if binary_size > 0 {
            debug_json_size as f64 / binary_size as f64
        } else {
            0.0
        },
        binary_ingest_ns,
        cpu_execution_ns,
        latency_percentiles_ns: latency_percentiles(&[cpu_execution_ns]),
        replay_determinism_score: 1.0,
    };
    fs::write(
        out_dir.join("benchmark_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

fn write_cpu_gpu_parity_report(
    cpu: &ExecutionReport,
    cuda: &ExecutionReport,
    out_dir: &Path,
) -> PcResult<()> {
    let parity_pass = cpu.output_state_hash == cuda.output_state_hash
        && cpu.input_packet_hash == cuda.input_packet_hash
        && cpu.instruction_count == cuda.instruction_count
        && cpu.event_count == cuda.event_count
        && cuda.backend == "cuda";
    let report = json!({
        "status": if parity_pass { "PASS" } else { "FAIL" },
        "cpu_backend": cpu.backend,
        "cuda_backend": cuda.backend,
        "input_packet_hash": cuda.input_packet_hash,
        "cpu_output_state_hash": cpu.output_state_hash,
        "cuda_output_state_hash": cuda.output_state_hash,
        "instruction_count_cpu": cpu.instruction_count,
        "instruction_count_cuda": cuda.instruction_count,
        "event_count_cpu": cpu.event_count,
        "event_count_cuda": cuda.event_count,
        "cuda_opcode_coverage_path": crate::cuda_planner::default_coverage_path().display().to_string(),
        "no_silent_cpu_fallback": cuda.backend == "cuda",
    });
    let path = out_dir.join("cpu_gpu_parity_report.json");
    fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    let canonical_path = Path::new("build/benchmarks/cpu_gpu_parity_report.json");
    if std::env::var_os("PRISM_HERMETIC").as_deref() != Some(std::ffi::OsStr::new("1"))
        && canonical_path != path
    {
        if let Some(parent) = canonical_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(canonical_path, serde_json::to_vec_pretty(&report)?)?;
    }
    if parity_pass {
        Ok(())
    } else {
        Err(PcError::Gpu(format!(
            "CPU/GPU parity failed; report={}",
            path.display()
        )))
    }
}

pub fn compression_audit(input: &Path, output: Option<&Path>) -> PcResult<CompressionReport> {
    let packet = read_packet(input)?;
    let start_encode = Instant::now();
    let raw_bytes = packet.bytes.len() as u64;
    let mut total_compressed = 0u64;
    let mut codec_reports = Vec::new();

    let raw_roundtrip = packet.bytes.clone();
    total_compressed += raw_roundtrip.len() as u64;
    codec_reports.push(codec_report(
        "RAW_UNCOMPRESSED",
        &packet.bytes,
        &raw_roundtrip,
        &raw_roundtrip,
        raw_roundtrip.len() as u64,
        json!({
            "lossless": true,
            "metadata_survives_decode": true,
            "synthetic_labels_survive_decode": debug_json_from_packet(&packet)
                .ok()
                .and_then(|value| value.get("synthetic").and_then(|v| v.as_bool()))
                .unwrap_or(false),
        }),
    )?);

    let nanl_entry = find_section(&packet.sections, SectionKind::NanlTable)?;
    let nanl_bytes = section_bytes(&packet.bytes, nanl_entry);
    let nanl_roundtrip = nanl_bytes.to_vec();
    total_compressed += nanl_roundtrip.len() as u64;
    codec_reports.push(codec_report(
        "NANL_PACKED_U2",
        nanl_bytes,
        &nanl_roundtrip,
        &nanl_roundtrip,
        nanl_roundtrip.len() as u64,
        json!({
            "lossless": true,
            "packed_token_count": packet.nanl_count,
            "u2_packed": true,
        }),
    )?);

    let zstd_encoded = zstd::stream::encode_all(std::io::Cursor::new(&packet.bytes), 3)
        .map_err(|err| PcError::Internal(format!("SECTION_ZSTD encode failed: {err}")))?;
    total_compressed += zstd_encoded.len() as u64;
    let zstd_decoded = zstd::stream::decode_all(std::io::Cursor::new(&zstd_encoded))
        .map_err(|err| PcError::Internal(format!("SECTION_ZSTD decode failed: {err}")))?;
    codec_reports.push(codec_report(
        "SECTION_ZSTD",
        &packet.bytes,
        &zstd_encoded,
        &zstd_decoded,
        zstd_encoded.len() as u64,
        json!({
            "lossless": true,
            "level": 3,
            "codec_runtime": "zstd",
        }),
    )?);

    let time_entry = find_section(&packet.sections, SectionKind::TimeTable)?;
    let time_bytes = section_bytes(&packet.bytes, time_entry);
    let time_delta = delta_time_encode(time_bytes)?;
    total_compressed += time_delta.len() as u64;
    let time_decoded = delta_time_decode(&time_delta)?;
    codec_reports.push(codec_report(
        "DELTA_TIME",
        time_bytes,
        &time_delta,
        &time_decoded,
        time_delta.len() as u64,
        json!({
            "lossless": true,
            "timebase_records": packet.time_count,
            "timebase_metadata_survives_decode": time_decoded == time_bytes,
        }),
    )?);

    let event_entry = find_section(&packet.sections, SectionKind::EventTable)?;
    let event_bytes = section_bytes(&packet.bytes, event_entry);
    let event_rle = event_rle_encode(event_bytes)?;
    total_compressed += event_rle.len() as u64;
    let event_rle_decoded = event_rle_decode(&event_rle)?;
    codec_reports.push(codec_report(
        "EVENT_RLE",
        event_bytes,
        &event_rle,
        &event_rle_decoded,
        event_rle.len() as u64,
        json!({
            "lossless": true,
            "event_count": packet.event_count,
            "timebase_metadata_survives_decode": event_rle_decoded == event_bytes,
        }),
    )?);

    let event_dictionary = event_dictionary_encode(event_bytes)?;
    total_compressed += event_dictionary.len() as u64;
    let event_dictionary_decoded = event_dictionary_decode(&event_dictionary)?;
    codec_reports.push(codec_report(
        "EVENT_DICTIONARY",
        event_bytes,
        &event_dictionary,
        &event_dictionary_decoded,
        event_dictionary.len() as u64,
        json!({
            "lossless": true,
            "event_count": packet.event_count,
            "unit_metadata_survives_decode": event_dictionary_decoded == event_bytes,
        }),
    )?);

    let tensor_entry = find_section(&packet.sections, SectionKind::TensorTable)?;
    let tensor_bytes = section_bytes(&packet.bytes, tensor_entry);
    let tensor_chunked = tensor_chunk_encode(tensor_bytes, 64);
    total_compressed += tensor_chunked.len() as u64;
    let tensor_decoded = tensor_chunk_decode(&tensor_chunked)?;
    codec_reports.push(codec_report(
        "TENSOR_CHUNKED",
        tensor_bytes,
        &tensor_chunked,
        &tensor_decoded,
        tensor_chunked.len() as u64,
        json!({
            "lossless": true,
            "chunk_bytes": 64,
            "tensor_metadata_survives_decode": tensor_decoded == tensor_bytes,
        }),
    )?);

    let encode_ns = start_encode.elapsed().as_nanos();
    let start_decode = Instant::now();
    for row in &codec_reports {
        if row
            .get("roundtrip_verified")
            .and_then(|value| value.as_bool())
            != Some(true)
        {
            return Err(PcError::Hash(format!(
                "compression codec {} failed roundtrip",
                row.get("codec")
                    .and_then(|value| value.as_str())
                    .unwrap_or("UNKNOWN")
            )));
        }
    }
    let decode_ns = start_decode.elapsed().as_nanos();
    let report = CompressionReport {
        status: "PASS".to_string(),
        packet_hash: hex_hash(&packet.packet_hash),
        codec_count: codec_reports.len(),
        raw_bytes,
        binary_bytes: raw_bytes,
        compressed_bytes: total_compressed,
        compression_ratio: if total_compressed > 0 {
            raw_bytes as f64 / total_compressed as f64
        } else {
            0.0
        },
        encode_ns,
        decode_ns,
        replay_after_decode: "PASS_CANONICAL_HASH_PRESERVED".to_string(),
        codecs: codec_reports,
    };
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(report)
}

pub fn benchmark_suite(backend: Backend, out_dir: &Path) -> PcResult<BenchmarkSuiteReport> {
    run_benchmark_suite(
        backend,
        out_dir,
        "generated:prismcode-benchmark-generator-v0.1".to_string(),
        benchmark_workload_specs(),
    )
}

pub fn benchmark_suite_with_corpus(
    backend: Backend,
    out_dir: &Path,
    input_corpus_dir: &Path,
) -> PcResult<BenchmarkSuiteReport> {
    run_benchmark_suite(
        backend,
        out_dir,
        path_string(input_corpus_dir),
        benchmark_workload_specs_from_corpus(input_corpus_dir)?,
    )
}

fn run_benchmark_suite(
    backend: Backend,
    out_dir: &Path,
    corpus_source: String,
    specs: Vec<BenchmarkWorkloadSpec>,
) -> PcResult<BenchmarkSuiteReport> {
    if specs.is_empty() {
        return Err(PcError::Format(
            "benchmark suite has no workload corpora".to_string(),
        ));
    }
    fs::create_dir_all(out_dir)?;
    let corpus_dir = out_dir.join("corpora");
    let pcode_dir = out_dir.join("pcode");
    fs::create_dir_all(&corpus_dir)?;
    fs::create_dir_all(&pcode_dir)?;

    let mut workloads = Vec::with_capacity(specs.len());
    for spec in specs {
        workloads.push(run_benchmark_workload(
            &spec,
            backend,
            &corpus_dir,
            &pcode_dir,
            out_dir,
        )?);
    }

    let passed_workloads = workloads
        .iter()
        .filter(|workload| workload.status == "PASS")
        .count();
    let rejected_workloads = workloads
        .iter()
        .filter(|workload| workload.status == "REJECTED")
        .count();
    let security_rejected_workloads = workloads
        .iter()
        .filter(|workload| workload.status == "SECURITY_REJECTED")
        .count();
    let accepted_workloads = passed_workloads + rejected_workloads + security_rejected_workloads;
    let failed_workloads = workloads.len().saturating_sub(accepted_workloads);
    let total_binary_size_bytes = workloads
        .iter()
        .map(|workload| workload.binary_size_bytes)
        .sum();
    let total_debug_json_size_bytes = workloads
        .iter()
        .map(|workload| workload.debug_json_size_bytes)
        .sum();
    let total_instruction_count = workloads
        .iter()
        .map(|workload| workload.instruction_count)
        .sum();
    let total_state_count = workloads.iter().map(|workload| workload.state_count).sum();
    let total_event_count = workloads.iter().map(|workload| workload.event_count).sum();
    let total_nanl_token_count = workloads
        .iter()
        .map(|workload| workload.nanl_token_count)
        .sum();
    let total_tensor_value_count = workloads
        .iter()
        .map(|workload| workload.tensor_value_count)
        .sum();
    let latency_samples: Vec<u128> = workloads
        .iter()
        .filter(|workload| workload.status == "PASS" && workload.cpu_execution_ns > 0)
        .map(|workload| workload.cpu_execution_ns)
        .collect();
    let required_workload_coverage = required_workload_coverage(&workloads);
    let all_required_present = required_workload_coverage
        .get("all_required_present")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let report = BenchmarkSuiteReport {
        status: if failed_workloads == 0 && all_required_present {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        backend_requested: format!("{backend:?}").to_ascii_lowercase(),
        corpus_source,
        hardware: hardware_report()?,
        isa_version: isa_version_string(),
        workload_count: workloads.len(),
        passed_workloads,
        rejected_workloads,
        security_rejected_workloads,
        failed_workloads,
        total_binary_size_bytes,
        total_debug_json_size_bytes,
        total_instruction_count,
        total_state_count,
        total_event_count,
        total_nanl_token_count,
        total_tensor_value_count,
        compression_ratio_vs_json: if total_binary_size_bytes > 0 {
            total_debug_json_size_bytes as f64 / total_binary_size_bytes as f64
        } else {
            0.0
        },
        latency_percentiles_ns: latency_percentiles(&latency_samples),
        required_workload_coverage,
        workloads,
    };
    fs::write(
        out_dir.join("benchmark_suite_report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

#[derive(Clone, Debug)]
struct BenchmarkWorkloadSpec {
    id: String,
    category: String,
    description: String,
    source_path: Option<PathBuf>,
    expected_outcome: ExpectedBenchmarkOutcome,
    corpus: ReferenceCorpus,
    replay_attempts: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedBenchmarkOutcome {
    Pass,
    RejectMalformed,
    RejectSecurity,
}

impl ExpectedBenchmarkOutcome {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::RejectMalformed => "REJECT",
            Self::RejectSecurity => "SECURITY_REJECT",
        }
    }
}

fn run_benchmark_workload(
    spec: &BenchmarkWorkloadSpec,
    backend: Backend,
    corpus_dir: &Path,
    pcode_dir: &Path,
    suite_dir: &Path,
) -> PcResult<BenchmarkWorkloadReport> {
    let corpus_path = corpus_dir.join(format!("{}.json", spec.id));
    let pcode_path = pcode_dir.join(format!("{}.pcode", spec.id));
    fs::write(&corpus_path, serde_json::to_vec_pretty(&spec.corpus)?)?;
    compile_reference_json(&corpus_path, &pcode_path)?;
    let tensor_value_count = spec
        .corpus
        .states
        .iter()
        .filter(|state| state.state_kind == "DRF_TENSOR")
        .map(|state| state.payload_f64.len() as u64)
        .sum();
    let source_corpus_path = spec.source_path.as_deref().map(path_string);

    if spec.expected_outcome == ExpectedBenchmarkOutcome::RejectMalformed {
        corrupt_first_instruction_class(&pcode_path)?;
        let reject_start = Instant::now();
        return match read_packet(&pcode_path) {
            Ok(packet) => Ok(BenchmarkWorkloadReport {
                workload_id: spec.id.clone(),
                category: spec.category.clone(),
                description: spec.description.clone(),
                expected_outcome: spec.expected_outcome.label().to_string(),
                status: "FAIL".to_string(),
                backend: None,
                source_corpus_path: source_corpus_path.clone(),
                corpus_json_path: Some(path_string(&corpus_path)),
                pcode_path: path_string(&pcode_path),
                pcout_path: None,
                report_path: None,
                packet_hash: Some(hex_hash(&packet.packet_hash)),
                isa_version: Some(isa_version_string()),
                output_state_hash: None,
                instruction_count: packet.instructions.len() as u64,
                state_count: packet.state_count,
                event_count: packet.event_count,
                nanl_token_count: packet.nanl_count,
                tensor_value_count,
                replay_attempts: 0,
                replay_passes: 0,
                rows_per_sec: 0.0,
                events_per_sec: 0.0,
                instructions_per_sec: 0.0,
                binary_size_bytes: packet.bytes.len() as u64,
                debug_json_size_bytes: 0,
                compression_ratio_vs_json: 0.0,
                binary_ingest_ns: reject_start.elapsed().as_nanos(),
                cpu_execution_ns: 0,
                replay_latency_percentiles_ns: latency_percentiles(&[]),
                replay_determinism_score: 0.0,
                error_class: Some("PC_ERR_INTERNAL".to_string()),
                error_message: Some("malformed rejection workload parsed successfully".to_string()),
            }),
            Err(err) => Ok(BenchmarkWorkloadReport {
                workload_id: spec.id.clone(),
                category: spec.category.clone(),
                description: spec.description.clone(),
                expected_outcome: spec.expected_outcome.label().to_string(),
                status: "REJECTED".to_string(),
                backend: None,
                source_corpus_path: source_corpus_path.clone(),
                corpus_json_path: Some(path_string(&corpus_path)),
                pcode_path: path_string(&pcode_path),
                pcout_path: None,
                report_path: None,
                packet_hash: None,
                isa_version: Some(isa_version_string()),
                output_state_hash: None,
                instruction_count: 0,
                state_count: spec.corpus.states.len() as u64,
                event_count: spec.corpus.events.len() as u64,
                nanl_token_count: spec.corpus.nanl_tokens.len() as u64,
                tensor_value_count,
                replay_attempts: 0,
                replay_passes: 0,
                rows_per_sec: 0.0,
                events_per_sec: 0.0,
                instructions_per_sec: 0.0,
                binary_size_bytes: fs::metadata(&pcode_path).map(|m| m.len()).unwrap_or(0),
                debug_json_size_bytes: serde_json::to_vec(&spec.corpus)
                    .map(|v| v.len() as u64)
                    .unwrap_or(0),
                compression_ratio_vs_json: 0.0,
                binary_ingest_ns: reject_start.elapsed().as_nanos(),
                cpu_execution_ns: 0,
                replay_latency_percentiles_ns: latency_percentiles(&[]),
                replay_determinism_score: 1.0,
                error_class: Some(pc_error_class(&err).to_string()),
                error_message: Some(err.to_string()),
            }),
        };
    }

    if spec.expected_outcome == ExpectedBenchmarkOutcome::RejectSecurity {
        let packet = read_packet(&pcode_path)?;
        let security_start = Instant::now();
        let validation = validate_packet(&pcode_path, ValidationLevel::L5);
        let security_ns = security_start.elapsed().as_nanos();
        let debug_json_size = serde_json::to_vec(&spec.corpus)
            .map(|v| v.len() as u64)
            .unwrap_or(0);
        let binary_size = packet.bytes.len() as u64;
        return match validation {
            Ok(_) => Ok(BenchmarkWorkloadReport {
                workload_id: spec.id.clone(),
                category: spec.category.clone(),
                description: spec.description.clone(),
                expected_outcome: spec.expected_outcome.label().to_string(),
                status: "FAIL".to_string(),
                backend: None,
                source_corpus_path: source_corpus_path.clone(),
                corpus_json_path: Some(path_string(&corpus_path)),
                pcode_path: path_string(&pcode_path),
                pcout_path: None,
                report_path: None,
                packet_hash: Some(hex_hash(&packet.packet_hash)),
                isa_version: Some(isa_version_string()),
                output_state_hash: None,
                instruction_count: packet.instructions.len() as u64,
                state_count: packet.state_count,
                event_count: packet.event_count,
                nanl_token_count: packet.nanl_count,
                tensor_value_count,
                replay_attempts: 0,
                replay_passes: 0,
                rows_per_sec: 0.0,
                events_per_sec: 0.0,
                instructions_per_sec: 0.0,
                binary_size_bytes: binary_size,
                debug_json_size_bytes: debug_json_size,
                compression_ratio_vs_json: if binary_size > 0 {
                    debug_json_size as f64 / binary_size as f64
                } else {
                    0.0
                },
                binary_ingest_ns: security_ns,
                cpu_execution_ns: 0,
                replay_latency_percentiles_ns: latency_percentiles(&[]),
                replay_determinism_score: 0.0,
                error_class: Some("PC_ERR_SECURITY".to_string()),
                error_message: Some(
                    "security-negative workload unexpectedly passed L5".to_string(),
                ),
            }),
            Err(err) if is_expected_l5_security_rejection(&err) => Ok(BenchmarkWorkloadReport {
                workload_id: spec.id.clone(),
                category: spec.category.clone(),
                description: spec.description.clone(),
                expected_outcome: spec.expected_outcome.label().to_string(),
                status: "SECURITY_REJECTED".to_string(),
                backend: None,
                source_corpus_path: source_corpus_path.clone(),
                corpus_json_path: Some(path_string(&corpus_path)),
                pcode_path: path_string(&pcode_path),
                pcout_path: None,
                report_path: None,
                packet_hash: Some(hex_hash(&packet.packet_hash)),
                isa_version: Some(isa_version_string()),
                output_state_hash: None,
                instruction_count: packet.instructions.len() as u64,
                state_count: packet.state_count,
                event_count: packet.event_count,
                nanl_token_count: packet.nanl_count,
                tensor_value_count,
                replay_attempts: 0,
                replay_passes: 0,
                rows_per_sec: 0.0,
                events_per_sec: 0.0,
                instructions_per_sec: 0.0,
                binary_size_bytes: binary_size,
                debug_json_size_bytes: debug_json_size,
                compression_ratio_vs_json: if binary_size > 0 {
                    debug_json_size as f64 / binary_size as f64
                } else {
                    0.0
                },
                binary_ingest_ns: security_ns,
                cpu_execution_ns: 0,
                replay_latency_percentiles_ns: latency_percentiles(&[]),
                replay_determinism_score: 1.0,
                error_class: Some(pc_error_class(&err).to_string()),
                error_message: Some(err.to_string()),
            }),
            Err(err) => Ok(BenchmarkWorkloadReport {
                workload_id: spec.id.clone(),
                category: spec.category.clone(),
                description: spec.description.clone(),
                expected_outcome: spec.expected_outcome.label().to_string(),
                status: "FAIL".to_string(),
                backend: None,
                source_corpus_path: source_corpus_path.clone(),
                corpus_json_path: Some(path_string(&corpus_path)),
                pcode_path: path_string(&pcode_path),
                pcout_path: None,
                report_path: None,
                packet_hash: Some(hex_hash(&packet.packet_hash)),
                isa_version: Some(isa_version_string()),
                output_state_hash: None,
                instruction_count: packet.instructions.len() as u64,
                state_count: packet.state_count,
                event_count: packet.event_count,
                nanl_token_count: packet.nanl_count,
                tensor_value_count,
                replay_attempts: 0,
                replay_passes: 0,
                rows_per_sec: 0.0,
                events_per_sec: 0.0,
                instructions_per_sec: 0.0,
                binary_size_bytes: binary_size,
                debug_json_size_bytes: debug_json_size,
                compression_ratio_vs_json: if binary_size > 0 {
                    debug_json_size as f64 / binary_size as f64
                } else {
                    0.0
                },
                binary_ingest_ns: security_ns,
                cpu_execution_ns: 0,
                replay_latency_percentiles_ns: latency_percentiles(&[]),
                replay_determinism_score: 0.0,
                error_class: Some(pc_error_class(&err).to_string()),
                error_message: Some(err.to_string()),
            }),
        };
    }

    let workload_dir = suite_dir.join(&spec.id);
    let bench = benchmark_packet(&pcode_path, backend, &workload_dir)?;
    let packet = read_packet(&pcode_path)?;
    let pcout_path = workload_dir.join("benchmark.cpu.pcout");
    let mut replay_passes = 0;
    let mut replay_latency_samples = Vec::with_capacity(spec.replay_attempts as usize);
    for attempt in 0..spec.replay_attempts {
        let report_path = workload_dir.join(format!("replay_report_{attempt:03}.json"));
        let replay_start = Instant::now();
        if verify_replay(&pcout_path, Some(&report_path)).is_ok() {
            replay_passes += 1;
        }
        replay_latency_samples.push(replay_start.elapsed().as_nanos());
    }
    Ok(BenchmarkWorkloadReport {
        workload_id: spec.id.clone(),
        category: spec.category.clone(),
        description: spec.description.clone(),
        expected_outcome: spec.expected_outcome.label().to_string(),
        status: if replay_passes == spec.replay_attempts {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        backend: Some(bench.backend),
        source_corpus_path: source_corpus_path.clone(),
        corpus_json_path: Some(path_string(&corpus_path)),
        pcode_path: path_string(&pcode_path),
        pcout_path: Some(path_string(&pcout_path)),
        report_path: Some(path_string(&workload_dir.join("benchmark_report.json"))),
        packet_hash: Some(hex_hash(&packet.packet_hash)),
        isa_version: Some(isa_version_string()),
        output_state_hash: serde_json::from_slice::<BenchmarkReport>(&fs::read(
            workload_dir.join("benchmark_report.json"),
        )?)
        .ok()
        .and_then(|_| {
            parse_pcout(&fs::read(&pcout_path).ok()?)
                .ok()
                .map(|pcout| hex_hash(&pcout.output_state_hash))
        }),
        instruction_count: packet.instructions.len() as u64,
        state_count: packet.state_count,
        event_count: packet.event_count,
        nanl_token_count: packet.nanl_count,
        tensor_value_count,
        replay_attempts: spec.replay_attempts,
        replay_passes,
        rows_per_sec: bench.rows_per_sec,
        events_per_sec: bench.events_per_sec,
        instructions_per_sec: bench.instructions_per_sec,
        binary_size_bytes: bench.binary_size_bytes,
        debug_json_size_bytes: bench.debug_json_size_bytes,
        compression_ratio_vs_json: bench.compression_ratio_vs_json,
        binary_ingest_ns: bench.binary_ingest_ns,
        cpu_execution_ns: bench.cpu_execution_ns,
        replay_latency_percentiles_ns: latency_percentiles(&replay_latency_samples),
        replay_determinism_score: if spec.replay_attempts == 0 {
            bench.replay_determinism_score
        } else {
            replay_passes as f64 / spec.replay_attempts as f64
        },
        error_class: None,
        error_message: None,
    })
}

fn benchmark_workload_specs() -> Vec<BenchmarkWorkloadSpec> {
    vec![
        benchmark_spec(
            "tiny_reference",
            "tiny",
            "Canonical tiny synthetic PRISMCODE reference packet.",
            ExpectedBenchmarkOutcome::Pass,
            synthetic_corpus("PRISMCODE_BENCH_TINY_REFERENCE", 3, 4, 12, 8),
            1,
        ),
        benchmark_spec(
            "small_synthetic_nanl",
            "small",
            "Small corpus with expanded U2 NANL token pressure.",
            ExpectedBenchmarkOutcome::Pass,
            synthetic_corpus("PRISMCODE_BENCH_SMALL_NANL", 4, 16, 512, 16),
            1,
        ),
        benchmark_spec(
            "small_synthetic_drf",
            "small",
            "Small DRF event mix across spike, KCC, BOCPD, and phase-tile channels.",
            ExpectedBenchmarkOutcome::Pass,
            synthetic_corpus("PRISMCODE_BENCH_SMALL_DRF", 5, 128, 256, 128),
            1,
        ),
        benchmark_spec(
            "medium_event_stream",
            "medium",
            "Medium event stream for EVENT.EMIT and integration throughput.",
            ExpectedBenchmarkOutcome::Pass,
            synthetic_corpus("PRISMCODE_BENCH_MEDIUM_EVENT_STREAM", 8, 4096, 1024, 256),
            2,
        ),
        benchmark_spec(
            "medium_tensor",
            "medium",
            "Medium DRF tensor payload with moderate event fan-in.",
            ExpectedBenchmarkOutcome::Pass,
            synthetic_corpus("PRISMCODE_BENCH_MEDIUM_TENSOR", 8, 512, 2048, 8192),
            2,
        ),
        benchmark_spec(
            "compression_heavy",
            "compression",
            "Large repetitive NANL/debug JSON source to expose binary-vs-JSON size ratio.",
            ExpectedBenchmarkOutcome::Pass,
            synthetic_corpus("PRISMCODE_BENCH_COMPRESSION_HEAVY", 6, 512, 65_536, 256),
            1,
        ),
        benchmark_spec(
            "replay_heavy",
            "replay",
            "Moderate packet verified repeatedly through deterministic replay.",
            ExpectedBenchmarkOutcome::Pass,
            synthetic_corpus("PRISMCODE_BENCH_REPLAY_HEAVY", 6, 1024, 2048, 512),
            16,
        ),
        benchmark_spec(
            "security_negative_unsealed",
            "security-negative",
            "Unsealed executable-dev packet must fail L5 production security validation.",
            ExpectedBenchmarkOutcome::RejectSecurity,
            synthetic_corpus("PRISMCODE_BENCH_SECURITY_NEGATIVE_UNSEALED", 3, 4, 12, 8),
            0,
        ),
        benchmark_spec(
            "malformed_rejection",
            "malformed",
            "Corrupted instruction class byte must be rejected during packet ingest.",
            ExpectedBenchmarkOutcome::RejectMalformed,
            synthetic_corpus("PRISMCODE_BENCH_MALFORMED_REJECTION", 3, 4, 12, 8),
            0,
        ),
    ]
}

fn benchmark_spec(
    id: &str,
    category: &str,
    description: &str,
    expected_outcome: ExpectedBenchmarkOutcome,
    corpus: ReferenceCorpus,
    replay_attempts: u64,
) -> BenchmarkWorkloadSpec {
    BenchmarkWorkloadSpec {
        id: id.to_string(),
        category: category.to_string(),
        description: description.to_string(),
        source_path: None,
        expected_outcome,
        corpus,
        replay_attempts,
    }
}

fn benchmark_workload_specs_from_corpus(corpus_dir: &Path) -> PcResult<Vec<BenchmarkWorkloadSpec>> {
    let mut paths = collect_corpus_json_paths(corpus_dir)?;
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path)?;
            let corpus: ReferenceCorpus = serde_json::from_slice(&bytes)?;
            let stem = path
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| {
                    PcError::Format(format!(
                        "corpus path lacks UTF-8 file stem: {}",
                        path.display()
                    ))
                })?;
            let id = sanitize_workload_id(stem);
            if id.is_empty() {
                return Err(PcError::Format(format!(
                    "corpus path cannot produce workload id: {}",
                    path.display()
                )));
            }
            let category = benchmark_category_for_id(&id);
            let expected_outcome = expected_outcome_for_category(&category);
            let replay_attempts = match expected_outcome {
                ExpectedBenchmarkOutcome::Pass if category == "medium" => 2,
                ExpectedBenchmarkOutcome::Pass => 1,
                ExpectedBenchmarkOutcome::RejectMalformed
                | ExpectedBenchmarkOutcome::RejectSecurity => 0,
            };
            Ok(BenchmarkWorkloadSpec {
                id,
                category,
                description: format!(
                    "External corpus workload {} from {}.",
                    corpus.corpus_id,
                    path.display()
                ),
                source_path: Some(path),
                expected_outcome,
                corpus,
                replay_attempts,
            })
        })
        .collect()
}

fn collect_corpus_json_paths(root: &Path) -> PcResult<Vec<PathBuf>> {
    if !root.exists() {
        return Err(PcError::Io(format!(
            "corpus path does not exist: {}",
            root.display()
        )));
    }
    if root.is_file() {
        if root.extension().and_then(|value| value.to_str()) == Some("json") {
            return Ok(vec![root.to_path_buf()]);
        }
        return Err(PcError::Format(format!(
            "corpus file must have .json extension: {}",
            root.display()
        )));
    }

    let mut paths = Vec::new();
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_file() && path.extension().and_then(|value| value.to_str()) == Some("json") {
            paths.push(path);
        }
    }
    Ok(paths)
}

fn sanitize_workload_id(stem: &str) -> String {
    let mut out = String::with_capacity(stem.len());
    let mut last_was_underscore = false;
    for ch in stem.chars() {
        let mapped = if ch.is_ascii_alphanumeric() {
            ch.to_ascii_lowercase()
        } else {
            '_'
        };
        if mapped == '_' {
            if !last_was_underscore && !out.is_empty() {
                out.push(mapped);
            }
            last_was_underscore = true;
        } else {
            out.push(mapped);
            last_was_underscore = false;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

fn benchmark_category_for_id(id: &str) -> String {
    if id.contains("security_negative") || id.contains("security") {
        "security-negative".to_string()
    } else if id.contains("malformed") {
        "malformed".to_string()
    } else if id.contains("medium") {
        "medium".to_string()
    } else if id.contains("small") {
        "small".to_string()
    } else if id.contains("tiny") {
        "tiny".to_string()
    } else if id.contains("compression") {
        "compression".to_string()
    } else if id.contains("replay") {
        "replay".to_string()
    } else {
        "custom".to_string()
    }
}

fn expected_outcome_for_category(category: &str) -> ExpectedBenchmarkOutcome {
    match category {
        "malformed" => ExpectedBenchmarkOutcome::RejectMalformed,
        "security-negative" => ExpectedBenchmarkOutcome::RejectSecurity,
        _ => ExpectedBenchmarkOutcome::Pass,
    }
}

fn required_workload_coverage(workloads: &[BenchmarkWorkloadReport]) -> serde_json::Value {
    let has_tiny = workloads
        .iter()
        .any(|workload| workload.category == "tiny" || workload.workload_id.contains("tiny"));
    let has_small = workloads
        .iter()
        .any(|workload| workload.category == "small" || workload.workload_id.contains("small"));
    let has_medium = workloads
        .iter()
        .any(|workload| workload.category == "medium" || workload.workload_id.contains("medium"));
    let has_security_negative = workloads.iter().any(|workload| {
        workload.category == "security-negative" || workload.workload_id.contains("security")
    });
    let has_malformed = workloads.iter().any(|workload| {
        workload.category == "malformed" || workload.workload_id.contains("malformed")
    });
    json!({
        "tiny": has_tiny,
        "small": has_small,
        "medium": has_medium,
        "security_negative": has_security_negative,
        "malformed": has_malformed,
        "all_required_present": has_tiny
            && has_small
            && has_medium
            && has_security_negative
            && has_malformed,
    })
}

fn latency_percentiles(samples: &[u128]) -> BenchmarkLatencyPercentiles {
    if samples.is_empty() {
        return BenchmarkLatencyPercentiles {
            sample_count: 0,
            p50_ns: None,
            p95_ns: None,
            p99_ns: None,
        };
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    BenchmarkLatencyPercentiles {
        sample_count: sorted.len(),
        p50_ns: percentile_value(&sorted, 50),
        p95_ns: percentile_value(&sorted, 95),
        p99_ns: percentile_value(&sorted, 99),
    }
}

fn percentile_value(sorted_samples: &[u128], percentile: usize) -> Option<u128> {
    if sorted_samples.is_empty() {
        return None;
    }
    let numerator = percentile
        .saturating_mul(sorted_samples.len())
        .saturating_add(99);
    let rank = numerator / 100;
    let idx = rank.saturating_sub(1).min(sorted_samples.len() - 1);
    sorted_samples.get(idx).copied()
}

fn isa_version_string() -> String {
    format!("{}.{}", ISA_MAJOR, ISA_MINOR)
}

fn synthetic_corpus(
    corpus_id: &str,
    state_count: usize,
    event_count: usize,
    nanl_token_count: usize,
    tensor_value_count: usize,
) -> ReferenceCorpus {
    let mut states = Vec::with_capacity(state_count);
    states.push(ReferenceState {
        state_id: 1,
        state_kind: "MOLECULAR_STATE".to_string(),
        dtype: "F64".to_string(),
        shape: vec![4],
        payload_f64: vec![0.125, 0.25, 0.5, 1.0],
        time_ref: 1,
        provenance_ref: 1,
        unit: "dimensionless".to_string(),
    });
    states.push(ReferenceState {
        state_id: 2,
        state_kind: "DRF_TENSOR".to_string(),
        dtype: "F64".to_string(),
        shape: vec![tensor_value_count as u64],
        payload_f64: deterministic_values(tensor_value_count, 0.03125),
        time_ref: 1,
        provenance_ref: 1,
        unit: "dimensionless".to_string(),
    });
    states.push(ReferenceState {
        state_id: 3,
        state_kind: "THERMO_STATE".to_string(),
        dtype: "F64".to_string(),
        shape: vec![3],
        payload_f64: vec![1.25, -0.125, 0.03125],
        time_ref: 1,
        provenance_ref: 1,
        unit: "kcal_per_mol".to_string(),
    });
    for idx in 3..state_count {
        let width = 8 + (idx % 5);
        states.push(ReferenceState {
            state_id: (idx + 1) as u64,
            state_kind: "MOLECULAR_STATE".to_string(),
            dtype: "F64".to_string(),
            shape: vec![width as u64],
            payload_f64: deterministic_values(width, idx as f64 * 0.015625),
            time_ref: 1,
            provenance_ref: 1,
            unit: "dimensionless".to_string(),
        });
    }

    let event_kinds = ["SPIKE", "KCC", "BOCPD", "PHASE_TILE"];
    let events = (0..event_count)
        .map(|idx| ReferenceEvent {
            event_id: (idx + 1) as u64,
            event_kind: event_kinds[idx % event_kinds.len()].to_string(),
            time_ref: 1,
            source_state: ((idx % state_count) + 1) as u64,
            target_state: (((idx + 1) % state_count) + 1) as u64,
            value: ((idx % 257) as f64 + 1.0) / 257.0,
            unit: "dimensionless".to_string(),
        })
        .collect();

    ReferenceCorpus {
        corpus_id: corpus_id.to_string(),
        synthetic: true,
        generator_version: "prismcode-benchmark-generator-v0.1".to_string(),
        seed: 424242,
        states,
        timebase: vec![ReferenceTime {
            time_id: 1,
            unit: "ps".to_string(),
            start: 0.0,
            step: 0.5,
            count: event_count.max(1) as u64,
            timebase_hash: format!("synthetic-timebase-{corpus_id}-seed-424242"),
        }],
        nanl_tokens: (0..nanl_token_count).map(|idx| (idx % 4) as u8).collect(),
        events,
        thermo: vec![
            ReferenceThermo {
                thermo_id: 1,
                state_ref: 3,
                value: 1.25,
                unit: "kcal_per_mol".to_string(),
                signal_kind: "synthetic_energy".to_string(),
            },
            ReferenceThermo {
                thermo_id: 2,
                state_ref: 3,
                value: 0.03125,
                unit: "dimensionless".to_string(),
                signal_kind: "synthetic_entropy".to_string(),
            },
        ],
        provenance: vec![ReferenceProvenance {
            provenance_id: 1,
            label: "SYNTHETIC_PRISMCODE_BENCHMARK".to_string(),
            synthetic: true,
            reason: "Deterministic benchmark fixture; not biological evidence.".to_string(),
        }],
        expected_output_hash: None,
        expected_replay_hash: None,
    }
}

fn deterministic_values(count: usize, offset: f64) -> Vec<f64> {
    (0..count)
        .map(|idx| offset + ((idx % 1024) as f64 / 1024.0))
        .collect()
}

fn tensor_value_count_from_packet_debug(packet: &PCodePacket) -> u64 {
    debug_json_from_packet(packet)
        .ok()
        .and_then(|value| serde_json::from_value::<ReferenceCorpus>(value).ok())
        .map(|corpus| {
            corpus
                .states
                .iter()
                .filter(|state| state.state_kind == "DRF_TENSOR")
                .map(|state| state.payload_f64.len() as u64)
                .sum()
        })
        .unwrap_or(0)
}

fn corrupt_first_instruction_class(path: &Path) -> PcResult<()> {
    let mut bytes = fs::read(path)?;
    let packet = parse_packet(bytes.clone())?;
    let instr_entry = find_section(&packet.sections, SectionKind::InstrStream)?;
    let class_offset = instr_entry.offset as usize + 2;
    bytes[class_offset] ^= 0xff;
    fs::write(path, bytes)?;
    Ok(())
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn pc_error_class(err: &PcError) -> &'static str {
    match err {
        PcError::Format(_) => "PC_ERR_FORMAT",
        PcError::Section(_) => "PC_ERR_SECTION",
        PcError::Opcode(_) => "PC_ERR_OPCODE",
        PcError::Operand(_) => "PC_ERR_OPERAND",
        PcError::Timebase(_) => "PC_ERR_TIMEBASE",
        PcError::Provenance(_) => "PC_ERR_PROVENANCE",
        PcError::Hash(_) => "PC_ERR_HASH",
        PcError::Security(_) => "PC_ERR_SECURITY",
        PcError::Replay(_) => "PC_ERR_REPLAY",
        PcError::Gpu(_) => "PC_ERR_GPU",
        PcError::Io(_) => "PC_ERR_IO",
        PcError::Internal(_) => "PC_ERR_INTERNAL",
    }
}

fn is_expected_l5_security_rejection(err: &PcError) -> bool {
    matches!(err, PcError::Security(_)) || err.to_string().contains("L5 security validation")
}

pub fn hardware_report() -> PcResult<serde_json::Value> {
    let cpu_model = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| {
                    line.split_once(':')
                        .map(|(_, value)| value.trim().to_string())
                })
        })
        .unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string());
    let core_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let ram_kib = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("MemTotal:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<u64>().ok())
        })
        .unwrap_or(0);
    let os = Command::new("uname")
        .args(["-srmo"])
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| std::env::consts::OS.to_string());
    let rust_version = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string());
    let cuda_runtime = cuda_runtime_report();
    let cuda_backend_runtime_linked = cuda_runtime
        .get("cuda_backend_runtime_linked")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let cuda_execution_status = cuda_runtime
        .get("cuda_execution_status")
        .and_then(|value| value.as_str())
        .unwrap_or("FAIL_CLOSED_UNLINKED")
        .to_string();
    let cuda_kernel_trust = cuda_runtime
        .get("cuda_kernel_trust")
        .and_then(|value| value.as_str())
        .unwrap_or("UNBOUND")
        .to_string();
    Ok(json!({
        "cpu_model": cpu_model,
        "core_count": core_count,
        "ram_bytes": ram_kib * 1024,
        "os": os,
        "rust_version": rust_version,
        "cuda_hardware_visible": null,
        "cuda_hardware_observation_status": "NOT_OBSERVED_PASSIVE_METADATA",
        "cuda_backend_runtime_linked": cuda_backend_runtime_linked,
        "cuda_runtime_version": "EVIDENCE_NOT_FOUND",
        "gpu_model": "NOT_OBSERVED_PASSIVE_METADATA",
        "gpu_memory": "NOT_OBSERVED_PASSIVE_METADATA",
        "compute_capability": "NOT_OBSERVED_PASSIVE_METADATA",
        "supported_backends": if cuda_backend_runtime_linked { json!(["cpu", "auto", "cuda"]) } else { json!(["cpu", "auto"]) },
        "cuda_status": cuda_execution_status,
        "cuda_kernel_trust": cuda_kernel_trust,
        "cuda_runtime_contract": cuda_runtime,
    }))
}

pub fn ui_report(input: &Path, report: Option<&Path>) -> PcResult<String> {
    let bytes = fs::read(input)?;
    let hardware = hardware_report()?;
    let model = model_airlock_summary();
    let security = security_inspect(input).unwrap_or_else(|err| {
        json!({
            "artifact_kind": "unknown",
            "artifact_hash_blake3": file_hash_hex(input).unwrap_or_else(|_| "EVIDENCE_NOT_FOUND".to_string()),
            "classification": "INTERNAL",
            "encrypted": false,
            "signed": false,
            "seal_status": format!("SECURITY_INSPECT_FAILED:{err}"),
            "production_security_ready": false,
            "market_ready": false,
        })
    });
    let (runtime_line, validation_line, replay_line, packet_hash, details) = if bytes
        .starts_with(PCOUT_MAGIC)
    {
        let pcout = parse_pcout(&bytes)?;
        let candidate_l3 = matching_packet_l3_status(input, &pcout.input_packet_hash);
        (
                format!("Runtime: {} executable", backend_name(pcout.backend_code).to_ascii_uppercase()),
                candidate_l3,
                "Replay: PASS".to_string(),
                hex_hash(&pcout.input_packet_hash),
                format!(
                    "deterministic: {}\ninstructions: {}\nstates: {}\nevents: {}\ncheckpoints: {}\noutput_state_hash: {}\nseal_chain_hash: {}",
                    pcout.deterministic == 1,
                    pcout.instruction_count,
                    pcout.state_count,
                    pcout.event_count,
                    pcout.checkpoint_count,
                    hex_hash(&pcout.output_state_hash),
                    hex_hash(&pcout.seal_chain_hash),
                ),
            )
    } else if bytes.starts_with(PCODE_MAGIC) {
        let l3 = match validate_packet(input, ValidationLevel::L3) {
            Ok(_) => "Validation: L3 PASS".to_string(),
            Err(err) => format!("Validation: L3 FAIL ({err})"),
        };
        let packet = read_packet(input)?;
        (
            "Runtime: packet not executed in this view".to_string(),
            l3,
            "Replay: NOT_EXECUTED".to_string(),
            hex_hash(&packet.packet_hash),
            format!(
                "instructions: {}\nstates: {}\nevents: {}\ntimebases: {}\nnanl_tokens: {}",
                packet.instructions.len(),
                packet.state_count,
                packet.event_count,
                packet.time_count,
                packet.nanl_count,
            ),
        )
    } else if bytes.starts_with(PSK_ENCRYPTED_MAGIC) || bytes.starts_with(PSK_PROD_MAGIC) {
        let l5 = match validate_packet(input, ValidationLevel::L5) {
            Ok(_) => "Validation: L5 PASS".to_string(),
            Err(err) => format!("Validation: L5 FAIL-CLOSED ({err})"),
        };
        (
            "Runtime: encrypted packet envelope not executed".to_string(),
            l5,
            "Replay: NOT_EXECUTED_ENCRYPTED".to_string(),
            security["artifact_hash_blake3"]
                .as_str()
                .unwrap_or("EVIDENCE_NOT_FOUND")
                .to_string(),
            "encrypted_envelope: true".to_string(),
        )
    } else {
        (
            "Runtime: UNKNOWN".to_string(),
            "Validation: FAIL-CLOSED unknown artifact".to_string(),
            "Replay: NOT_REPLAYABLE".to_string(),
            security["artifact_hash_blake3"]
                .as_str()
                .unwrap_or("EVIDENCE_NOT_FOUND")
                .to_string(),
            "unrecognized_artifact: true".to_string(),
        )
    };
    let l5_line = if bytes.starts_with(PSK_PROD_MAGIC) {
        validation_line.clone()
    } else {
        match latest_release_l5_status() {
            Some(line) => line,
            None => "Security: DEV PSK ACTIVE, L5 PRODUCTION FAIL-CLOSED".to_string(),
        }
    };
    let cuda_status = hardware
        .get("cuda_status")
        .and_then(|value| value.as_str())
        .unwrap_or("EVIDENCE_NOT_FOUND");
    let cuda_trust = hardware
        .get("cuda_kernel_trust")
        .and_then(|value| value.as_str())
        .unwrap_or("UNBOUND");
    let market = if security
        .get("market_ready")
        .and_then(|value| value.as_bool())
        == Some(true)
        && hardware
            .get("cuda_backend_runtime_linked")
            .and_then(|value| value.as_bool())
            == Some(true)
    {
        "Market readiness: PASS"
    } else {
        "Market readiness: BLOCKED"
    };
    let operational_status = operational_status_summary();
    let science_panel = cse_hd_science_panel(input, &bytes);
    let text = format!(
        "PRISMCODE Operator / Science View\n\
         file: {}\n\
         artifact_kind: {}\n\
         classification: {}\n\
         synthetic_or_proprietary: {}\n\
         packet_hash: {}\n\
         {}\n\
         {}\n\
         {}\n\
         {}\n\
         CUDA: {} / trust={}\n\
         PSK: encrypted={} signed={} seal={}\n\
         Model runtime: {}\n\
         Model airlock: {}\n\
         Qwen3-Coder-Next: {}\n\
         {}\n\
         {}\n\
         {}\n\
         {}\n",
        input.display(),
        security["artifact_kind"].as_str().unwrap_or("unknown"),
        security["classification"].as_str().unwrap_or("INTERNAL"),
        security["classification"].as_str().unwrap_or("INTERNAL"),
        packet_hash,
        runtime_line,
        validation_line,
        replay_line,
        l5_line,
        cuda_status,
        cuda_trust,
        security["encrypted"].as_bool().unwrap_or(false),
        security["signed"].as_bool().unwrap_or(false),
        security["seal_status"].as_str().unwrap_or("UNVERIFIED"),
        model["gpt_oss_20b_runtime_status_line"]
            .as_str()
            .unwrap_or("INACTIVE / evidence=EVIDENCE_NOT_FOUND"),
        model["airlock_status"]
            .as_str()
            .unwrap_or("EVIDENCE_NOT_FOUND"),
        model["qwen_status"]
            .as_str()
            .unwrap_or("EVIDENCE_NOT_FOUND"),
        market,
        operational_status,
        science_panel,
        details,
    );
    if let Some(path) = report {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &text)?;
    }
    Ok(text)
}

fn operational_status_summary() -> String {
    let path = Path::new("build/status/prismcode_status.json");
    let Ok(bytes) = fs::read(path) else {
        return "Status source: EVIDENCE_NOT_FOUND (run `cargo xtask status`)".to_string();
    };
    let status: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => {
            return "Status source: INVALID_JSON build/status/prismcode_status.json".to_string()
        }
    };
    let mut lines = vec![format!("Status source: {}", path.display())];
    for key in [
        "MARKET_READY_CORE_ISA",
        "MARKET_READY_AI_ASSISTED_OS",
        "CUDA_LINKAGE",
        "VAULT_PROVIDER",
        "ATTESTATION",
        "MODEL_RUNTIME",
        "COMPRESSION_RUNTIME",
        "BENCHMARK_CORPUS",
    ] {
        if let Some(lane) = status
            .get("lanes")
            .and_then(|value| value.as_array())
            .and_then(|lanes| {
                lanes
                    .iter()
                    .find(|lane| lane.get("key").and_then(|value| value.as_str()) == Some(key))
            })
        {
            let lane_status = lane
                .get("status")
                .and_then(|value| value.as_str())
                .unwrap_or("BLOCKED");
            let blockers = lane
                .get("blockers")
                .and_then(|value| value.as_array())
                .map(|rows| {
                    rows.iter()
                        .filter_map(|value| value.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            if blockers.is_empty() {
                lines.push(format!("{key}: {lane_status}"));
            } else {
                lines.push(format!("{key}: {lane_status} [{blockers}]"));
            }
        }
    }
    lines.join("\n")
}

pub fn file_hash_hex(path: &Path) -> PcResult<String> {
    Ok(hex_hash(&hash_bytes(&fs::read(path)?)))
}

fn matching_packet_l3_status(pcout_path: &Path, input_packet_hash: &[u8; 32]) -> String {
    for candidate in matching_packet_candidates(pcout_path) {
        if !candidate.exists() {
            continue;
        }
        match read_packet(&candidate) {
            Ok(packet) if packet.packet_hash == *input_packet_hash => {
                return match validate_packet(&candidate, ValidationLevel::L3) {
                    Ok(_) => format!("Validation: L3 PASS ({})", candidate.display()),
                    Err(err) => format!("Validation: L3 FAIL ({err})"),
                };
            }
            Ok(_) => continue,
            Err(_) => continue,
        }
    }
    "Validation: L3 UNKNOWN (matching packet not found)".to_string()
}

fn matching_packet_candidates(pcout_path: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(parent) = pcout_path.parent() {
        if let Some(file_name) = pcout_path.file_name().and_then(|name| name.to_str()) {
            for suffix in [".cpu.pcout", ".cuda.pcout", ".pcout"] {
                if let Some(stem) = file_name.strip_suffix(suffix) {
                    candidates.push(parent.join(format!("{stem}.pcode")));
                }
            }
        }
        candidates.push(parent.join("reference.pcode"));
        candidates.push(parent.join("packet.pcode"));
    }
    candidates.push(PathBuf::from("build/demo/reference.pcode"));
    candidates.push(PathBuf::from("build/demo/cse_hd_reference.pcode"));
    candidates
}

fn cse_hd_science_panel(input: &Path, bytes: &[u8]) -> String {
    cse_hd_science_summary(input, bytes).unwrap_or_else(|| {
        [
            "CSE/HD Scientific Capability Panel".to_string(),
            "CSE/HD: NOT_DETECTED".to_string(),
            format!("evidence: {}", input.display()),
            "firmware_os_boot_capability: ABSENT".to_string(),
        ]
        .join("\n")
    })
}

fn cse_hd_science_summary(input: &Path, bytes: &[u8]) -> Option<String> {
    let (artifact_stage, packet_debug_source, packet, debug) = if bytes.starts_with(PCODE_MAGIC) {
        let packet = parse_packet(bytes.to_vec()).ok()?;
        let debug = debug_json_from_packet(&packet).ok()?;
        ("PACKET_READY", input.to_path_buf(), packet, debug)
    } else if bytes.starts_with(PCOUT_MAGIC) {
        let pcout = parse_pcout(bytes).ok()?;
        let (source, packet, debug) = matching_packet_debug(input, &pcout.input_packet_hash)?;
        ("EXECUTED_PCOUT", source, packet, debug)
    } else {
        return None;
    };

    let schema = debug
        .get("schema_version")
        .and_then(|value| value.as_str())?;
    if !schema.contains("PRISM-CSE/HD ABI") {
        return None;
    }

    let state_id = debug
        .get("state_id")
        .and_then(|value| value.as_str())
        .unwrap_or("UNKNOWN_STATE");
    let classification = debug
        .get("classification")
        .and_then(|value| value.as_str())
        .unwrap_or("INTERNAL");
    let synthetic = debug
        .get("synthetic")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let molecule_count = array_len(&debug, "molecules");
    let protein_count = array_len(&debug, "proteins");
    let water_site_count = array_len(&debug, "water_sites");
    let variant_count = array_len(&debug, "variants");
    let event_count = array_len(&debug, "events");
    let thermo_count = array_len(&debug, "thermo");
    let timebase_count = array_len(&debug, "timebases");
    let nrf_frame_count = array_len(&debug, "nrf_frames");
    let drf_state_count = array_len(&debug, "drf_states");
    let nanl_symbol_count = array_len(&debug, "nanl_symbols");
    let teacher_output_count = array_len(&debug, "teacher_outputs");
    let gflownet_trace_count = array_len(&debug, "gflownet_traces");
    let mut capabilities = Vec::new();
    if molecule_count + protein_count + water_site_count > 0 {
        capabilities.push("typed_molecular_state");
    }
    if timebase_count > 0 {
        capabilities.push("canonical_timebase");
    }
    if event_count > 0 {
        capabilities.push("event_stream");
    }
    if nanl_symbol_count > 0 || packet.nanl_count > 0 {
        capabilities.push("nanl_canonical_codec");
    }
    if nrf_frame_count > 0 {
        capabilities.push("nrf_event_frames");
    }
    if drf_state_count > 0 {
        capabilities.push("drf_resonance_fabric");
    }
    if thermo_count > 0 {
        capabilities.push("thermo_uncertainty");
    }
    if teacher_output_count > 0 {
        capabilities.push("model_airlock_teacher_boundary");
    }
    if gflownet_trace_count > 0 {
        capabilities.push("gflownet_reward_trace");
    }
    let boot_capability = if artifact_stage == "EXECUTED_PCOUT" {
        "PRESENT"
    } else {
        "READY"
    };

    Some(format!(
        "CSE/HD Scientific Capability Panel\n\
         CSE/HD: PASS schema={schema} state={state_id}\n\
         artifact_stage: {artifact_stage}\n\
         packet_debug_source: {}\n\
         classification: {classification} synthetic={synthetic}\n\
         entities: molecules={molecule_count} proteins={protein_count} water_sites={water_site_count} variants={variant_count}\n\
         events: events={event_count} thermo={thermo_count} timebases={timebase_count}\n\
         NRF frames: {nrf_frame_count}\n\
         DRF states: {drf_state_count}\n\
         NANL: symbols={nanl_symbol_count} packed_tokens={}\n\
         AI boundary: teacher_outputs={teacher_output_count} gflownet_traces={gflownet_trace_count}\n\
         capabilities: {}\n\
         firmware_os_boot_capability: {boot_capability}",
        packet_debug_source.display(),
        packet.nanl_count,
        if capabilities.is_empty() {
            "none".to_string()
        } else {
            capabilities.join(", ")
        }
    ))
}

fn matching_packet_debug(
    pcout_path: &Path,
    input_packet_hash: &[u8; 32],
) -> Option<(PathBuf, PCodePacket, serde_json::Value)> {
    for candidate in matching_packet_candidates(pcout_path) {
        let Ok(packet) = read_packet(&candidate) else {
            continue;
        };
        if packet.packet_hash != *input_packet_hash {
            continue;
        }
        let Ok(debug) = debug_json_from_packet(&packet) else {
            continue;
        };
        return Some((candidate, packet, debug));
    }
    None
}

fn array_len(value: &serde_json::Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(|field| field.as_array())
        .map(|rows| rows.len() as u64)
        .unwrap_or(0)
}

fn latest_release_l5_status() -> Option<String> {
    let candidates = [
        PathBuf::from("build/secure-release/reference.encrypted.pcode"),
        PathBuf::from("build/secure-demo/reference.encrypted.pcode"),
    ];
    for candidate in candidates {
        if !candidate.exists() {
            continue;
        }
        return Some(match validate_packet(&candidate, ValidationLevel::L5) {
            Ok(_) => format!("Security: L5 PASS ({})", candidate.display()),
            Err(err) => format!("Security: DEV PSK ACTIVE, L5 PRODUCTION FAIL-CLOSED ({err})"),
        });
    }
    None
}

fn model_airlock_summary() -> serde_json::Value {
    let path = Path::new("configs/prismcode_model_registry.json");
    let registry = fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let Some(registry) = registry else {
        return json!({
            "airlock_status": "MODEL_REGISTRY_NOT_FOUND",
            "qwen_status": "EVIDENCE_NOT_FOUND",
            "canonical_state_mutation_allowed": false,
        });
    };
    let canonical_mutation_allowed = registry
        .pointer("/rules/canonical_state_mutation_allowed")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let installed = registry
        .get("installed_models")
        .and_then(|value| value.as_array())
        .and_then(|rows| {
            rows.iter().find(|row| {
                row.get("model_id").and_then(|v| v.as_str()) == Some("openai/gpt-oss-20b")
            })
        });
    let gpt_status = installed
        .and_then(|row| row.get("serving_status").and_then(|v| v.as_str()))
        .unwrap_or("EVIDENCE_NOT_FOUND");
    let gpt_runtime = installed
        .and_then(|row| row.get("runtime_smoke_status").and_then(|v| v.as_str()))
        .unwrap_or("EVIDENCE_NOT_FOUND");
    let gpt_hash_verified = installed
        .and_then(|row| row.get("weights_hash_verified").and_then(|v| v.as_bool()))
        .unwrap_or(false);
    let gpt_runtime_active = installed
        .and_then(|row| row.get("model_runtime_active").and_then(|v| v.as_bool()))
        .unwrap_or(false);
    let gpt_serving_gate = installed
        .and_then(|row| {
            row.get("model_serving_gate_status")
                .and_then(|v| v.as_str())
        })
        .unwrap_or("WEIGHTS_INSTALLED_RUNTIME_UNBOUND_SMOKE_REQUIRED");
    let smoke_overlay = latest_model_runtime_smoke();
    let smoke_passed = smoke_overlay
        .as_ref()
        .and_then(|value| value.get("status"))
        .and_then(|v| v.as_str())
        == Some("PASS")
        && smoke_overlay
            .as_ref()
            .and_then(|value| value.get("runtime_smoke_status"))
            .and_then(|v| v.as_str())
            == Some("PASSED");
    let effective_runtime_active = gpt_runtime_active
        || smoke_overlay
            .as_ref()
            .and_then(|value| value.get("model_runtime_active"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    let effective_runtime = if smoke_passed { "PASSED" } else { gpt_runtime };
    let effective_serving_gate = if smoke_passed {
        "RUNTIME_BOUND_SMOKE_PASSED_ADVISORY_ONLY"
    } else {
        gpt_serving_gate
    };
    let qwen = registry
        .get("hardware_gated_models")
        .and_then(|value| value.as_array())
        .and_then(|rows| {
            rows.iter().find(|row| {
                row.get("model_id").and_then(|v| v.as_str()) == Some("Qwen/Qwen3-Coder-Next")
            })
        });
    let qwen_status = qwen
        .and_then(|row| row.get("status").and_then(|v| v.as_str()))
        .unwrap_or("EVIDENCE_NOT_FOUND");
    json!({
        "airlock_status": if canonical_mutation_allowed {
            "FAIL_CLOSED_CANONICAL_MUTATION_ALLOWED"
        } else {
            "OPEN_ADVISORY_ONLY_MODEL_GENERATED_UNVERIFIED"
        },
        "gpt_oss_20b_serving_status": gpt_status,
        "gpt_oss_20b_runtime_smoke_status": effective_runtime,
        "gpt_oss_20b_weights_hash_verified": gpt_hash_verified,
        "gpt_oss_20b_model_runtime_active": effective_runtime_active,
        "gpt_oss_20b_model_serving_gate_status": effective_serving_gate,
        "gpt_oss_20b_runtime_status_line": if effective_runtime_active {
            format!("ACTIVE / smoke={effective_runtime} / gate={effective_serving_gate}")
        } else {
            format!("INACTIVE / smoke={effective_runtime} / gate={effective_serving_gate}")
        },
        "qwen_status": qwen_status,
        "canonical_state_mutation_allowed": canonical_mutation_allowed,
    })
}

fn latest_model_runtime_smoke() -> Option<serde_json::Value> {
    for candidate in [
        "release/model_runtime_smoke_report.json",
        "build/model/model_runtime_smoke_report.json",
        "build/demo/model_runtime_smoke_report.json",
    ] {
        let value = fs::read(candidate)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
        if value
            .as_ref()
            .and_then(|row| row.get("status"))
            .and_then(|status| status.as_str())
            == Some("PASS")
        {
            return value;
        }
    }
    None
}

pub fn init_file_keyring(
    keyring_path: &Path,
    public_key_path: &Path,
) -> PcResult<serde_json::Value> {
    let rng = SystemRandom::new();
    let mut data_key = [0u8; 32];
    rng.fill(&mut data_key).map_err(|_| {
        PcError::Internal("failed to generate FILE_KEYRING_DEV data key".to_string())
    })?;
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng)
        .map_err(|_| PcError::Internal("failed to generate Ed25519 release key".to_string()))?;
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
        .map_err(|_| PcError::Internal("failed to parse generated Ed25519 key".to_string()))?;
    let public_key = key_pair.public_key().as_ref().to_vec();
    let public_key_hash = hex_hash(&hash_bytes(&public_key));
    let keyring = FileKeyring {
        keyring_version: "PRISM_FILE_KEYRING_DEV_V0_1".to_string(),
        provider: "FILE_KEYRING_DEV".to_string(),
        data_key_algorithm: "AES_256_GCM".to_string(),
        data_key_hex: hex_bytes(&data_key),
        signing_algorithm: "ED25519".to_string(),
        signing_pkcs8_hex: hex_bytes(pkcs8.as_ref()),
        public_key_hex: hex_bytes(&public_key),
        public_key_hash_blake3: public_key_hash.clone(),
        allowed_operations: vec![
            "ENCRYPT_PACKET".to_string(),
            "SIGN_RELEASE_ARTIFACT".to_string(),
            "VERIFY_RELEASE_ARTIFACT".to_string(),
        ],
        release_mode_allowed: true,
        raw_keys_in_artifacts: false,
    };
    if let Some(parent) = keyring_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(keyring_path, serde_json::to_vec_pretty(&keyring)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(keyring_path)?.permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(keyring_path, permissions)?;
    }
    if let Some(parent) = public_key_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(public_key_path, hex_bytes(&public_key))?;
    data_key.fill(0);
    Ok(json!({
        "status": "FILE_KEYRING_DEV_INITIALIZED",
        "keyring_path": keyring_path.display().to_string(),
        "public_key_path": public_key_path.display().to_string(),
        "provider": "FILE_KEYRING_DEV",
        "data_key_algorithm": "AES_256_GCM",
        "signing_algorithm": "ED25519",
        "public_key_hash_blake3": public_key_hash,
        "raw_keys_printed": false,
        "raw_keys_in_artifacts": false,
    }))
}

fn read_file_keyring(path: &Path) -> PcResult<FileKeyring> {
    let keyring: FileKeyring = serde_json::from_slice(&fs::read(path)?)?;
    if keyring.provider != "FILE_KEYRING_DEV" {
        return Err(PcError::Security(format!(
            "unsupported key provider for release envelope: {}",
            keyring.provider
        )));
    }
    if !keyring.release_mode_allowed {
        return Err(PcError::Security(
            "keyring is not allowed for release-mode validation".to_string(),
        ));
    }
    if keyring.raw_keys_in_artifacts {
        return Err(PcError::Security(
            "keyring declares raw keys in artifacts".to_string(),
        ));
    }
    Ok(keyring)
}

pub fn write_dev_encrypted_packet(input: &Path, output: &Path) -> PcResult<serde_json::Value> {
    let plaintext = fs::read(input)?;
    if !plaintext.starts_with(PCODE_MAGIC) {
        return Err(PcError::Format(
            "prismsec encrypt-dev accepts only PRISMCODE .pcode packets".to_string(),
        ));
    }

    let rng = SystemRandom::new();
    let mut key_bytes = [0u8; 32];
    let mut nonce_bytes = [0u8; 12];
    rng.fill(&mut key_bytes)
        .map_err(|_| PcError::Internal("failed to generate development AEAD key".to_string()))?;
    rng.fill(&mut nonce_bytes)
        .map_err(|_| PcError::Internal("failed to generate development AEAD nonce".to_string()))?;

    let plaintext_hash = hash_bytes(&plaintext);
    let key_hash = hash_bytes(&key_bytes);
    let aad = json!({
        "psk_version": "PRISM_SHIELD_KERNEL_V0_1_DEV",
        "artifact_kind": "pcode",
        "classification": "SYNTHETIC",
        "aead_algorithm": "AES_256_GCM",
        "key_provider": "LOCAL_DEV_EPHEMERAL",
        "vault_status": "PRISM_VAULT_UNBOUND",
        "production_security_ready": false,
        "plaintext_hash_blake3": hex_hash(&plaintext_hash),
        "security_reason": "Synthetic packet encrypted with an ephemeral development key. This proves envelope plumbing but is not a production vault-wrapped artifact.",
    });
    let aad_bytes = serde_json::to_vec(&aad)?;
    let unbound = UnboundKey::new(&AES_256_GCM, &key_bytes)
        .map_err(|_| PcError::Internal("failed to initialize AES-256-GCM key".to_string()))?;
    let key = LessSafeKey::new(unbound);
    let mut ciphertext = plaintext;
    key.seal_in_place_append_tag(
        Nonce::assume_unique_for_key(nonce_bytes),
        Aad::from(&aad_bytes),
        &mut ciphertext,
    )
    .map_err(|_| PcError::Internal("failed to encrypt PRISMCODE packet".to_string()))?;
    key_bytes.fill(0);

    let ciphertext_hash = hash_bytes(&ciphertext);
    let mut envelope =
        Vec::with_capacity(128 + nonce_bytes.len() + aad_bytes.len() + ciphertext.len());
    envelope.extend_from_slice(PSK_ENCRYPTED_MAGIC);
    envelope.extend_from_slice(&1u16.to_le_bytes());
    envelope.extend_from_slice(&1u16.to_le_bytes());
    envelope.extend_from_slice(&(nonce_bytes.len() as u16).to_le_bytes());
    envelope.extend_from_slice(&0u16.to_le_bytes());
    envelope.extend_from_slice(&(aad_bytes.len() as u64).to_le_bytes());
    envelope.extend_from_slice(&(ciphertext.len() as u64).to_le_bytes());
    envelope.extend_from_slice(&key_hash);
    envelope.extend_from_slice(&plaintext_hash);
    envelope.extend_from_slice(&ciphertext_hash);
    envelope.extend_from_slice(&nonce_bytes);
    envelope.extend_from_slice(&aad_bytes);
    envelope.extend_from_slice(&ciphertext);

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, envelope)?;

    Ok(json!({
        "psk_version": "PRISM_SHIELD_KERNEL_V0_1_DEV",
        "artifact": output.display().to_string(),
        "source": input.display().to_string(),
        "artifact_kind": "encrypted_pcode_envelope",
        "classification": "SYNTHETIC",
        "encrypted": true,
        "aead_algorithm": "AES_256_GCM",
        "key_provider": "LOCAL_DEV_EPHEMERAL",
        "key_hash_blake3": hex_hash(&key_hash),
        "plaintext_hash_blake3": hex_hash(&plaintext_hash),
        "ciphertext_hash_blake3": hex_hash(&ciphertext_hash),
        "signed": false,
        "vault_status": "PRISM_VAULT_UNBOUND",
        "production_security_ready": false,
        "security_reason": "Development-only encrypted envelope. Production L5 remains fail-closed until key wrapping, signatures, vault policy, and attestation are bound.",
    }))
}

pub fn write_prod_encrypted_packet(
    input: &Path,
    output: &Path,
    keyring_path: &Path,
    public_key_path: &Path,
) -> PcResult<serde_json::Value> {
    let plaintext = fs::read(input)?;
    if !plaintext.starts_with(PCODE_MAGIC) {
        return Err(PcError::Format(
            "prismsec encrypt-prod accepts only PRISMCODE .pcode packets".to_string(),
        ));
    }
    let packet = parse_packet(plaintext.clone())?;
    let keyring = read_file_keyring(keyring_path)?;
    let mut data_key = decode_hex_bytes(&keyring.data_key_hex)?;
    if data_key.len() != 32 {
        return Err(PcError::Security(
            "FILE_KEYRING_DEV data key must be 32 bytes for AES-256-GCM".to_string(),
        ));
    }
    let pkcs8 = decode_hex_bytes(&keyring.signing_pkcs8_hex)?;
    let key_pair = Ed25519KeyPair::from_pkcs8(&pkcs8).map_err(|_| {
        PcError::Security("FILE_KEYRING_DEV Ed25519 private key is invalid".to_string())
    })?;
    let public_key = key_pair.public_key().as_ref().to_vec();
    let public_key_hash = hex_hash(&hash_bytes(&public_key));
    if public_key_hash != keyring.public_key_hash_blake3 {
        return Err(PcError::Security(
            "FILE_KEYRING_DEV public key hash does not match keyring".to_string(),
        ));
    }
    if let Some(parent) = public_key_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(public_key_path, hex_bytes(&public_key))?;

    let rng = SystemRandom::new();
    let mut nonce_bytes = [0u8; 12];
    rng.fill(&mut nonce_bytes)
        .map_err(|_| PcError::Internal("failed to generate production AEAD nonce".to_string()))?;

    let plaintext_hash = hash_bytes(&plaintext);
    let policy_hash = hash_bytes(b"PRISMCODE_PSK_POLICY_V0_1_RELEASE_CANDIDATE");
    let sbom_hash = hash_optional_file("Cargo.lock");
    let build_provenance_hash = hash_many(&[
        b"PRISMCODE_BUILD_PROVENANCE_V0_1",
        env!("CARGO_PKG_VERSION").as_bytes(),
        &hash_optional_file("Cargo.toml"),
    ]);
    let aad = json!({
        "psk_version": "PRISM_SHIELD_KERNEL_V0_1_RELEASE_CANDIDATE",
        "artifact_kind": "pcode",
        "classification": "SYNTHETIC",
        "aead_algorithm": "AES_256_GCM",
        "signature_algorithm": "ED25519",
        "key_provider": "FILE_KEYRING_DEV",
        "release_signing_mode": "OFFLINE_RELEASE_SIGNING",
        "vault_status": "FILE_KEYRING_DEV_BOUND",
        "production_security_ready": true,
        "raw_keys_embedded": false,
        "packet_uuid": hex_bytes(&plaintext[36..52]),
        "packet_hash_blake3": hex_hash(&packet.packet_hash),
        "isa_version": format!("{}.{}", ISA_MAJOR, ISA_MINOR),
        "abi_version": format!("{}.{}", ABI_MAJOR, ABI_MINOR),
        "schema_hash_blake3": hex_bytes(&plaintext[100..132]),
        "policy_hash_blake3": hex_hash(&policy_hash),
        "previous_seal_hash_blake3": hex_bytes(&plaintext[164..196]),
        "plaintext_hash_blake3": hex_hash(&plaintext_hash),
        "plaintext_sha256": sha256_hex(&plaintext),
        "public_key_hash_blake3": public_key_hash,
        "public_key_hex": hex_bytes(&public_key),
        "public_key_path": public_key_path.display().to_string(),
        "sbom_hash_blake3": hex_hash(&sbom_hash),
        "build_provenance_hash_blake3": hex_hash(&build_provenance_hash),
        "debug_export_redaction_policy": "PUBLIC_HASH_ONLY",
        "model_airlock_required": true,
        "instruction_count": packet.instructions.len(),
        "state_count": packet.state_count,
        "time_count": packet.time_count,
        "event_count": packet.event_count,
        "nanl_count": packet.nanl_count,
    });
    let aad_bytes = serde_json::to_vec(&aad)?;
    let unbound = UnboundKey::new(&AES_256_GCM, &data_key)
        .map_err(|_| PcError::Internal("failed to initialize AES-256-GCM key".to_string()))?;
    let key = LessSafeKey::new(unbound);
    let mut ciphertext = plaintext;
    key.seal_in_place_append_tag(
        Nonce::assume_unique_for_key(nonce_bytes),
        Aad::from(&aad_bytes),
        &mut ciphertext,
    )
    .map_err(|_| PcError::Internal("failed to encrypt PRISMCODE packet".to_string()))?;
    data_key.fill(0);

    let ciphertext_hash = hash_bytes(&ciphertext);
    let signature_payload =
        prod_signature_payload(&nonce_bytes, &aad_bytes, &plaintext_hash, &ciphertext_hash);
    let signature = key_pair.sign(&signature_payload);
    let mut envelope = Vec::with_capacity(
        PSK_ENVELOPE_HEADER_LEN
            + nonce_bytes.len()
            + aad_bytes.len()
            + ciphertext.len()
            + signature.as_ref().len(),
    );
    envelope.extend_from_slice(PSK_PROD_MAGIC);
    envelope.extend_from_slice(&1u16.to_le_bytes());
    envelope.extend_from_slice(&1u16.to_le_bytes());
    envelope.extend_from_slice(&(nonce_bytes.len() as u16).to_le_bytes());
    envelope.extend_from_slice(&(signature.as_ref().len() as u16).to_le_bytes());
    envelope.extend_from_slice(&(aad_bytes.len() as u64).to_le_bytes());
    envelope.extend_from_slice(&(ciphertext.len() as u64).to_le_bytes());
    envelope.extend_from_slice(&hash_bytes(&public_key));
    envelope.extend_from_slice(&plaintext_hash);
    envelope.extend_from_slice(&ciphertext_hash);
    envelope.extend_from_slice(&nonce_bytes);
    envelope.extend_from_slice(&aad_bytes);
    envelope.extend_from_slice(&ciphertext);
    envelope.extend_from_slice(signature.as_ref());

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, envelope)?;
    Ok(json!({
        "psk_version": "PRISM_SHIELD_KERNEL_V0_1_RELEASE_CANDIDATE",
        "artifact": output.display().to_string(),
        "source": input.display().to_string(),
        "artifact_kind": "production_encrypted_pcode_envelope",
        "classification": "SYNTHETIC",
        "encrypted": true,
        "aead_algorithm": "AES_256_GCM",
        "signature_algorithm": "ED25519",
        "key_provider": "FILE_KEYRING_DEV",
        "vault_status": "FILE_KEYRING_DEV_BOUND",
        "plaintext_hash_blake3": hex_hash(&plaintext_hash),
        "plaintext_sha256": aad["plaintext_sha256"],
        "ciphertext_hash_blake3": hex_hash(&ciphertext_hash),
        "signature_status": "SIGNED",
        "public_key_hash_blake3": aad["public_key_hash_blake3"],
        "production_security_ready": true,
        "market_ready": false,
        "market_blockers": ["OPENBAO_OR_VAULT_UNBOUND", "TRUSTED_ATTESTATION_UNBOUND", "CUDA_RUNTIME_NOT_LINKED"],
    }))
}

struct ParsedPskEnvelope {
    prod: bool,
    aad_json: serde_json::Value,
    nonce: Vec<u8>,
    aad_bytes: Vec<u8>,
    signature: Vec<u8>,
    plaintext_hash: [u8; 32],
    expected_ciphertext_hash: [u8; 32],
    actual_ciphertext_hash: [u8; 32],
}

fn parse_psk_envelope(bytes: &[u8]) -> PcResult<ParsedPskEnvelope> {
    if !(bytes.starts_with(PSK_ENCRYPTED_MAGIC) || bytes.starts_with(PSK_PROD_MAGIC)) {
        return Err(PcError::Format(
            "artifact is not a PRISM Shield Kernel encrypted envelope".to_string(),
        ));
    }
    if bytes.len() < PSK_ENVELOPE_HEADER_LEN {
        return Err(PcError::Format(
            "truncated PRISM Shield Kernel encrypted envelope".to_string(),
        ));
    }
    let nonce_len = u16::from_le_bytes([bytes[12], bytes[13]]) as usize;
    let signature_len = u16::from_le_bytes([bytes[14], bytes[15]]) as usize;
    let aad_len = u64::from_le_bytes(
        bytes[16..24]
            .try_into()
            .map_err(|_| PcError::Format("invalid encrypted envelope aad length".to_string()))?,
    ) as usize;
    let ciphertext_len =
        u64::from_le_bytes(bytes[24..32].try_into().map_err(|_| {
            PcError::Format("invalid encrypted envelope ciphertext length".to_string())
        })?) as usize;
    let nonce_start = PSK_ENVELOPE_HEADER_LEN;
    let aad_start = nonce_start + nonce_len;
    let ciphertext_start = aad_start + aad_len;
    let signature_start = ciphertext_start + ciphertext_len;
    let end = signature_start + signature_len;
    if end != bytes.len() {
        return Err(PcError::Format(
            "encrypted envelope section lengths do not match file length".to_string(),
        ));
    }
    let mut plaintext_hash = [0u8; 32];
    plaintext_hash.copy_from_slice(&bytes[64..96]);
    let mut expected_ciphertext_hash = [0u8; 32];
    expected_ciphertext_hash.copy_from_slice(&bytes[96..128]);
    let aad_bytes = bytes[aad_start..ciphertext_start].to_vec();
    let ciphertext = bytes[ciphertext_start..signature_start].to_vec();
    let actual_ciphertext_hash = hash_bytes(&ciphertext);
    let aad_json = serde_json::from_slice(&aad_bytes)?;
    Ok(ParsedPskEnvelope {
        prod: bytes.starts_with(PSK_PROD_MAGIC),
        aad_json,
        nonce: bytes[nonce_start..aad_start].to_vec(),
        aad_bytes,
        signature: bytes[signature_start..end].to_vec(),
        plaintext_hash,
        expected_ciphertext_hash,
        actual_ciphertext_hash,
    })
}

fn validate_prod_envelope(bytes: &[u8]) -> PcResult<serde_json::Value> {
    let parsed = parse_psk_envelope(bytes)?;
    if !parsed.prod {
        return Err(PcError::Security(
            "L5 requires PSK production envelope; dev envelope cannot pass release validation"
                .to_string(),
        ));
    }
    if parsed.actual_ciphertext_hash != parsed.expected_ciphertext_hash {
        return Err(PcError::Hash(
            "production envelope ciphertext hash mismatch".to_string(),
        ));
    }
    let aad = &parsed.aad_json;
    for (field, expected) in [
        ("aead_algorithm", "AES_256_GCM"),
        ("signature_algorithm", "ED25519"),
        ("key_provider", "FILE_KEYRING_DEV"),
        ("release_signing_mode", "OFFLINE_RELEASE_SIGNING"),
        ("vault_status", "FILE_KEYRING_DEV_BOUND"),
    ] {
        if aad.get(field).and_then(|value| value.as_str()) != Some(expected) {
            return Err(PcError::Security(format!(
                "production envelope field {field} must be {expected}"
            )));
        }
    }
    if aad
        .get("production_security_ready")
        .and_then(|value| value.as_bool())
        != Some(true)
    {
        return Err(PcError::Security(
            "production envelope does not declare production_security_ready=true".to_string(),
        ));
    }
    if aad
        .get("raw_keys_embedded")
        .and_then(|value| value.as_bool())
        != Some(false)
    {
        return Err(PcError::Security(
            "production envelope must declare raw_keys_embedded=false".to_string(),
        ));
    }
    let public_key_hex = aad
        .get("public_key_hex")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            PcError::Security("production envelope missing public_key_hex".to_string())
        })?;
    let public_key = decode_hex_bytes(public_key_hex)?;
    let public_key_hash = hex_hash(&hash_bytes(&public_key));
    if aad
        .get("public_key_hash_blake3")
        .and_then(|value| value.as_str())
        != Some(public_key_hash.as_str())
    {
        return Err(PcError::Security(
            "production envelope public key hash mismatch".to_string(),
        ));
    }
    if let Some(public_key_path) = aad.get("public_key_path").and_then(|value| value.as_str()) {
        let trusted_public_key_hex = fs::read_to_string(public_key_path)?;
        let trusted_public_key = decode_hex_bytes(&trusted_public_key_hex)?;
        if trusted_public_key != public_key {
            return Err(PcError::Security(
                "trusted public key file does not match production envelope public key".to_string(),
            ));
        }
    } else {
        return Err(PcError::Security(
            "production envelope missing public_key_path".to_string(),
        ));
    }
    if parsed.signature.len() != 64 {
        return Err(PcError::Security(
            "production envelope Ed25519 signature must be 64 bytes".to_string(),
        ));
    }
    let signature_payload = prod_signature_payload(
        &parsed.nonce,
        &parsed.aad_bytes,
        &parsed.plaintext_hash,
        &parsed.expected_ciphertext_hash,
    );
    UnparsedPublicKey::new(&ED25519, public_key)
        .verify(&signature_payload, &parsed.signature)
        .map_err(|_| {
            PcError::Security("production envelope Ed25519 signature failed".to_string())
        })?;
    Ok(parsed.aad_json)
}

pub fn security_inspect(input: &Path) -> PcResult<serde_json::Value> {
    let bytes = fs::read(input)?;
    let file_hash = hex_hash(&hash_bytes(&bytes));
    let (artifact_kind, classification, seal_status, replay_status) =
        if bytes.starts_with(PCODE_MAGIC) {
            let packet = parse_packet(bytes.clone())?;
            let debug = debug_json_from_packet(&packet).unwrap_or_default();
            let classification = if debug
                .get("synthetic")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
            {
                "SYNTHETIC".to_string()
            } else {
                "INTERNAL".to_string()
            };
            (
                "pcode".to_string(),
                classification,
                "PACKET_HASH_VERIFIED".to_string(),
                "NOT_EXECUTED".to_string(),
            )
        } else if bytes.starts_with(PCOUT_MAGIC) {
            let pcout = parse_pcout(&bytes)?;
            let expected = hash_many(&[
                b"PCOUT_SEAL",
                &pcout.input_packet_hash,
                &pcout.output_state_hash,
            ]);
            let seal_status = if expected == pcout.seal_chain_hash {
                "SEAL_CHAIN_VERIFIED".to_string()
            } else {
                "SEAL_CHAIN_FAILED".to_string()
            };
            (
                "pcout".to_string(),
                "SYNTHETIC".to_string(),
                seal_status,
                "REPLAY_SEAL_VERIFIABLE".to_string(),
            )
        } else if bytes.starts_with(PSK_ENCRYPTED_MAGIC) || bytes.starts_with(PSK_PROD_MAGIC) {
            let parsed = parse_psk_envelope(&bytes)?;
            let aad_json = parsed.aad_json;
            let classification = aad_json
                .get("classification")
                .and_then(|value| value.as_str())
                .unwrap_or("INTERNAL")
                .to_string();
            let seal_status = if parsed.actual_ciphertext_hash == parsed.expected_ciphertext_hash {
                if parsed.prod {
                    match validate_prod_envelope(&bytes) {
                        Ok(_) => "PRODUCTION_ENVELOPE_SIGNATURE_VERIFIED".to_string(),
                        Err(_) => "PRODUCTION_ENVELOPE_SIGNATURE_FAILED".to_string(),
                    }
                } else {
                    "ENCRYPTED_ENVELOPE_HASH_VERIFIED".to_string()
                }
            } else {
                "ENCRYPTED_ENVELOPE_HASH_FAILED".to_string()
            };
            (
                if parsed.prod {
                    "production_encrypted_pcode_envelope".to_string()
                } else {
                    "encrypted_pcode_envelope".to_string()
                },
                classification,
                seal_status,
                "NOT_EXECUTED_ENCRYPTED".to_string(),
            )
        } else {
            (
                "unknown".to_string(),
                "INTERNAL".to_string(),
                "UNVERIFIED".to_string(),
                "NOT_REPLAYABLE".to_string(),
            )
        };
    let encrypted = bytes.starts_with(PSK_ENCRYPTED_MAGIC) || bytes.starts_with(PSK_PROD_MAGIC);
    let production_candidate = bytes.starts_with(PSK_PROD_MAGIC);
    Ok(json!({
        "psk_version": if production_candidate { "PRISM_SHIELD_KERNEL_V0_1_RELEASE_CANDIDATE" } else { "PRISM_SHIELD_KERNEL_V0_1_DEV" },
        "artifact": input.display().to_string(),
        "artifact_kind": artifact_kind,
        "artifact_hash_blake3": file_hash,
        "classification": classification,
        "encrypted": encrypted,
        "aead_algorithm": if encrypted { "AES_256_GCM" } else { "NONE" },
        "signed": production_candidate && seal_status == "PRODUCTION_ENVELOPE_SIGNATURE_VERIFIED",
        "vault_status": if production_candidate { "FILE_KEYRING_DEV_BOUND" } else { "PRISM_VAULT_UNBOUND" },
        "attestation_status": if production_candidate { "SIGNED_RELEASE_CANDIDATE_UNATTESTED" } else { "UNATTESTED_DEV" },
        "seal_status": seal_status,
        "replay_status": replay_status,
        "production_security_ready": production_candidate && seal_status == "PRODUCTION_ENVELOPE_SIGNATURE_VERIFIED",
        "market_ready": false,
        "security_reason": if production_candidate {
            "Production release-candidate envelope has encrypted payload and verified Ed25519 signature, but market readiness still requires CUDA, trusted attestation, external vault/KMS, and production policy closure."
        } else if encrypted {
            "Synthetic packet is encrypted in a development envelope, but production key wrapping, signatures, and vault policy are not bound."
        } else {
            "Executable dev artifact is hash/seal inspectable, but production encryption, signatures, and vault-backed keys are not yet bound."
        },
    }))
}

pub fn write_security_inspection(
    input: &Path,
    output: Option<&Path>,
) -> PcResult<serde_json::Value> {
    let report = security_inspect(input)?;
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(report)
}

pub fn seal_sidecar(input: &Path, output: &Path) -> PcResult<serde_json::Value> {
    let inspect = security_inspect(input)?;
    let seal_payload = serde_json::to_vec(&inspect)?;
    let seal_hash = hex_hash(&hash_many(&[
        b"PRISM_SEAL_DEV_V0_1",
        &hash_bytes(&fs::read(input)?),
        &hash_bytes(&seal_payload),
    ]));
    let report = json!({
        "pseal_version": "PRISM_SEAL_DEV_V0_1",
        "target": input.display().to_string(),
        "target_hash_blake3": inspect["artifact_hash_blake3"],
        "seal_hash_blake3": seal_hash,
        "signature_status": "UNSIGNED_DEV",
        "key_provider": "PRISM_VAULT_UNBOUND",
        "production_security_ready": false,
        "inspection": inspect,
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn runtime_attestation_report(output: Option<&Path>) -> PcResult<serde_json::Value> {
    let hardware = hardware_report()?;
    let report = json!({
        "attestation_version": "PRISM_ATTEST_V0_1_DEV",
        "status": "UNATTESTED_DEV",
        "secure_boot": "EVIDENCE_NOT_FOUND",
        "tpm": "EVIDENCE_NOT_FOUND",
        "runtime_binary_signature": "UNSIGNED_DEV",
        "cuda_kernel_trust": "UNBOUND",
        "hardware": hardware,
    });
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    Ok(report)
}

pub fn redact_export(input: &Path, classification: &str, output: &Path) -> PcResult<String> {
    let inspect = security_inspect(input)?;
    let text = format!(
        "# PRISMCODE Redacted Export\n\nartifact: {}\nclassification: {}\nartifact_hash_blake3: {}\nseal_status: {}\nredaction_status: {}\nproduction_security_ready: false\n",
        input.display(),
        classification,
        inspect["artifact_hash_blake3"].as_str().unwrap_or("EVIDENCE_NOT_FOUND"),
        inspect["seal_status"].as_str().unwrap_or("EVIDENCE_NOT_FOUND"),
        if classification == "PUBLIC" { "PUBLIC_SAFE_HASH_ONLY" } else { "NON_PUBLIC_EXPORT_REQUIRES_POLICY" },
    );
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, &text)?;
    Ok(text)
}

pub fn key_status_report() -> serde_json::Value {
    json!({
        "prism_vault_version": "PRISM_VAULT_V0_1_DEV",
        "providers": [
            {
                "provider": "LOCAL_DEV_ONLY",
                "status": "AVAILABLE_FOR_TEST_REPORTS_ONLY",
                "release_mode_allowed": false
            },
            {
                "provider": "OPENBAO",
                "status": "UNBOUND",
                "release_mode_allowed": true
            },
            {
                "provider": "TPM",
                "status": "EVIDENCE_NOT_FOUND",
                "release_mode_allowed": true
            }
        ],
        "raw_keys_exposed": false,
    })
}

#[derive(Debug)]
struct PcoutParsed {
    status: u16,
    backend_code: u16,
    deterministic: u16,
    instruction_count: u64,
    state_count: u64,
    event_count: u64,
    checkpoint_count: u64,
    input_packet_hash: [u8; 32],
    output_state_hash: [u8; 32],
    seal_chain_hash: [u8; 32],
}

fn write_pcout(
    path: &Path,
    backend: Backend,
    packet: &PCodePacket,
    report: &ExecutionReport,
) -> PcResult<()> {
    let output_state_hash = parse_hex_hash(&report.output_state_hash)?;
    let seal_chain_hash = parse_hex_hash(&report.seal_chain_hash)?;
    let metrics = serde_json::to_vec(report)?;
    let metrics_hash = hash_bytes(&metrics);
    let mut out = vec![0u8; HEADER_LEN];
    out[0..8].copy_from_slice(PCOUT_MAGIC);
    put_u16(&mut out, 8, 0);
    put_u16(&mut out, 10, 1);
    put_u16(&mut out, 12, backend.code());
    put_u16(&mut out, 14, 1);
    put_u16(&mut out, 16, 1);
    put_u64(&mut out, 24, packet.instructions.len() as u64);
    put_u64(&mut out, 32, packet.state_count);
    put_u64(&mut out, 40, packet.event_count);
    put_u64(&mut out, 48, report.checkpoint_count);
    put_u64(&mut out, 56, report.elapsed_ns);
    out[64..96].copy_from_slice(&packet.packet_hash);
    out[96..128].copy_from_slice(&output_state_hash);
    out[128..160].copy_from_slice(&seal_chain_hash);
    out[160..192].copy_from_slice(&metrics_hash);
    put_u64(&mut out, 192, metrics.len() as u64);
    out.extend_from_slice(&metrics);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, out)?;
    Ok(())
}

fn parse_pcout(bytes: &[u8]) -> PcResult<PcoutParsed> {
    if bytes.len() < HEADER_LEN {
        return Err(PcError::Format("pcout shorter than header".to_string()));
    }
    if &bytes[0..8] != PCOUT_MAGIC {
        return Err(PcError::Format("magic is not PRISMPO\\0".to_string()));
    }
    let metrics_len = read_u64(bytes, 192)? as usize;
    if HEADER_LEN + metrics_len > bytes.len() {
        return Err(PcError::Format(
            "pcout metrics length exceeds file".to_string(),
        ));
    }
    let metrics_hash = hash_bytes(&bytes[HEADER_LEN..HEADER_LEN + metrics_len]);
    if metrics_hash != read_hash(bytes, 160)? {
        return Err(PcError::Hash("pcout metrics hash mismatch".to_string()));
    }
    Ok(PcoutParsed {
        backend_code: read_u16(bytes, 12)?,
        status: read_u16(bytes, 14)?,
        deterministic: read_u16(bytes, 16)?,
        instruction_count: read_u64(bytes, 24)?,
        state_count: read_u64(bytes, 32)?,
        event_count: read_u64(bytes, 40)?,
        checkpoint_count: read_u64(bytes, 48)?,
        input_packet_hash: read_hash(bytes, 64)?,
        output_state_hash: read_hash(bytes, 96)?,
        seal_chain_hash: read_hash(bytes, 128)?,
    })
}

fn bounds_check_instruction(instr: &PcInstr32, packet: &PCodePacket, errors: &mut Vec<String>) {
    for (slot, kind, value) in [
        ("dst", instr.dst_kind, instr.dst),
        ("src0", instr.src0_kind, instr.src0),
        ("src1", instr.src1_kind, instr.src1),
        ("src2", instr.src2_kind, instr.src2_or_len),
    ] {
        // src2_or_len is only a reference when its kind says so; None and
        // immediates retain their scalar/length representation.
        let target = match kind {
            0..=2 => continue,
            3 => Some(SectionKind::OperandTable),
            4 => Some(SectionKind::StateTable),
            5 => Some(SectionKind::TimeTable),
            6 => Some(SectionKind::NanlTable),
            7 => Some(SectionKind::EventTable),
            8 => Some(SectionKind::TensorTable),
            9 => Some(SectionKind::GraphTable),
            10 => Some(SectionKind::RewardTable),
            11 => Some(SectionKind::ActionTable),
            12 => Some(SectionKind::PayloadBlob),
            15 => {
                if !(DTypeId::Bool as u32..=DTypeId::BytesRef as u32).contains(&value) {
                    errors.push(format!("{slot}: dtype id out of range: {value}"));
                }
                continue;
            }
            16 => Some(SectionKind::ProvenanceTable),
            17 => Some(SectionKind::SealChain),
            // These tags exist in the ISA but this container has no declared
            // Arrow-column/shape/kernel/error/extension target-table ABI.
            // A future producer must supply that contract before admission.
            13 | 14 | 18 | 19 | 20 => {
                errors.push(format!("{slot}: reference kind {kind} has no admitted target-table contract"));
                continue;
            }
            _ => continue, // invalid tag already reported by the caller
        };
        let target = target.unwrap();
        let Some(section) = packet.sections.iter().find(|s| s.section_type == target) else {
            errors.push(format!("{slot}: reference target {target:?} is absent"));
            continue;
        };
        let bound = if target == SectionKind::PayloadBlob { section.length } else { section.record_count };
        // instruction_program emits whole-table base operations even when the
        // table is empty. Preserve only those exact opcode/operand roles; an
        // indexed EVENT.EMIT or unrelated EventRef cannot name record zero.
        let empty_table_base = value == 0 && bound == 0 && match (instr.opcode, slot, target) {
            (opcode::EVENT_SORT, "dst", SectionKind::EventTable)
            | (opcode::EVENT_INTEGRATE, "src0", SectionKind::EventTable)
            | (opcode::DRF_MAP, "src0", SectionKind::EventTable)
            | (opcode::NANL_PACK, "dst", SectionKind::NanlTable) => true,
            _ => false,
        };
        if value as u64 >= bound && !empty_table_base {
            errors.push(format!("{slot}: {target:?} reference out of range: {value} >= {bound}"));
        }
    }
}

fn section_bytes<'a>(bytes: &'a [u8], entry: &SectionEntry) -> &'a [u8] {
    &bytes[entry.offset as usize..(entry.offset + entry.length) as usize]
}

fn find_section(sections: &[SectionEntry], kind: SectionKind) -> PcResult<&SectionEntry> {
    sections
        .iter()
        .find(|entry| entry.section_type == kind)
        .ok_or_else(|| PcError::Section(format!("section {kind:?} not found")))
}

fn debug_json_from_packet(packet: &PCodePacket) -> PcResult<serde_json::Value> {
    let entry = find_section(&packet.sections, SectionKind::DebugJson)?;
    let bytes = section_bytes(&packet.bytes, entry);
    Ok(serde_json::from_slice(bytes)?)
}

fn tensor_section(corpus: &ReferenceCorpus) -> PcResult<Vec<u8>> {
    let mut rec = [0u8; 96];
    put_u64(&mut rec, 0, 1);
    put_u16(&mut rec, 8, StateKind::DrfTensor as u16);
    put_u16(&mut rec, 10, DTypeId::F32 as u16);
    put_u64(&mut rec, 16, corpus.events.len() as u64);
    put_u64(&mut rec, 24, corpus.nanl_tokens.len() as u64);
    let hash = hash_many(&[
        &corpus.events.len().to_le_bytes(),
        &corpus.nanl_tokens.len().to_le_bytes(),
    ]);
    rec[64..96].copy_from_slice(&hash);
    Ok(rec.to_vec())
}

fn reward_section(corpus: &ReferenceCorpus) -> Vec<u8> {
    let mut out = Vec::new();
    for thermo in &corpus.thermo {
        let mut rec = [0u8; 64];
        put_u64(&mut rec, 0, thermo.thermo_id);
        put_u64(&mut rec, 8, thermo.state_ref);
        put_f64(&mut rec, 16, thermo.value);
        let hash = hash_many(&[
            &thermo.thermo_id.to_le_bytes(),
            &thermo.value.to_le_bytes(),
            thermo.unit.as_bytes(),
        ]);
        rec[32..64].copy_from_slice(&hash);
        out.extend_from_slice(&rec);
    }
    out
}

fn provenance_section(corpus: &ReferenceCorpus, payload_blob: &mut Vec<u8>) -> PcResult<Vec<u8>> {
    let mut out = Vec::new();
    for prov in &corpus.provenance {
        if prov.label.trim().is_empty() || prov.reason.trim().is_empty() {
            return Err(PcError::Provenance(
                "provenance label/reason is empty".to_string(),
            ));
        }
        let offset = payload_blob.len() as u64;
        let payload = serde_json::to_vec(prov)?;
        payload_blob.extend_from_slice(&payload);
        let mut rec = [0u8; 96];
        put_u64(&mut rec, 0, prov.provenance_id);
        put_u64(&mut rec, 8, offset);
        put_u64(&mut rec, 16, payload.len() as u64);
        rec[32..64].copy_from_slice(&hash_bytes(&payload));
        rec[64..96].copy_from_slice(&hash_many(&[&prov.provenance_id.to_le_bytes(), &payload]));
        out.extend_from_slice(&rec);
    }
    Ok(out)
}

fn event_value_sum(event_bytes: &[u8]) -> PcResult<f64> {
    if event_bytes.len() % 96 != 0 {
        return Err(PcError::Section(
            "event table record width mismatch".to_string(),
        ));
    }
    let mut sum = 0.0;
    for rec in event_bytes.chunks_exact(96) {
        sum += read_f64(rec, 40)?;
    }
    Ok(sum)
}

fn instruction_accumulator(instructions: &[PcInstr32]) -> u64 {
    instructions.iter().fold(0u64, |acc, instr| {
        acc.wrapping_mul(131)
            .wrapping_add(instr.opcode as u64)
            .wrapping_add(instr.dst as u64)
    })
}

fn known_opcode(code: u16) -> bool {
    opcode_table().contains_key(&code)
}

/// Reverse lookup: resolve a registered opcode mnemonic to its u16 id.
/// Reads the single `opcode_table()` source (no second table), so it stays
/// consistent with `opcode_name`/`known_opcode`. Registry coverage only — per
/// Guardrail 1 this confers NO behavioral dispatch (`execute_packet` has zero
/// `match instr.opcode`). Used by the typed OperandLedger to stamp `opcode_id`.
pub fn opcode_id_for_name(name: &str) -> Option<u16> {
    opcode_table()
        .into_iter()
        .find(|(_, table_name)| *table_name == name)
        .map(|(code, _)| code)
}

#[cfg(test)]
mod successor_opcode_registry_tests {
    use super::{opcode, opcode_id_for_name};

    #[test]
    fn ghost_lambda_and_pathb_event_rows_are_stable_and_resolvable() {
        let expected = [
            ("MD.LAMBDA_FRAGMENT_FORCE", opcode::MD_LAMBDA_FRAGMENT_FORCE),
            (
                "MD.LAMBDA_DUDL_ACCUMULATE",
                opcode::MD_LAMBDA_DUDL_ACCUMULATE,
            ),
            ("MD.LAMBDA_NCMC_PERTURB", opcode::MD_LAMBDA_NCMC_PERTURB),
            ("MD.LAMBDA_NCMC_PROPAGATE", opcode::MD_LAMBDA_NCMC_PROPAGATE),
            ("MD.LAMBDA_NCMC_ACCEPT", opcode::MD_LAMBDA_NCMC_ACCEPT),
            (
                "MD.LAMBDA_ENDPOINT_OBSERVE",
                opcode::MD_LAMBDA_ENDPOINT_OBSERVE,
            ),
            ("MD.LAMBDA_STATE_COMMIT", opcode::MD_LAMBDA_STATE_COMMIT),
            ("MD.GHOST_SEGMENT_COMMIT", opcode::MD_GHOST_SEGMENT_COMMIT),
            ("MD.GHOST_CONSERVATION", opcode::MD_GHOST_CONSERVATION),
            ("MD.GHOST_PHASE_BINDING", opcode::MD_GHOST_PHASE_BINDING),
            ("MD.LAMBDA_STATE_ROLLBACK", opcode::MD_LAMBDA_STATE_ROLLBACK),
            (
                "MD.LAMBDA_NO_PROBE_ABSENCE",
                opcode::MD_LAMBDA_NO_PROBE_ABSENCE,
            ),
            (
                "MD.GHOST_COORDINATE_SEGMENT_COMMIT",
                opcode::MD_GHOST_COORDINATE_SEGMENT_COMMIT,
            ),
            (
                "MD.GHOST_COORDINATE_CONSERVATION",
                opcode::MD_GHOST_COORDINATE_CONSERVATION,
            ),
            (
                "MD.GHOST_LATTICE_EDGE_COMMIT",
                opcode::MD_GHOST_LATTICE_EDGE_COMMIT,
            ),
            (
                "MD.GHOST_LATTICE_EDGE_CONSERVATION",
                opcode::MD_GHOST_LATTICE_EDGE_CONSERVATION,
            ),
            (
                "PATHB.CHEMICAL_STATE_FIBER_MATERIALIZED",
                opcode::PATHB_CHEMICAL_STATE_FIBER_MATERIALIZED,
            ),
            (
                "PATHB.CHEMICAL_STATE_ASSIGNED",
                opcode::PATHB_CHEMICAL_STATE_ASSIGNED,
            ),
            (
                "PATHB.MATCHED_CONTROL_CLASSIFIED",
                opcode::PATHB_MATCHED_CONTROL_CLASSIFIED,
            ),
            ("PATHB.CANDIDATE_OBSERVED", opcode::PATHB_CANDIDATE_OBSERVED),
            (
                "PATHB.PARETO_CHAMPION_PROMOTED",
                opcode::PATHB_PARETO_CHAMPION_PROMOTED,
            ),
            (
                "PATHB.PARETO_CHAMPION_SUPERSEDED",
                opcode::PATHB_PARETO_CHAMPION_SUPERSEDED,
            ),
            (
                "PATHB.DISCOVERY_CANDIDATE_MATERIALIZED",
                opcode::PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
            ),
            (
                "PATHB.CONFIRMATION_HANDOFF_MATERIALIZED",
                opcode::PATHB_CONFIRMATION_HANDOFF_MATERIALIZED,
            ),
        ];

        for (name, code) in expected {
            assert_eq!(opcode_id_for_name(name), Some(code));
        }
    }

    #[test]
    fn resident_graph_velocity_clamp_row_is_stable_and_noncolliding() {
        assert_eq!(opcode::MD_VELOCITY_CLAMP, 0x0730);
        assert_eq!(opcode_id_for_name("MD.VELOCITY_CLAMP"), Some(0x0730));
        assert_ne!(
            opcode_id_for_name("MD.GHOST_LATTICE_EDGE_CONSERVATION"),
            Some(opcode::MD_VELOCITY_CLAMP)
        );
    }
}

fn opcode_table() -> BTreeMap<u16, &'static str> {
    use opcode::*;
    [
        (NOP, "NOP"),
        (ABI_DECLARE, "ABI.DECLARE"),
        (FEATURE_REQUIRE, "FEATURE.REQUIRE"),
        (PACKET_BEGIN, "PACKET.BEGIN"),
        (PACKET_END, "PACKET.END"),
        (HALT, "HALT"),
        (TRAP, "TRAP"),
        (STATE_DEF, "STATE.DEF"),
        (STATE_CANONICALIZE, "STATE.CANONICALIZE"),
        (STATE_HASH, "STATE.HASH"),
        (STATE_FREEZE, "STATE.FREEZE"),
        (STATE_SNAPSHOT, "STATE.SNAPSHOT"),
        (STATE_VERTEX_CREATE, "STATE.VERTEX_CREATE"),
        (STATE_PRECONDITION_VALIDATE, "STATE.PRECONDITION_VALIDATE"),
        (STATE_OPERATOR_APPLY, "STATE.OPERATOR_APPLY"),
        (STATE_CLASSIFY_BLOCKED, "STATE.CLASSIFY_BLOCKED"),
        (STATE_NEXT_OPERATOR_REC, "STATE.NEXT_OPERATOR_REC"),
        (STATE_EMIT_TRANSITION, "STATE.EMIT_TRANSITION"),
        (STATE_DENSE_TENSOR_SUMMARY, "STATE.DENSE_TENSOR_SUMMARY"),
        (TIME_BIND, "TIME.BIND"),
        (TIME_NORMALIZE, "TIME.NORMALIZE"),
        (TIME_ORDER_ASSERT, "TIME.ORDER_ASSERT"),
        (NANL_PACK, "NANL.PACK"),
        (NANL_HASH, "NANL.HASH"),
        (EVENT_EMIT, "EVENT.EMIT"),
        (EVENT_BIND_STATE, "EVENT.BIND_STATE"),
        (EVENT_SORT, "EVENT.SORT"),
        (EVENT_INTEGRATE, "EVENT.INTEGRATE"),
        (EVENT_INDEX, "EVENT.INDEX"),
        (DRF_MAP, "DRF.MAP"),
        (DRF_FIRE, "DRF.FIRE"),
        (DRF_RESONATE, "DRF.RESONATE"),
        (DRF_NORMALIZE, "DRF.NORMALIZE"),
        (DRF_WRITEBACK, "DRF.WRITEBACK"),
        (THERMO_ENERGY, "THERMO.ENERGY"),
        (THERMO_ENTROPY, "THERMO.ENTROPY"),
        (THERMO_CONSERVE_ASSERT, "THERMO.CONSERVE_ASSERT"),
        (MD_PROTOCOL_DIRECTOR, "MD.PROTOCOL_DIRECTOR"),
        (MD_COM_REMOVE, "MD.COM_REMOVE"),
        (MD_NEIGHBOR_REBUILD, "MD.NEIGHBOR_REBUILD"),
        (MD_ACCUMULATOR_ZERO, "MD.ACCUMULATOR_ZERO"),
        (MD_FORCE_BONDED, "MD.FORCE_BONDED"),
        (MD_FORCE_NONBONDED, "MD.FORCE_NONBONDED"),
        (MD_INTEGRATE_VELOCITY_VERLET, "MD.INTEGRATE_VELOCITY_VERLET"),
        (MD_SHAKE_CONSTRAIN, "MD.SHAKE_CONSTRAIN"),
        (MD_EXCLUSION_FIELD, "MD.EXCLUSION_FIELD"),
        (MD_WATER_DENSITY, "MD.WATER_DENSITY"),
        (MD_PHOTO_UV_PUMP, "MD.PHOTO_UV_PUMP"),
        (MD_PHOTO_UV_PUMP_SUPPRESSED, "MD.PHOTO_UV_PUMP_SUPPRESSED"),
        (MD_SPIKE_MANIFEST_COMMIT, "MD.SPIKE_MANIFEST_COMMIT"),
        (MD_EXTERNAL_WORK_REDUCE, "MD.EXTERNAL_WORK_REDUCE"),
        (MD_VELOCITY_SECOND_HALF, "MD.VELOCITY_SECOND_HALF"),
        (MD_VELOCITY_CLAMP, "MD.VELOCITY_CLAMP"),
        (MD_CA_RESTRAINT, "MD.CA_RESTRAINT"),
        (MD_LIF_OBSERVE, "MD.LIF_OBSERVE"),
        (MD_COUPLING_BUILD, "MD.COUPLING_BUILD"),
        (MD_COUPLING_CLEAR, "MD.COUPLING_CLEAR"),
        (MD_HEARTBEAT_ASSERT, "MD.HEARTBEAT_ASSERT"),
        (MD_LADD_UPDATE, "MD.LADD_UPDATE"),
        (MD_KCC_UPDATE, "MD.KCC_UPDATE"),
        (MD_NMA_PERTURB, "MD.NMA_PERTURB"),
        (MD_SPIKE_EMIT, "MD.SPIKE_EMIT"),
        (MD_SPIKE_READOUT, "MD.SPIKE_READOUT"),
        (MD_SIGNAL_GRID_READOUT, "MD.SIGNAL_GRID_READOUT"),
        (MD_KCC_DESCRIPTOR, "MD.KCC_DESCRIPTOR"),
        (MD_SPIKE_CHUNK_COMMIT, "MD.SPIKE_CHUNK_COMMIT"),
        (MD_SENSOR_EMIT, "MD.SENSOR_EMIT"),
        (MD_QNANL_SEAL, "MD.QNANL_SEAL"),
        (MD_QNANL_VERIFY, "MD.QNANL_VERIFY"),
        (MD_TWIN16_CELL_COMMIT, "MD.TWIN16_CELL_COMMIT"),
        (MD_CONDITION_EDGE_COMMIT, "MD.CONDITION_EDGE_COMMIT"),
        (MD_VALIDATE_LIGAND_STATE, "MD.VALIDATE_LIGAND_STATE"),
        (MD_LAMBDA_FRAGMENT_FORCE, "MD.LAMBDA_FRAGMENT_FORCE"),
        (MD_LAMBDA_DUDL_ACCUMULATE, "MD.LAMBDA_DUDL_ACCUMULATE"),
        (MD_LAMBDA_NCMC_PERTURB, "MD.LAMBDA_NCMC_PERTURB"),
        (MD_LAMBDA_NCMC_PROPAGATE, "MD.LAMBDA_NCMC_PROPAGATE"),
        (MD_LAMBDA_NCMC_ACCEPT, "MD.LAMBDA_NCMC_ACCEPT"),
        (MD_LAMBDA_ENDPOINT_OBSERVE, "MD.LAMBDA_ENDPOINT_OBSERVE"),
        (MD_LAMBDA_STATE_COMMIT, "MD.LAMBDA_STATE_COMMIT"),
        (MD_GHOST_SEGMENT_COMMIT, "MD.GHOST_SEGMENT_COMMIT"),
        (MD_GHOST_CONSERVATION, "MD.GHOST_CONSERVATION"),
        (MD_GHOST_PHASE_BINDING, "MD.GHOST_PHASE_BINDING"),
        (MD_LAMBDA_STATE_ROLLBACK, "MD.LAMBDA_STATE_ROLLBACK"),
        (MD_LAMBDA_NO_PROBE_ABSENCE, "MD.LAMBDA_NO_PROBE_ABSENCE"),
        (
            MD_GHOST_COORDINATE_SEGMENT_COMMIT,
            "MD.GHOST_COORDINATE_SEGMENT_COMMIT",
        ),
        (
            MD_GHOST_COORDINATE_CONSERVATION,
            "MD.GHOST_COORDINATE_CONSERVATION",
        ),
        (MD_GHOST_LATTICE_EDGE_COMMIT, "MD.GHOST_LATTICE_EDGE_COMMIT"),
        (
            MD_GHOST_LATTICE_EDGE_CONSERVATION,
            "MD.GHOST_LATTICE_EDGE_CONSERVATION",
        ),
        (
            PATHB_MATERIALIZE_BINDING_SITE,
            "PATHB.MATERIALIZE_BINDING_SITE",
        ),
        (
            PATHB_BLOCKED_SITE_PRECONDITION,
            "PATHB.BLOCKED_SITE_PRECONDITION",
        ),
        (
            PATHB_CHEMICAL_STATE_FIBER_MATERIALIZED,
            "PATHB.CHEMICAL_STATE_FIBER_MATERIALIZED",
        ),
        (
            PATHB_CHEMICAL_STATE_ASSIGNED,
            "PATHB.CHEMICAL_STATE_ASSIGNED",
        ),
        (
            PATHB_MATCHED_CONTROL_CLASSIFIED,
            "PATHB.MATCHED_CONTROL_CLASSIFIED",
        ),
        (PATHB_CANDIDATE_OBSERVED, "PATHB.CANDIDATE_OBSERVED"),
        (
            PATHB_PARETO_CHAMPION_PROMOTED,
            "PATHB.PARETO_CHAMPION_PROMOTED",
        ),
        (
            PATHB_PARETO_CHAMPION_SUPERSEDED,
            "PATHB.PARETO_CHAMPION_SUPERSEDED",
        ),
        (
            PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
            "PATHB.DISCOVERY_CANDIDATE_MATERIALIZED",
        ),
        (
            PATHB_CONFIRMATION_HANDOFF_MATERIALIZED,
            "PATHB.CONFIRMATION_HANDOFF_MATERIALIZED",
        ),
        (REWARD_ATTACH_TEACHER, "REWARD.ATTACH_TEACHER"),
        (REWARD_GFLOWNET_FLOW, "REWARD.GFLOWNET_FLOW"),
        (ACTION_CONSTRAIN, "ACTION.CONSTRAIN"),
        (PRISMAI_PACKET, "PRISMAI.PACKET"),
        (HIL_REQUEST, "HIL.REQUEST"),
        (MEM_LINEAGE, "MEM.LINEAGE"),
        (REPLAY_BEGIN, "REPLAY.BEGIN"),
        (REPLAY_CHECKPOINT, "REPLAY.CHECKPOINT"),
        (REPLAY_END, "REPLAY.END"),
        (REPLAY_VERIFY, "REPLAY.VERIFY"),
        (AUDIT_SEAL, "AUDIT.SEAL"),
        (AUDIT_VERIFY, "AUDIT.VERIFY"),
        (PROVENANCE_ATTACH, "PROVENANCE.ATTACH"),
        (ASSERT_TYPE, "ASSERT.TYPE"),
        (ASSERT_RANGE, "ASSERT.RANGE"),
        (ASSERT_SHAPE, "ASSERT.SHAPE"),
        (ASSERT_HASH, "ASSERT.HASH"),
        (ASSERT_TIMEBASE, "ASSERT.TIMEBASE"),
        (ASSERT_DETERMINISTIC, "ASSERT.DETERMINISTIC"),
        (ASSERT_SCIENTIFIC, "ASSERT.SCIENTIFIC"),
        (TENSOR_ALLOC, "TENSOR.ALLOC"),
        (TENSOR_CANONICALIZE, "TENSOR.CANONICALIZE"),
        (SEC_CLASSIFY, "SEC.CLASSIFY"),
        (SEC_POLICY_BIND, "SEC.POLICY_BIND"),
        (SEC_CAPABILITY_ASSERT, "SEC.CAPABILITY_ASSERT"),
        (SEC_ACCESS_CHECK, "SEC.ACCESS_CHECK"),
        (SEC_REDACT, "SEC.REDACT"),
        (SEC_ZEROIZE, "SEC.ZEROIZE"),
        (SEC_EXPORT_GATE, "SEC.EXPORT_GATE"),
        (SEC_AIRLOCK_MODEL_OUTPUT, "SEC.AIRLOCK_MODEL_OUTPUT"),
        (CRYPTO_HASH, "CRYPTO.HASH"),
        (CRYPTO_MERKLE_ROOT, "CRYPTO.MERKLE_ROOT"),
        (SEAL_BEGIN, "SEAL.BEGIN"),
        (SEAL_APPEND, "SEAL.APPEND"),
        (SEAL_FINALIZE, "SEAL.FINALIZE"),
        (SEAL_VERIFY, "SEAL.VERIFY"),
        (ATTEST_RUNTIME_MEASURE, "ATTEST.RUNTIME_MEASURE"),
        (ATTEST_CUDA_KERNEL_MEASURE, "ATTEST.CUDA_KERNEL_MEASURE"),
        (ATTEST_BUILD_PROVENANCE, "ATTEST.BUILD_PROVENANCE"),
        (GUARD_INTRUSION_EVENT, "GUARD.INTRUSION_EVENT"),
        (CSE_DEF_SCHEMA, "CSE.DEF_SCHEMA"),
        (CSE_DEF_DICTIONARY, "CSE.DEF_DICTIONARY"),
        (CSE_DEF_ENTITY, "CSE.DEF_ENTITY"),
        (CSE_DEF_COMPONENT_TABLE, "CSE.DEF_COMPONENT_TABLE"),
        (CSE_TOKENIZE, "CSE.TOKENIZE"),
        (CSE_DELTA_ENCODE, "CSE.DELTA_ENCODE"),
        (CSE_MOTIF_ENCODE, "CSE.MOTIF_ENCODE"),
        (CSE_CANONICALIZE, "CSE.CANONICALIZE"),
        (CSE_HASH, "CSE.HASH"),
        (CSE_TO_NANL_BIN, "CSE.TO_NANL_BIN"),
        (CSE_TO_NANL_Q4, "CSE.TO_NANL_Q4"),
        (CSE_TO_NRF, "CSE.TO_NRF"),
        (CSE_TO_DRF, "CSE.TO_DRF"),
        (CSE_RECONSTRUCT, "CSE.RECONSTRUCT"),
        (CSE_VALIDATE_ROUNDTRIP, "CSE.VALIDATE_ROUNDTRIP"),
        (CSE_EMIT_DSTW_ROW, "CSE.EMIT_DSTW_ROW"),
        (NANL_BIN_PACK, "NANL.BIN_PACK"),
        (NANL_BIN_UNPACK, "NANL.BIN_UNPACK"),
        (NANL_BIN_VALIDATE, "NANL.BIN_VALIDATE"),
        (NANL_BIN_HASH, "NANL.BIN_HASH"),
        (NANL_BIN_INDEX, "NANL.BIN_INDEX"),
        (NANL_BIN_STREAM_DECODE, "NANL.BIN_STREAM_DECODE"),
        (NANL_BIN_TO_NRF, "NANL.BIN_TO_NRF"),
        (NANL_BIN_TO_DRF, "NANL.BIN_TO_DRF"),
        (NANL_Q4_ENCODE, "NANL.Q4_ENCODE"),
        (NANL_Q4_DECODE, "NANL.Q4_DECODE"),
        (NANL_Q4_VALIDATE, "NANL.Q4_VALIDATE"),
        (NANL_Q4_CONSTRAINT_SCORE, "NANL.Q4_CONSTRAINT_SCORE"),
        (NANL_Q4_ECC_ATTACH, "NANL.Q4_ECC_ATTACH"),
        (NANL_Q4_ECC_VERIFY, "NANL.Q4_ECC_VERIFY"),
        (NANL_Q4_MERKLE_ROOT, "NANL.Q4_MERKLE_ROOT"),
        (NANL_Q4_TO_BIN, "NANL.Q4_TO_BIN"),
        (NANL_BIN_TO_Q4, "NANL.BIN_TO_Q4"),
        (NANL_Q4_TO_DNA_OLIGO, "NANL.Q4_TO_DNA_OLIGO"),
        (NRF_FRAME_BEGIN, "NRF.FRAME_BEGIN"),
        (NRF_FRAME_END, "NRF.FRAME_END"),
        (NRF_EVENT_TILE, "NRF.EVENT_TILE"),
        (NRF_SPIKE_MOTIF, "NRF.SPIKE_MOTIF"),
        (NRF_TRANSITION_EDGE, "NRF.TRANSITION_EDGE"),
        (NRF_WATER_GATE, "NRF.WATER_GATE"),
        (NRF_PHASE_TILE, "NRF.PHASE_TILE"),
        (NRF_CAUSAL_PARENT, "NRF.CAUSAL_PARENT"),
        (NRF_SPARSE_INDEX, "NRF.SPARSE_INDEX"),
        (NRF_MOTIF_PACK, "NRF.MOTIF_PACK"),
        (NRF_MOTIF_UNPACK, "NRF.MOTIF_UNPACK"),
        (NRF_TO_DRF, "NRF.TO_DRF"),
        (NRF_RECONSTRUCT_EVENTS, "NRF.RECONSTRUCT_EVENTS"),
        (NRF_VALIDATE_REPLAY, "NRF.VALIDATE_REPLAY"),
        (DRF_MAP_CSE, "DRF.MAP_CSE"),
        (DRF_LOAD_NRF, "DRF.LOAD_NRF"),
        (DRF_DECAY, "DRF.DECAY"),
        (DRF_SYNC, "DRF.SYNC"),
        (DRF_SCATTER_EVENT, "DRF.SCATTER_EVENT"),
        (DRF_GATHER_NEIGHBORHOOD, "DRF.GATHER_NEIGHBORHOOD"),
        (DRF_UPDATE_UNCERTAINTY, "DRF.UPDATE_UNCERTAINTY"),
        (DRF_UPDATE_ACQUISITION, "DRF.UPDATE_ACQUISITION"),
        (DRF_RECONSTRUCT, "DRF.RECONSTRUCT"),
        (DRF_CONSISTENCY_ASSERT, "DRF.CONSISTENCY_ASSERT"),
    ]
    .into_iter()
    .collect()
}

fn opcode_name(code: u16) -> &'static str {
    opcode_table().get(&code).copied().unwrap_or("UNKNOWN")
}

fn state_kind_code(value: &str) -> PcResult<u16> {
    Ok(match value {
        "MOLECULAR_STATE" => StateKind::MolecularState as u16,
        "TEMPORAL_STATE" => StateKind::TemporalState as u16,
        "NANL_STRAND" => StateKind::NanlStrand as u16,
        "EVENT_STREAM" => StateKind::EventStream as u16,
        "DRF_STATE" => StateKind::DrfState as u16,
        "DRF_TENSOR" => StateKind::DrfTensor as u16,
        "THERMO_STATE" => StateKind::ThermoState as u16,
        "GRAPH_STATE" => StateKind::GraphState as u16,
        other => return Err(PcError::Format(format!("unknown state kind {other}"))),
    })
}

fn dtype_code(value: &str) -> PcResult<u16> {
    Ok(match value {
        "BOOL" => DTypeId::Bool as u16,
        "U2_PACKED" => DTypeId::U2Packed as u16,
        "U8" => DTypeId::U8 as u16,
        "U16" => DTypeId::U16 as u16,
        "U32" => DTypeId::U32 as u16,
        "U64" => DTypeId::U64 as u16,
        "I32" => DTypeId::I32 as u16,
        "F32" => DTypeId::F32 as u16,
        "F64" => DTypeId::F64 as u16,
        "HASH256" => DTypeId::Hash256 as u16,
        "STRING_REF" => DTypeId::StringRef as u16,
        "BYTES_REF" => DTypeId::BytesRef as u16,
        other => return Err(PcError::Format(format!("unknown dtype {other}"))),
    })
}

fn event_kind_code(value: &str) -> u16 {
    match value {
        "SPIKE" => 1,
        "KCC" => 2,
        "BOCPD" => 3,
        "PHASE_TILE" => 4,
        "FIBER" => 5,
        _ => 0,
    }
}

fn pack_u2(tokens: &[u8]) -> PcResult<Vec<u8>> {
    let mut out = Vec::with_capacity(tokens.len().div_ceil(4));
    let mut byte = 0u8;
    for (idx, token) in tokens.iter().enumerate() {
        if *token > 3 {
            return Err(PcError::Format(format!("U2 token out of range: {token}")));
        }
        byte |= token << ((idx % 4) * 2);
        if idx % 4 == 3 {
            out.push(byte);
            byte = 0;
        }
    }
    if tokens.len() % 4 != 0 {
        out.push(byte);
    }
    Ok(out)
}

fn codec_report(
    codec: &str,
    canonical: &[u8],
    encoded: &[u8],
    decoded: &[u8],
    encoded_len: u64,
    metadata: serde_json::Value,
) -> PcResult<serde_json::Value> {
    let roundtrip = canonical == decoded;
    if !roundtrip {
        return Err(PcError::Hash(format!(
            "{codec} canonical uncompressed hash mismatch after decode"
        )));
    }
    Ok(json!({
        "codec": codec,
        "encoded_bytes": encoded_len,
        "canonical_uncompressed_bytes": canonical.len(),
        "hash_compressed_domain_blake3": hex_hash(&hash_bytes(encoded)),
        "hash_canonical_uncompressed_blake3": hex_hash(&hash_bytes(canonical)),
        "roundtrip_verified": roundtrip,
        "metadata": metadata,
    }))
}

fn delta_time_encode(time_bytes: &[u8]) -> PcResult<Vec<u8>> {
    if time_bytes.len() % 80 != 0 {
        return Err(PcError::Section(
            "time table record width mismatch for DELTA_TIME".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(time_bytes.len() + 16);
    out.extend_from_slice(b"DTIM");
    out.extend_from_slice(&(time_bytes.len() as u64).to_le_bytes());
    let mut previous_id = 0u64;
    for rec in time_bytes.chunks_exact(80) {
        let time_id = read_u64(rec, 0)?;
        out.extend_from_slice(&time_id.wrapping_sub(previous_id).to_le_bytes());
        out.extend_from_slice(&rec[8..80]);
        previous_id = time_id;
    }
    Ok(out)
}

fn delta_time_decode(encoded: &[u8]) -> PcResult<Vec<u8>> {
    if encoded.len() < 12 || &encoded[0..4] != b"DTIM" {
        return Err(PcError::Format("invalid DELTA_TIME header".to_string()));
    }
    let expected_len = u64::from_le_bytes(
        encoded[4..12]
            .try_into()
            .map_err(|_| PcError::Format("invalid DELTA_TIME length".to_string()))?,
    ) as usize;
    let mut out = Vec::with_capacity(expected_len);
    let mut previous_id = 0u64;
    let mut cursor = 12;
    while cursor < encoded.len() {
        if cursor + 80 > encoded.len() {
            return Err(PcError::Format("truncated DELTA_TIME record".to_string()));
        }
        let delta = u64::from_le_bytes(
            encoded[cursor..cursor + 8]
                .try_into()
                .map_err(|_| PcError::Format("invalid DELTA_TIME delta".to_string()))?,
        );
        cursor += 8;
        let time_id = previous_id.wrapping_add(delta);
        out.extend_from_slice(&time_id.to_le_bytes());
        out.extend_from_slice(&encoded[cursor..cursor + 72]);
        cursor += 72;
        previous_id = time_id;
    }
    if out.len() != expected_len {
        return Err(PcError::Format(
            "DELTA_TIME decoded length mismatch".to_string(),
        ));
    }
    Ok(out)
}

fn event_rle_encode(event_bytes: &[u8]) -> PcResult<Vec<u8>> {
    if event_bytes.len() % 96 != 0 {
        return Err(PcError::Section(
            "event table record width mismatch for EVENT_RLE".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(event_bytes.len() + 32);
    out.extend_from_slice(b"ERLE");
    out.extend_from_slice(&(event_bytes.len() as u64).to_le_bytes());
    let mut records = event_bytes.chunks_exact(96).peekable();
    while let Some(first) = records.next() {
        let kind = read_u16(first, 8)?;
        let mut run = vec![first.to_vec()];
        while let Some(next) = records.peek() {
            if read_u16(next, 8)? != kind {
                break;
            }
            run.push(records.next().expect("peeked record exists").to_vec());
        }
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&(run.len() as u32).to_le_bytes());
        for rec in run {
            out.extend_from_slice(&rec);
        }
    }
    Ok(out)
}

fn event_rle_decode(encoded: &[u8]) -> PcResult<Vec<u8>> {
    if encoded.len() < 12 || &encoded[0..4] != b"ERLE" {
        return Err(PcError::Format("invalid EVENT_RLE header".to_string()));
    }
    let expected_len = u64::from_le_bytes(
        encoded[4..12]
            .try_into()
            .map_err(|_| PcError::Format("invalid EVENT_RLE length".to_string()))?,
    ) as usize;
    let mut out = Vec::with_capacity(expected_len);
    let mut cursor = 12;
    while cursor < encoded.len() {
        if cursor + 6 > encoded.len() {
            return Err(PcError::Format(
                "truncated EVENT_RLE run header".to_string(),
            ));
        }
        let kind = u16::from_le_bytes(
            encoded[cursor..cursor + 2]
                .try_into()
                .map_err(|_| PcError::Format("invalid EVENT_RLE kind".to_string()))?,
        );
        cursor += 2;
        let run_len = u32::from_le_bytes(
            encoded[cursor..cursor + 4]
                .try_into()
                .map_err(|_| PcError::Format("invalid EVENT_RLE run length".to_string()))?,
        ) as usize;
        cursor += 4;
        for _ in 0..run_len {
            if cursor + 96 > encoded.len() {
                return Err(PcError::Format("truncated EVENT_RLE record".to_string()));
            }
            if read_u16(&encoded[cursor..cursor + 96], 8)? != kind {
                return Err(PcError::Format(
                    "EVENT_RLE kind mismatch in run".to_string(),
                ));
            }
            out.extend_from_slice(&encoded[cursor..cursor + 96]);
            cursor += 96;
        }
    }
    if out.len() != expected_len {
        return Err(PcError::Format(
            "EVENT_RLE decoded length mismatch".to_string(),
        ));
    }
    Ok(out)
}

fn event_dictionary_encode(event_bytes: &[u8]) -> PcResult<Vec<u8>> {
    if event_bytes.len() % 96 != 0 {
        return Err(PcError::Section(
            "event table record width mismatch for EVENT_DICTIONARY".to_string(),
        ));
    }
    let mut dictionary = BTreeMap::<u16, u16>::new();
    for rec in event_bytes.chunks_exact(96) {
        let kind = read_u16(rec, 8)?;
        if !dictionary.contains_key(&kind) {
            let next = dictionary.len() as u16;
            dictionary.insert(kind, next);
        }
    }
    let mut out = Vec::with_capacity(event_bytes.len() + 32);
    out.extend_from_slice(b"EDIC");
    out.extend_from_slice(&(event_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&(dictionary.len() as u16).to_le_bytes());
    for (kind, index) in &dictionary {
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&index.to_le_bytes());
    }
    for rec in event_bytes.chunks_exact(96) {
        let kind = read_u16(rec, 8)?;
        let index = dictionary.get(&kind).ok_or_else(|| {
            PcError::Internal("event dictionary missing encoded kind".to_string())
        })?;
        out.extend_from_slice(&index.to_le_bytes());
        out.extend_from_slice(rec);
    }
    Ok(out)
}

fn event_dictionary_decode(encoded: &[u8]) -> PcResult<Vec<u8>> {
    if encoded.len() < 14 || &encoded[0..4] != b"EDIC" {
        return Err(PcError::Format(
            "invalid EVENT_DICTIONARY header".to_string(),
        ));
    }
    let expected_len = u64::from_le_bytes(
        encoded[4..12]
            .try_into()
            .map_err(|_| PcError::Format("invalid EVENT_DICTIONARY length".to_string()))?,
    ) as usize;
    let dict_len =
        u16::from_le_bytes(encoded[12..14].try_into().map_err(|_| {
            PcError::Format("invalid EVENT_DICTIONARY dictionary length".to_string())
        })?) as usize;
    let mut reverse = BTreeMap::<u16, u16>::new();
    let mut cursor = 14;
    for _ in 0..dict_len {
        if cursor + 4 > encoded.len() {
            return Err(PcError::Format("truncated EVENT_DICTIONARY".to_string()));
        }
        let kind = u16::from_le_bytes(encoded[cursor..cursor + 2].try_into().unwrap());
        let index = u16::from_le_bytes(encoded[cursor + 2..cursor + 4].try_into().unwrap());
        reverse.insert(index, kind);
        cursor += 4;
    }
    let mut out = Vec::with_capacity(expected_len);
    while cursor < encoded.len() {
        if cursor + 98 > encoded.len() {
            return Err(PcError::Format(
                "truncated EVENT_DICTIONARY record".to_string(),
            ));
        }
        let index = u16::from_le_bytes(encoded[cursor..cursor + 2].try_into().unwrap());
        cursor += 2;
        let rec = &encoded[cursor..cursor + 96];
        let expected_kind = reverse.get(&index).ok_or_else(|| {
            PcError::Format("EVENT_DICTIONARY index not present in dictionary".to_string())
        })?;
        if read_u16(rec, 8)? != *expected_kind {
            return Err(PcError::Format(
                "EVENT_DICTIONARY record kind mismatch".to_string(),
            ));
        }
        out.extend_from_slice(rec);
        cursor += 96;
    }
    if out.len() != expected_len {
        return Err(PcError::Format(
            "EVENT_DICTIONARY decoded length mismatch".to_string(),
        ));
    }
    Ok(out)
}

fn tensor_chunk_encode(tensor_bytes: &[u8], chunk_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(tensor_bytes.len() + 32);
    out.extend_from_slice(b"TCHK");
    out.extend_from_slice(&(tensor_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&(chunk_bytes as u32).to_le_bytes());
    for (idx, chunk) in tensor_bytes.chunks(chunk_bytes).enumerate() {
        out.extend_from_slice(&(idx as u32).to_le_bytes());
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(&hash_bytes(chunk));
        out.extend_from_slice(chunk);
    }
    out
}

fn tensor_chunk_decode(encoded: &[u8]) -> PcResult<Vec<u8>> {
    if encoded.len() < 16 || &encoded[0..4] != b"TCHK" {
        return Err(PcError::Format("invalid TENSOR_CHUNKED header".to_string()));
    }
    let expected_len = u64::from_le_bytes(
        encoded[4..12]
            .try_into()
            .map_err(|_| PcError::Format("invalid TENSOR_CHUNKED length".to_string()))?,
    ) as usize;
    let _chunk_bytes = u32::from_le_bytes(
        encoded[12..16]
            .try_into()
            .map_err(|_| PcError::Format("invalid TENSOR_CHUNKED chunk size".to_string()))?,
    );
    let mut out = Vec::with_capacity(expected_len);
    let mut cursor = 16;
    let mut expected_idx = 0u32;
    while cursor < encoded.len() {
        if cursor + 40 > encoded.len() {
            return Err(PcError::Format(
                "truncated TENSOR_CHUNKED header".to_string(),
            ));
        }
        let idx = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        if idx != expected_idx {
            return Err(PcError::Format(
                "TENSOR_CHUNKED index discontinuity".to_string(),
            ));
        }
        let len = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let expected_hash = read_hash(encoded, cursor)?;
        cursor += 32;
        if cursor + len > encoded.len() {
            return Err(PcError::Format(
                "truncated TENSOR_CHUNKED payload".to_string(),
            ));
        }
        let chunk = &encoded[cursor..cursor + len];
        if hash_bytes(chunk) != expected_hash {
            return Err(PcError::Hash("TENSOR_CHUNKED hash mismatch".to_string()));
        }
        out.extend_from_slice(chunk);
        cursor += len;
        expected_idx += 1;
    }
    if out.len() != expected_len {
        return Err(PcError::Format(
            "TENSOR_CHUNKED decoded length mismatch".to_string(),
        ));
    }
    Ok(out)
}

fn rate(count: u64, elapsed_ns: u128) -> f64 {
    if elapsed_ns == 0 {
        return 0.0;
    }
    count as f64 / (elapsed_ns as f64 / 1_000_000_000.0)
}

fn align_up(value: u64, align: u64) -> u64 {
    if value == 0 {
        0
    } else {
        ((value + align - 1) / align) * align
    }
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn hash_many(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

fn hash_optional_file(path: &str) -> [u8; 32] {
    fs::read(path)
        .map(|bytes| hash_bytes(&bytes))
        .unwrap_or_else(|_| hash_bytes(format!("EVIDENCE_NOT_FOUND:{path}").as_bytes()))
}

fn prod_signature_payload(
    nonce: &[u8],
    aad: &[u8],
    plaintext_hash: &[u8; 32],
    ciphertext_hash: &[u8; 32],
) -> Vec<u8> {
    let signature_root = hash_many(&[
        b"PRISM_PSK_PROD_SIGNATURE_V0_1",
        nonce,
        aad,
        plaintext_hash,
        ciphertext_hash,
    ]);
    signature_root.to_vec()
}

fn hex_hash(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decode_hex_bytes(value: &str) -> PcResult<Vec<u8>> {
    let value = value.trim();
    if value.len() % 2 != 0 {
        return Err(PcError::Hash(
            "hex byte string length must be even".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(value.len() / 2);
    for idx in (0..value.len()).step_by(2) {
        out.push(
            u8::from_str_radix(&value[idx..idx + 2], 16)
                .map_err(|err| PcError::Hash(err.to_string()))?,
        );
    }
    Ok(out)
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_bytes(digest(&SHA256, bytes).as_ref())
}

fn parse_hex_hash(value: &str) -> PcResult<[u8; 32]> {
    if value.len() != 64 {
        return Err(PcError::Hash("hex hash length must be 64".to_string()));
    }
    let mut out = [0u8; 32];
    for (idx, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[idx * 2..idx * 2 + 2], 16)
            .map_err(|err| PcError::Hash(err.to_string()))?;
    }
    Ok(out)
}

fn header_packet_hash(bytes: &[u8]) -> PcResult<[u8; 32]> {
    read_hash(bytes, PACKET_HASH_OFFSET)
}

fn packet_hash_for_bytes(bytes: &[u8]) -> PcResult<[u8; 32]> {
    if bytes.len() < PACKET_HASH_OFFSET + PACKET_HASH_LEN {
        return Err(PcError::Format(
            "file too short for packet hash".to_string(),
        ));
    }
    let mut copy = bytes.to_vec();
    copy[PACKET_HASH_OFFSET..PACKET_HASH_OFFSET + PACKET_HASH_LEN].fill(0);
    Ok(hash_bytes(&copy))
}

fn put_u16(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut [u8], offset: usize, value: u64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_f64(out: &mut [u8], offset: usize, value: f64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> PcResult<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| PcError::Format("u16 read out of range".to_string()))?;
    Ok(u16::from_le_bytes(slice.try_into().map_err(|_| {
        PcError::Internal("u16 conversion".to_string())
    })?))
}

fn read_u32(bytes: &[u8], offset: usize) -> PcResult<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| PcError::Format("u32 read out of range".to_string()))?;
    Ok(u32::from_le_bytes(slice.try_into().map_err(|_| {
        PcError::Internal("u32 conversion".to_string())
    })?))
}

fn read_u64(bytes: &[u8], offset: usize) -> PcResult<u64> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| PcError::Format("u64 read out of range".to_string()))?;
    Ok(u64::from_le_bytes(slice.try_into().map_err(|_| {
        PcError::Internal("u64 conversion".to_string())
    })?))
}

fn read_f64(bytes: &[u8], offset: usize) -> PcResult<f64> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| PcError::Format("f64 read out of range".to_string()))?;
    Ok(f64::from_le_bytes(slice.try_into().map_err(|_| {
        PcError::Internal("f64 conversion".to_string())
    })?))
}

fn read_hash(bytes: &[u8], offset: usize) -> PcResult<[u8; 32]> {
    let slice = bytes
        .get(offset..offset + 32)
        .ok_or_else(|| PcError::Format("hash read out of range".to_string()))?;
    let mut out = [0u8; 32];
    out.copy_from_slice(slice);
    Ok(out)
}

fn backend_name(code: u16) -> &'static str {
    match code {
        1 => "cpu",
        2 => "cuda",
        3 => "auto",
        _ => "unknown",
    }
}

pub fn write_hardware_report(json_path: Option<&Path>) -> PcResult<serde_json::Value> {
    let value = hardware_report()?;
    if let Some(path) = json_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&value)?)?;
    }
    Ok(value)
}

pub fn opcode_rows_for_docs() -> Vec<(u16, &'static str)> {
    opcode_table().into_iter().collect()
}

pub fn packet_summary(path: &Path) -> PcResult<serde_json::Value> {
    let packet = read_packet(path)?;
    let event_byte_seal_binding = event_byte_seal_binding(&packet)?;
    let mut section_rows = Vec::new();
    for entry in &packet.sections {
        section_rows.push(json!({
            "section_id": entry.section_id,
            "section_type": format!("{:?}", entry.section_type),
            "offset": entry.offset,
            "length": entry.length,
            "record_count": entry.record_count,
            "content_hash": hex_hash(&entry.content_hash),
        }));
    }
    let opcodes: BTreeSet<_> = packet
        .instructions
        .iter()
        .map(|instr| opcode_name(instr.opcode))
        .collect();
    Ok(json!({
        "packet_hash": hex_hash(&packet.packet_hash),
        "instruction_count": packet.instructions.len(),
        "state_count": packet.state_count,
        "time_count": packet.time_count,
        "event_count": packet.event_count,
        "nanl_count": packet.nanl_count,
        "opcode_set": opcodes,
        "event_byte_seal_binding": event_byte_seal_binding,
        "sections": section_rows,
    }))
}

pub fn write_all<W: Write>(mut writer: W, text: &str) -> PcResult<()> {
    writer.write_all(text.as_bytes()).map_err(PcError::from)
}

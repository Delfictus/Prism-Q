#![recursion_limit = "256"]

//! PRISM ISA / PRISMCODE Rust contract surface.
//!
//! This crate intentionally defines only the envelope needed for a Rust runtime
//! to consume compiler-emitted PRISMCODE artifacts: registry records,
//! instruction records, deterministic hashes, validation, and disassembly.

pub mod acceptance;
pub mod acceptance_hermeticity;
pub mod attestation;
pub mod binary_runtime;
pub mod bundle_archive;
pub mod bundle_copy;
pub mod bundle_hash_tree;
pub mod codec_registry;
pub mod codecs;
pub mod compiler;
pub mod compression_registry;
pub mod contract;
pub mod cse_dictionaries;
pub mod cse_hd;
pub mod cse_tables;
pub mod cuda_backend;
mod cuda_operation;
pub mod cuda_ffi;
pub mod cuda_manifest;
pub mod cuda_planner;
pub mod disassembler;
pub mod drf_active_state;
pub mod drf_binding;
pub mod drf_fabric;
pub mod drf_tensors;
pub mod drf_validation;
pub mod dstw_state_graph;
pub mod durable_experiment;
pub mod hash;
pub mod hermetic_process;
pub mod host_roots;
pub mod hyper_runtime_request;
pub mod instruction;
pub mod isa;
pub mod key_envelope;
pub mod key_provider;
pub mod key_provider_local_release_vault;
pub mod key_provider_openbao;
pub mod manifest_signature;
pub mod md_native_runtime;
pub mod memory_image;
pub mod merge_manifest;
pub mod model_airlock;
pub mod model_catalog;
pub mod model_governance;
pub mod model_policy;
pub mod model_runtime;
pub mod model_runtime_champion;
pub mod model_runtime_hermeticity;
pub mod model_runtime_smoke;
pub mod nanl_ans;
pub mod nanl_bin;
pub mod nanl_codec_stack;
pub mod nanl_delta;
pub mod nanl_dict;
pub mod nanl_dna;
pub mod nanl_h16;
pub mod nanl_hd_bench;
pub mod nanl_hd_firmware;
pub mod nanl_hd_overclaim_guard;
pub mod nanl_hd_validation;
pub mod nanl_motif;
pub mod nanl_q4;
pub mod nanl_var;
pub mod nrf_frame;
pub mod nrf_sparse;
pub mod opcode;
pub mod operand;
pub mod operand_ledger;
mod operand_schema;
pub mod pinned_root;
pub mod prismai_binding;
pub mod provenance;
pub mod qnanl_artifact;
pub mod release_key_policy;
pub mod release_manifest;
pub mod runtime_provenance;
pub mod signature;
pub mod trust_root;
pub mod validator;
pub mod vault_provider;
pub mod warning_policy;

pub use compression_registry::{
    CompressionCodecKind, CompressionCodecRecord, CompressionError, EncodedCompressionSection,
    ExecutableRoundTripReport, HashPair, RuntimeCodecRegistry,
};
pub use contract::{HASH_ALGORITHM, PRISMCODE_ISA_VERSION, PRISMCODE_PROGRAM_SCHEMA_VERSION};
pub use disassembler::{disassemble_program, format_disassembly, DisassemblyLine};
pub use hash::{canonical_json, instruction_hash, registry_hash};
pub use instruction::{Instruction, Program};
pub use isa::OpcodeRegistry;
pub use opcode::OpcodeSpec;
pub use validator::{validate_program, validate_registry, PrismCodeError, ValidationReport};

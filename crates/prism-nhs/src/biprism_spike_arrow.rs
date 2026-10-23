//! Paired Arrow V5 and lossless V10S ledger export from one decoded queue.
//!
//! Each native row and companion row have the same (replica_seed, spike_id).
//! Row IDs start at zero within an explicitly named queue export. The runtime
//! owner acknowledges retained exports before reusing its queue. The companion retains the complete 96-byte record,
//! including original atom coordinates and every binary32 ledger bit pattern.

use crate::biprism_spike::{
    decode_spike_queue_for_arrow_and_pathb_in_frame, BiprismDecodeLimits, BiprismEventFrame,
    BiprismProducerGrid, DecodedVoronoiSpikeQueue, NativeLifObservation,
};
use crate::spike_arrow_writer::{build_spike_record_batch, SpikeClassification};
use arrow_array::{
    ArrayRef, BinaryArray, Float32Array, Int32Array, RecordBatch, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use std::{collections::HashMap, sync::Arc};

/// The native rows remain available for PathB; exporting never consumes or
/// replaces their original, fully validated ledger.
pub struct VoronoiSpikeExport {
    pub decoded: DecodedVoronoiSpikeQueue,
    pub native_v5: RecordBatch,
    pub ledger: RecordBatch,
}

#[allow(clippy::too_many_arguments)]
pub fn decode_voronoi_spike_export(
    raw: &[u8],
    count: u32,
    capacity: u32,
    limits: BiprismDecodeLimits,
    observations: &[NativeLifObservation],
    geometry: crate::sdst_bridge::SdstGridGeometry,
    classifications: &[SpikeClassification],
    replica_seed: u64,
    leaf_stream_ids: &[u8; 8],
) -> Result<VoronoiSpikeExport, String> {
    decode_voronoi_spike_export_in_frame(
        raw,
        count,
        capacity,
        limits,
        observations,
        geometry,
        classifications,
        replica_seed,
        leaf_stream_ids,
        BiprismEventFrame::AtomicBohr,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_voronoi_spike_export_in_frame(
    raw: &[u8],
    count: u32,
    capacity: u32,
    limits: BiprismDecodeLimits,
    observations: &[NativeLifObservation],
    geometry: crate::sdst_bridge::SdstGridGeometry,
    classifications: &[SpikeClassification],
    replica_seed: u64,
    leaf_stream_ids: &[u8; 8],
    frame: BiprismEventFrame,
) -> Result<VoronoiSpikeExport, String> {
    validate_leaf_stream_ids(leaf_stream_ids)?;
    let producer_grid = BiprismProducerGrid::from_native_geometry(geometry, frame)?;
    if producer_grid.total_voxels != limits.total_voxels {
        return Err("V10S frame geometry differs from the decoder's voxel population".into());
    }
    let decoded = decode_spike_queue_for_arrow_and_pathb_in_frame(
        raw,
        count,
        capacity,
        limits,
        observations,
        geometry,
        frame,
    )?;
    if classifications.len() != decoded.ledger.len() {
        return Err("each V10S row requires a matched classification".into());
    }
    for (row, (record, class)) in decoded.ledger.iter().zip(classifications).enumerate() {
        if class.stream_id != leaf_stream_ids[usize::from(record.event.leaf_id)] {
            return Err(format!(
                "V10S row {row}: classification stream differs from the owner leaf map"
            ));
        }
    }
    let native = build_spike_record_batch(&decoded.native_events, classifications, replica_seed)
        .map_err(|e| e.to_string())?;
    let mut metadata = native.schema().metadata().clone();
    metadata.insert("prism.event_producer".into(), "voronoi_lif_v10s".into());
    metadata.insert("prism.source_event_frame".into(), frame.name().into());
    metadata.insert("prism.native_position_unit".into(), "angstrom".into());
    metadata.insert(
        "prism.ledger_required".into(),
        "voronoi_lif_ledger_v2".into(),
    );
    metadata.insert(
        "prism.position_projection".into(),
        "voxel_center; original_atom_xyz_high_and_low_in_ledger".into(),
    );
    metadata.insert(
        "prism.event_count_semantics".into(),
        "emission_groups; integer_grids_count_multiplicity".into(),
    );
    let native_v5 = RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            native.schema().fields().clone(),
            metadata,
        )),
        native.columns().to_vec(),
    )
    .map_err(|e| e.to_string())?;

    let rows = &decoded.ledger;
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    macro_rules! column {
        ($name:literal, $kind:ident, $array:ident, $value:expr) => {{
            fields.push(Field::new($name, DataType::$kind, false));
            columns.push(Arc::new($array::from(
                rows.iter().map($value).collect::<Vec<_>>(),
            )));
        }};
    }
    column!(
        "spike_id",
        UInt64,
        UInt64Array,
        |r: &crate::biprism_spike::VoronoiSpikeLedgerRecord| r.queue_row
    );
    column!("replica_seed", UInt64, UInt64Array, |_| replica_seed);
    column!("step_tau", UInt32, UInt32Array, |r| r.event.step_tau);
    column!("leaf_id", UInt16, UInt16Array, |r| r.event.leaf_id);
    column!("vertex_id", UInt16, UInt16Array, |r| r.event.vertex_id);
    column!("atom_or_site_id", UInt32, UInt32Array, |r| r
        .event
        .atom_or_site_id);
    column!("pocket_tile_id", UInt32, UInt32Array, |r| r
        .event
        .pocket_tile_id);
    column!("residue_id", Int32, Int32Array, |r| r.event.residue_id);
    column!("voxel_cell_idx", Int32, Int32Array, |r| r
        .event
        .voxel_cell_idx);
    column!("atom_x", Float32, Float32Array, |r| r.event.pos_xyz[0]);
    column!("atom_y", Float32, Float32Array, |r| r.event.pos_xyz[1]);
    column!("atom_z", Float32, Float32Array, |r| r.event.pos_xyz[2]);
    column!("spike_amplitude", Float32, Float32Array, |r| r
        .event
        .spike_amplitude);
    column!("holonomy_deficit", Float32, Float32Array, |r| r
        .event
        .uv_or_holonomy_val);
    column!("voronoi_dual_vol", Float32, Float32Array, |r| r
        .event
        .voronoi_dual_vol);
    column!("multiplicity_m", UInt32, UInt32Array, |r| r
        .event
        .multiplicity_m);
    column!("v_pre_reset", Float32, Float32Array, |r| r
        .event
        .v_pre_reset);
    column!("q_discharged", Float32, Float32Array, |r| r
        .event
        .q_discharged);
    column!("q_discharged_lo", Float32, Float32Array, |r| r
        .event
        .q_discharged_lo);
    column!("r_post_reset", Float32, Float32Array, |r| r
        .event
        .r_post_reset);
    column!("lif_threshold_used", Float32, Float32Array, |r| r
        .event
        .lif_threshold_used);
    column!("q_lo_product_raw_bits", UInt32, UInt32Array, |r| r
        .event
        .q_lo_product_raw_bits);
    column!("produced_force_x", Float32, Float32Array, |r| r
        .event
        .coupled_force_xyz[0]);
    column!("produced_force_y", Float32, Float32Array, |r| r
        .event
        .coupled_force_xyz[1]);
    column!("produced_force_z", Float32, Float32Array, |r| r
        .event
        .coupled_force_xyz[2]);
    column!("event_schema_magic", UInt32, UInt32Array, |r| r
        .event
        .event_schema_magic);
    column!("atom_x_low", Float32, Float32Array, |r| observations[r.queue_row as usize].producer_position_lo[0]);
    column!("atom_y_low", Float32, Float32Array, |r| observations[r.queue_row as usize].producer_position_lo[1]);
    column!("atom_z_low", Float32, Float32Array, |r| observations[r.queue_row as usize].producer_position_lo[2]);
    column!("coordinate_magic", UInt32, UInt32Array, |r| observations[r.queue_row as usize].coordinate_magic);
    column!("coordinate_generation", UInt64, UInt64Array, |r| observations[r.queue_row as usize].coordinate_generation);
    column!("coordinate_unit_frame", UInt32, UInt32Array, |r| observations[r.queue_row as usize].coordinate_unit_frame);
    let companion_bytes: Vec<_> = observations.iter().map(|r|r.to_le_bytes()).collect();
    fields.push(Field::new("raw_nob2_64", DataType::Binary, false));
    columns.push(Arc::new(BinaryArray::from_iter_values(companion_bytes.iter().map(|b|b.as_slice()))));
    let bytes: Vec<_> = rows.iter().map(|r| r.event.to_le_bytes()).collect();
    fields.push(Field::new("raw_v10s_96", DataType::Binary, false));
    columns.push(Arc::new(BinaryArray::from_iter_values(
        bytes.iter().map(|b| b.as_slice()),
    )));
    let metadata = HashMap::from([
        ("prism.schema".into(), "voronoi_lif_ledger_v2".into()),
        ("prism.event_frame".into(), frame.name().into()),
        ("prism.position_unit".into(), frame.position_unit().into()),
        ("prism.force_unit".into(), frame.force_unit().into()),
        (
            "prism.dual_volume_unit".into(),
            frame.dual_volume_unit().into(),
        ),
        ("prism.native_position_unit".into(), "angstrom".into()),
        (
            "prism.join".into(),
            "paired export identity,replica_seed,spike_id".into(),
        ),
        (
            "prism.force_semantics".into(),
            "produced_endpoint_force; not a delivery or work receipt".into(),
        ),
        (
            "prism.quantum_semantics".into(),
            "q_hi+q_lo+r=v_pre; emitted groups carry multiplicity_m".into(),
        ),
    ]);
    let ledger = RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(fields, metadata)),
        columns,
    )
    .map_err(|e| e.to_string())?;
    Ok(VoronoiSpikeExport {
        decoded,
        native_v5,
        ledger,
    })
}

pub(crate) fn validate_leaf_stream_ids(ids: &[u8; 8]) -> Result<(), String> {
    for (i, id) in ids.iter().enumerate() {
        if ids[..i].contains(id) {
            return Err("foliated leaf stream map must be one-to-one".into());
        }
    }
    Ok(())
}

impl VoronoiSpikeExport {
    /// Bind the actual eight physical leaf seeds to each native/ledger row.
    /// The full map remains schema metadata, including leaves with no emissions.
    /// This changes no native fields, raw96 ledger bytes, row ordinals or queue identity.
    pub fn bind_replica_seeds(&mut self, leaf_seeds: [u64; 8]) -> Result<(), String> {
        let seeds: Vec<u64> = self
            .decoded
            .ledger
            .iter()
            .map(|row| {
                leaf_seeds
                    .get(usize::from(row.event.leaf_id))
                    .copied()
                    .ok_or_else(|| "decoded event leaf exceeds the seed map".to_string())
            })
            .collect::<Result<_, _>>()?;
        let metadata_map = serde_json::to_string(&leaf_seeds).map_err(|e| e.to_string())?;
        let mut replacements = Vec::new();
        for batch in [&self.native_v5, &self.ledger] {
            if batch.num_rows() != seeds.len() {
                return Err("paired export row population differs from the seed map".into());
            }
            let idx = batch
                .schema()
                .index_of("replica_seed")
                .map_err(|e| e.to_string())?;
            let mut columns = batch.columns().to_vec();
            columns[idx] = Arc::new(UInt64Array::from(seeds.clone()));
            let mut metadata = batch.schema().metadata().clone();
            metadata.insert("prism.leaf_replica_seeds".into(), metadata_map.clone());
            metadata.insert(
                "prism.replica_seed_binding".into(),
                "raw96.leaf_id selects physical leaf seed".into(),
            );
            replacements.push(
                RecordBatch::try_new(
                    Arc::new(Schema::new_with_metadata(
                        batch.schema().fields().clone(),
                        metadata,
                    )),
                    columns,
                )
                .map_err(|e| e.to_string())?,
            );
        }
        self.ledger = replacements.pop().unwrap();
        self.native_v5 = replacements.pop().unwrap();
        Ok(())
    }
    /// Bind both IPC schemas to the same owner allocation and queue generation.
    /// Callers supply the run-unique allocation identity when creating the owner.
    pub(crate) fn bind_queue_export(&mut self, owner: &str, generation: u64) -> Result<(), String> {
        for batch in [&mut self.native_v5, &mut self.ledger] {
            let mut metadata = batch.schema().metadata().clone();
            metadata.insert("prism.queue_owner".into(), owner.into());
            metadata.insert("prism.queue_generation".into(), generation.to_string());
            *batch = RecordBatch::try_new(
                Arc::new(Schema::new_with_metadata(
                    batch.schema().fields().clone(),
                    metadata,
                )),
                batch.columns().to_vec(),
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    /// Serialize both required artifacts with the real Arrow IPC writer. A
    /// caller must retain both buffers; no ledger is hidden in residue slots.
    pub fn to_paired_ipc(&self) -> Result<(Vec<u8>, Vec<u8>), String> {
        fn encode(batch: &RecordBatch) -> Result<Vec<u8>, String> {
            let mut bytes = Vec::new();
            {
                let mut writer =
                    arrow_ipc::writer::FileWriter::try_new(&mut bytes, &batch.schema())
                        .map_err(|e| e.to_string())?;
                writer.write(batch).map_err(|e| e.to_string())?;
                writer.finish().map_err(|e| e.to_string())?;
            }
            Ok(bytes)
        }
        Ok((encode(&self.native_v5)?, encode(&self.ledger)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biprism_spike::{GpuSpikeEventCanonical96Ffi, SPIKE_EVENT_V10_SCHEMA_MAGIC};
    fn fixture() -> (
        GpuSpikeEventCanonical96Ffi,
        BiprismDecodeLimits,
        NativeLifObservation,
        crate::sdst_bridge::SdstGridGeometry,
        SpikeClassification,
    ) {
        let th = 1.1f32;
        let hi = 3.0 * th;
        let lo = 3.0f32.mul_add(th, -hi);
        let e = GpuSpikeEventCanonical96Ffi {
            pos_xyz: [0.25; 3],
            step_tau: 17,
            atom_or_site_id: 0,
            spike_amplitude: hi,
            uv_or_holonomy_val: 0.0,
            voronoi_dual_vol: 2.0,
            pocket_tile_id: 0,
            residue_id: 0,
            multiplicity_m: 3,
            v_pre_reset: 3.5,
            q_discharged: hi,
            q_discharged_lo: lo,
            r_post_reset: (-3.0f32).mul_add(th, 3.5),
            lif_threshold_used: th,
            leaf_id: 0,
            vertex_id: 0,
            voxel_cell_idx: 0,
            event_schema_magic: SPIKE_EVENT_V10_SCHEMA_MAGIC,
            coupled_force_xyz: [1.0, -1.0, 0.0],
            q_lo_product_raw_bits: lo.to_bits(),
            reserved_v5: 0,
        };
        let limits = BiprismDecodeLimits {
            num_atoms: 256,
            num_residues: 32,
            total_voxels: 8,
            expected_step: Some(17),
        };
        let o = NativeLifObservation {
            step_tau: 17,
            leaf_id: 0,
            vertex_id: 0,
            atom_or_site_id: 0,
            voxel_cell_idx: 0,
            water_density: 0.75,
            wd_change: 0.125,
            n_nearby_excited: 0,
            phase_bits: 0,
            producer_position_lo: [0.0;3], coordinate_magic: crate::biprism_spike::NATIVE_LIF_OBSERVATION_MAGIC,
            coordinate_generation: 1, coordinate_unit_frame: 2, coordinate_reserved: 0,
        };
        let g = crate::sdst_bridge::SdstGridGeometry {
            dimension: 2,
            spacing_a: 1.0,
            origin_a: [0.0; 3],
        };
        let c = SpikeClassification {
            stream_id: 0,
            group_id: 0,
            chunk_idx: 0,
            ccns_phase: 0,
            site_id: -1,
            nearest_site_id: -1,
            nearest_site_dist: 0.0,
            background_class: 0,
            burial_score: 0.125,
            intensity_percentile: 50,
        };
        (e, limits, o, g, c)
    }
    #[test]
    fn interleaved_leaf_seeds_preserve_paired_identity_and_raw_ledger() {
        use arrow_array::UInt64Array;
        let (e, l, o, g, c) = fixture();
        let mut e7 = e;
        e7.leaf_id = 7;
        let mut o7 = o;
        o7.leaf_id = 7;
        let mut c7 = c.clone();
        c7.stream_id = 7;
        let mut raw = e.to_le_bytes().to_vec();
        raw.extend_from_slice(&e7.to_le_bytes());
        let mut export = decode_voronoi_spike_export(
            &raw,
            2,
            2,
            l,
            &[o, o7],
            g,
            &[c, c7],
            999,
            &[0, 1, 2, 3, 4, 5, 6, 7],
        )
        .unwrap();
        let original = export
            .decoded
            .ledger
            .iter()
            .map(|r| r.event.to_le_bytes())
            .collect::<Vec<_>>();
        export
            .bind_replica_seeds([11, 22, 33, 44, 55, 66, 77, 88])
            .unwrap();
        for batch in [&export.native_v5, &export.ledger] {
            let seeds = batch
                .column_by_name("replica_seed")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(seeds.values().as_ref(), &[11, 88]);
            assert_eq!(
                batch
                    .schema()
                    .metadata()
                    .get("prism.leaf_replica_seeds")
                    .unwrap(),
                "[11,22,33,44,55,66,77,88]"
            );
        }
        assert_eq!(
            export
                .decoded
                .ledger
                .iter()
                .map(|r| r.event.to_le_bytes())
                .collect::<Vec<_>>(),
            original
        );
    }

    #[test]
    fn real_arrow_v5_and_full_companion_round_trip() {
        let (e, l, o, g, c) = fixture();
        let mut export = decode_voronoi_spike_export(
            &e.to_le_bytes(),
            1,
            1,
            l,
            &[o],
            g,
            &[c],
            42,
            &[0, 1, 2, 3, 4, 5, 6, 7],
        )
        .unwrap();
        export.bind_queue_export("test-run/queue-0", 3).unwrap();
        assert_eq!(export.native_v5.num_columns(), 30);
        let native = &export.decoded.native_events[0];
        assert_eq!(native.position, [0.5; 3]);
        assert_eq!(native.nearby_residues, [0, -1, -1, -1, -1, -1, -1, -1]);
        assert_eq!(native.water_density, 0.75);
        assert_eq!(native.spike_source, 6);
        assert_eq!(
            export.decoded.event_frame,
            BiprismEventFrame::AtomicBohr
        );
        assert_eq!(
            export.ledger.schema().metadata()["prism.force_unit"],
            "hartree/bohr"
        );
        assert_eq!(
            crate::spike_arrow_writer::mechanism_tag_for_spike(native),
            "VORONOI_LIF"
        );
        let (nbytes, lbytes) = export.to_paired_ipc().unwrap();
        for bytes in [&nbytes, &lbytes] {
            let mut reader =
                arrow_ipc::reader::FileReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
            assert_eq!(
                reader
                    .schema()
                    .metadata()
                    .get("prism.queue_generation")
                    .unwrap(),
                "3"
            );
            assert_eq!(
                reader.schema().metadata().get("prism.queue_owner").unwrap(),
                "test-run/queue-0"
            );
            assert_eq!(reader.next().unwrap().unwrap().num_rows(), 1);
            assert!(reader.next().is_none());
        }
        let mut reader =
            arrow_ipc::reader::FileReader::try_new(std::io::Cursor::new(lbytes), None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        let raw = batch
            .column_by_name("raw_v10s_96")
            .unwrap()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(raw.value(0), e.to_le_bytes());
        assert_eq!(
            GpuSpikeEventCanonical96Ffi::from_le_bytes(raw.value(0), l).unwrap(),
            e
        );
    }
    #[test]
    fn projection_refuses_wrong_context_and_center_overflow() {
        let (mut e, mut l, mut o, mut g, c) = fixture();
        o.step_tau += 1;
        assert!(decode_voronoi_spike_export(
            &e.to_le_bytes(),
            1,
            1,
            l,
            &[o],
            g,
            &[c],
            42,
            &[0, 1, 2, 3, 4, 5, 6, 7]
        )
        .is_err());
        o.step_tau = 17;
        o.n_nearby_excited = 256;
        assert!(decode_voronoi_spike_export(
            &e.to_le_bytes(),
            1,
            1,
            l,
            &[o],
            g,
            &[c],
            42,
            &[0, 1, 2, 3, 4, 5, 6, 7]
        )
        .is_err());
        o.n_nearby_excited = 0;
        e.pos_xyz = [3e38; 3];
        g.origin_a = [3e38; 3];
        g.spacing_a = 3e38;
        l.total_voxels = 8;
        assert!(decode_voronoi_spike_export(
            &e.to_le_bytes(),
            1,
            1,
            l,
            &[o],
            g,
            &[c],
            42,
            &[0, 1, 2, 3, 4, 5, 6, 7]
        )
        .err()
        .unwrap()
        .contains("overflows"));
    }
    #[test]
    fn projection_binds_atom_cell_and_owner_stream_mapping() {
        let (e, l, mut o, g, mut c) = fixture();
        let map = [8, 9, 10, 11, 12, 13, 14, 15];
        assert!(
            decode_voronoi_spike_export(&e.to_le_bytes(), 1, 1, l, &[o], g, &[c], 42, &map)
                .is_err()
        );
        c.stream_id = 8;
        assert!(
            decode_voronoi_spike_export(&e.to_le_bytes(), 1, 1, l, &[o], g, &[c], 42, &map).is_ok()
        );
        o.atom_or_site_id = 1;
        assert!(
            decode_voronoi_spike_export(&e.to_le_bytes(), 1, 1, l, &[o], g, &[c], 42, &map)
                .is_err()
        );
        o.atom_or_site_id = 0;
        o.voxel_cell_idx = 1;
        assert!(
            decode_voronoi_spike_export(&e.to_le_bytes(), 1, 1, l, &[o], g, &[c], 42, &map)
                .is_err()
        );
        assert!(validate_leaf_stream_ids(&[0; 8]).is_err());
    }
    #[test]
    fn atomic_frame_keeps_raw_ledger_and_native_angstrom_centers() {
        let (mut e, l, o, g, c) = fixture();
        e.pos_xyz = e
            .pos_xyz
            .map(|x| (f64::from(x) * crate::biprism_spike::BIPRISM_ANGSTROM_TO_BOHR) as f32);
        let raw = e.to_le_bytes();
        let export = decode_voronoi_spike_export_in_frame(
            &raw,
            1,
            1,
            l,
            &[o],
            g,
            &[c],
            42,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            BiprismEventFrame::AtomicBohr,
        )
        .unwrap();
        assert_eq!(export.native_v5.num_columns(), 30);
        assert_eq!(export.ledger.num_columns(), 34);
        assert_eq!(export.decoded.native_events[0].position, [0.5; 3]);
        assert_eq!(
            export.decoded.native_events[0].water_density,
            o.water_density
        );
        assert_eq!(export.decoded.ledger[0].event.to_le_bytes(), raw);
        assert_eq!(export.decoded.event_frame, BiprismEventFrame::AtomicBohr);
        let (_, ledger_ipc) = export.to_paired_ipc().unwrap();
        let mut reader =
            arrow_ipc::reader::FileReader::try_new(std::io::Cursor::new(ledger_ipc), None).unwrap();
        assert_eq!(
            reader.schema().metadata()["prism.event_frame"],
            "atomic_bohr"
        );
        assert_eq!(reader.schema().metadata()["prism.position_unit"], "bohr");
        assert_eq!(
            reader.schema().metadata()["prism.force_unit"],
            "hartree/bohr"
        );
        let batch = reader.next().unwrap().unwrap();
        let retained = batch
            .column_by_name("raw_v10s_96")
            .unwrap()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(retained.value(0), raw);
    }
    #[test]
    fn atomic_grid_boundary_uses_producer_arithmetic_without_roundtrip() {
        let (mut e, mut l, mut o, mut g, c) = fixture();
        g.dimension = 16;
        g.spacing_a = 0.1;
        l.total_voxels = 4096;
        // Actual binary32 counterexample: direct AU mapping gives x-cell2;
        // converting this same packet position back to Å before floor gives3.
        e.pos_xyz = [f32::from_bits(0x3f112187), 0.05, 0.05];
        e.voxel_cell_idx = 2;
        o.voxel_cell_idx = 2;
        let back =
            (f64::from(e.pos_xyz[0]) / crate::biprism_spike::BIPRISM_ANGSTROM_TO_BOHR) as f32;
        assert_eq!((back * (1.0f32 / g.spacing_a)).floor(), 3.0);
        let producer =
            BiprismProducerGrid::from_native_geometry(g, BiprismEventFrame::AtomicBohr).unwrap();
        assert_eq!(producer.recorded_cell(e.pos_xyz).unwrap(), 2);
        let export = decode_voronoi_spike_export_in_frame(
            &e.to_le_bytes(),
            1,
            1,
            l,
            &[o],
            g,
            &[c],
            42,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            BiprismEventFrame::AtomicBohr,
        )
        .unwrap();
        assert_eq!(
            export.decoded.native_events[0].position,
            [
                2.5f32.mul_add(g.spacing_a, 0.0),
                0.5 * g.spacing_a,
                0.5 * g.spacing_a
            ]
        );
        assert!(decode_voronoi_spike_export(
            &e.to_le_bytes(),
            1,
            1,
            l,
            &[o],
            g,
            &[c],
            42,
            &[0, 1, 2, 3, 4, 5, 6, 7]
        )
        .is_err());
        e.voxel_cell_idx = 3;
        o.voxel_cell_idx = 3;
        assert!(decode_voronoi_spike_export_in_frame(
            &e.to_le_bytes(),
            1,
            1,
            l,
            &[o],
            g,
            &[c],
            42,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            BiprismEventFrame::AtomicBohr
        )
        .is_err());
        g.origin_a[0] = 3e38;
        assert!(
            BiprismProducerGrid::from_native_geometry(g, BiprismEventFrame::AtomicBohr).is_err()
        );
    }
}

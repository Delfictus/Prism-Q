//! Gate G1 — Multi-seed Arrow file merger.
//!
//! Concatenates multiple `<prefix>.topology.spike_events.arrow` files
//! (each from a separate `--replica-seed` canonical run) into a single
//! multi-seed Arrow IPC stream. Per-spike provenance is preserved
//! through the existing `replica_seed: u64` schema column (column 2
//! in [`crate::spike_arrow_writer`]'s 30-column layout); the merged
//! file simply concatenates all input batches without rewriting the
//! `replica_seed` field — every spike retains its source-run seed.
//!
//! # Schema invariant
//!
//! All input files MUST share the same Arrow schema (the canonical
//! 30-column spike-events schema produced by
//! [`crate::spike_arrow_writer::build_spike_schema`]). The merger
//! verifies this on open and returns
//! [`MergeError::SchemaMismatch`] on the first divergence.
//!
//! # No producer-side changes
//!
//! This module is strictly a post-MD consumer/producer. Reads three
//! Arrow files, writes one Arrow file. No GPU, no Python, no CUDA.
//! No data-plane Python; all logic in Rust (per §1).
//!
//! # Determinism
//!
//! Output row order = concatenation of inputs in caller-supplied
//! order. No reordering, no merge-sort. Two invocations on the same
//! input set in the same order produce byte-identical output (modulo
//! Arrow IPC's internal alignment padding, which is itself
//! deterministic).
//!
//! # Verification
//!
//! Acceptance gate: the merged file should contain at least
//! `n_inputs` distinct `replica_seed` values (i.e., one per input
//! file). The `MergeStats` return value reports the per-seed row
//! count and the set of distinct seeds observed.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::Schema;

/// Errors produced by the merger.
#[derive(Debug)]
pub enum MergeError {
    /// I/O error opening, reading, or writing an Arrow file.
    Io(std::io::Error),
    /// Arrow IPC reader / writer error (schema parse, batch decode).
    Arrow(arrow_schema::ArrowError),
    /// At least two input files have non-equal schemas. The
    /// producer-side schema is fixed by
    /// [`crate::spike_arrow_writer::build_spike_schema`]; if this
    /// fires, one of the inputs was emitted by a different schema
    /// version and the merger refuses to silently mix them.
    SchemaMismatch {
        /// Path of the file whose schema diverged.
        path: std::path::PathBuf,
        /// Index of the file in the caller-supplied input list.
        input_index: usize,
    },
    /// The schema does not contain a `replica_seed: u64` column,
    /// breaking the merger's per-spike provenance assumption. This
    /// indicates the input file was not produced by
    /// [`crate::spike_arrow_writer`].
    MissingReplicaSeedColumn {
        /// Path of the offending file.
        path: std::path::PathBuf,
    },
    /// The caller supplied zero input files.
    NoInputs,
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::Io(e) => write!(f, "I/O error: {e}"),
            MergeError::Arrow(e) => write!(f, "Arrow error: {e}"),
            MergeError::SchemaMismatch { path, input_index } => write!(
                f,
                "schema mismatch on input #{input_index} ({}): does not match the first file's schema",
                path.display()
            ),
            MergeError::MissingReplicaSeedColumn { path } => write!(
                f,
                "input file {} lacks the `replica_seed: u64` column required for multi-seed merge",
                path.display()
            ),
            MergeError::NoInputs => write!(f, "merger requires >= 1 input file"),
        }
    }
}

impl std::error::Error for MergeError {}

impl From<std::io::Error> for MergeError {
    fn from(e: std::io::Error) -> Self {
        MergeError::Io(e)
    }
}

impl From<arrow_schema::ArrowError> for MergeError {
    fn from(e: arrow_schema::ArrowError) -> Self {
        MergeError::Arrow(e)
    }
}

/// Statistics emitted by [`merge_arrow_files`] after a successful
/// merge. Used by the dossier script's `replica_axis_active` check
/// (active iff `distinct_seeds.len() > 1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeStats {
    /// Total number of rows (spikes) written to the merged output.
    pub total_rows: u64,
    /// Number of input batches concatenated. Each input file
    /// contributes one or more batches.
    pub total_batches: u64,
    /// Per-seed row counts. Keys are the distinct `replica_seed`
    /// values observed across all inputs; values are the count of
    /// rows tagged with that seed in the merged output.
    pub per_seed_rows: BTreeMap<u64, u64>,
    /// Convenience: sorted list of distinct seeds. `replica_axis_active`
    /// is true iff this has length >= 2.
    pub distinct_seeds: Vec<u64>,
}

impl MergeStats {
    /// True iff the merged output contains rows from at least two
    /// distinct `replica_seed` values. The G1 acceptance gate.
    #[inline]
    pub fn replica_axis_active(&self) -> bool {
        self.distinct_seeds.len() >= 2
    }
}

/// Merge `inputs` into a single Arrow IPC file at `output`. Returns
/// per-seed row counts.
///
/// All input files are opened in caller-supplied order; their
/// batches are concatenated to the output in the same order. The
/// output schema is taken verbatim from the first input file; any
/// subsequent input whose schema does not match raises
/// [`MergeError::SchemaMismatch`].
pub fn merge_arrow_files(
    inputs: &[impl AsRef<Path>],
    output: impl AsRef<Path>,
) -> Result<MergeStats, MergeError> {
    if inputs.is_empty() {
        return Err(MergeError::NoInputs);
    }

    // Open the first file to fix the schema.
    let first_path = inputs[0].as_ref();
    let first_file = File::open(first_path)?;
    let first_reader = FileReader::try_new(first_file, None)?;
    let schema: Arc<Schema> = first_reader.schema();

    // Locate the `replica_seed: u64` column index. Fail loudly if
    // missing — the merger's provenance preservation assumption is
    // broken without it.
    let replica_seed_col =
        schema
            .index_of("replica_seed")
            .map_err(|_| MergeError::MissingReplicaSeedColumn {
                path: first_path.to_path_buf(),
            })?;

    // Open the writer with the schema we just fixed.
    let out_file = File::create(output.as_ref())?;
    let mut writer = FileWriter::try_new(out_file, &schema)?;

    let mut total_rows: u64 = 0;
    let mut total_batches: u64 = 0;
    let mut per_seed_rows: BTreeMap<u64, u64> = BTreeMap::new();

    // Re-process the first file (we already consumed its reader to
    // get the schema, but FileReader's iterator is not Clone). Open
    // a fresh reader.
    drop(first_reader);
    for (idx, path) in inputs.iter().enumerate() {
        let path_ref = path.as_ref();
        let f = File::open(path_ref)?;
        let reader = FileReader::try_new(f, None)?;
        // Schema check (idx == 0 trivially passes; idx > 0 must match).
        if reader.schema().as_ref() != schema.as_ref() {
            return Err(MergeError::SchemaMismatch {
                path: path_ref.to_path_buf(),
                input_index: idx,
            });
        }
        for batch_res in reader {
            let batch: RecordBatch = batch_res?;
            // Tally per-seed row counts.
            let seeds = batch
                .column(replica_seed_col)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("replica_seed column declared u64; downcast is safe");
            for i in 0..seeds.len() {
                *per_seed_rows.entry(seeds.value(i)).or_insert(0) += 1;
            }
            total_rows += batch.num_rows() as u64;
            total_batches += 1;
            writer.write(&batch)?;
        }
    }

    writer.finish()?;

    let distinct_seeds: Vec<u64> = per_seed_rows.keys().copied().collect();
    Ok(MergeStats {
        total_rows,
        total_batches,
        per_seed_rows,
        distinct_seeds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float32Array, RecordBatch as ArrowBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    /// Minimal test schema: two columns. `replica_seed: u64` mirrors
    /// the production schema; `intensity: f32` is a stand-in for the
    /// other 28 columns. Tests do not need the full 30-column schema
    /// to exercise the merger logic.
    fn test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("replica_seed", DataType::UInt64, false),
            Field::new("intensity", DataType::Float32, false),
        ]))
    }

    fn write_test_file(path: &Path, seed: u64, intensities: &[f32]) -> Result<(), MergeError> {
        let schema = test_schema();
        let seeds: Vec<u64> = (0..intensities.len()).map(|_| seed).collect();
        let batch = ArrowBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(seeds)),
                Arc::new(Float32Array::from(intensities.to_vec())),
            ],
        )?;
        let f = File::create(path)?;
        let mut writer = FileWriter::try_new(f, &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
        Ok(())
    }

    fn tmp_path(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        dir.join(format!(
            "g1_merge_{}_{}_{}",
            label,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn merge_two_seeds_produces_concatenated_output() {
        let p1 = tmp_path("seed42");
        let p2 = tmp_path("seed43");
        let pout = tmp_path("merged");
        write_test_file(&p1, 42, &[1.0, 2.0, 3.0]).unwrap();
        write_test_file(&p2, 43, &[4.0, 5.0, 6.0, 7.0]).unwrap();

        let stats = merge_arrow_files(&[&p1, &p2], &pout).unwrap();
        assert_eq!(stats.total_rows, 7);
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.per_seed_rows.get(&42), Some(&3));
        assert_eq!(stats.per_seed_rows.get(&43), Some(&4));
        assert_eq!(stats.distinct_seeds, vec![42, 43]);
        assert!(stats.replica_axis_active());

        // Verify the merged file roundtrips with the same row count.
        let f = File::open(&pout).unwrap();
        let reader = FileReader::try_new(f, None).unwrap();
        let mut total = 0u64;
        for b in reader {
            total += b.unwrap().num_rows() as u64;
        }
        assert_eq!(total, 7);

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
        let _ = std::fs::remove_file(&pout);
    }

    #[test]
    fn merge_three_seeds_reports_all_three_distinct() {
        let p1 = tmp_path("s42");
        let p2 = tmp_path("s43");
        let p3 = tmp_path("s44");
        let pout = tmp_path("m3");
        write_test_file(&p1, 42, &[1.0]).unwrap();
        write_test_file(&p2, 43, &[2.0, 3.0]).unwrap();
        write_test_file(&p3, 44, &[4.0, 5.0, 6.0]).unwrap();

        let stats = merge_arrow_files(&[&p1, &p2, &p3], &pout).unwrap();
        assert_eq!(stats.total_rows, 6);
        assert_eq!(stats.distinct_seeds, vec![42, 43, 44]);
        assert!(stats.replica_axis_active());

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
        let _ = std::fs::remove_file(&p3);
        let _ = std::fs::remove_file(&pout);
    }

    #[test]
    fn merge_single_seed_has_inactive_replica_axis() {
        let p1 = tmp_path("single");
        let pout = tmp_path("singleout");
        write_test_file(&p1, 42, &[1.0, 2.0]).unwrap();

        let stats = merge_arrow_files(&[&p1], &pout).unwrap();
        assert_eq!(stats.distinct_seeds, vec![42]);
        assert!(!stats.replica_axis_active());

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&pout);
    }

    #[test]
    fn merge_zero_inputs_errors() {
        let pout = tmp_path("empty");
        let inputs: &[&Path] = &[];
        let err = merge_arrow_files(inputs, &pout).unwrap_err();
        match err {
            MergeError::NoInputs => {}
            other => panic!("expected NoInputs, got {other:?}"),
        }
    }

    #[test]
    fn merge_rejects_schema_mismatch() {
        let p1 = tmp_path("good_schema");
        let p2 = tmp_path("bad_schema");
        let pout = tmp_path("rejout");
        write_test_file(&p1, 42, &[1.0]).unwrap();

        // Build a different schema: one column of UInt32 instead of UInt64
        // for replica_seed. This is a true schema mismatch even though it
        // names the same column.
        let bad_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("replica_seed", DataType::UInt32, false),
            Field::new("intensity", DataType::Float32, false),
        ]));
        let bad_batch = ArrowBatch::try_new(
            bad_schema.clone(),
            vec![
                Arc::new(arrow_array::UInt32Array::from(vec![43u32])),
                Arc::new(Float32Array::from(vec![2.0f32])),
            ],
        )
        .unwrap();
        let f = File::create(&p2).unwrap();
        let mut w = FileWriter::try_new(f, &bad_schema).unwrap();
        w.write(&bad_batch).unwrap();
        w.finish().unwrap();

        let err = merge_arrow_files(&[&p1, &p2], &pout).unwrap_err();
        match err {
            MergeError::SchemaMismatch { input_index, .. } => assert_eq!(input_index, 1),
            other => panic!("expected SchemaMismatch, got {other:?}"),
        }

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
        let _ = std::fs::remove_file(&pout);
    }

    #[test]
    fn merge_rejects_missing_replica_seed_column() {
        let p1 = tmp_path("noseed");
        let pout = tmp_path("noseedout");
        let bad_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "intensity",
            DataType::Float32,
            false,
        )]));
        let batch = ArrowBatch::try_new(
            bad_schema.clone(),
            vec![Arc::new(Float32Array::from(vec![1.0f32]))],
        )
        .unwrap();
        let f = File::create(&p1).unwrap();
        let mut w = FileWriter::try_new(f, &bad_schema).unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();

        let err = merge_arrow_files(&[&p1], &pout).unwrap_err();
        match err {
            MergeError::MissingReplicaSeedColumn { .. } => {}
            other => panic!("expected MissingReplicaSeedColumn, got {other:?}"),
        }

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&pout);
    }
}

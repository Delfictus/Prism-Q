//! Exact elapsed durations for the original native KCC winning pairs.
//! Selection, lag estimator, signed MD-step values, and source events stay intact.
use super::physical_time::VerifiedPhysicalTimeAxis;
use super::*;
use crate::causome::KccLagSupportArchive;

pub(crate) fn decode_support_record(
    encoded: &[u8],
    expected: &Value,
) -> Result<KccLagSupportArchive> {
    ensure!(
        encoded.len() >= 32 && &encoded[..8] == b"PRKLSREC",
        "KCC support record ABI"
    );
    let size = |offset| -> Result<usize> {
        Ok(usize::try_from(u64::from_le_bytes(
            encoded[offset..offset + 8].try_into()?,
        ))?)
    };
    let metadata_len = size(8)?;
    let raw_len = size(16)?;
    let compressed_len = size(24)?;
    let split = 32usize
        .checked_add(metadata_len)
        .context("KCC metadata size overflow")?;
    ensure!(
        split.checked_add(compressed_len) == Some(encoded.len()),
        "KCC encoded support length mismatch"
    );
    let metadata: Value = serde_json::from_slice(&encoded[32..split])?;
    ensure!(
        metadata == *expected
            && metadata["schema_kind"] == "prism.kcc_lag_support_zstd_record.v1"
            && metadata["codec"] == "zstd"
            && u(&metadata["raw_json_bytes"])? == raw_len as u64
            && u(&metadata["compressed_bytes"])? == compressed_len as u64,
        "KCC support metadata differs from exact source reference"
    );
    let compressed = &encoded[split..];
    ensure!(
        format!("{:x}", Sha256::digest(compressed)) == text(&metadata["compressed_sha256"])?,
        "KCC compressed support hash mismatch"
    );
    let raw = zstd::bulk::decompress(compressed, raw_len)?;
    ensure!(
        raw.len() == raw_len
            && format!("{:x}", Sha256::digest(&raw)) == text(&metadata["raw_json_sha256"])?,
        "KCC raw support hash mismatch"
    );
    let archive: KccLagSupportArchive = serde_json::from_slice(&raw)?;
    ensure!(
        archive.schema_kind == "prism.kcc_lag_support.v1"
            && archive.schema_version == 1
            && metadata["run_id"] == archive.run_id
            && metadata["stream_id"] == archive.stream_id
            && metadata["measurement_step"] == json!(archive.measurement_step)
            && u(&metadata["row_count"])? == archive.rows.len() as u64,
        "KCC decoded archive identity mismatch"
    );
    Ok(archive)
}

pub(crate) fn winning_pair_times(
    archive: &KccLagSupportArchive,
    axis: &VerifiedPhysicalTimeAxis,
    expected_run: &str,
    support_ref: &Value,
    nr: usize,
) -> Result<Value> {
    ensure!(
        archive.run_id == expected_run
            && archive.stream_id as u64 == axis.authority.source_stream_id
            && archive.rows.len() == nr,
        "KCC physical-duration source identity mismatch"
    );
    let endpoint = archive
        .measurement_step
        .context("KCC support lacks measurement endpoint")?;
    axis.cumulative_time_ps(endpoint)?;
    let mut residues = Vec::with_capacity(nr);
    let mut total = 0usize;
    for (index, row) in archive.rows.iter().enumerate() {
        ensure!(
            row.residue_index as usize == index
                && row.source_stream_id == Some(archive.stream_id)
                && row.measurement_step == Some(endpoint)
                && row.winning_pair_count as usize == row.winning_pairs.len(),
            "KCC source residue/stream/endpoint/pair-count mismatch"
        );
        let mut pairs = Vec::with_capacity(row.winning_pairs.len());
        for (ordinal, pair) in row.winning_pairs.iter().enumerate() {
            let causal = row
                .ordered_events
                .get(pair.causal_event_ordered_index as usize)
                .context("KCC causal event correspondence absent")?;
            let response = row
                .ordered_events
                .get(pair.response_event_ordered_index as usize)
                .context("KCC response event correspondence absent")?;
            ensure!(
                causal.ordered_index == pair.causal_event_ordered_index
                    && response.ordered_index == pair.response_event_ordered_index
                    && causal.step == pair.causal_step
                    && response.step == pair.response_step
                    && causal.ring_slot == pair.causal_ring_slot
                    && response.ring_slot == pair.response_ring_slot
                    && pair.actual_step_delta_response_minus_causal
                        == pair.response_step as i64 - pair.causal_step as i64
                    && (pair.causal_step as u64) <= endpoint
                    && (pair.response_step as u64) <= endpoint,
                "KCC winning pair differs from recorded event endpoints"
            );
            let tc = axis.cumulative_time_ps(pair.causal_step as u64)?;
            let tr = axis.cumulative_time_ps(pair.response_step as u64)?;
            // Compact row retains exact ordinal into full pinned winning_pairs.
            // Both endpoints are original physical steps, never sample-index lag.
            pairs.push(
                json!({"source_winning_pair_ordinal":ordinal,"causal_step":pair.causal_step,
                "response_step":pair.response_step,"causal_time_ps":tc,"response_time_ps":tr,
                "signed_duration_response_minus_causal_ps":tr-tc,
                "causal_clock_line_ordinal":pair.causal_step.checked_sub(1),
                "response_clock_line_ordinal":pair.response_step.checked_sub(1)}),
            );
        }
        total += pairs.len();
        residues.push(json!({"source_residue_ordinal":index,"residue_index":row.residue_index,
            "native_estimator_origin":row.estimator_origin,"native_lag_unit":row.lag_unit,
            "native_availability":row.availability,"native_final_causal_lag_f32_bits":row.final_causal_lag_f32_bits,
            "native_winning_pair_count":row.winning_pair_count,"winning_pair_physical_times":pairs}));
    }
    let mut value = json!({"schema_kind":"prism.native_kcc_winning_pair_physical_times.v1",
        "run_id":archive.run_id,"stream_id":archive.stream_id,"measurement_step":endpoint,
        "source_support_record":support_ref,"physical_time_authority":axis.authority,
        "source_response_semantics":archive.source_response_semantics,"residue_rows":residues,
        "winning_pair_count":total,"native_estimator_reselected":false,"native_lag_overwritten":false,
        "time_domain":"exact cumulative native time difference for each original winning pair",
        "zero_step_semantics":"declared zero-time origin; null clock row ordinal is not an interpolated row",
        "duration_is_demonstrated_causal_influence":false});
    value["record_sha256"] = json!(canonical_sha256(&value)?);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires an actual saved native KCC endpoint index"]
    fn actual_kcc_support_record_decodes_with_full_identity() {
        let path = std::env::var("PRISM_KCC_TIME_SOURCE_INDEX").unwrap();
        let row: Value = serde_json::from_str(
            &BufReader::new(File::open(path).unwrap())
                .lines()
                .next()
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let r = &row["lag_support_ref"];
        let mut file = File::open(text(&r["path"]).unwrap()).unwrap();
        file.seek(SeekFrom::Start(u(&r["byte_offset"]).unwrap()))
            .unwrap();
        let mut raw = vec![0; u(&r["byte_length"]).unwrap() as usize];
        file.read_exact(&mut raw).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&raw)),
            text(&r["encoded_record_sha256"]).unwrap()
        );
        let archive = decode_support_record(&raw, &r["record_metadata"]).unwrap();
        assert_eq!(archive.measurement_step, Some(u(&row["step"]).unwrap()));
        assert_eq!(
            archive.rows.len(),
            u(&r["record_metadata"]["row_count"]).unwrap() as usize
        );
        println!(
            "{}",
            json!({"source_support":r,"residue_rows":archive.rows.len(),
            "winning_pairs":archive.rows.iter().map(|r|r.winning_pairs.len()).sum::<usize>(),"physical_duration_not_invented":true})
        );
        let mut bad = r["record_metadata"].clone();
        bad["measurement_step"] = json!(0);
        assert!(decode_support_record(&raw, &bad).is_err());
    }
}

/// Reuse already evaluated native KCC endpoint indices without replaying
/// coordinates, forces, geometry, or the complete mechanistic evaluator.
pub fn materialize_recorded_pair_times(manifest_path: &Path, out: &Path) -> Result<Value> {
    ensure!(
        !out.exists(),
        "preserve existing KCC physical-duration output"
    );
    let manifest: Value = serde_json::from_reader(File::open(manifest_path)?)?;
    ensure!(
        manifest["schema_kind"] == "prism.native_kcc_winning_pair_time_manifest.v1",
        "KCC physical-duration manifest schema"
    );
    let entries = manifest["streams"]
        .as_array()
        .context("explicit KCC stream input list missing")?;
    ensure!(
        !entries.is_empty(),
        "empty KCC physical-duration stream population"
    );
    let mut ids = BTreeSet::new();
    let mut inputs = BTreeMap::new();
    inputs.insert(manifest_path.to_path_buf(), pin(manifest_path)?);
    collect_pins(&manifest, &mut inputs)?;
    for p in inputs.values() {
        p.verify()?;
    }
    for entry in entries {
        let arm = text(&entry["arm"])?;
        ensure!(
            matches!(arm, "treatment" | "control"),
            "unknown KCC physical arm"
        );
        let stream = u16::try_from(u(&entry["stream_id"])?)?;
        ensure!(
            ids.insert((arm.to_string(), stream)),
            "duplicate KCC physical arm/stream"
        );
        ensure!(
            !text(&entry["run_id"])?.is_empty() && u(&entry["n_residues"])? > 0,
            "KCC physical input identity missing"
        );
    }
    fs::create_dir_all(out)?;
    let mut receipts = Vec::new();
    for entry in entries {
        let arm = text(&entry["arm"])?;
        let stream = u(&entry["stream_id"])?;
        let run = text(&entry["run_id"])?;
        let authority: super::physical_time::PhysicalTimeJournalAuthority =
            serde_json::from_value(entry["physical_time_journal"].clone())?;
        ensure!(
            authority.source_stream_id == stream,
            "KCC physical journal stream mismatch"
        );
        let axis = VerifiedPhysicalTimeAxis::load(&authority)?;
        let index = source(&entry["kcc_endpoint_index"])?;
        let support = source(&entry["kcc_support"])?;
        let mut pack = File::open(&support.path)?;
        let pack_len = pack.metadata()?.len();
        let path = out.join(format!(
            "{arm}_stream{stream:02}_kcc_winning_pair_physical_times.jsonl.zst"
        ));
        let mut writer =
            zstd::stream::write::Encoder::new(BufWriter::new(File::create(&path)?), 1)?;
        let mut rows = 0u64;
        let mut pairs = 0u64;
        let mut previous = None;
        for line_in in BufReader::new(File::open(&index.path)?).lines() {
            let raw = line_in?;
            let row: Value = serde_json::from_str(&raw)?;
            let step = u(&row["step"])?;
            ensure!(
                row["run_id"] == run && row["arm"] == arm && previous.is_none_or(|p| p < step),
                "KCC source endpoint identity/order mismatch"
            );
            previous = Some(step);
            let r = &row["lag_support_ref"];
            ensure!(
                Path::new(text(&r["path"])?) == support.path && u(&r["measurement_step"])? == step,
                "KCC source pack/step mismatch"
            );
            let offset = u(&r["byte_offset"])?;
            let bytes = u(&r["byte_length"])?;
            ensure!(
                offset.checked_add(bytes).is_some_and(|end| end <= pack_len),
                "KCC source byte range outside pack"
            );
            pack.seek(SeekFrom::Start(offset))?;
            let mut encoded = vec![0; usize::try_from(bytes)?];
            pack.read_exact(&mut encoded)?;
            ensure!(
                format!("{:x}", Sha256::digest(&encoded)) == text(&r["encoded_record_sha256"])?,
                "KCC exact source record hash mismatch"
            );
            let archive = decode_support_record(&encoded, &r["record_metadata"])?;
            let mut value = winning_pair_times(
                &archive,
                &axis,
                run,
                r,
                usize::try_from(u(&entry["n_residues"])?)?,
            )?;
            pairs += u(&value["winning_pair_count"])?;
            value.as_object_mut().unwrap().remove("record_sha256");
            value["source_kcc_endpoint_index"] = json!({"source":index,"line_ordinal":rows,"canonical_row_sha256":canonical_sha256(&row)?});
            value["record_sha256"] = json!(canonical_sha256(&value)?);
            line(&mut writer, &value)?;
            rows += 1;
        }
        writer.finish()?.flush()?;
        ensure!(rows > 0, "empty source KCC measurement index");
        receipts.push(json!({"arm":arm,"stream_id":stream,"run_id":run,"physical_time_authority":authority,
            "artifact":pin(&path)?,"measurement_endpoints":rows,"winning_pairs":pairs,"source_kcc_endpoint_index":index,"source_kcc_support":support}));
    }
    for p in inputs.values() {
        p.verify()?;
    }
    let receipt = json!({"schema_kind":"prism.native_kcc_winning_pair_time_execution.v1","status":"IMPLEMENTATION_VERIFIED",
        "input_manifest":inputs.get(manifest_path),"source_pins":inputs.values().collect::<Vec<_>>(),"streams":receipts,
        "MD_executed":false,"coordinate_or_geometry_replay_executed":false,"native_lag_estimator_reselected":false,
        "scope":"exact duration of each recorded winning pair; native step-domain estimator and associations retained"});
    write_json(&out.join("EXECUTION_RECEIPT.json"), &receipt)?;
    Ok(receipt)
}

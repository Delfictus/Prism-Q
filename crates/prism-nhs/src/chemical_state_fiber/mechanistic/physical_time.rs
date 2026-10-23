//! Source-bound physical time from the native post-integrator per-step journal.
//! No end-dt multiplication, interpolation, cross-arm clock conflation, or
//! reconstruction of absent historical adaptive-step records is performed.
use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicalTimeJournalAuthority {
    pub source: MeasurementSourcePin,
    pub source_stream_id: u64,
    pub allocation_generation: u64,
    pub epoch_id: String,
    #[serde(flatten)]
    pub completion: PhysicalTimeCompletion,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysicalTimeCompletion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_step: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_step: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cumulative_simulated_time_ps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity_steps: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_origin_ps: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativePhysicalTimeRecord {
    pub schema_kind: String,
    pub schema_version: u32,
    pub source_stream_id: u64,
    pub allocation_generation: u64,
    pub epoch_id: String,
    pub step: u64,
    pub applied_dt_ps: f64,
    pub cumulative_simulated_time_ps: f64,
    pub origin: String,
    pub step_origin: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordedPhysicalTimeEndpoint {
    pub source: MeasurementSourcePin,
    pub source_line_ordinal: u64,
    pub source_line_byte_offset: u64,
    pub source_line_byte_length: u64,
    pub source_line_sha256: String,
    pub native_record: NativePhysicalTimeRecord,
}

/// Validate the entire physical journal but retain only requested endpoints.
/// A journal is a single native allocation/epoch with origin step=0,time=0;
/// missing or repeated steps are rejected rather than repaired downstream.
pub fn load_physical_time_endpoints(
    authority: &PhysicalTimeJournalAuthority,
    requested: &BTreeSet<u64>,
) -> Result<BTreeMap<u64, RecordedPhysicalTimeEndpoint>> {
    ensure!(
        !requested.is_empty() && !requested.contains(&0),
        "empty/zero requested physical time axis"
    );
    let mut endpoints = BTreeMap::new();
    scan_native_physical_time_journal(authority, |row, ordinal, offset, raw| {
        if requested.contains(&row.step) {
            endpoints.insert(
                row.step,
                RecordedPhysicalTimeEndpoint {
                    source: authority.source.clone(),
                    source_line_ordinal: ordinal,
                    source_line_byte_offset: offset,
                    source_line_byte_length: raw.len() as u64,
                    source_line_sha256: format!("{:x}", Sha256::digest(raw)),
                    native_record: row.clone(),
                },
            );
        }
        Ok(())
    })?;
    ensure!(
        endpoints.len() == requested.len(),
        "requested coordinate step absent from native physical time journal"
    );
    Ok(endpoints)
}

/// Compact immutable index for repeated exact event-pair joins. The complete
/// journal is validated once; no per-KCC-record full journal scan is required.
pub struct VerifiedPhysicalTimeAxis {
    pub authority: PhysicalTimeJournalAuthority,
    times: Vec<f64>,
    dt: Vec<f64>,
    offsets: Vec<u64>,
    lengths: Vec<u32>,
    hashes: Vec<[u8; 32]>,
}
impl VerifiedPhysicalTimeAxis {
    pub fn load(authority: &PhysicalTimeJournalAuthority) -> Result<Self> {
        let mut axis = Self {
            authority: authority.clone(),
            times: vec![0.0],
            dt: Vec::new(),
            offsets: Vec::new(),
            lengths: Vec::new(),
            hashes: Vec::new(),
        };
        scan_native_physical_time_journal(authority, |row, _, offset, raw| {
            axis.times.push(row.cumulative_simulated_time_ps);
            axis.dt.push(row.applied_dt_ps);
            axis.offsets.push(offset);
            axis.lengths.push(
                u32::try_from(raw.len())
                    .context("physical time row exceeds byte-range representation")?,
            );
            axis.hashes.push(Sha256::digest(raw).into());
            Ok(())
        })?;
        ensure!(axis.times.len() > 1, "empty native physical time journal");
        Ok(axis)
    }
    pub fn cumulative_time_ps(&self, step: u64) -> Result<f64> {
        Ok(*self
            .times
            .get(usize::try_from(step)?)
            .context("exact event endpoint absent from native journal")?)
    }
    pub fn endpoint(&self, step: u64) -> Result<RecordedPhysicalTimeEndpoint> {
        ensure!(step > 0, "step zero is declared origin, not a journal row");
        let time = self.cumulative_time_ps(step)?;
        let i = usize::try_from(step - 1)?;
        Ok(RecordedPhysicalTimeEndpoint {
            source: self.authority.source.clone(),
            source_line_ordinal: step - 1,
            source_line_byte_offset: self.offsets[i],
            source_line_byte_length: self.lengths[i] as u64,
            source_line_sha256: self.hashes[i].iter().map(|b| format!("{b:02x}")).collect(),
            native_record: NativePhysicalTimeRecord {
                schema_kind: "prism_native_physical_time".into(),
                schema_version: 1,
                source_stream_id: self.authority.source_stream_id,
                allocation_generation: self.authority.allocation_generation,
                epoch_id: self.authority.epoch_id.clone(),
                step,
                applied_dt_ps: self.dt[i],
                cumulative_simulated_time_ps: time,
                origin: "native_physical_guard_post_integrator_pre_transition".into(),
                step_origin: "owning_ProtocolState.current_step".into(),
            },
        })
    }
}

fn scan_native_physical_time_journal(
    authority: &PhysicalTimeJournalAuthority,
    mut consume: impl FnMut(&NativePhysicalTimeRecord, u64, u64, &[u8]) -> Result<()>,
) -> Result<()> {
    ensure!(!authority.epoch_id.is_empty(), "empty native time epoch");
    authority.source.verify()?;
    let file = File::open(&authority.source.path)?;
    let mut reader = BufReader::new(file);
    let mut raw = Vec::new();
    let mut digest = Sha256::new();
    let mut count = 0u64;
    let mut offset = 0u64;
    let mut cumulative = 0.0;
    loop {
        raw.clear();
        let bytes = reader.read_until(b'\n', &mut raw)?;
        if bytes == 0 {
            break;
        }
        ensure!(
            raw.last() == Some(&b'\n'),
            "truncated native physical time journal row"
        );
        digest.update(&raw);
        let row: NativePhysicalTimeRecord = serde_json::from_slice(&raw)?;
        ensure!(
            row.schema_kind == "prism_native_physical_time"
                && row.schema_version == 1
                && row.origin == "native_physical_guard_post_integrator_pre_transition"
                && row.step_origin == "owning_ProtocolState.current_step",
            "native physical time schema/origin mismatch"
        );
        ensure!(
            row.source_stream_id == authority.source_stream_id
                && row.allocation_generation == authority.allocation_generation
                && row.epoch_id == authority.epoch_id,
            "native physical time allocation/epoch/stream mismatch"
        );
        count = count
            .checked_add(1)
            .context("physical time row count overflow")?;
        ensure!(
            row.step == count,
            "noncontiguous/duplicate native physical step"
        );
        ensure!(
            row.applied_dt_ps.is_finite()
                && row.applied_dt_ps > 0.0
                && (row.applied_dt_ps as f32) as f64 == row.applied_dt_ps,
            "native dt is not the exact positive f32 integrator value"
        );
        let previous = cumulative;
        cumulative += row.applied_dt_ps;
        ensure!(
            cumulative.is_finite()
                && cumulative > previous
                && row.cumulative_simulated_time_ps == cumulative,
            "native cumulative time differs from complete per-step dt sum"
        );
        consume(&row, count - 1, offset, &raw)?;
        offset = offset
            .checked_add(bytes as u64)
            .context("physical time offset overflow")?;
    }
    ensure!(
        format!("{:x}", digest.finalize()) == authority.source.sha256,
        "native physical time journal changed during decode"
    );
    let completion = &authority.completion;
    ensure!(
        completion.record_count.is_none_or(|n| n == count)
            && completion.first_step.is_none_or(|n| n == 1)
            && completion.last_step.is_none_or(|n| n == count)
            && completion
                .cumulative_simulated_time_ps
                .is_none_or(|t| t == cumulative)
            && completion.capacity_steps.is_none_or(|n| count <= n)
            && completion.time_origin_ps.is_none_or(|t| t == 0.0),
        "native physical time completion manifest differs from journal population/clock"
    );
    Ok(())
}

/// Select the owning arm's clock. An explicit but incomplete arm map fails;
/// only old rows without a native map may use their recorded shared clock.
pub(crate) fn timeline_arm_time(timeline: &Value, arm: &str) -> Result<f64> {
    ensure!(
        matches!(arm, "treatment" | "control"),
        "unknown physical time arm"
    );
    let exact = &timeline["arm_physical_time"];
    let time = if exact.is_null() || exact.as_object().is_some_and(|o| o.is_empty()) {
        ensure!(
            timeline["time_axis_scope"] != "exact_native_per_arm_step_sum",
            "declared exact time has no per-arm authority"
        );
        timeline["physical_time_ps"]
            .as_f64()
            .context("historical recorded physical time absent")?
    } else {
        let e: RecordedPhysicalTimeEndpoint = serde_json::from_value(exact[arm].clone())?;
        ensure!(
            e.native_record.step == u(&timeline["step"])?,
            "physical time endpoint step mismatch"
        );
        e.native_record.cumulative_simulated_time_ps
    };
    ensure!(
        time.is_finite() && time >= 0.0,
        "invalid recorded physical time"
    );
    Ok(time)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn temp_file(
        label: &str,
        rows: &[NativePhysicalTimeRecord],
    ) -> (PathBuf, PhysicalTimeJournalAuthority) {
        let root = std::env::temp_dir().join(format!(
            "prism_time_contract_{}_{}_{}",
            std::process::id(),
            label,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut w = File::create(&root).unwrap();
        for r in rows {
            line(&mut w, r).unwrap();
        }
        let authority = PhysicalTimeJournalAuthority {
            source: pin(&root).unwrap(),
            source_stream_id: 0,
            allocation_generation: 7,
            epoch_id: "transport_contract_fixture".into(),
            completion: PhysicalTimeCompletion::default(),
        };
        (root, authority)
    }
    fn records() -> Vec<NativePhysicalTimeRecord> {
        // Serialization/clock contract fixture only; no molecular observations.
        let mut cumulative = 0.0;
        [0.002f32, 0.004, 0.001]
            .into_iter()
            .enumerate()
            .map(|(i, dt)| {
                cumulative += dt as f64;
                NativePhysicalTimeRecord {
                    schema_kind: "prism_native_physical_time".into(),
                    schema_version: 1,
                    source_stream_id: 0,
                    allocation_generation: 7,
                    epoch_id: "transport_contract_fixture".into(),
                    step: i as u64 + 1,
                    applied_dt_ps: dt as f64,
                    cumulative_simulated_time_ps: cumulative,
                    origin: "native_physical_guard_post_integrator_pre_transition".into(),
                    step_origin: "owning_ProtocolState.current_step".into(),
                }
            })
            .collect()
    }
    #[test]
    fn native_time_preserves_variable_dt_and_rejects_corruption() {
        let rows = records();
        let requested = BTreeSet::from([1, 3]);
        let (path, authority) = temp_file("good", &rows);
        let out = load_physical_time_endpoints(&authority, &requested).unwrap();
        assert_eq!(
            out[&3].native_record.cumulative_simulated_time_ps,
            rows[2].cumulative_simulated_time_ps
        );
        assert_ne!(
            out[&3].native_record.cumulative_simulated_time_ps,
            3.0 * rows[2].applied_dt_ps
        );
        assert_eq!(out[&3].source_line_ordinal, 2);
        let axis = VerifiedPhysicalTimeAxis::load(&authority).unwrap();
        assert_eq!(
            serde_json::to_value(axis.endpoint(3).unwrap()).unwrap(),
            serde_json::to_value(&out[&3]).unwrap()
        );
        assert_eq!(axis.cumulative_time_ps(0).unwrap(), 0.0);
        assert!(axis.endpoint(0).is_err());
        assert!(axis.cumulative_time_ps(4).is_err());
        assert!(axis.cumulative_time_ps(1).unwrap() - axis.cumulative_time_ps(3).unwrap() < 0.0);

        let mut wrong_completion = authority.clone();
        wrong_completion.completion.record_count = Some(4);
        assert!(load_physical_time_endpoints(&wrong_completion, &requested).is_err());
        fs::remove_file(path).unwrap();
        for failure in 0..4 {
            let mut bad = rows.clone();
            match failure {
                0 => bad[1].step = 1,
                1 => bad[1].allocation_generation = 8,
                2 => bad[2].cumulative_simulated_time_ps = 3.0 * bad[2].applied_dt_ps,
                _ => {
                    bad.remove(1);
                }
            }
            let (path, a) = temp_file(&format!("bad{failure}"), &bad);
            assert!(load_physical_time_endpoints(&a, &requested).is_err());
            fs::remove_file(path).unwrap();
        }
    }
    #[test]
    fn paired_time_axes_remain_distinct_and_historical_absence_stays_historical() {
        let rows = records();
        let (path, a) = temp_file("arms", &rows);
        let out = load_physical_time_endpoints(&a, &BTreeSet::from([3])).unwrap();
        let treatment = out[&3].clone();
        let mut control = treatment.clone();
        control.native_record.cumulative_simulated_time_ps *= 2.0;
        let frame = json!({"step":3,"physical_time_ps":null,"time_axis_scope":"exact_native_per_arm_step_sum",
            "arm_physical_time":{"treatment":treatment,"control":control}});
        assert_ne!(
            timeline_arm_time(&frame, "treatment").unwrap(),
            timeline_arm_time(&frame, "control").unwrap()
        );
        let mut missing = frame.clone();
        missing["arm_physical_time"]
            .as_object_mut()
            .unwrap()
            .remove("control");
        assert!(timeline_arm_time(&missing, "control").is_err());
        assert_eq!(
            timeline_arm_time(&json!({"step":3,"physical_time_ps":0.006}), "control").unwrap(),
            0.006
        );
        fs::remove_file(path).unwrap();
    }
    #[test]
    #[ignore = "requires an actual completed native journal authority JSON"]
    fn actual_native_physical_time_journal_is_consumed_without_reconstruction() {
        let manifest = std::env::var("PRISM_NATIVE_TIME_AUTHORITY").unwrap();
        let authority: PhysicalTimeJournalAuthority =
            serde_json::from_reader(File::open(manifest).unwrap()).unwrap();
        let end: u64 = std::env::var("PRISM_NATIVE_TIME_LAST_STEP")
            .unwrap()
            .parse()
            .unwrap();
        let endpoints =
            load_physical_time_endpoints(&authority, &BTreeSet::from([1, end])).unwrap();
        assert_eq!(endpoints[&end].native_record.step, end);
        println!(
            "{}",
            serde_json::to_string(
                &json!({"native_journal":authority,"endpoints":endpoints,"MD_executed":false})
            )
            .unwrap()
        );
    }
}

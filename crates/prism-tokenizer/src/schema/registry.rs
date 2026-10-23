//! §6 — versioned `(ArtifactFamily, field) → FieldSpec` registry with factored-namespace
//! assignment. Field-slots are assigned in table order per family, so the 13-bit namespace
//! (family_slot, field_slot) is stable and same-value-different-family ⇒ different token (INV-2).

use std::collections::HashMap;

use crate::schema::families::field_table;
use crate::types::{pack_namespace, ArtifactFamily, FieldSpec};

pub struct FieldRegistry {
    specs: Vec<FieldSpec>,
    by_key: HashMap<(u8, &'static str), usize>,
    slots: HashMap<(u8, &'static str), u8>,
}

impl FieldRegistry {
    /// Build the registry from the §7 field table. Panics if any family declares more than 64
    /// fields (the field-slot is 6 bits) — a build-time guard, never a runtime surprise.
    pub fn build() -> Self {
        let specs = field_table();
        let mut next_slot: HashMap<u8, u8> = HashMap::new();
        let mut by_key = HashMap::new();
        let mut slots = HashMap::new();
        for (i, s) in specs.iter().enumerate() {
            let fam = s.family.family_slot();
            let slot = *next_slot.get(&fam).unwrap_or(&0);
            assert!(
                slot < 64,
                "family {:?} exceeds 64 fields (field-slot overflow)",
                s.family
            );
            next_slot.insert(fam, slot + 1);
            assert!(
                by_key.insert((fam, s.field), i).is_none(),
                "duplicate (family, field): {:?}/{}",
                s.family,
                s.field
            );
            slots.insert((fam, s.field), slot);
        }
        Self {
            specs,
            by_key,
            slots,
        }
    }

    pub fn lookup(&self, family: ArtifactFamily, field: &str) -> Option<&FieldSpec> {
        self.by_key
            .get(&(family.family_slot(), field))
            .map(|&i| &self.specs[i])
    }

    /// The 13-bit token namespace for `(family, field)`, or None if the field is unregistered
    /// (caller routes unknown fields to `Opaque` + sidecar, §13).
    pub fn namespace(&self, family: ArtifactFamily, field: &str) -> Option<u16> {
        self.slots
            .get(&(family.family_slot(), field))
            .map(|&slot| pack_namespace(family.family_slot(), slot))
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = &FieldSpec> {
        self.specs.iter()
    }
}

impl Default for FieldRegistry {
    fn default() -> Self {
        Self::build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FieldGeometry as G;
    use std::collections::HashSet;

    #[test]
    fn builds_and_is_nonempty() {
        let r = FieldRegistry::build();
        assert!(r.len() >= 40, "expected the full §7 table, got {}", r.len());
    }

    #[test]
    fn spike_source_is_enum_of_seven() {
        // 0=NONE 1=UV 2=LIF 3=EFP 4=LADD 5=COFIRE 6=VORONOI_LIF.
        let r = FieldRegistry::build();
        let s = r
            .lookup(ArtifactFamily::SpikeEvent, "spike_source")
            .unwrap();
        match s.geometry {
            G::Enum { cardinality } => assert_eq!(cardinality, 7),
            ref g => panic!("expected Enum, got {g:?}"),
        }
    }

    #[test]
    fn gflownet_reward_is_not_canonical_safe() {
        // INV-3: generative reward is firewalled (verified-gated), never canonical.
        let r = FieldRegistry::build();
        let s = r
            .lookup(ArtifactFamily::GflownetTrajectory, "reward")
            .unwrap();
        assert!(!s.canonical_safe);
    }

    #[test]
    fn receipt_status_is_closed_three_way() {
        // INV-5: PASS / PARTIAL / FAIL only — no inferred-PASS token.
        let r = FieldRegistry::build();
        let s = r.lookup(ArtifactFamily::ReceiptStatus, "status").unwrap();
        match s.geometry {
            G::Enum { cardinality } => assert_eq!(cardinality, 3),
            ref g => panic!("expected Enum(3), got {g:?}"),
        }
    }

    #[test]
    fn namespaces_are_unique_across_the_whole_table() {
        // INV-2: every (family, field) maps to a distinct namespace.
        let r = FieldRegistry::build();
        let mut seen = HashSet::new();
        for s in r.iter() {
            let ns = r.namespace(s.family, s.field).unwrap();
            assert!(
                seen.insert(ns),
                "namespace collision at {:?}/{}",
                s.family,
                s.field
            );
        }
        assert_eq!(seen.len(), r.len());
    }

    #[test]
    fn aromatic_residue_id_is_sentinel_wrapped() {
        let r = FieldRegistry::build();
        let s = r
            .lookup(ArtifactFamily::SpikeEvent, "aromatic_residue_id")
            .unwrap();
        match &s.geometry {
            G::Sentinel { value, inner } => {
                assert_eq!(*value, -1.0);
                assert!(matches!(**inner, G::JoinKey));
            }
            g => panic!("expected Sentinel, got {g:?}"),
        }
    }

    #[test]
    fn unknown_field_has_no_namespace() {
        let r = FieldRegistry::build();
        assert!(r
            .namespace(ArtifactFamily::SpikeEvent, "no_such_field")
            .is_none());
    }

    #[test]
    fn kcc_coherence_replaced_by_real_residue_fields() {
        // The fictional `coherence` field (present in NO real artifact) is gone; the 16 real
        // residues[] descriptor fields from kcc_visualization.json are registered instead.
        use crate::schema::families::AA_NAMES;
        let r = FieldRegistry::build();
        let f = ArtifactFamily::KccDescriptor;
        assert!(
            r.lookup(f, "coherence").is_none(),
            "fictional `coherence` must be removed"
        );
        // every real field is present
        for field in [
            "residue_id",
            "residue_name",
            "ca_position",
            "kcc_score",
            "direction_score",
            "motion_efficiency",
            "burst_motion",
            "causal_lag",
            "lag_corr_peak",
            "local_cov",
            "sum_motion",
            "active_causal_steps",
            "total_steps",
            "net_dx",
            "net_dy",
            "net_dz",
        ] {
            assert!(
                r.lookup(f, field).is_some(),
                "real KCC field `{field}` missing"
            );
        }
        // ca_position is a 3-D lattice, not three scalars; residue_name is the AA enum (exact codes).
        assert!(matches!(
            r.lookup(f, "ca_position").unwrap().geometry,
            G::LatticeVq3d { .. }
        ));
        match r.lookup(f, "residue_name").unwrap().geometry {
            G::Enum { cardinality } => assert_eq!(cardinality as usize, AA_NAMES.len()),
            ref g => panic!("residue_name should be Enum, got {g:?}"),
        }
        assert!(matches!(
            r.lookup(f, "residue_id").unwrap().geometry,
            G::JoinKey
        ));
    }

    #[test]
    fn kcc_descriptor_native_16_fields_registered() {
        // RAW PRKCC001 kcc_v2full.bin per-residue carrier (discriminant 43): 2 JoinKeys + 14 decoded
        // columns. DISTINCT namespace from KccDescriptor (same-named fields get different tokens).
        let r = FieldRegistry::build();
        let f = ArtifactFamily::KccDescriptorNative;
        for field in [
            "residue_id",
            "stream_id",
            "temporal_corr",
            "direction_score",
            "motion_efficiency",
            "burst_motion",
            "phase_shift",
            "causal_lag",
            "lag_corr_peak",
            "local_cov",
            "net_dx",
            "net_dy",
            "net_dz",
            "sum_m",
            "residue_count",
            "active_causal",
        ] {
            assert!(
                r.lookup(f, field).is_some(),
                "KccDescriptorNative field `{field}` missing"
            );
        }
        // residue_id/stream_id are JoinKeys.
        assert!(matches!(
            r.lookup(f, "residue_id").unwrap().geometry,
            G::JoinKey
        ));
        // REVIEW FIX: burst_motion + active_causal have measured min==0 ⇒ linear (0 representable),
        // NOT log-magnitude (whose floor>0 would quantize a genuine zero to a nonzero bin).
        assert!(matches!(
            r.lookup(f, "burst_motion").unwrap().geometry,
            G::LinearBounded { lo, .. } if lo == 0.0
        ));
        assert!(matches!(
            r.lookup(f, "active_causal").unwrap().geometry,
            G::LinearBounded { lo, .. } if lo == 0.0
        ));
        // REVIEW FIX: net_dx/dy/dz KEPT at the existing KccDescriptor lin(-10,10) range.
        for field in ["net_dx", "net_dy", "net_dz"] {
            assert!(matches!(
                r.lookup(f, field).unwrap().geometry,
                G::LinearBounded { lo, hi, .. } if lo == -10.0 && hi == 10.0
            ));
        }
        // DISTINCT namespace from KccDescriptor for a same-named field (INV-2).
        assert_ne!(
            r.namespace(ArtifactFamily::KccDescriptorNative, "net_dx"),
            r.namespace(ArtifactFamily::KccDescriptor, "net_dx"),
        );
    }

    #[test]
    fn teacher_fields_use_exact_parquet_column_names() {
        // The registry field names must be the EXACT parquet columns (corpus shard sha 2146c1ee…03a5cc):
        // the two aliases are renamed, and the three additional real columns are registered.
        let r = FieldRegistry::build();
        let f = ArtifactFamily::TeacherStudentConsensus;
        // renamed to the real column names
        assert!(r.lookup(f, "phase_manifold_score_mean").is_some());
        assert!(r.lookup(f, "hard_negative_mask").is_some());
        // the old (non-parquet) aliases are gone
        assert!(
            r.lookup(f, "phase_manifold_score").is_none(),
            "stale alias must be removed"
        );
        assert!(
            r.lookup(f, "hard_negative").is_none(),
            "stale alias must be removed"
        );
        // additional real columns now routed
        for field in ["phase_manifold_score_std", "student_weight", "train_mask"] {
            assert!(
                r.lookup(f, field).is_some(),
                "real Teacher column `{field}` missing"
            );
        }
        // is_bimodal is a real ensemble_aggregates.parquet column → kept as-is.
        assert!(r.lookup(f, "is_bimodal").is_some());
    }

    #[test]
    fn v11_teacher_bridge_fields_appended_without_shifting_v10_slots() {
        // Directive v6.1 Track B: the per-pair bridge labels are APPENDED LAST. The 11 v10
        // per-residue teacher fields MUST keep field-slots 0..10 (INV-2 namespace stability), and
        // the new fields MUST occupy slots >= 11. No existing field altered.
        let r = FieldRegistry::build();
        let f = ArtifactFamily::TeacherStudentConsensus;
        let fam = f.family_slot();
        // v10 fields keep their original slot order (0..10).
        let v10 = [
            "cryptic_likelihood_label",
            "phase_manifold_score_mean",
            "phase_manifold_score_std",
            "rhat",
            "ess",
            "label_confidence",
            "is_bimodal",
            "hard_negative_mask",
            "train_mask",
            "student_weight",
            "residue_id",
        ];
        for (i, field) in v10.iter().enumerate() {
            let ns = r.namespace(f, field).expect("v10 field present");
            // low 6 bits of the namespace are the field-slot.
            assert_eq!((ns & 0x3F) as usize, i, "v10 `{field}` slot drifted");
        }
        // the NEW bridge fields are present and all sit at slots >= 11.
        for field in [
            "pair_label",
            "pair_weight",
            "abstention_target",
            "teacher_target",
            "teacher_loss_weight",
            "distillation_loss_weight",
            "uncertainty_target",
            "positive_target",
            "negative_target",
            "model_uncertainty",
            "model_coverage",
            "evidence_quality_score",
            "integrated_score",
            "phase_coherence_score",
            "gflownet_reward",
            "gflownet_reward_fusion_target",
            "fusion_kcc_coherence",
            "fusion_kinetic_durability",
            "fusion_hydration_gate_behavior",
            "fusion_uncertainty",
            "fusion_evidence_coverage_score",
            "fusion_force_stress",
            "fusion_warp_strain",
            "fusion_receptor_recovery",
            "fusion_raw_spike_activity",
            "fusion_raw_spike_entropy",
            "fusion_productive_activation_score",
            "fusion_quiet_lock_risk",
            "fusion_variant_rescue_signal",
            "fusion_receptor_state_coherence",
            "fusion_fiber_transport_coherence",
            "fusion_chronology_alignment",
            "delta_functional_score",
            "delta_survival_auc",
            "delta_temporal_score",
            "delta_thermodynamic_score",
            "delta_water_score_v2",
            "evidence_role",
            "phase_evidence_tier",
            "water_class_v2",
            "negative_family",
            "contradiction_flag",
            "ood_or_missing_family_flag",
            "do_not_discard",
            "pair_id",
            "molecule_id",
        ] {
            let ns = r
                .namespace(f, field)
                .unwrap_or_else(|| panic!("new field `{field}` missing"));
            assert!((ns & 0x3F) >= 11, "new `{field}` collided with a v10 slot");
            assert_eq!((ns >> 6) as u8, fam, "new `{field}` wrong family slot");
        }
        // teacher-side labels are canonical_safe=true (label evidence, NOT a generative firewall).
        assert!(r.lookup(f, "teacher_loss_weight").unwrap().canonical_safe);
        assert!(r.lookup(f, "pair_label").unwrap().canonical_safe);
    }

    #[test]
    fn native_epistemic_sequence_fields_are_registered() {
        let r = FieldRegistry::build();
        let cases = [
            (ArtifactFamily::EpistemicUncertainty, "sigma", 72, 0),
            (ArtifactFamily::EpistemicUncertainty, "variance", 72, 1),
            (ArtifactFamily::TeacherConsensus, "entropy", 73, 0),
            (ArtifactFamily::TeacherConsensus, "disagreement", 73, 1),
            (ArtifactFamily::TeacherConsensus, "vote_state", 73, 2),
            (ArtifactFamily::ThermodynamicAction, "tsh_state", 74, 0),
            (ArtifactFamily::ThermodynamicAction, "dt_fs", 74, 1),
        ];
        for (family, field, fam_slot, field_slot) in cases {
            let spec = r.lookup(family, field).expect("native field registered");
            assert!(spec.canonical_safe, "{family:?}/{field} canonical-safe");
            let ns = r.namespace(family, field).expect("native namespace");
            assert_eq!((ns >> 6) as u8, fam_slot, "{family:?}/{field} family slot");
            assert_eq!(
                (ns & 0x3F) as u8,
                field_slot,
                "{family:?}/{field} field slot"
            );
        }
    }

    #[test]
    fn inv7_every_field_has_a_coherent_distortion_contract() {
        use crate::types::{DecisionSpec, DistortionBound};
        let r = FieldRegistry::build();
        for s in r.iter() {
            // Threshold fields: non-empty points, all strictly inside a well-formed region.
            if let DecisionSpec::Threshold { points, region } = s.decision_relevance {
                assert!(
                    !points.is_empty(),
                    "{}/{} empty threshold points",
                    s.family.family_slot(),
                    s.field
                );
                assert!(region.0 < region.1, "{} bad region", s.field);
                for &p in points {
                    assert!(
                        region.0 <= p && p <= region.1,
                        "{} point {p} outside region",
                        s.field
                    );
                }
            }
            // Categorical decisions only on exact-class geometries.
            if matches!(s.decision_relevance, DecisionSpec::Categorical) {
                assert!(
                    matches!(s.geometry, G::Enum { .. } | G::SpectralLine { .. }),
                    "{} Categorical but geometry isn't enum/spectral",
                    s.field
                );
            }
            // Exact bound only on geometries that can round-trip exactly.
            if matches!(s.max_distortion, DistortionBound::Exact) {
                assert!(
                    matches!(
                        s.geometry,
                        G::Enum { .. }
                            | G::SpectralLine { .. }
                            | G::Boolean
                            | G::JoinKey
                            | G::Sentinel { .. }
                            | G::SmilesGrammar
                            | G::OpcodeIsa
                            | G::Opaque
                    ),
                    "{} Exact bound on a lossy geometry",
                    s.field
                );
            }
        }
    }
}

use prism_forge::scoring::{
    score_molecule, score_molecule_with_continuity, score_molecule_with_continuity_and_pose,
    ContinuityVoxelProfile, LoadedContinuityMaps, LoadedSignalGrid, VoxelField,
};
use std::collections::{HashMap, HashSet};

fn activated_voxel() -> VoxelField {
    VoxelField {
        complement: 3.0,
        clash: 0.0,
        raw_clash: 0.0,
        cryptic_bonus: 0.0,
        stable_occupied: false,
        thermally_destabilized: false,
        thermally_activated: true,
        thermally_released: false,
        coherence_score: 1.0,
        coherence_factor: 1.0,
        coherence_missing: false,
        primary_residue_idx: None,
        cold_mean: 0.0,
        warm_mean: 3.0,
        delta: 3.0,
        consensus_complement_bonus: 0.5,
        on_activation_pathway: true,
    }
}

#[test]
fn continuity_penalties_reduce_reward_and_emit_columns() {
    let grid = LoadedSignalGrid {
        field: HashMap::from([(0_u64, activated_voxel())]),
        origin: [0.0, 0.0, 0.0],
        spacing: 1.0,
        dims: [10, 10, 10],
        condition_id: "test".to_owned(),
    };
    let continuity = LoadedContinuityMaps {
        by_voxel: HashMap::from([(
            0_u64,
            ContinuityVoxelProfile {
                nma_disruption_penalty: 2.0,
                hydration_blockade_penalty: 0.25,
                thermodynamic_trap_penalty: 0.5,
            },
        )]),
        global_nma_penalty: 0.0,
        global_hydration_penalty: 0.0,
        global_thermodynamic_penalty: 0.0,
        provenance: "L3_DERIVED_TEST".to_owned(),
    };

    let base = score_molecule(&[(0.1, 0.1, 0.1)], &grid, &HashSet::new(), &HashMap::new());
    let scored = score_molecule_with_continuity(
        &[(0.1, 0.1, 0.1)],
        &grid,
        &HashSet::new(),
        &HashMap::new(),
        &continuity,
    );

    assert!(scored.reward < base.reward);
    assert_eq!(scored.nma_disruption_penalty, 2.0);
    assert_eq!(scored.hydration_blockade_penalty, 0.25);
    assert_eq!(scored.thermodynamic_trap_penalty, 0.5);
    assert!(scored.continuity_admissibility);
    assert_eq!(scored.continuity_provenance, "L3_DERIVED_TEST");
}

#[test]
fn continuity_global_fallback_is_runtime_scored_not_python_only() {
    let grid = LoadedSignalGrid {
        field: HashMap::from([(0_u64, activated_voxel())]),
        origin: [0.0, 0.0, 0.0],
        spacing: 1.0,
        dims: [10, 10, 10],
        condition_id: "test".to_owned(),
    };
    let continuity = LoadedContinuityMaps {
        by_voxel: HashMap::new(),
        global_nma_penalty: 1.0,
        global_hydration_penalty: 0.1,
        global_thermodynamic_penalty: 0.9,
        provenance: "L3_DERIVED_GLOBAL_FALLBACK".to_owned(),
    };

    let scored = score_molecule_with_continuity(
        &[(0.1, 0.1, 0.1)],
        &grid,
        &HashSet::new(),
        &HashMap::new(),
        &continuity,
    );

    assert_eq!(scored.nma_disruption_penalty, 1.0);
    assert_eq!(scored.hydration_blockade_penalty, 0.1);
    assert_eq!(scored.thermodynamic_trap_penalty, 0.9);
    assert!(!scored.continuity_admissibility);
    assert!(scored.continuity_reward_v1 > 0.0);
}

#[test]
fn continuity_pose_penalty_is_runtime_scored() {
    let grid = LoadedSignalGrid {
        field: HashMap::from([(0_u64, activated_voxel())]),
        origin: [0.0, 0.0, 0.0],
        spacing: 1.0,
        dims: [10, 10, 10],
        condition_id: "test".to_owned(),
    };
    let continuity = LoadedContinuityMaps {
        by_voxel: HashMap::new(),
        global_nma_penalty: 0.0,
        global_hydration_penalty: 0.0,
        global_thermodynamic_penalty: 0.0,
        provenance: "L3_DERIVED_POSE_TEST".to_owned(),
    };

    let without_pose = score_molecule_with_continuity(
        &[(0.1, 0.1, 0.1)],
        &grid,
        &HashSet::new(),
        &HashMap::new(),
        &continuity,
    );
    let with_pose = score_molecule_with_continuity_and_pose(
        &[(0.1, 0.1, 0.1)],
        &grid,
        &HashSet::new(),
        &HashMap::new(),
        &continuity,
        1.5,
    );

    assert_eq!(with_pose.u_pose, 1.5);
    assert!(with_pose.continuity_reward_v1 < without_pose.continuity_reward_v1);
}

//! Exact native Ghost/Path-B connectivity association for recorded physical
//! measurements. Reuses the existing accepted-edge constructor; chemical lane
//! estimation and periodic-solvent contracts are not invoked or redefined.
use super::*;
use crate::ghost_phase_lattice::{load_verified_ghost_lattice_edges,
    load_verified_ghost_segments_from_dir, StreamProtocolPhaseBinding};
use crate::pathb_chemical_state::{build_exact_pathb_connectivity,NativePathBLatticeObservation};
use crate::pathb_evidence_emitter::adapt_verified_ghost_load_for_pathb;

pub(super) struct NativeGraphAssociation {
    arm:String,
    lineages:Vec<GhostByteLineage>,
    by_lineage:BTreeMap<String,usize>,
    component_by_node:Vec<usize>,
    components:Vec<Value>,
    observation_hashes:Vec<String>,
    node_source:MeasurementSourcePin,
    graph_reference:MeasurementSourcePin,
    source_pins:Vec<MeasurementSourcePin>,
    matched:BTreeSet<usize>,
    unmatched_observations:u64,
    edge_count:u64,
    connectivity_sha256:String,
}

fn source_in_artifact(a:&MechanisticMeasurementArtifact,value:&Value)->Result<MeasurementSourcePin> {
    let p=source(value)?;
    ensure!(a.source_pins.contains(&p),"native graph source outside authenticated measurement population: {}",p.path.display());
    p.verify()?;Ok(p)
}

impl NativeGraphAssociation {
    pub(super) fn load(a:&MechanisticMeasurementArtifact,arm:&str,out:&Path)->Result<Self> {
        let manifest_pin=a.source_pins.iter().find(|p|p.sha256==a.manifest_sha256)
            .context("native graph measurement manifest absent")?.clone();
        let manifest:Value=read_pin(&manifest_pin)?;
        let input=&manifest["mechanistic_sources"]["arms"][arm];
        let lattice_pin=source_in_artifact(a,&input["native_lattice"])?;
        let edge_pin=source_in_artifact(a,&input["ghost_accepted_edges"])?;
        let lineage_pin=source_in_artifact(a,&input["ghost_lineage"])?;
        let segment_pin=source_in_artifact(a,&input["ghost_segment_manifest"])?;
        let phase_pin=source_in_artifact(a,&input["ghost_phase_binding"])?;
        let lattice:Value=read_pin(&lattice_pin)?;
        ensure!(lattice["run_id"]==a.arm_run_ids[arm]
            && lattice["run_id"]==input["ghost_producer_run_id"]
            && lattice["ghost_record_byte_lineage_artifact"]["sha256"]==lineage_pin.sha256
            && lattice["ghost_lattice_accepted_edge_artifact"]["sha256"]==edge_pin.sha256
            && lattice["ghost_lattice_accepted_edge_artifact"]["node_lineage_artifact_sha256"]==lineage_pin.sha256,
            "native graph artifacts do not share exact recorded source identity");
        let binding:StreamProtocolPhaseBinding=read_pin(&phase_pin)?;binding.verify()?;
        ensure!(binding.stream_id==a.stream_id,"native graph arm stream differs");
        let suffix=format!("_stream{:02}_ghost_phase_binding.json",a.stream_id);
        let stem=phase_pin.path.file_name().and_then(|p|p.to_str()).and_then(|p|p.strip_suffix(suffix.as_str()))
            .context("native graph phase path differs from owning stream")?;
        let root=segment_pin.path.parent().and_then(Path::parent).context("native graph segmented root absent")?;
        let load=load_verified_ghost_segments_from_dir(root,stem,u(&lattice["config"]["step_bucket_size"])?)?;
        ensure!(load.overflow_conservation_pass && load.equality_pass,
            "native graph raw population conservation failed");
        let inputs=adapt_verified_ghost_load_for_pathb(&load,&lattice_pin.sha256)?;
        ensure!(u(&lattice["ghost_record_byte_lineage_artifact"]["row_count"])?==inputs.lineages.len() as u64,
            "native graph loaded node count differs from persisted lineage population");
        for lineage in &inputs.lineages {
            ensure!(lineage.run_id==a.arm_run_ids[arm] && lineage.topology_sha256==a.topology_sha256,
                "native graph occurrence belongs to a different run/topology");
            if lineage.key.stream_id==a.stream_id {
                ensure!(lineage.segment_manifest_sha256==segment_pin.sha256
                    && lineage.phase_schedule_digest_sha256==binding.phase_schedule_digest_sha256
                    && lineage.protocol_state_hash==binding.protocol_state_hash,
                    "native graph stream authority differs from recorded physical arm");
            }
        }
        let accepted=load_verified_ghost_lattice_edges(&edge_pin.path,&load.nodes,&edge_pin.sha256,
            u(&lattice["ghost_lattice_accepted_edge_artifact"]["bytes"])?,&lineage_pin.sha256)?;
        ensure!(accepted.run_id==a.arm_run_ids[arm],"accepted-edge run differs from recorded arm");
        let authority=build_exact_pathb_connectivity(&load.nodes,&inputs.lineages,&accepted)?;
        ensure!(authority.selected_node_count==load.nodes.len() as u64
            && authority.selected_edge_count==accepted.produced_edges,
            "whole native graph association must retain every node and accepted edge");
        // The native constructor verifies all edge predicates, populations,
        // component membership and canonical hashes. Preserve original edges by
        // their full source pin, rather than writing another multi-GiB body.
        let mut by_lineage=BTreeMap::new();
        for (index,lineage) in inputs.lineages.iter().enumerate() {
            ensure!(by_lineage.insert(lineage.lineage_sha256()?,index).is_none(),
                "duplicate native graph Ghost occurrence");
        }
        let mut component_by_node=vec![usize::MAX;inputs.lineages.len()];
        let mut components=Vec::with_capacity(authority.components.len());
        for (ordinal,component) in authority.components.iter().enumerate() {
            for lineage in &component.node_ghost_lineage_sha256 {
                let index=*by_lineage.get(lineage).context("native component node outside loaded population")?;
                ensure!(component_by_node[index]==usize::MAX,"native component duplicates node");
                component_by_node[index]=ordinal;
            }
            components.push(json!({"component_ordinal":ordinal,"component_sha256":component.component_sha256,
                "node_count":component.node_ghost_lineage_sha256.len(),"accepted_edge_count":component.edge_sha256.len(),
                "singleton":component.node_ghost_lineage_sha256.len()==1 && component.edge_sha256.is_empty()}));
        }
        ensure!(component_by_node.iter().all(|v|*v!=usize::MAX),"native graph lost singleton/component node");
        let edge_count=authority.selected_edge_count;
        let connectivity_sha256=authority.connectivity_sha256.clone();
        let graph_path=out.join(format!("{arm}_native_pathb_connectivity_reference.json"));
        write_json(&graph_path,&json!({
            "schema_kind":"prism.native_pathb_connectivity_source_reference.v1",
            "native_connectivity_sha256":connectivity_sha256,
            "native_connectivity_schema_kind":authority.schema_kind,"native_connectivity_schema_version":authority.schema_version,
            "native_constructor":"prism_nhs::pathb_chemical_state::build_exact_pathb_connectivity",
            "measurement_source_manifest":manifest_pin,"native_lattice":lattice_pin,
            "original_accepted_edge_population":edge_pin,"original_node_lineage_population":lineage_pin,
            "segment_manifest":segment_pin,"phase_binding":phase_pin,
            "native_source_edge_artifact_sha256":authority.source_edge_artifact_sha256,
            "native_source_edge_payload_sha256":authority.source_edge_payload_sha256,
            "selected_node_count":authority.selected_node_count,"selected_edge_count":edge_count,
            "source_produced_edges":authority.source_produced_edges,"components":components,
            "edge_values_location":"complete original authenticated accepted-edge artifact; no second edge population created",
            "reconstruction":"reload verified native segments and original accepted edges, adapt exact Ghost lineages, call named native constructor with all loaded nodes",
            "all_original_accepted_edges_retained":true,"thresholds_changed":false,"chemical_lanes_created":false,
            "markov_transition_probabilities_inferred":false
        }))?;
        let graph_reference=pin(&graph_path)?;
        drop(authority);drop(accepted);
        let node_path=out.join(format!("{arm}_native_pathb_lattice_observations.jsonl.zst"));
        let mut writer=zstd::stream::write::Encoder::new(BufWriter::new(File::create(&node_path)?),1)?;
        let mut observation_hashes=Vec::with_capacity(load.nodes.len());
        for (ordinal,(node,lineage)) in load.nodes.iter().zip(&inputs.lineages).enumerate() {
            let observation=NativePathBLatticeObservation::from_native_node(node,lineage)?;
            observation_hashes.push(observation.observation_sha256.clone());
            let component=component_by_node[ordinal];
            line(&mut writer,&json!({"schema_kind":"prism.native_pathb_lattice_measurement_node.v1",
                "node_ordinal":ordinal,"native_observation":observation,
                "native_component":components[component],"native_connectivity":graph_reference}))?;
        }
        writer.finish()?.flush()?;
        let node_source=pin(&node_path)?;
        Ok(Self{arm:arm.to_string(),lineages:inputs.lineages,by_lineage,component_by_node,
            components,observation_hashes,node_source,graph_reference,
            source_pins:vec![manifest_pin,lattice_pin,edge_pin,lineage_pin,segment_pin,phase_pin],
            matched:BTreeSet::new(),unmatched_observations:0,edge_count,connectivity_sha256})
    }

    pub(super) fn associate(&mut self,reference:&RecordedFiberReference,fiber:&RecordedMeasurementFiber)->Result<Value> {
        ensure!(fiber.body.arm==self.arm,"native graph association crossed recorded arms");
        let Some(&index)=self.by_lineage.get(&reference.ghost_lineage_sha256) else {
            self.unmatched_observations+=1;
            return Ok(json!({"status":"UNMATCHED_OBSERVATION_RETAINED","native_connectivity":self.graph_reference}));
        };
        fiber.resolve_ghost(&self.lineages[index],&fiber.body.resolved_physical_phase)?;
        ensure!(self.matched.insert(index),"recorded population maps twice to one native graph occurrence");
        let component=self.component_by_node[index];
        Ok(json!({"status":"EXACT_NATIVE_GHOST_ASSOCIATION","native_connectivity":self.graph_reference,
            "native_observation_source":self.node_source,"native_observation_line_ordinal":index,
            "native_observation_sha256":self.observation_hashes[index],"native_component":self.components[component],
            "native_lattice_node_index":self.lineages[index].lattice_node_index,
            "ghost_lineage_sha256":reference.ghost_lineage_sha256,"recorded_fiber_sha256":fiber.fiber_sha256,
            "completed_step":fiber.body.completed_step,"physical_time_ps":fiber.body.physical_time_ps,
            "resolved_physical_phase":fiber.body.resolved_physical_phase}))
    }

    pub(super) fn finish(self,out:&Path)->Result<Value> {
        let unmatched_path=out.join(format!("{}_native_pathb_nodes_without_recorded_fiber.jsonl.zst",self.arm));
        let mut writer=zstd::stream::write::Encoder::new(BufWriter::new(File::create(&unmatched_path)?),1)?;
        let mut unmatched=0u64;
        for (index,lineage) in self.lineages.iter().enumerate() {
            if self.matched.contains(&index){continue;}
            unmatched+=1;
            line(&mut writer,&json!({"status":"UNMATCHED_NATIVE_NODE_RETAINED","ghost":lineage,
                "native_observation_source":self.node_source,"native_observation_line_ordinal":index,
                "native_observation_sha256":self.observation_hashes[index],
                "native_component":self.components[self.component_by_node[index]],"native_connectivity":self.graph_reference}))?;
        }
        writer.finish()?.flush()?;
        for source in &self.source_pins {source.verify()?;}
        ensure!(self.matched.len() as u64+unmatched==self.lineages.len() as u64,
            "native observation/graph population association lost a node");
        Ok(json!({"schema_kind":"prism.native_recorded_pathb_graph_association.v1","status":"IMPLEMENTATION_VERIFIED",
            "arm":self.arm,"native_connectivity_sha256":self.connectivity_sha256,
            "native_connectivity":self.graph_reference,"native_lattice_observations":self.node_source,
            "native_nodes_without_recorded_fiber":pin(&unmatched_path)?,"native_nodes":self.lineages.len(),
            "accepted_edges":self.edge_count,"components":self.components.len(),
            "singleton_components":self.components.iter().filter(|c|c["singleton"]==true).count(),
            "exact_recorded_fiber_matches":self.matched.len(),"unmatched_native_nodes_retained":unmatched,
            "unmatched_recorded_observations_retained":self.unmatched_observations,
            "all_native_edges_source_preserved":true,"thresholds_changed":false,"chemical_lanes_created":false,
            "markov_or_free_energy_estimates_created":false,"native_tensor_fields_collapsed":false,"MD_executed":false}))
    }
}

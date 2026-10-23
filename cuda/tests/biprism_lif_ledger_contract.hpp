#pragma once
#include <cstring>

// Actual candidate-packet readback checks for an isolated, initially empty
// observation window. This is not the production Arrow/PathB packet contract.
// Include the candidate kernel and biprism_host_contract.hpp first.
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace biprism_qc {
struct LifReadbackReport {
    ContractReport contract;
    long double max_expanded_ledger_error = 0;
    long double max_single_float_ledger_error = 0;
    long double max_force_sum_change = 0;
    uint64_t total_multiplicity = 0;
};

inline int64_t signal_cell_host(const float* position, const SignalGridSpecFfi& grid) {
    uint64_t cell[3];
    for (unsigned k=0;k<3;++k) {
        const float x=(position[k]-grid.origin_xyz[k])*grid.inv_voxel_spacing;
        if (!std::isfinite(x) || x<0 || x>=float(grid.dim_xyz[k])) return -1;
        cell[k]=uint64_t(std::floor(x));
    }
    const uint64_t index=(cell[2]*grid.dim_xyz[1]+cell[1])*grid.dim_xyz[0]+cell[0];
    return index<grid.total_voxels ? int64_t(index) : -1;
}

inline LifReadbackReport validate_lif_window(
    const std::vector<GpuSpikeEventCanonical96>& events, uint32_t queue_count,
    const std::vector<float>& voltage,
    const std::vector<ContactShellSimplicialStateFfi>& tiles,
    const std::vector<BiprismConvergenceRecordFfi>& convergence,
    const std::vector<int32_t>& atom_residues,
    const std::vector<float3>& coordinates,
    const std::vector<float3>& forces_before, const std::vector<float3>& forces_after,
    const SignalGridSpecFfi& grid,
    const std::vector<int32_t>& hits, const std::vector<int32_t>& coupled,
    const std::vector<int32_t>& primary, const std::vector<int32_t>& primary_count,
    uint32_t expected_step, float expected_threshold,
    bool require_requested_absolute_bounds
) {
    LifReadbackReport result;
    auto& report=result.contract;
    const size_t n=atom_residues.size();
    uint64_t cells=uint64_t(grid.dim_xyz[0])*grid.dim_xyz[1];
    const bool grid_product_safe=grid.dim_xyz[2]!=0 && cells<=uint64_t(INT32_MAX)/grid.dim_xyz[2];
    if (grid_product_safe) cells*=grid.dim_xyz[2];
    if (!n || tiles.size()!=8 || convergence.size()!=8 || voltage.size()!=8*TILE_TOTAL_VERTICES ||
        coordinates.size()!=8*n || forces_before.size()!=8*n || forces_after.size()!=8*n ||
        !grid_product_safe || !cells || cells!=grid.total_voxels ||
        !std::isfinite(grid.inv_voxel_spacing) || grid.inv_voxel_spacing<=0 ||
        !std::isfinite(grid.origin_xyz[0]) || !std::isfinite(grid.origin_xyz[1]) || !std::isfinite(grid.origin_xyz[2]) ||
        hits.size()!=cells || coupled.size()!=cells || primary.size()!=cells || primary_count.size()!=cells ||
        !std::isfinite(expected_threshold) || expected_threshold<=0) {
        report.add("LIF_READBACK_SHAPE", "Readback allocation, grid or threshold contract failed.",0);
        return result;
    }
    if (queue_count>events.size()) {
        report.add("LIF_QUEUE_TRUNCATED", "A complete ledger requires every emitted record; queue overflow is not accepted.",queue_count);
        return result;
    }
    std::vector<uint64_t> expected_hits(cells,0),expected_coupled(cells,0);
    std::map<std::pair<uint32_t,int32_t>,uint64_t> residue_counts;
    std::set<std::pair<uint32_t,uint32_t>> seen;
    for (uint32_t i=0;i<queue_count;++i) {
        const auto& ev=events[i];
        if (ev.event_schema_magic!=SPIKE_EVENT_V10_SCHEMA_MAGIC || ev.reserved_v5!=0)
            report.add("LIF_EVENT_SCHEMA", "Versioned packet discriminator or reserved bits failed.",i);
        if (ev.leaf_id>=8 || ev.vertex_id>=TILE_CORE_VERTICES ||
            ev.atom_or_site_id>=n || ev.multiplicity_m==0 || ev.multiplicity_m>65535u ||
            !seen.emplace(ev.leaf_id,ev.vertex_id).second) {
            report.add("LIF_EVENT_IDENTITY", "Each firing core vertex needs one uniquely identified, bounded-multiplicity record.",i);
            continue;
        }
        const uint32_t atom=UNPACK_GLOBAL_ATOM_ID(tiles[ev.leaf_id].global_atom_idx[ev.vertex_id]);
        if (atom!=ev.atom_or_site_id || atom_residues[ev.atom_or_site_id]<-1 ||
            atom_residues[ev.atom_or_site_id]!=ev.residue_id || ev.step_tau!=expected_step)
            report.add("LIF_EVENT_PROVENANCE", "Event atom, residue or authoritative step differs from source state.",i);
        const auto& tile=tiles[ev.leaf_id];
        const auto& conv=convergence[ev.leaf_id];
        if (ev.pocket_tile_id!=tile.active_pocket_id ||
            !std::isfinite(tile.star0_vol[ev.vertex_id]) ||
            ev.voronoi_dual_vol!=std::max(tile.star0_vol[ev.vertex_id],1e-4f) ||
            !std::isfinite(conv.w_x_lambda_trace) ||
            ev.uv_or_holonomy_val!=std::max(2.f-conv.w_x_lambda_trace,0.f) ||
            conv.terminal_fault_latch!=0 || conv.epoch_cluster_fault!=0)
            report.add("LIF_OBSERVABLE_PROVENANCE", "Pocket, dual volume, holonomy or admitted fault state differs from the event source.",i);
        const auto pos=coordinates[size_t(ev.leaf_id)*n+ev.atom_or_site_id];
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(ev.pos_xyz[0]) || !std::isfinite(ev.pos_xyz[1]) || !std::isfinite(ev.pos_xyz[2]) ||
            ev.pos_xyz[0]!=pos.x || ev.pos_xyz[1]!=pos.y || ev.pos_xyz[2]!=pos.z)
            report.add("LIF_EVENT_POSITION", "Event position is not the emitting leaf's coordinate.",i);
        if (!std::isfinite(ev.v_pre_reset) || !std::isfinite(ev.q_discharged) ||
            !std::isfinite(ev.q_discharged_lo) || !std::isfinite(ev.r_post_reset) ||
            !std::isfinite(ev.spike_amplitude) || !std::isfinite(ev.uv_or_holonomy_val) ||
            ev.lif_threshold_used!=expected_threshold || ev.v_pre_reset<expected_threshold ||
            ev.r_post_reset<0 || ev.r_post_reset>=expected_threshold) {
            report.add("LIF_LEDGER_DOMAIN", "Firing ledger must be finite and have a bounded nonnegative remainder at the actual threshold.",i);
            continue;
        }
        const long double exact_quantum=static_cast<long double>(ev.multiplicity_m)*expected_threshold;
        const long double exact_pre=ev.v_pre_reset;
        const float rounded_quantum=float(ev.multiplicity_m)*expected_threshold;
        if (std::floor(exact_pre/static_cast<long double>(expected_threshold))!=ev.multiplicity_m ||
            ev.q_discharged!=rounded_quantum ||
            static_cast<long double>(ev.q_discharged)+ev.q_discharged_lo!=exact_quantum)
            report.add("LIF_QUOTIENT_OR_QUANTUM", "Quotient or high/low discharged quantum does not match the represented inputs.",i);
        const float product_low=std::fma(float(ev.multiplicity_m),expected_threshold,-rounded_quantum);
        uint32_t product_bits=0;std::memcpy(&product_bits,&product_low,4);
        if (ev.q_lo_product_raw_bits!=product_bits || ev.q_discharged_lo!=product_low ||
            ev.r_post_reset!=std::fma(-float(ev.multiplicity_m),expected_threshold,ev.v_pre_reset))
            report.add("LIF_CANONICAL_PARTITION", "Low product witness or canonical FMA remainder failed.",i);
        const long double expanded_error=std::fabs(exact_pre-(static_cast<long double>(ev.q_discharged)+ev.q_discharged_lo+ev.r_post_reset));
        const long double single_error=std::fabs(exact_pre-(static_cast<long double>(ev.q_discharged)+ev.r_post_reset));
        result.max_expanded_ledger_error=std::max(result.max_expanded_ledger_error,expanded_error);
        result.max_single_float_ledger_error=std::max(result.max_single_float_ledger_error,single_error);
        if (expanded_error!=0)
            report.add("LIF_EXPANDED_LEDGER", "Discharged high/low quantum plus retained remainder differs from represented pre-reset voltage.",i);
        if (require_requested_absolute_bounds && single_error>1e-6L)
            report.add("LIF_REQUESTED_SINGLE_FLOAT_BOUND", "The requested single-float q+r absolute 1e-6 bound failed; high/low ledger is reported separately.",i);
        if (voltage[ev.leaf_id*TILE_TOTAL_VERTICES+ev.vertex_id]!=ev.r_post_reset)
            report.add("LIF_MEMBRANE_LEDGER", "Retained membrane voltage differs from this event's remainder.",i);
        const float expected_amplitude=ev.q_discharged*(1.f+.5f*ev.uv_or_holonomy_val);
        if (std::fabs(ev.spike_amplitude-expected_amplitude)>4*std::numeric_limits<float>::epsilon()*std::max(1.f,std::fabs(expected_amplitude)))
            report.add("LIF_EMITTED_PAYLOAD", "Payload does not match discharged quantum and declared holonomy weighting.",i);
        result.total_multiplicity+=ev.multiplicity_m;
        const int64_t cell=signal_cell_host(ev.pos_xyz,grid);
        if (ev.voxel_cell_idx!=cell)
            report.add("LIF_EVENT_CELL", "Event cell differs from its endpoint coordinate and grid.",i);
        for (float f:ev.coupled_force_xyz) if (!std::isfinite(f))
            report.add("LIF_EVENT_FORCE", "Produced force is nonfinite.",i);
        if (cell>=0) {
            expected_hits[size_t(cell)]+=ev.multiplicity_m;
            if (atom_residues[ev.atom_or_site_id]>=0)
                residue_counts[{uint32_t(cell),atom_residues[ev.atom_or_site_id]}]+=ev.multiplicity_m;
            const size_t a=size_t(ev.leaf_id)*n+ev.atom_or_site_id;
            const auto f0=forces_before[a], f1=forces_after[a];
            // This diagnostic starts from zero forces, so deposited nonzero
            // force and nonzero observed change have the same predicate.
            if (f0.x!=0 || f0.y!=0 || f0.z!=0)
                report.add("LIF_WINDOW_FORCE_BASELINE", "Coupled-count readback requires the declared zero-force diagnostic baseline.",i);
            if (f1.x!=f0.x || f1.y!=f0.y || f1.z!=f0.z) expected_coupled[size_t(cell)]+=ev.multiplicity_m;
        }
    }
    for (size_t c=0;c<cells;++c) {
        if (hits[c]<0 || coupled[c]<0 || primary_count[c]<0 ||
            uint64_t(hits[c])!=expected_hits[c] || uint64_t(coupled[c])!=expected_coupled[c] ||
            coupled[c]>hits[c])
            report.add("LIF_MULTIPLICITY_GRIDS", "Integer grids differ from complete summed event multiplicity or overflowed.",uint32_t(c));
        const auto first_residue=residue_counts.lower_bound({uint32_t(c),0});
        const bool attributable=first_residue!=residue_counts.end() && first_residue->first.first==c;
        if (!attributable) {
            if (primary_count[c]!=0 || primary[c]!=-1)
                report.add("LIF_EMPTY_PRIMARY", "A cell with no attributable protein events must preserve signed -1 owner and zero count.",uint32_t(c));
        } else {
            const auto found=residue_counts.find({uint32_t(c),primary[c]});
            if (found==residue_counts.end() || primary_count[c]<=0 || uint64_t(primary_count[c])!=found->second || primary_count[c]>hits[c])
                report.add("LIF_PRIMARY_ATTRIBUTION", "Primary count must equal multiplicity of the actual chosen residue, not every event's multiplicity.",uint32_t(c));
        }
    }
    for (unsigned leaf=0;leaf<8;++leaf) {
        long double delta[3]={0,0,0};
        for (size_t a=0;a<n;++a) {
            const auto before=forces_before[size_t(leaf)*n+a],after=forces_after[size_t(leaf)*n+a];
            const float b[3]={before.x,before.y,before.z},f[3]={after.x,after.y,after.z};
            for (unsigned k=0;k<3;++k) {
                if (!std::isfinite(b[k]) || !std::isfinite(f[k]))
                    report.add("LIF_FORCE_FINITE", "Force readback contains a nonfinite value.",uint32_t(a));
                else delta[k]+=static_cast<long double>(f[k])-b[k];
            }
        }
        for (auto d:delta) {
            result.max_force_sum_change=std::max(result.max_force_sum_change,std::fabs(d));
            if (require_requested_absolute_bounds && std::fabs(d)>1e-7L)
                report.add("LIF_REQUESTED_FORCE_BOUND", "Observed force-buffer sum change exceeds requested absolute 1e-7; integer accumulator closure is a different property.",leaf);
        }
    }
    return result;
}
} // namespace biprism_qc

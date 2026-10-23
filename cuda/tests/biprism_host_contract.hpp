#pragma once

// Host admission checks for the biprism native diagnostic. Include the actual
// nhs_biprism_voronoi_lif.cuh first. These checks never manufacture topology,
// reinterpret a different event packet, launch CUDA, or grant physics approval.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace biprism_qc {

struct HConstraint {
    uint32_t central;
    std::vector<uint32_t> hydrogens;
};

struct ContractIssue {
    std::string code;
    std::string message;
    uint32_t subject;
};

struct ContractReport {
    std::vector<ContractIssue> issues;
    uint64_t total_failures = 0;
    bool ok() const { return total_failures == 0; }
    void add(const char* code, const std::string& message, uint32_t subject) {
        ++total_failures;
        // Keep diagnostics bounded; total_failures preserves the full count.
        if (issues.size() < 128) issues.push_back({code, message, subject});
    }
};

// These offsets are the existing sealed native packet contract, not a new
// layout: fused_engine.rs::GpuSpikeEvent, nhs_amber_fused.cu::SpikeEvent,
// ring_buffer.cu::GpuSpikeEvent and the little-endian Rust decoder agree.
// Source identities and offset checks are retained in this task's receipt.
struct NativeSpikePacketContract {
    static constexpr size_t bytes = 96;
    static constexpr size_t rust_allocation_alignment = 32;
    static constexpr size_t timestep = 0;
    static constexpr size_t voxel_idx = 4;
    static constexpr size_t position = 8;
    static constexpr size_t intensity = 20;
    static constexpr size_t nearby_residues = 24;
    static constexpr size_t n_residues = 56;
    static constexpr size_t spike_source = 60;
    static constexpr size_t phase_bits = 92;
};

inline ContractReport validate_candidate_spike_abi() {
    ContractReport report;
    if (sizeof(GpuSpikeEventCanonical96) != NativeSpikePacketContract::bytes)
        report.add("SPIKE_PACKET_SIZE", "Candidate packet size differs from native96.", 0);
    if (alignof(GpuSpikeEventCanonical96) != NativeSpikePacketContract::rust_allocation_alignment)
        report.add("SPIKE_PACKET_ALIGNMENT", "Candidate allocation alignment differs from native Rust allocation.", 0);
    if (offsetof(GpuSpikeEventCanonical96, pos_xyz) != NativeSpikePacketContract::position)
        report.add("SPIKE_POSITION_OFFSET", "Candidate position begins at0; native position begins at8.", 0);
    if (offsetof(GpuSpikeEventCanonical96, step_tau) != NativeSpikePacketContract::timestep)
        report.add("SPIKE_CLOCK_OFFSET", "Candidate step begins at12; native protocol timestep begins at0.", 0);
    if (offsetof(GpuSpikeEventCanonical96, spike_amplitude) != NativeSpikePacketContract::intensity)
        report.add("SPIKE_INTENSITY_OFFSET", "Candidate amplitude offset differs from native intensity.", 0);
    report.add("SPIKE_SPATIAL_IDENTITY", "Candidate atom_or_site_id has no native voxel_idx identity contract; do not reinterpret atom IDs as Cartesian voxel IDs.", 0);
    report.add("SPIKE_CLOCK_CONTRACT_UNVERIFIED", "The selected LIF v2 candidate requires the protocol step pointer and has no committed_substeps fallback. Its production capture path and sole accepted-step clock writer remain unverified.", 0);
    report.add("SPIKE_REQUIRED_METADATA", "Candidate omits native residue membership, source channel and phase_bits; holonomy/dual volume cannot occupy residue fields.", 0);
    return report;
}

inline void check_core_constraint_closure(
    const std::array<uint32_t, TILE_TOTAL_VERTICES>& atoms,
    uint32_t n_atoms, const std::vector<HConstraint>& groups,
    ContractReport& report
) {
    std::set<uint32_t> core;
    for (uint32_t i = 0; i < TILE_CORE_VERTICES; ++i) core.insert(atoms[i]);
    for (uint32_t gi = 0; gi < groups.size(); ++gi) {
        const HConstraint& group = groups[gi];
        bool valid = group.central < n_atoms;
        std::set<uint32_t> members;
        members.insert(group.central);
        for (uint32_t hydrogen : group.hydrogens) {
            if (hydrogen >= n_atoms || !members.insert(hydrogen).second) valid = false;
        }
        if (!valid) {
            report.add("CONSTRAINT_GROUP_INVALID", "Constraint group has an invalid or duplicate atom ID.", gi);
            continue;
        }
        size_t inside = 0;
        for (uint32_t atom : members) inside += core.count(atom);
        if (inside != 0 && inside != members.size())
            report.add("CONSTRAINT_CORE_SPLIT", "Constraint group crosses the core boundary (including atoms outside the256-vertex tile).", gi);
    }
}

inline size_t link_component_count(const std::map<uint32_t, std::set<uint32_t>>& graph) {
    std::set<uint32_t> pending;
    for (const auto& node : graph) pending.insert(node.first);
    size_t count = 0;
    while (!pending.empty()) {
        ++count;
        std::vector<uint32_t> todo{*pending.begin()};
        while (!todo.empty()) {
            const uint32_t v = todo.back();
            todo.pop_back();
            if (!pending.erase(v)) continue;
            const auto found = graph.find(v);
            if (found != graph.end())
                for (uint32_t neighbor : found->second) todo.push_back(neighbor);
        }
    }
    return count;
}

inline ContractReport validate_seed_inputs(
    const std::vector<int>& neighbor_list,
    const std::vector<int>& neighbor_counts,
    uint32_t n_atoms, uint32_t center,
    const std::vector<HConstraint>& groups
) {
    ContractReport report;
    constexpr size_t stride = 1024;
    if (n_atoms == 0 || center >= n_atoms) {
        report.add("SEED_CENTER_BOUNDS", "Center must index the actual atom allocation.", center);
        return report;
    }
    if (n_atoms > 0x01000000u)
        report.add("SEED_PACKED_ATOM_RANGE", "Atom allocation exceeds the24-bit packed atom-ID domain.", n_atoms);
    if (neighbor_counts.size() != n_atoms ||
        n_atoms > std::numeric_limits<size_t>::max() / stride ||
        neighbor_list.size() != size_t(n_atoms) * stride) {
        report.add("SEED_NEIGHBOR_ALLOCATION", "Native neighbor buffers must be N counts and N*1024 signed IDs.", center);
        return report;
    }
    const int count = neighbor_counts[center];
    if (count < int(TILE_TOTAL_VERTICES - 1) || count >= int(stride)) {
        report.add("SEED_NEIGHBOR_COUNT", "Need255 distinct valid neighbors and an unsaturated native row; no center+tid padding is admitted.", center);
        return report;
    }
    std::set<uint32_t> row_ids;
    bool row_valid = true;
    for (int k = 0; k < count; ++k) {
        const int atom = neighbor_list[size_t(center) * stride + size_t(k)];
        if (atom < 0 || uint32_t(atom) >= n_atoms || uint32_t(atom) == center) {
            report.add("SEED_NEIGHBOR_ID", "Neighbor row contains a negative, out-of-range or self atom ID.", uint32_t(k));
            row_valid = false;
        } else if (!row_ids.insert(uint32_t(atom)).second) {
            report.add("SEED_NEIGHBOR_DUPLICATE", "Native neighbor row contains duplicate atom IDs.", uint32_t(k));
            row_valid = false;
        }
    }
    if (row_valid) {
        std::array<uint32_t, TILE_TOTAL_VERTICES> selected{};
        selected[0] = center;
        for (uint32_t i = 1; i < TILE_TOTAL_VERTICES; ++i)
            selected[i] = uint32_t(neighbor_list[size_t(center) * stride + i - 1]);
        check_core_constraint_closure(selected, n_atoms, groups, report);
    }
    return report;
}

inline ContractReport validate_tile_topology(
    const ContactShellSimplicialStateFfi& tile,
    uint32_t n_atoms, const std::vector<HConstraint>& groups
) {
    ContractReport report;
    std::array<uint32_t, TILE_TOTAL_VERTICES> atoms{};
    std::set<uint32_t> unique_atoms;
    for (uint32_t v = 0; v < TILE_TOTAL_VERTICES; ++v) {
        atoms[v] = UNPACK_GLOBAL_ATOM_ID(tile.global_atom_idx[v]);
        if (atoms[v] >= n_atoms)
            report.add("TILE_ATOM_BOUNDS", "Tile global atom ID is outside the actual allocation.", v);
        if (!unique_atoms.insert(atoms[v]).second)
            report.add("TILE_ATOM_DUPLICATE", "Two tile vertices alias the same global atom.", v);
    }
    check_core_constraint_closure(atoms, n_atoms, groups, report);
    const uint32_t ne = tile.num_active_edges;
    const uint32_t nt = tile.num_active_tets;
    if (ne == 0 || ne > TILE_MAX_EDGES || nt == 0 || nt > TILE_MAX_TETS) {
        report.add("TILE_ACTIVE_COUNTS", "Active tetrahedron/edge counts must be nonzero and within the actual arrays.", 0);
        return report;
    }
    using Edge = std::array<uint32_t, 2>;
    using Face = std::array<uint32_t, 3>;
    using Tet = std::array<uint32_t, 4>;
    using FaceOwner = std::pair<uint32_t, uint32_t>;
    std::map<Edge, uint32_t> edge_owner;
    for (uint32_t e = 0; e < ne; ++e) {
        const uint32_t u = tile.edge_u8[e][0], v = tile.edge_u8[e][1];
        if (u >= TILE_TOTAL_VERTICES || v >= TILE_TOTAL_VERTICES || u >= v)
            report.add("EDGE_ENDPOINTS", "Edge endpoints must be distinct valid vertices in the supplied lower-ID orientation.", e);
        const Edge key{{std::min(u, v), std::max(u, v)}};
        if (!edge_owner.emplace(key, e).second)
            report.add("EDGE_DUPLICATE", "A simplicial edge occurs more than once in the active edge table.", e);
    }
    constexpr uint32_t pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    std::set<Tet> seen_tets;
    std::map<Face, std::vector<FaceOwner>> faces;
    std::map<Edge, std::vector<Edge>> edge_links;
    std::map<uint32_t, std::vector<Face>> vertex_links;
    for (uint32_t t = 0; t < nt; ++t) {
        Tet vertices{};
        for (uint32_t k = 0; k < 4; ++k) vertices[k] = tile.tet_v[t][k];
        Tet sorted = vertices;
        std::sort(sorted.begin(), sorted.end());
        const bool distinct = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
        if (!distinct || sorted[3] >= TILE_TOTAL_VERTICES) {
            report.add("TET_VERTICES", "Tetrahedron must have four distinct valid vertices.", t);
            continue;
        }
        if (!seen_tets.insert(sorted).second)
            report.add("TET_DUPLICATE", "An active tetrahedron is duplicated under a different ordering.", t);
        for (uint32_t k = 0; k < 6; ++k) {
            Edge edge{{vertices[pairs[k][0]], vertices[pairs[k][1]]}};
            std::sort(edge.begin(), edge.end());
            Edge opposite{};
            uint32_t at = 0;
            for (uint32_t j = 0; j < 4; ++j)
                if (j != pairs[k][0] && j != pairs[k][1]) opposite[at++] = vertices[j];
            std::sort(opposite.begin(), opposite.end());
            edge_links[edge].push_back(opposite);
        }
        for (uint32_t k = 0; k < 6; ++k) {
            const uint32_t eid = tile.tet_edges[t][k];
            if (eid >= ne) {
                report.add("TET_EDGE_BOUNDS", "Tetrahedron references an inactive or out-of-range edge.", t);
                continue;
            }
            const uint32_t a = vertices[pairs[k][0]], b = vertices[pairs[k][1]];
            if (tile.edge_u8[eid][0] != std::min(a,b) || tile.edge_u8[eid][1] != std::max(a,b))
                report.add("TET_EDGE_INCIDENCE", "Tetrahedron edge slots must name01,02,03,12,13,23 of that tetrahedron.", t);
        }
        for (uint32_t f = 0; f < 4; ++f) {
            Face face{};
            uint32_t at = 0;
            for (uint32_t k = 0; k < 4; ++k) if (k != f) face[at++] = vertices[k];
            std::sort(face.begin(), face.end());
            faces[face].push_back({t, f});
            vertex_links[vertices[f]].push_back(face);
        }
    }
    for (const auto& entry : faces) {
        const auto& owners = entry.second;
        if (owners.size() > 2) {
            report.add("TET_FACE_NONMANIFOLD", "More than two tetrahedra share one face.", owners.front().first);
            continue;
        }
        for (size_t i = 0; i < owners.size(); ++i) {
            const uint32_t t = owners[i].first, f = owners[i].second;
            const uint32_t actual = tile.tet_opp_tet[t][f];
            if (owners.size() == 1) {
                if (actual != 0xffffu)
                    report.add("TET_FACE_BOUNDARY", "Boundary face adjacency must be0xffff; referenced tetrahedron does not share the face.", t);
            } else {
                const uint32_t expected = owners[1-i].first;
                if (actual != expected)
                    report.add("TET_FACE_RECIPROCAL", "Shared face must name the actual adjacent tetrahedron reciprocally at its opposite-vertex slot.", t);
            }
        }
    }

    // Face multiplicity alone admits pinched or disconnected tetrahedral stars.
    // An edge of a 3-manifold has a connected circular or interval link.
    std::set<uint32_t> vertices_with_bad_edge_links;
    for (const auto& entry : edge_links) {
        const std::set<Edge> unique(entry.second.begin(), entry.second.end());
        std::map<uint32_t, std::set<uint32_t>> graph;
        for (const auto& e : unique) {
            graph[e[0]].insert(e[1]);
            graph[e[1]].insert(e[0]);
        }
        size_t degree_one = 0;
        bool degrees_valid = true;
        for (const auto& node : graph) {
            degree_one += node.second.size() == 1;
            degrees_valid = degrees_valid && (node.second.size() == 1 || node.second.size() == 2);
        }
        if (unique.size() != entry.second.size() || link_component_count(graph) != 1 ||
            !degrees_valid || (degree_one != 0 && degree_one != 2)) {
            vertices_with_bad_edge_links.insert(entry.first[0]);
            vertices_with_bad_edge_links.insert(entry.first[1]);
            report.add("EDGE_LINK_NONMANIFOLD", "Incident tetrahedra must give every edge one circular or interval link; face multiplicity alone is insufficient.", entry.first[0] * TILE_TOTAL_VERTICES + entry.first[1]);
        }
    }

    // A vertex link must be a sphere (interior) or disk (boundary). The edge
    // links above establish its local manifold condition; connectedness,
    // boundary-loop count and Euler characteristic distinguish these surfaces.
    for (const auto& entry : vertex_links) {
        std::map<Edge, uint32_t> link_edges;
        std::set<Face> unique_faces(entry.second.begin(), entry.second.end());
        for (const auto& tri : entry.second) {
            ++link_edges[Edge{{tri[0], tri[1]}}];
            ++link_edges[Edge{{tri[0], tri[2]}}];
            ++link_edges[Edge{{tri[1], tri[2]}}];
        }
        std::map<uint32_t, std::set<uint32_t>> graph, boundary;
        bool multiplicity_valid = true;
        for (const auto& e : link_edges) {
            graph[e.first[0]].insert(e.first[1]);
            graph[e.first[1]].insert(e.first[0]);
            multiplicity_valid = multiplicity_valid && e.second <= 2;
            if (e.second == 1) {
                boundary[e.first[0]].insert(e.first[1]);
                boundary[e.first[1]].insert(e.first[0]);
            }
        }
        bool boundary_valid = !boundary.empty() && link_component_count(boundary) == 1;
        for (const auto& node : boundary) boundary_valid = boundary_valid && node.second.size() == 2;
        const int64_t chi = int64_t(graph.size()) - int64_t(link_edges.size()) + int64_t(entry.second.size());
        const bool sphere_or_disk = boundary.empty() ? chi == 2 : (boundary_valid && chi == 1);
        if (link_component_count(graph) != 1 || !multiplicity_valid ||
            unique_faces.size() != entry.second.size() || vertices_with_bad_edge_links.count(entry.first) || !sphere_or_disk)
            report.add("VERTEX_LINK_NONMANIFOLD", "Incident tetrahedra must give every used vertex a connected sphere or disk link.", entry.first);
    }
    return report;
}

} // namespace biprism_qc

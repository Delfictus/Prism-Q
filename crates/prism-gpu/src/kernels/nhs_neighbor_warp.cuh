#pragma once

// Included by nhs_amber_fused.cu after the scalar neighbor builder. Constants,
// float3 and PRISM_PHYSICS_TRAP are supplied by that translation unit.
static_assert(NEIGHBOR_LIST_SIZE == 1024, "warp neighbor shared row capacity");

// Same 13-argument interface and pair membership as build_neighbor_list.
// Eight independent warps own eight atoms. Cell insertion order is immaterial:
// the final ascending atom-ID order is identical to the scalar builder.
extern "C" __global__ void __launch_bounds__(256) nhs_build_neighbor_list_warp(
    const float3* __restrict__ positions,
    const int* __restrict__ cell_list,
    const int* __restrict__ cell_counts,
    const int* __restrict__ atom_cell,
    const int* __restrict__ excl_list,
    const int* __restrict__ excl_offsets,
    int* __restrict__ neighbor_list,
    int* __restrict__ n_neighbors,
    int nx, int ny, int nz,
    int n_atoms,
    float cutoff_sq
) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) {
        if (threadIdx.x == 0) PRISM_PHYSICS_TRAP();
        return;
    }
    constexpr unsigned FULL_WARP = 0xffffffffu;
    constexpr int PAD = 0x7fffffff;
    __shared__ int rows[8][NEIGHBOR_LIST_SIZE]; // 32 KiB, no global sort traffic
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int atom = blockIdx.x * 8 + warp;
    // Entire warps return together; remaining warps need no block barriers.
    if (atom >= n_atoms) return;
    int* row = rows[warp];
    const float3 pi = positions[atom];
    const int cell = atom_cell[atom];
    const int ix = cell % nx, iy = (cell / nx) % ny, iz = cell / (nx * ny);
    const int excl_begin = excl_offsets[atom], excl_end = excl_offsets[atom+1];
    int count = 0;

    for (int dz=-1; dz<=1; ++dz) {
        const int z = iz+dz;
        if (z<0 || z>=nz) continue;
        for (int dy=-1; dy<=1; ++dy) {
            const int y = iy+dy;
            if (y<0 || y>=ny) continue;
            for (int dx=-1; dx<=1; ++dx) {
                const int x = ix+dx;
                if (x<0 || x>=nx) continue;
                const int other_cell = x+y*nx+z*nx*ny;
                const int cell_count = cell_counts[other_cell];
                if (cell_count > MAX_ATOMS_PER_CELL) {
                    if (lane==0) PRISM_PHYSICS_TRAP();
                    return;
                }
                for (int base=0; base<cell_count; base+=32) {
                    const int slot = base+lane;
                    int j = PAD;
                    bool keep = false;
                    if (slot<cell_count) {
                        j = cell_list[other_cell*MAX_ATOMS_PER_CELL+slot];
                        if (j!=atom) {
                            const float dx_ij=positions[j].x-pi.x;
                            const float dy_ij=positions[j].y-pi.y;
                            const float dz_ij=positions[j].z-pi.z;
                            const float r2=dx_ij*dx_ij+dy_ij*dy_ij+dz_ij*dz_ij;
                            // Keep the scalar builder's exact comparison,
                            // including equality at the buffered cutoff.
                            keep = !(r2>cutoff_sq);
                            if (keep) {
                                for (int e=excl_begin; e<excl_end; ++e) {
                                    if (excl_list[e]==j) { keep=false; break; }
                                }
                            }
                        }
                    }
                    const unsigned mask = __ballot_sync(FULL_WARP, keep);
                    const int added = __popc(mask);
                    if (count+added>NEIGHBOR_LIST_SIZE) {
                        if (lane==0) PRISM_PHYSICS_TRAP();
                        return;
                    }
                    if (keep) {
                        const unsigned lower_lanes = (1u<<lane)-1u;
                        row[count+__popc(mask & lower_lanes)] = j;
                    }
                    count += added;
                }
            }
        }
    }

    // Pad only the next power-of-two extent; all live atom IDs precede PAD.
    int sort_count=1;
    while (sort_count<count) sort_count<<=1;
    for (int i=count+lane; i<sort_count; i+=32) row[i]=PAD;
    __syncwarp(FULL_WARP);
    for (int width=2; width<=sort_count; width<<=1) {
        for (int stride=width>>1; stride>0; stride>>=1) {
            for (int i=lane; i<sort_count; i+=32) {
                const int partner=i^stride;
                // Each pair has exactly one writer in this network stage.
                if (partner>i) {
                    const int a=row[i], b=row[partner];
                    const bool ascending=(i & width)==0;
                    if ((ascending && a>b) || (!ascending && a<b)) {
                        row[i]=b;row[partner]=a;
                    }
                }
            }
            __syncwarp(FULL_WARP);
        }
    }
    int* out=neighbor_list+atom*NEIGHBOR_LIST_SIZE;
    for (int i=lane; i<count; i+=32) out[i]=row[i];
    __syncwarp(FULL_WARP);
    if (lane==0) n_neighbors[atom]=count;
}

// Explicit native hi+lo profile. The13 legacy arguments retain their order.
extern "C" __global__ void __launch_bounds__(256) nhs_build_neighbor_list_warp_compensated_v1(
    const float3* __restrict__ positions,
    const int* __restrict__ cell_list,
    const int* __restrict__ cell_counts,
    const int* __restrict__ atom_cell,
    const int* __restrict__ excl_list,
    const int* __restrict__ excl_offsets,
    int* __restrict__ neighbor_list,
    int* __restrict__ n_neighbors,
    int nx, int ny, int nz,
    int n_atoms,
    float cutoff_sq,
    const BiprismCoordinateOwnerFfi* coordinate_owner,
    uint32_t* precision_fault
) {
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1) {
        if (threadIdx.x == 0) PRISM_PHYSICS_TRAP();
        return;
    }
    constexpr unsigned FULL_WARP = 0xffffffffu;
    const bool owner_valid=bp_native_coordinate_owner_valid(coordinate_owner,
        reinterpret_cast<const float*>(positions),n_atoms,BP_COORD_CURRENT_INITIALIZED,precision_fault);
    // A sticky fault can arrive while lanes enter. Broadcast before any
    // full-mask ballot so no lane leaves a still-active warp by itself.
    if(!__all_sync(FULL_WARP,owner_valid))return;
    if(nx<=0||ny<=0||nz<=0||!isfinite(cutoff_sq)||cutoff_sq<=0.0f) {
        if(threadIdx.x==0)bp_native_coordinate_fault(precision_fault);return;
    }

    constexpr int PAD = 0x7fffffff;
    __shared__ int rows[8][NEIGHBOR_LIST_SIZE]; // 32 KiB, no global sort traffic
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int atom = blockIdx.x * 8 + warp;
    // Entire warps return together; remaining warps need no block barriers.
    if (atom >= n_atoms) return;
    int* row = rows[warp];

    const auto center=bp_coord_load(*coordinate_owner,BiprismCoordinatePlane::Current,(uint32_t)atom);
    if(!__all_sync(FULL_WARP,bp_coord_finite(center))) {
        if(lane==0)bp_native_coordinate_fault(precision_fault);return;
    }
    const int cell = atom_cell[atom];
    if(cell<0||(long long)cell>=(long long)nx*ny*nz) {
        if(lane==0)bp_native_coordinate_fault(precision_fault);return;
    }
    const int ix = cell % nx, iy = (cell / nx) % ny, iz = cell / (nx * ny);
    const int excl_begin = excl_offsets[atom], excl_end = excl_offsets[atom+1];
    int count = 0;

    for (int dz=-1; dz<=1; ++dz) {
        const int z = iz+dz;
        if (z<0 || z>=nz) continue;
        for (int dy=-1; dy<=1; ++dy) {
            const int y = iy+dy;
            if (y<0 || y>=ny) continue;
            for (int dx=-1; dx<=1; ++dx) {
                const int x = ix+dx;
                if (x<0 || x>=nx) continue;
                const int other_cell = x+y*nx+z*nx*ny;
                const int cell_count = cell_counts[other_cell];
                if (cell_count < 0 || cell_count > MAX_ATOMS_PER_CELL) {
                    if (lane==0) {bp_native_coordinate_fault(precision_fault);PRISM_PHYSICS_TRAP();}
                    return;
                }
                for (int base=0; base<cell_count; base+=32) {
                    const int slot = base+lane;
                    int j = PAD;
                    bool keep = false;
                    bool invalid = false;
                    if (slot<cell_count) {
                        j = cell_list[other_cell*MAX_ATOMS_PER_CELL+slot];
                        if(j<0||j>=n_atoms)invalid=true;
                        else if (j!=atom) {
                            const auto d=BpNativeGeometry<true>::difference(positions,j,atom,
                                coordinate_owner,BiprismCoordinatePlane::Current,precision_fault);
                            const double r2=d.x*d.x+d.y*d.y+d.z*d.z;
                            invalid=!isfinite(r2);
                            // Keep the scalar builder's exact comparison,
                            // including equality at the buffered cutoff.
                            keep = !invalid && !(r2>(double)cutoff_sq);
                            if (keep) {
                                for (int e=excl_begin; e<excl_end; ++e) {
                                    if (excl_list[e]==j) { keep=false; break; }
                                }
                            }
                        }
                    }
                    if(__any_sync(FULL_WARP,invalid)) {
                        if(lane==0)bp_native_coordinate_fault(precision_fault);return;
                    }
                    const unsigned mask = __ballot_sync(FULL_WARP, keep);
                    const int added = __popc(mask);
                    if (count+added>NEIGHBOR_LIST_SIZE) {
                        if (lane==0) {bp_native_coordinate_fault(precision_fault);PRISM_PHYSICS_TRAP();}
                        return;
                    }
                    if (keep) {
                        const unsigned lower_lanes = (1u<<lane)-1u;
                        row[count+__popc(mask & lower_lanes)] = j;
                    }
                    count += added;
                }
            }
        }
    }

    // Pad only the next power-of-two extent; all live atom IDs precede PAD.
    int sort_count=1;
    while (sort_count<count) sort_count<<=1;
    for (int i=count+lane; i<sort_count; i+=32) row[i]=PAD;
    __syncwarp(FULL_WARP);
    for (int width=2; width<=sort_count; width<<=1) {
        for (int stride=width>>1; stride>0; stride>>=1) {
            for (int i=lane; i<sort_count; i+=32) {
                const int partner=i^stride;
                // Each pair has exactly one writer in this network stage.
                if (partner>i) {
                    const int a=row[i], b=row[partner];
                    const bool ascending=(i & width)==0;
                    if ((ascending && a>b) || (!ascending && a<b)) {
                        row[i]=b;row[partner]=a;
                    }
                }
            }
            __syncwarp(FULL_WARP);
        }
    }
    int* out=neighbor_list+atom*NEIGHBOR_LIST_SIZE;
    for (int i=lane; i<count; i+=32) out[i]=row[i];
    __syncwarp(FULL_WARP);
    if (lane==0) n_neighbors[atom]=count;
}

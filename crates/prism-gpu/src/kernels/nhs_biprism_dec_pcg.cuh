#pragma once
#include <float.h>
#include <math_constants.h>
#include <type_traits>
#include "nhs_biprism_ordered_edges.cuh"

// Included after cta_dot_product_256. The operator is the represented
// positive-weight Dirichlet graph Laplacian; the diagonal offset belongs
// only to its Jacobi preconditioner.
constexpr uint32_t BIPRISM_DEC_MAX_ITERS = TILE_CORE_VERTICES;
constexpr uint32_t BIPRISM_DEC_RESTART_INTERVAL = 16u;
constexpr float BIPRISM_DEC_TOL_SQ = 1e-12f;

// product[0..15] is dead during a true-residual check. Store the two raw
// binary64 words through its declared float type, avoiding an aliased or
// potentially misaligned double pointer. All readers finish before reuse.
__device__ __forceinline__ double biprism_dec_sum_f64_256(double value, float* product) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(FULL_WARP_MASK, value, offset);
    if (lane == 0u) {
        product[2u * warp] = __int_as_float(__double2loint(value));
        product[2u * warp + 1u] = __int_as_float(__double2hiint(value));
    }
    __syncthreads();
    if (warp == 0u) {
        double sum = lane < 8u ? __hiloint2double(
            __float_as_int(product[2u * lane + 1u]),
            __float_as_int(product[2u * lane])) : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(FULL_WARP_MASK, sum, offset);
        if (lane == 0u) {
            product[0] = __int_as_float(__double2loint(sum));
            product[1] = __int_as_float(__double2hiint(sum));
        }
    }
    __syncthreads();
    const double result = __hiloint2double(__float_as_int(product[1]), __float_as_int(product[0]));
    __syncthreads();
    return result;
}

template<class EdgeWeight, class EdgeTraversal, class ValueScalar>
__device__ __forceinline__ double biprism_dec_row_f64_traversal(
    const uint8_t* edge_u, const uint8_t* edge_v, uint32_t edge_stride,
    const EdgeWeight& weights, const EdgeTraversal& traversal, const ValueScalar* value, uint32_t vertex
) {
    double row = 0.0;
    if (vertex < TILE_CORE_VERTICES) {
        const double own = static_cast<double>(value[vertex]);
        #pragma unroll 1
        for (uint32_t cursor = traversal.begin(vertex); cursor < traversal.end(vertex); ++cursor) {
            const uint32_t e = traversal.edge(cursor);
            const uint32_t u = edge_u[e * edge_stride];
            const uint32_t v = edge_v[e * edge_stride];
            if (u == vertex) row += static_cast<double>(weights(e)) * (own - static_cast<double>(value[v]));
            else if (v == vertex) row += static_cast<double>(weights(e)) * (own - static_cast<double>(value[u]));
        }
    }
    return row;
}

// Legacy traversal remains the same increasing all-edge scan.
template<class EdgeWeight>
__device__ __forceinline__ double biprism_dec_row_f64_accessor(
    const uint8_t* edge_u, const uint8_t* edge_v, uint32_t edge_stride,
    const EdgeWeight& weights, uint32_t num_edges, const float* value, uint32_t vertex
) {
    return biprism_dec_row_f64_traversal(edge_u, edge_v, edge_stride, weights,
        BiprismLinearEdgeTraversal{num_edges}, value, vertex);
}

template<class EdgeWeight, class EdgeTraversal, class PotentialScalar>
__device__ __forceinline__ bool biprism_dec_pcg_256_traversal(
    const uint8_t* edge_u, const uint8_t* edge_v, uint32_t edge_stride,
    const EdgeWeight& weights, uint32_t num_edges, const EdgeTraversal& traversal, double charge,
    PotentialScalar* phi, float* residual, float* preconditioned, float* direction,
    float* product, float* inverse_diagonal, float* reduction,
    float& out_residual, uint32_t& out_iters,
    double internal_tol_sq = static_cast<double>(BIPRISM_DEC_TOL_SQ)
) {
    using PotentialValue = typename std::remove_cv<PotentialScalar>::type;
    static_assert(std::is_same<PotentialValue,float>::value
        ||std::is_same<PotentialValue,double>::value,"DEC potential must be FP32 or retained FP64");
    const uint32_t tid = threadIdx.x;
    const bool core = tid < TILE_CORE_VERTICES;
    out_residual = CUDART_INF_F;
    out_iters = 0u;
    // An internal solve target may be tighter than the external admission
    // ceiling. It may never relax that ceiling or change the iteration cap.
    bool valid = isfinite(internal_tol_sq) && internal_tol_sq > 0.0
        && internal_tol_sq <= static_cast<double>(BIPRISM_DEC_TOL_SQ)
        && isfinite(phi[tid]) && (!core || isfinite(charge));
    #pragma unroll 1
    for (uint32_t e = tid; e < num_edges; e += 256u) {
        valid = valid && isfinite(weights(e)) && weights(e) > 0.0f
            && edge_u[e * edge_stride] != edge_v[e * edge_stride];
    }
    if (!__syncthreads_and(valid)) return false;

    double diagonal = 1e-4;
    if (core) {
        #pragma unroll 1
        for (uint32_t cursor = traversal.begin(tid); cursor < traversal.end(tid); ++cursor) {
            const uint32_t e = traversal.edge(cursor);
            if (edge_u[e * edge_stride] == tid || edge_v[e * edge_stride] == tid)
                diagonal += static_cast<double>(weights(e));
        }
    }
    inverse_diagonal[tid] = core ? static_cast<float>(1.0 / diagonal) : 0.0f;
    double true_r = core ? static_cast<double>(charge)
        - biprism_dec_row_f64_traversal(edge_u, edge_v, edge_stride, weights, traversal, phi, tid) : 0.0;
    if (!__syncthreads_and(isfinite(true_r) && fabs(true_r) <= static_cast<double>(FLT_MAX))) return false;
    double true_rr = biprism_dec_sum_f64_256(true_r * true_r, product);
    if (!isfinite(true_rr)) return false;
    out_residual = static_cast<float>(sqrt(true_rr));
    if (true_rr <= internal_tol_sq) return true;
    residual[tid] = static_cast<float>(true_r);
    preconditioned[tid] = inverse_diagonal[tid] * residual[tid];
    direction[tid] = preconditioned[tid];
    __syncthreads();
    float rr = cta_dot_product_256(residual[tid] * residual[tid], reduction);
    float rz = cta_dot_product_256(residual[tid] * preconditioned[tid], reduction);
    if (!isfinite(rr) || !isfinite(rz)) return false;

    #pragma unroll 1
    for (uint32_t iter = 0u; iter < BIPRISM_DEC_MAX_ITERS; ++iter) {
        out_iters = iter + 1u;
        const double ap = biprism_dec_row_f64_traversal(edge_u, edge_v, edge_stride, weights, traversal, direction, tid);
        if (!__syncthreads_and(isfinite(ap) && fabs(ap) <= static_cast<double>(FLT_MAX))) return false;
        product[tid] = static_cast<float>(ap);
        const float pap = cta_dot_product_256(direction[tid] * product[tid], reduction);
        if (!(pap > 0.0f) || !(rz > 0.0f) || !isfinite(pap) || !isfinite(rz)) return false;
        const float alpha = rz / pap;
        if (!isfinite(alpha)) return false;
        if (core) {
            if constexpr (std::is_same<PotentialValue,float>::value) {
                // Keep the historical FP32 update and its contraction exact.
                phi[tid] = __fmaf_rn(alpha, direction[tid], phi[tid]);
            } else {
                // Retain all potential bits through residual/work and commit.
                phi[tid] = __fma_rn(static_cast<double>(alpha),
                    static_cast<double>(direction[tid]),static_cast<double>(phi[tid]));
            }
            residual[tid] = __fmaf_rn(-alpha, product[tid], residual[tid]);
        }
        __syncthreads();
        if (!__syncthreads_and(isfinite(phi[tid]) && isfinite(residual[tid]))) return false;
        rr = cta_dot_product_256(residual[tid] * residual[tid], reduction);
        if (!isfinite(rr)) return false;
        const bool restart = (out_iters % BIPRISM_DEC_RESTART_INTERVAL == 0u)
            || static_cast<double>(rr) <= internal_tol_sq || out_iters == BIPRISM_DEC_MAX_ITERS;
        if (restart) {
            true_r = core ? static_cast<double>(charge)
                - biprism_dec_row_f64_traversal(edge_u, edge_v, edge_stride, weights, traversal, phi, tid) : 0.0;
            if (!__syncthreads_and(isfinite(true_r) && fabs(true_r) <= static_cast<double>(FLT_MAX))) return false;
            true_rr = biprism_dec_sum_f64_256(true_r * true_r, product);
            if (!isfinite(true_rr)) return false;
            residual[tid] = static_cast<float>(true_r);
            rr = cta_dot_product_256(residual[tid] * residual[tid], reduction);
            if (!isfinite(rr)) return false;
            out_residual = static_cast<float>(sqrt(true_rr));
            if (true_rr <= internal_tol_sq) return true;
        }
        preconditioned[tid] = inverse_diagonal[tid] * residual[tid];
        const float next_rz = cta_dot_product_256(residual[tid] * preconditioned[tid], reduction);
        if (!(next_rz > 0.0f) || !isfinite(next_rz)) return false;
        const float beta = restart ? 0.0f : next_rz / rz;
        direction[tid] = core ? __fmaf_rn(beta, direction[tid], preconditioned[tid]) : 0.0f;
        rz = next_rz;
        __syncthreads();
    }
    return false;
}

// Both policies share every solver operation, reduction and refusal predicate.
// Indexed callers must first publish the current invocation's complete CSR.
// Full-precision Krylov realization for the retained FP64 potential. The
// operator, Jacobi diagonal, halo values, true-residual gate and192-step bound
// are identical. Checking the residual does not itself discard conjugacy.
// Replacement restarts only when the measured recurrence gap is significant
// or a recursive stopping estimate is disproved by the true residual.
// Only direction[] is read by neighboring lanes; the other Krylov scalars
// have one-thread ownership. reduction[16] is a distinct stage scratch view.
template<class EdgeWeight, class EdgeTraversal>
__device__ __forceinline__ bool biprism_dec_pcg_256_precise_traversal(
    const uint8_t* edge_u, const uint8_t* edge_v, uint32_t edge_stride,
    const EdgeWeight& weights, uint32_t num_edges, const EdgeTraversal& traversal,
    double charge, volatile double* phi, double* direction, float* reduction,
    float& out_residual, uint32_t& out_iters,
    double internal_tol_sq = static_cast<double>(BIPRISM_DEC_TOL_SQ)
) {
    const uint32_t tid=threadIdx.x;const bool core=tid<TILE_CORE_VERTICES;
    out_residual=CUDART_INF_F;out_iters=0u;
    bool valid=isfinite(internal_tol_sq)&&internal_tol_sq>0.
        &&internal_tol_sq<=double(BIPRISM_DEC_TOL_SQ)&&isfinite(phi[tid])
        &&(!core||isfinite(charge));
    #pragma unroll 1
    for(uint32_t e=tid;e<num_edges;e+=256u)
        valid=valid&&isfinite(weights(e))&&weights(e)>0.
            &&edge_u[e*edge_stride]!=edge_v[e*edge_stride];
    if(!__syncthreads_and(valid))return false;
    double diagonal=1e-4;
    if(core){
        #pragma unroll 1
        for(uint32_t cursor=traversal.begin(tid);cursor<traversal.end(tid);++cursor){
            const uint32_t e=traversal.edge(cursor);
            if(edge_u[e*edge_stride]==tid||edge_v[e*edge_stride]==tid)diagonal+=double(weights(e));
        }
    }
    const double inverse=core?1./diagonal:0.;
    double r=core?charge-biprism_dec_row_f64_traversal(edge_u,edge_v,edge_stride,weights,traversal,phi,tid):0.;
    if(!__syncthreads_and(isfinite(r)&&isfinite(inverse)&&inverse>=0.))return false;
    double rr=biprism_dec_sum_f64_256(r*r,reduction);
    if(!isfinite(rr))return false;
    out_residual=float(sqrt(rr));if(rr<=internal_tol_sq)return true;
    direction[tid]=inverse*r;
    double rz=biprism_dec_sum_f64_256(r*direction[tid],reduction);
    #pragma unroll 1
    for(uint32_t iter=0u;iter<BIPRISM_DEC_MAX_ITERS;++iter){
        out_iters=iter+1u;
        const double ap=biprism_dec_row_f64_traversal(edge_u,edge_v,edge_stride,weights,traversal,direction,tid);
        const double pap=biprism_dec_sum_f64_256(direction[tid]*ap,reduction);
        if(!(pap>0.)||!(rz>0.)||!isfinite(pap)||!isfinite(rz))return false;
        const double alpha=rz/pap;
        if(!isfinite(alpha))return false;
        if(core){phi[tid]=__fma_rn(alpha,direction[tid],double(phi[tid]));r=__fma_rn(-alpha,ap,r);}
        __syncthreads();
        if(!__syncthreads_and(isfinite(phi[tid])&&isfinite(r)))return false;
        rr=biprism_dec_sum_f64_256(r*r,reduction);
        if(!isfinite(rr))return false;
        bool restart=false;
        if(out_iters%BIPRISM_DEC_RESTART_INTERVAL==0u||rr<=internal_tol_sq||out_iters==BIPRISM_DEC_MAX_ITERS){
            const double actual=core?charge-biprism_dec_row_f64_traversal(edge_u,edge_v,edge_stride,weights,traversal,phi,tid):0.;
            if(!__syncthreads_and(isfinite(actual)))return false;
            const double actual_rr=biprism_dec_sum_f64_256(actual*actual,reduction);
            const double gap=actual-r;
            const double gap_rr=biprism_dec_sum_f64_256(gap*gap,reduction);
            if(!isfinite(actual_rr)||!isfinite(gap_rr))return false;
            out_residual=float(sqrt(actual_rr));
            if(actual_rr<=internal_tol_sq)return true;
            restart=rr<=internal_tol_sq||gap_rr>.01*actual_rr;
            if(restart)r=actual;
        }
        const double z=inverse*r;
        const double next_rz=biprism_dec_sum_f64_256(r*z,reduction);
        if(!(next_rz>0.)||!isfinite(next_rz))return false;
        const double beta=restart?0.:next_rz/rz;
        direction[tid]=core?__fma_rn(beta,direction[tid],z):0.;
        rz=next_rz;
        __syncthreads();
    }
    return false;
}

template<class EdgeWeight>
__device__ __forceinline__ bool biprism_dec_pcg_256_accessor(
    const uint8_t* edge_u, const uint8_t* edge_v, uint32_t edge_stride,
    const EdgeWeight& weights, uint32_t num_edges, double charge,
    float* phi, float* residual, float* preconditioned, float* direction,
    float* product, float* inverse_diagonal, float* reduction,
    float& out_residual, uint32_t& out_iters
) {
    return biprism_dec_pcg_256_traversal(edge_u,edge_v,edge_stride,weights,num_edges,
        BiprismLinearEdgeTraversal{num_edges},charge,phi,residual,preconditioned,
        direction,product,inverse_diagonal,reduction,out_residual,out_iters);
}
template<class EdgeWeight>
__device__ __forceinline__ bool biprism_dec_pcg_256_indexed_accessor(
    const uint8_t* edge_u, const uint8_t* edge_v, uint32_t edge_stride,
    const EdgeWeight& weights, uint32_t num_edges, const BiprismOrderedEdgesFfi* index,
    double charge, float* phi, float* residual, float* preconditioned, float* direction,
    float* product, float* inverse_diagonal, float* reduction,
    float& out_residual, uint32_t& out_iters
) {
    return biprism_dec_pcg_256_traversal(edge_u,edge_v,edge_stride,weights,num_edges,
        BiprismIncidentEdgeTraversal{index},charge,phi,residual,preconditioned,
        direction,product,inverse_diagonal,reduction,out_residual,out_iters);
}

// Precise-profile entry points. Only the potential storage/update changes;
// Krylov vectors, preconditioning, ordered rows, reductions, restart/iteration
// bounds and true-residual admission share the solver body above.
// A caller must retain all potential bits through physical consumers and the
// accepted owner. A later FP32 compatibility projection is not authority.
template<class EdgeWeight>
__device__ __forceinline__ bool biprism_dec_pcg_256_f64_accessor(
    const uint8_t* edge_u, const uint8_t* edge_v, uint32_t edge_stride,
    const EdgeWeight& weights, uint32_t num_edges, double charge,
    volatile double* phi, float* residual, float* preconditioned, float* direction,
    float* product, float* inverse_diagonal, float* reduction,
    float& out_residual, uint32_t& out_iters
) {
    return biprism_dec_pcg_256_traversal(edge_u,edge_v,edge_stride,weights,num_edges,
        BiprismLinearEdgeTraversal{num_edges},charge,phi,residual,preconditioned,
        direction,product,inverse_diagonal,reduction,out_residual,out_iters);
}
template<class EdgeWeight>
__device__ __forceinline__ bool biprism_dec_pcg_256_f64_indexed_accessor(
    const uint8_t* edge_u, const uint8_t* edge_v, uint32_t edge_stride,
    const EdgeWeight& weights, uint32_t num_edges, const BiprismOrderedEdgesFfi* index,
    double charge, volatile double* phi, float* residual, float* preconditioned,
    float* direction, float* product, float* inverse_diagonal, float* reduction,
    float& out_residual, uint32_t& out_iters,
    double internal_tol_sq = static_cast<double>(BIPRISM_DEC_TOL_SQ)
) {
    return biprism_dec_pcg_256_traversal(edge_u,edge_v,edge_stride,weights,num_edges,
        BiprismIncidentEdgeTraversal{index},charge,phi,residual,preconditioned,
        direction,product,inverse_diagonal,reduction,out_residual,out_iters,internal_tol_sq);
}

// Existing callers retain the same array-based ABI and represented operator.
template<typename Weight> struct BiprismDecArrayWeightsT {
    const Weight* data;
    __device__ __forceinline__ Weight operator()(uint32_t e) const { return data[e]; }
};
using BiprismDecArrayWeights = BiprismDecArrayWeightsT<float>;
__device__ __forceinline__ double biprism_dec_row_f64(
    const uint8_t* edge_u,const uint8_t* edge_v,uint32_t edge_stride,
    const float* weights,uint32_t num_edges,const float* value,uint32_t vertex
) {
    return biprism_dec_row_f64_accessor(edge_u,edge_v,edge_stride,
        BiprismDecArrayWeights{weights},num_edges,value,vertex);
}
__device__ __forceinline__ bool biprism_dec_pcg_256(
    const uint8_t* edge_u,const uint8_t* edge_v,uint32_t edge_stride,
    const float* weights,uint32_t num_edges,double charge,
    float* phi,float* residual,float* preconditioned,float* direction,
    float* product,float* inverse_diagonal,float* reduction,
    float& out_residual,uint32_t& out_iters
) {
    return biprism_dec_pcg_256_accessor(edge_u,edge_v,edge_stride,
        BiprismDecArrayWeights{weights},num_edges,charge,phi,residual,
        preconditioned,direction,product,inverse_diagonal,reduction,out_residual,out_iters);
}
template<typename Weight>
__device__ __forceinline__ bool biprism_dec_pcg_256_f64(
    const uint8_t* edge_u,const uint8_t* edge_v,uint32_t edge_stride,
    const Weight* weights,uint32_t num_edges,double charge,
    volatile double* phi,float* residual,float* preconditioned,float* direction,
    float* product,float* inverse_diagonal,float* reduction,
    float& out_residual,uint32_t& out_iters
) {
    return biprism_dec_pcg_256_f64_accessor(edge_u,edge_v,edge_stride,
        BiprismDecArrayWeightsT<Weight>{weights},num_edges,charge,phi,residual,
        preconditioned,direction,product,inverse_diagonal,reduction,out_residual,out_iters);
}

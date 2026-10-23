#ifndef NHS_BIPRISM_COORDINATE_PRECISION_CUH
#define NHS_BIPRISM_COORDINATE_PRECISION_CUH

// Coordinate authority for the explicit Biprism precision profile.
// The physical coordinate is the sum of TWO stored FP32 components. A high
// plane alone is a projection, not the complete state. Reconstruct in FP64,
// or difference expansions before geometry; never collapse hi+lo to FP32.
//
// Each descriptor binds dense 3*atom_count planes in one declared unit frame.
// Native and atomic arrays are distinct and share leaf/count/generation identity.
// MeshReference is the immutable cold/reseed Biprism geometry reference. It is
// NOT the native CA/NMA restraint reference. Those retain their own authority.
//
// This is an allocation/representation descriptor, not a step-acceptance token.
// Existing bridge, selected-step and collective records own chronology. Callers
// must latch invalid descriptors/values in their explicit precision-fault sink.
// A helper failure must never be interpreted as a completed physical phase.

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <limits>

#if defined(__CUDACC__)
#define BP_COORD_HD __host__ __device__ __forceinline__
#else
#define BP_COORD_HD inline
#endif

static constexpr uint32_t BP_COORDINATE_MAGIC = 0x31504342u; // BCP1
static constexpr uint32_t BP_COORDINATE_VERSION = 1u;
static constexpr uint32_t BP_COORD_NATIVE_ANGSTROM = 1u;
static constexpr uint32_t BP_COORD_ATOMIC_BOHR = 2u;
static constexpr uint32_t BP_COORD_CURRENT_INITIALIZED = 1u;
static constexpr uint32_t BP_COORD_START_INITIALIZED = 2u;
static constexpr uint32_t BP_COORD_REFERENCE_INITIALIZED = 4u;
static constexpr uint32_t BP_COORD_ALL_INITIALIZED = 7u;

enum class BiprismCoordinatePlane : uint32_t {
    Current = 1u,
    StepStart = 2u,
    MeshReference = 4u
};

struct alignas(16) BiprismCoordinateOwnerFfi {
    uint32_t abi_magic;
    uint32_t abi_version;
    uint32_t leaf_id;
    uint32_t atom_count;
    uint64_t generation;
    uint32_t unit_frame;
    uint32_t initialized_mask;
    float* current_hi;
    float* current_lo;
    float* step_start_hi;
    float* step_start_lo;
    float* reference_hi;
    float* reference_lo;
    uint64_t reserved[2];
};
static_assert(sizeof(void*) == 8, "Coordinate owner requires the native 64-bit ABI");
static_assert(sizeof(BiprismCoordinateOwnerFfi) == 96, "Coordinate owner ABI");
static_assert(alignof(BiprismCoordinateOwnerFfi) == 16, "Coordinate owner alignment");
static_assert(offsetof(BiprismCoordinateOwnerFfi,generation) == 16, "Coordinate generation offset");
static_assert(offsetof(BiprismCoordinateOwnerFfi,unit_frame) == 24, "Coordinate frame offset");
static_assert(offsetof(BiprismCoordinateOwnerFfi,current_hi) == 32, "Coordinate current high offset");
static_assert(offsetof(BiprismCoordinateOwnerFfi,current_lo) == 40, "Coordinate current low offset");
static_assert(offsetof(BiprismCoordinateOwnerFfi,step_start_hi) == 48, "Coordinate start high offset");
static_assert(offsetof(BiprismCoordinateOwnerFfi,step_start_lo) == 56, "Coordinate start low offset");
static_assert(offsetof(BiprismCoordinateOwnerFfi,reference_hi) == 64, "Coordinate reference high offset");
static_assert(offsetof(BiprismCoordinateOwnerFfi,reference_lo) == 72, "Coordinate reference low offset");
static_assert(offsetof(BiprismCoordinateOwnerFfi,reserved) == 80, "Coordinate reserved offset");

struct BiprismCoordinatePairF32 { float hi, lo; };
struct BiprismCoordinateExpansion3 { BiprismCoordinatePairF32 x, y, z; };
struct BiprismCoordinate3F64 { double x, y, z; };
struct BiprismCoordinateAccumulatorF64 { double hi, lo; };

BP_COORD_HD bool bp_coord_finite(double x) {
    return x == x && x <= std::numeric_limits<double>::max()
        && x >= -std::numeric_limits<double>::max();
}
BP_COORD_HD bool bp_coord_finite(BiprismCoordinatePairF32 x) {
    return bp_coord_finite(double(x.hi)) && bp_coord_finite(double(x.lo));
}
BP_COORD_HD bool bp_coord_finite(BiprismCoordinateExpansion3 x) {
    return bp_coord_finite(x.x) && bp_coord_finite(x.y) && bp_coord_finite(x.z);
}
BP_COORD_HD bool bp_coord_finite(BiprismCoordinate3F64 x) {
    return bp_coord_finite(x.x) && bp_coord_finite(x.y) && bp_coord_finite(x.z);
}
BP_COORD_HD double bp_coord_add_rn(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a,b);
#else
    return a+b;
#endif
}
BP_COORD_HD double bp_coord_sub_rn(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(a,b);
#else
    return a-b;
#endif
}
BP_COORD_HD double bp_coord_mul_rn(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a,b);
#else
    return a*b;
#endif
}
BP_COORD_HD double bp_coord_div_rn(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rn(a,b);
#else
    return a/b;
#endif
}
BP_COORD_HD double bp_coord_fma_rn(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
    return __fma_rn(a,b,c);
#else
    return ::fma(a,b,c);
#endif
}
BP_COORD_HD float bp_coord_round_f32(double a) {
#if defined(__CUDA_ARCH__)
    return __double2float_rn(a);
#else
    return static_cast<float>(a);
#endif
}
BP_COORD_HD BiprismCoordinatePairF32 bp_coord_invalid() {
    return {std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()};
}
BP_COORD_HD BiprismCoordinateExpansion3 bp_coord_invalid3() {
    return {bp_coord_invalid(),bp_coord_invalid(),bp_coord_invalid()};
}

// Knuth TwoSum. Exact decomposition of a finite non-overflowing binary64 sum
// under RN arithmetic; callers reject nonfinite results. It is not a claim that
// an arbitrary exact coordinate is representable by two FP32 numbers.
BP_COORD_HD BiprismCoordinateAccumulatorF64 bp_coord_two_sum(double a, double b) {
    const double s=bp_coord_add_rn(a,b);
    const double bb=bp_coord_sub_rn(s,a);
    const double e=bp_coord_add_rn(bp_coord_sub_rn(a,bp_coord_sub_rn(s,bb)),bp_coord_sub_rn(b,bb));
    return {s,e};
}
BP_COORD_HD BiprismCoordinatePairF32 bp_coord_pack(BiprismCoordinateAccumulatorF64 v) {
    if(!bp_coord_finite(v.hi)||!bp_coord_finite(v.lo))return bp_coord_invalid();
    const auto n=bp_coord_two_sum(v.hi,v.lo);
    const float hi=bp_coord_round_f32(n.hi);
    if(!bp_coord_finite(double(hi)))return bp_coord_invalid();
    const double rest=bp_coord_add_rn(bp_coord_sub_rn(n.hi,double(hi)),n.lo);
    const float lo=bp_coord_round_f32(rest);
    const BiprismCoordinatePairF32 result={hi,lo};
    return bp_coord_finite(result)?result:bp_coord_invalid();
}
BP_COORD_HD BiprismCoordinatePairF32 bp_coord_split(double value) {
    return bp_coord_pack({value,0.});
}
BP_COORD_HD BiprismCoordinateExpansion3 bp_coord_split(BiprismCoordinate3F64 value) {
    return {bp_coord_split(value.x),bp_coord_split(value.y),bp_coord_split(value.z)};
}
BP_COORD_HD double bp_coord_value(BiprismCoordinatePairF32 p) {
    return bp_coord_add_rn(double(p.hi),double(p.lo));
}
BP_COORD_HD BiprismCoordinate3F64 bp_coord_value(BiprismCoordinateExpansion3 p) {
    return {bp_coord_value(p.x),bp_coord_value(p.y),bp_coord_value(p.z)};
}
BP_COORD_HD double bp_coord_difference(BiprismCoordinatePairF32 a,BiprismCoordinatePairF32 b) {
    const auto high=bp_coord_two_sum(double(a.hi),-double(b.hi));
    const auto low=bp_coord_two_sum(double(a.lo),-double(b.lo));
    const auto joined=bp_coord_two_sum(high.hi,low.hi);
    return bp_coord_add_rn(joined.hi,bp_coord_add_rn(joined.lo,bp_coord_add_rn(high.lo,low.lo)));
}
BP_COORD_HD BiprismCoordinate3F64 bp_coord_difference(BiprismCoordinateExpansion3 a,BiprismCoordinateExpansion3 b) {
    return {bp_coord_difference(a.x,b.x),bp_coord_difference(a.y,b.y),bp_coord_difference(a.z,b.z)};
}
BP_COORD_HD BiprismCoordinatePairF32 bp_coord_add(BiprismCoordinatePairF32 p,double delta) {
    if(!bp_coord_finite(p)||!bp_coord_finite(delta))return bp_coord_invalid();
    const auto a=bp_coord_two_sum(double(p.hi),delta);
    const auto b=bp_coord_two_sum(a.hi,bp_coord_add_rn(a.lo,double(p.lo)));
    return bp_coord_pack(b);
}
BP_COORD_HD BiprismCoordinateExpansion3 bp_coord_add(BiprismCoordinateExpansion3 p,BiprismCoordinate3F64 d) {
    return {bp_coord_add(p.x,d.x),bp_coord_add(p.y,d.y),bp_coord_add(p.z,d.z)};
}
BP_COORD_HD BiprismCoordinatePairF32 bp_coord_scale(BiprismCoordinatePairF32 p,double factor) {
    if(!bp_coord_finite(p)||!bp_coord_finite(factor))return bp_coord_invalid();
    const double h=bp_coord_mul_rn(double(p.hi),factor),l=bp_coord_mul_rn(double(p.lo),factor);
    if(!bp_coord_finite(h)||!bp_coord_finite(l))return bp_coord_invalid();
    const auto joined=bp_coord_two_sum(h,l);
    const double error=bp_coord_add_rn(bp_coord_fma_rn(double(p.hi),factor,-h),bp_coord_fma_rn(double(p.lo),factor,-l));
    return bp_coord_pack({joined.hi,bp_coord_add_rn(joined.lo,error)});
}
BP_COORD_HD BiprismCoordinateExpansion3 bp_coord_scale(BiprismCoordinateExpansion3 p,double factor) {
    return {bp_coord_scale(p.x,factor),bp_coord_scale(p.y,factor),bp_coord_scale(p.z,factor)};
}
BP_COORD_HD BiprismCoordinatePairF32 bp_coord_divide(BiprismCoordinatePairF32 p,double divisor) {
    if(!bp_coord_finite(p)||!bp_coord_finite(divisor)||divisor==0.)return bp_coord_invalid();
    const double high=bp_coord_div_rn(double(p.hi),divisor);
    if(!bp_coord_finite(high))return bp_coord_invalid();
    const double residual=bp_coord_add_rn(bp_coord_fma_rn(-high,divisor,double(p.hi)),double(p.lo));
    return bp_coord_pack({high,bp_coord_div_rn(residual,divisor)});
}
BP_COORD_HD BiprismCoordinateExpansion3 bp_coord_divide(BiprismCoordinateExpansion3 p,double divisor) {
    return {bp_coord_divide(p.x,divisor),bp_coord_divide(p.y,divisor),bp_coord_divide(p.z,divisor)};
}
// Add scale*value without discarding the binary64 product residual.
BP_COORD_HD BiprismCoordinatePairF32 bp_coord_fma(BiprismCoordinatePairF32 p,double scale,double value) {
    if(!bp_coord_finite(p)||!bp_coord_finite(scale)||!bp_coord_finite(value))return bp_coord_invalid();
    const double product=bp_coord_mul_rn(scale,value);
    if(!bp_coord_finite(product))return bp_coord_invalid();
    const double error=bp_coord_fma_rn(scale,value,-product);
    const auto a=bp_coord_two_sum(double(p.hi),product);
    const auto b=bp_coord_two_sum(a.hi,bp_coord_add_rn(a.lo,bp_coord_add_rn(double(p.lo),error)));
    return bp_coord_pack(b);
}

BP_COORD_HD bool bp_coord_pointer_valid(const float* p) {
    return p && (reinterpret_cast<uintptr_t>(p)&(alignof(float)-1u))==0u;
}
BP_COORD_HD bool bp_coord_owner_valid(
    const BiprismCoordinateOwnerFfi* o,uint32_t leaf,uint32_t atom_count,
    uint64_t generation,uint32_t frame,uint32_t required_mask,
    const float* expected_current_hi,const float* expected_start_hi=nullptr,
    const float* expected_reference_hi=nullptr) {
    if(!o||(reinterpret_cast<uintptr_t>(o)&15u)!=0u||leaf>=8u||atom_count==0u||generation==0u
        ||(frame!=BP_COORD_NATIVE_ANGSTROM&&frame!=BP_COORD_ATOMIC_BOHR)
        ||(required_mask&~BP_COORD_ALL_INITIALIZED)!=0u)return false;
    if(o->abi_magic!=BP_COORDINATE_MAGIC||o->abi_version!=BP_COORDINATE_VERSION
        ||o->leaf_id!=leaf||o->atom_count!=atom_count||o->generation!=generation||o->unit_frame!=frame
        ||(o->initialized_mask&~BP_COORD_ALL_INITIALIZED)!=0u
        ||(o->initialized_mask&required_mask)!=required_mask||o->reserved[0]||o->reserved[1]
        ||!expected_current_hi||o->current_hi!=expected_current_hi)return false;
    if(expected_start_hi&&o->step_start_hi!=expected_start_hi)return false;
    if(expected_reference_hi&&o->reference_hi!=expected_reference_hi)return false;
    const float* planes[6]={o->current_hi,o->current_lo,o->step_start_hi,o->step_start_lo,o->reference_hi,o->reference_lo};
    for(uint32_t i=0;i<6u;++i){
        if((required_mask&(1u<<(i/2u)))==0u)continue;
        if(!bp_coord_pointer_valid(planes[i]))return false;
        for(uint32_t j=0;j<i;++j)
            if((required_mask&(1u<<(j/2u)))!=0u&&planes[i]==planes[j])return false;
    }
    // Full extent/non-overlap and independent leaf/generation binding are host
    // admission obligations. A pointer alone cannot prove allocation extent.
    return true;
}
BP_COORD_HD BiprismCoordinateExpansion3 bp_coord_load(
    const BiprismCoordinateOwnerFfi& o,BiprismCoordinatePlane plane,uint32_t atom) {
    const uint32_t bit=static_cast<uint32_t>(plane);
    if(atom>=o.atom_count||(bit!=1u&&bit!=2u&&bit!=4u)||(o.initialized_mask&bit)==0u)return bp_coord_invalid3();
    const float* hi=bit==1u?o.current_hi:(bit==2u?o.step_start_hi:o.reference_hi);
    const float* lo=bit==1u?o.current_lo:(bit==2u?o.step_start_lo:o.reference_lo);
    if(!bp_coord_pointer_valid(hi)||!bp_coord_pointer_valid(lo)||hi==lo)return bp_coord_invalid3();
    const size_t i=size_t(atom)*3u;
    const BiprismCoordinateExpansion3 result={{hi[i],lo[i]},{hi[i+1u],lo[i+1u]},{hi[i+2u],lo[i+2u]}};
    return bp_coord_finite(result)?result:bp_coord_invalid3();
}
// This is a per-atom store, NOT a collective transaction or rollback mechanism.
// The caller owns its applicable producer/admission gate, exclusive writes and
// absence of concurrent readers. Native attempted drift/constraint stages can
// write before endpoint acceptance; main's accepted scatter uses its all-leaf
// gate. All six components are validated before this helper's first write.
BP_COORD_HD bool bp_coord_store_current(
    const BiprismCoordinateOwnerFfi& o,uint32_t atom,BiprismCoordinateExpansion3 value) {
    if(atom>=o.atom_count||(o.initialized_mask&BP_COORD_CURRENT_INITIALIZED)==0u
        ||!bp_coord_pointer_valid(o.current_hi)||!bp_coord_pointer_valid(o.current_lo)
        ||o.current_hi==o.current_lo||!bp_coord_finite(value))return false;
    const size_t i=size_t(atom)*3u;
    o.current_hi[i]=value.x.hi;o.current_lo[i]=value.x.lo;
    o.current_hi[i+1u]=value.y.hi;o.current_lo[i+1u]=value.y.lo;
    o.current_hi[i+2u]=value.z.hi;o.current_lo[i+2u]=value.z.lo;
    return true;
}

#undef BP_COORD_HD
#endif

// CPU execution of the actual shared coordinate-expansion helper.
// No CUDA dispatch, engine trajectory or native precision acceptance is implied.
#include "../../crates/prism-gpu/src/kernels/nhs_biprism_coordinate_precision.cuh"
#include <array>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

struct Input { float hi,lo; double h,v,factor; };
struct Output {
    BiprismCoordinatePairF32 next;
    double value,difference;
    BiprismCoordinatePairF32 scaled,back,four,added;
    double divided_value;
};
static_assert(sizeof(Input)==32,"reference input ABI");
static_assert(sizeof(Output)==64,"reference output ABI");
static_assert(offsetof(Output,scaled)==24&&offsetof(Output,divided_value)==56,"reference field ABI");

static unsigned checks=0;
static void require(bool ok,const char* name) {
    ++checks;
    if(!ok){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);}
}

static void ownership_controls() {
    std::array<std::array<float,9>,6> p{};
    BiprismCoordinateOwnerFfi o={BP_COORDINATE_MAGIC,BP_COORDINATE_VERSION,3,3,23,
        BP_COORD_NATIVE_ANGSTROM,BP_COORD_ALL_INITIALIZED,p[0].data(),p[1].data(),
        p[2].data(),p[3].data(),p[4].data(),p[5].data(),{0,0}};
    auto valid=[&](const BiprismCoordinateOwnerFfi* a){return bp_coord_owner_valid(a,3,3,23,
        BP_COORD_NATIVE_ANGSTROM,7,p[0].data(),p[2].data(),p[4].data());};
    require(valid(&o),"complete owner");
    require(!valid(nullptr),"null owner");
    auto bad=o;bad.abi_magic^=1;require(!valid(&bad),"magic");
    bad=o;bad.abi_version=2;require(!valid(&bad),"version");
    bad=o;bad.leaf_id=4;require(!valid(&bad),"leaf");
    bad=o;bad.atom_count=2;require(!valid(&bad),"extent identity");
    bad=o;bad.generation=24;require(!valid(&bad),"generation");
    bad=o;bad.unit_frame=BP_COORD_ATOMIC_BOHR;require(!valid(&bad),"unit frame");
    bad=o;bad.initialized_mask=3;require(!valid(&bad),"missing mesh reference initialization");
    bad=o;bad.initialized_mask=15;require(!valid(&bad),"unknown initialization bit");
    bad=o;bad.reserved[0]=1;require(!valid(&bad),"reserved0");
    bad=o;bad.reserved[1]=1;require(!valid(&bad),"reserved1");
    bad=o;bad.current_hi=p[2].data();require(!valid(&bad),"current high binding");
    bad=o;bad.step_start_hi=p[4].data();require(!valid(&bad),"start high binding");
    bad=o;bad.reference_hi=p[0].data();require(!valid(&bad),"mesh high binding");
    bad=o;bad.current_lo=nullptr;require(!valid(&bad),"missing low authority");
    bad=o;bad.current_lo=bad.current_hi;require(!valid(&bad),"high low alias");
    bad=o;bad.reference_lo=bad.step_start_lo;require(!valid(&bad),"snapshot plane alias");
    bad=o;bad.current_lo=reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(p[1].data())+1);
    require(!valid(&bad),"misaligned low plane");
    require(!bp_coord_owner_valid(&o,3,3,0,1,7,p[0].data()),"zero expected generation");
    require(!bp_coord_owner_valid(&o,3,3,23,1,8,p[0].data()),"unknown requested plane");
    require(!bp_coord_finite(bp_coord_load(o,BiprismCoordinatePlane::Current,3)),"out of range load");
    require(!bp_coord_finite(bp_coord_load(o,static_cast<BiprismCoordinatePlane>(8),0)),"unknown plane load");
    bad=o;bad.initialized_mask=0;
    require(!bp_coord_finite(bp_coord_load(bad,BiprismCoordinatePlane::Current,0)),"uninitialized load");
    p[1][0]=std::numeric_limits<float>::infinity();
    require(!bp_coord_finite(bp_coord_load(o,BiprismCoordinatePlane::Current,0)),"nonfinite low load");p[1][0]=0;
    const auto point=bp_coord_split(BiprismCoordinate3F64{100.+1e-6,-10.-1e-7,2.+1e-8});
    require(bp_coord_store_current(o,1,point),"complete pair store");
    const auto read=bp_coord_load(o,BiprismCoordinatePlane::Current,1);
    require(std::memcmp(&read,&point,sizeof(read))==0,"pair store readback bits");
    const auto before=p;
    auto invalid=point;invalid.z.lo=std::numeric_limits<float>::quiet_NaN();
    require(!bp_coord_store_current(o,0,invalid),"last component NaN refuses store");
    require(std::memcmp(&p,&before,sizeof(p))==0,"no partial invalid store");
    require(!bp_coord_store_current(o,3,point),"out of range store");
    require(!bp_coord_finite(bp_coord_split(std::numeric_limits<double>::infinity())),"infinite split");
    require(!bp_coord_finite(bp_coord_split(std::numeric_limits<double>::max())),"unrepresentable split");
    require(!bp_coord_finite(bp_coord_scale(BiprismCoordinatePairF32{std::numeric_limits<float>::max(),0},2.)),"scale range refusal");
    require(!bp_coord_finite(bp_coord_divide(BiprismCoordinatePairF32{1,0},0.)),"zero divisor refusal");
    require(!bp_coord_finite(bp_coord_add({1,0},std::numeric_limits<double>::quiet_NaN())),"NaN increment refusal");
    require(!bp_coord_finite(bp_coord_fma({1,0},std::numeric_limits<double>::max(),2.)),"product range refusal");
    const BiprismCoordinatePairF32 a={1e30f,1e-20f},b={1e30f,0.f};
    require(bp_coord_value(a)==bp_coord_value(b),"absolute reconstruction cancellation control");
    require(bp_coord_difference(a,b)==double(a.lo),"difference preserves low after cancellation");
    BiprismCoordinatePairF32 accumulated={100.f,0.f};
    for(unsigned i=0;i<40000;++i)accumulated=bp_coord_add(accumulated,1e-10);
    require(std::fabs(bp_coord_value(accumulated)-(100.+40000.*1e-10))<2e-9,
        "persistent low accumulation finite budget");
    std::printf("coordinate_owner_controls PASS %u\n",checks);
    std::printf("accumulation_40000 hi=%.9g lo=%.9g value=%.17g\n",accumulated.hi,accumulated.lo,bp_coord_value(accumulated));
}

int main(int argc,char** argv) {
    if(argc!=3){std::fprintf(stderr,"usage: %s input32b output64b\n",argv[0]);return 2;}
    require(std::fegetround()==FE_TONEAREST,"host RN rounding mode");
    ownership_controls();
    std::ifstream input(argv[1],std::ios::binary);
    std::ofstream output(argv[2],std::ios::binary|std::ios::trunc);
    require(bool(input)&&bool(output),"reference files");
    uint64_t count=0;
    for(Input in;input.read(reinterpret_cast<char*>(&in),sizeof(in));){
        const BiprismCoordinatePairF32 p={in.hi,in.lo};
        Output out{};out.next=bp_coord_fma(p,in.h,in.v);
        out.value=bp_coord_value(out.next);out.difference=bp_coord_difference(out.next,p);
        out.scaled=bp_coord_scale(out.next,in.factor);out.back=bp_coord_divide(out.scaled,in.factor);
        out.four=p;for(unsigned i=0;i<4;++i)out.four=bp_coord_fma(out.four,in.h,in.v);
        out.added=bp_coord_add(p,bp_coord_mul_rn(in.h,in.v));
        out.divided_value=bp_coord_value(bp_coord_divide(out.next,in.factor));
        output.write(reinterpret_cast<const char*>(&out),sizeof(out));++count;
    }
    require(input.eof()&&input.gcount()==0,"complete input records");
    require(bool(output),"complete output writes");
    std::printf("coordinate_actual_helper_records %llu\n",static_cast<unsigned long long>(count));
    std::printf("CPU helper execution only; native consumers and trajectory acceptance remain separate.\n");
    return 0;
}

#pragma once
// Native Bowyer-Watson construction with separately owned global scratch.
// Original atom coordinates/IDs are preserved. Temporary supervertices use u16.
// A successful export is an admitted, peeled Delaunay subdomain, not a claim that
// the exported tile integrates the entire original convex hull. The host owns
// selection/constraint closure and must admit every leaf before downstream work.
// CPU_REFERENCE exposes these same algorithm bodies for independent testing;
// it is not a substitute for execution of nhs_seed_bowyer_watson_256_v1 on GPU.
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>

#if defined(__CUDACC__)
#define BPBW_HD __host__ __device__ __forceinline__
#else
#define BPBW_HD inline
#endif

static constexpr uint32_t BPBW_VERTICES = 256u;
static constexpr uint32_t BPBW_TEMP_VERTICES = 260u;
static constexpr uint32_t BPBW_TET_CAPACITY = 4096u;
static constexpr uint32_t BPBW_FACE_SLOTS = 32768u;
static constexpr uint32_t BPBW_MAX_PEEL_PASSES = 256u;
static constexpr uint32_t BPBW_NONE = 0xffffffffu;

enum BiprismBwError : uint32_t {
    BPBW_OK=0u, BPBW_BAD_INPUT=1u, BPBW_UNCERTAIN_PREDICATE=2u,
    BPBW_TET_CAPACITY_EXCEEDED=3u, BPBW_FACE_CAPACITY_EXCEEDED=4u,
    BPBW_EMPTY_CAVITY=5u, BPBW_INVALID_COMPLEX=6u,
    BPBW_HULL_NOT_CLOSED=7u, BPBW_PEEL_BUDGET_UNREACHABLE=8u,
    BPBW_EXPORT_CAPACITY_EXCEEDED=9u, BPBW_NONFINITE=10u
};

struct BiprismBwPoint { double x,y,z; };
struct BiprismBwTet { uint16_t a,b,c,d; double radius2; };
struct BiprismBwFace { uint32_t key,count,owner0,owner1; };
static constexpr uint32_t BPBW_STATUS_MAGIC = 0x42574231u;
struct alignas(16) BiprismBwStatusFfi {
    uint32_t abi_magic, leaf_id, error, phase;
    uint32_t inserted_vertices, peak_tets;
    uint32_t unpeeled_tets, unpeeled_edges, final_tets, final_edges;
    uint32_t peel_passes, removed_tets, uncertain_predicates, super_attempt;
    uint32_t diagnostic_tet, diagnostic_vertex, published, reserved[3];
};
static_assert(sizeof(BiprismBwStatusFfi)==80, "BW status ABI");

struct alignas(128) BiprismBwScratchFfi {
    BiprismBwPoint points[BPBW_TEMP_VERTICES];
    BiprismBwTet tetrahedra[2][BPBW_TET_CAPACITY];
    BiprismBwFace faces[BPBW_FACE_SLOTS];
    uint32_t adjacency[BPBW_TET_CAPACITY][4];
    uint32_t order[BPBW_TET_CAPACITY];
    uint32_t remap[BPBW_TET_CAPACITY];
    uint32_t queue[BPBW_TET_CAPACITY];
    uint32_t seen[BPBW_TET_CAPACITY];
    uint32_t vertex_degree[BPBW_VERTICES];
    uint32_t boundary_vertex_degree[BPBW_VERTICES];
    uint32_t edge_degree[BPBW_VERTICES*BPBW_VERTICES];
    uint32_t boundary_edge_degree[BPBW_VERTICES*BPBW_VERTICES];
    uint16_t link_edges[BPBW_VERTICES*BPBW_VERTICES];
    uint32_t link_degree[BPBW_VERTICES];
    uint32_t link_present[BPBW_VERTICES];
    uint8_t active[BPBW_TET_CAPACITY];
    uint8_t bad[BPBW_TET_CAPACITY];
    uint32_t selected_ids[BPBW_VERTICES];
    // Serial-owner predicate workspace; never aliases scientific output state.
    double predicate_work[40];
};
static_assert(sizeof(BiprismBwScratchFfi)==1461760, "BW dedicated scratch ABI");
static_assert(alignof(BiprismBwScratchFfi)==128, "BW scratch alignment");

// Directed roundoff enclosures. GPU uses directed double instructions; host
// uses one representable outward step after each rounded basic operation.
// The filter returns only a certified strict sign; intervals containing zero
// refuse the attempt. This is not an adaptive exact-predicate implementation.
struct BpBwInterval { double lo,hi; };
BPBW_HD BpBwInterval bp_bw_exact(double x) { return {x,x}; }
BPBW_HD double bp_bw_lo_add(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rd(a,b);
#else
    volatile double v=a+b; return nextafter(v,-INFINITY);
#endif
}
BPBW_HD double bp_bw_hi_add(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_ru(a,b);
#else
    volatile double v=a+b; return nextafter(v,INFINITY);
#endif
}
BPBW_HD double bp_bw_lo_mul(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rd(a,b);
#else
    volatile double v=a*b; return nextafter(v,-INFINITY);
#endif
}
BPBW_HD double bp_bw_hi_mul(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_ru(a,b);
#else
    volatile double v=a*b; return nextafter(v,INFINITY);
#endif
}
BPBW_HD BpBwInterval bp_bw_add(BpBwInterval a,BpBwInterval b) {
    return {bp_bw_lo_add(a.lo,b.lo),bp_bw_hi_add(a.hi,b.hi)};
}
BPBW_HD BpBwInterval bp_bw_sub(BpBwInterval a,BpBwInterval b) {
    return {bp_bw_lo_add(a.lo,-b.hi),bp_bw_hi_add(a.hi,-b.lo)};
}
BPBW_HD BpBwInterval bp_bw_mul(BpBwInterval a,BpBwInterval b) {
    const double l0=bp_bw_lo_mul(a.lo,b.lo), l1=bp_bw_lo_mul(a.lo,b.hi);
    const double l2=bp_bw_lo_mul(a.hi,b.lo), l3=bp_bw_lo_mul(a.hi,b.hi);
    const double h0=bp_bw_hi_mul(a.lo,b.lo), h1=bp_bw_hi_mul(a.lo,b.hi);
    const double h2=bp_bw_hi_mul(a.hi,b.lo), h3=bp_bw_hi_mul(a.hi,b.hi);
    return {fmin(fmin(l0,l1),fmin(l2,l3)),fmax(fmax(h0,h1),fmax(h2,h3))};
}
BPBW_HD BpBwInterval bp_bw_square(BpBwInterval a) {
    const double l0=bp_bw_lo_mul(a.lo,a.lo), l1=bp_bw_lo_mul(a.hi,a.hi);
    const double h0=bp_bw_hi_mul(a.lo,a.lo), h1=bp_bw_hi_mul(a.hi,a.hi);
    return {(a.lo<=0.0 && a.hi>=0.0)?0.0:fmin(l0,l1),fmax(h0,h1)};
}
BPBW_HD BpBwInterval bp_bw_det3(
    BpBwInterval ax,BpBwInterval ay,BpBwInterval az,
    BpBwInterval bx,BpBwInterval by,BpBwInterval bz,
    BpBwInterval cx,BpBwInterval cy,BpBwInterval cz) {
    return bp_bw_add(bp_bw_sub(
        bp_bw_mul(ax,bp_bw_sub(bp_bw_mul(by,cz),bp_bw_mul(bz,cy))),
        bp_bw_mul(ay,bp_bw_sub(bp_bw_mul(bx,cz),bp_bw_mul(bz,cx)))),
        bp_bw_mul(az,bp_bw_sub(bp_bw_mul(bx,cy),bp_bw_mul(by,cx))));
}
BPBW_HD int bp_bw_sign(BpBwInterval a) {
    if (!isfinite(a.lo)||!isfinite(a.hi)) return 0;
    return a.lo>0.0?1:(a.hi<0.0?-1:0);
}
BPBW_HD void bp_bw_store_interval(BiprismBwScratchFfi& s,uint32_t index,BpBwInterval v) {
    volatile double* w=s.predicate_work;w[2*index]=v.lo;w[2*index+1]=v.hi;
}
BPBW_HD BpBwInterval bp_bw_load_interval(BiprismBwScratchFfi& s,uint32_t index) {
    volatile double* w=s.predicate_work;return {w[2*index],w[2*index+1]};
}
BPBW_HD BpBwInterval bp_bw_workspace_det3(BiprismBwScratchFfi& s,uint32_t a,uint32_t b,uint32_t c) {
    return bp_bw_det3(bp_bw_load_interval(s,a),bp_bw_load_interval(s,a+1),bp_bw_load_interval(s,a+2),
        bp_bw_load_interval(s,b),bp_bw_load_interval(s,b+1),bp_bw_load_interval(s,b+2),
        bp_bw_load_interval(s,c),bp_bw_load_interval(s,c+1),bp_bw_load_interval(s,c+2));
}
BPBW_HD void bp_bw_translated_row(BiprismBwScratchFfi& s,uint32_t row,uint32_t point,uint32_t origin,bool lift) {
    const BiprismBwPoint p=s.points[point],o=s.points[origin];
    const BpBwInterval x=bp_bw_sub(bp_bw_exact(p.x),bp_bw_exact(o.x));
    const BpBwInterval y=bp_bw_sub(bp_bw_exact(p.y),bp_bw_exact(o.y));
    const BpBwInterval z=bp_bw_sub(bp_bw_exact(p.z),bp_bw_exact(o.z));
    bp_bw_store_interval(s,row,x);bp_bw_store_interval(s,row+1,y);bp_bw_store_interval(s,row+2,z);
    if(lift)bp_bw_store_interval(s,row+3,bp_bw_add(bp_bw_add(bp_bw_square(x),bp_bw_square(y)),bp_bw_square(z)));
}
BPBW_HD int bp_bw_orientation(BiprismBwScratchFfi& s,uint32_t a,uint32_t b,uint32_t c,uint32_t d) {
    bp_bw_translated_row(s,0,b,a,false);bp_bw_translated_row(s,4,c,a,false);bp_bw_translated_row(s,8,d,a,false);
    return bp_bw_sign(bp_bw_workspace_det3(s,0,4,8));
}
BPBW_HD int bp_bw_insphere(BiprismBwScratchFfi& s,uint32_t a,uint32_t b,uint32_t c,uint32_t d,uint32_t e) {
    // Same ordered interval determinant as the retained initial implementation.
    // Intermediate storage reduces register liveness without changing arithmetic.
    bp_bw_translated_row(s,0,a,e,true);bp_bw_translated_row(s,4,b,e,true);
    bp_bw_translated_row(s,8,c,e,true);bp_bw_translated_row(s,12,d,e,true);
    bp_bw_store_interval(s,16,bp_bw_mul(bp_bw_load_interval(s,15),bp_bw_workspace_det3(s,0,4,8)));
    bp_bw_store_interval(s,16,bp_bw_sub(bp_bw_load_interval(s,16),bp_bw_mul(bp_bw_load_interval(s,11),bp_bw_workspace_det3(s,0,4,12))));
    bp_bw_store_interval(s,16,bp_bw_add(bp_bw_load_interval(s,16),bp_bw_mul(bp_bw_load_interval(s,7),bp_bw_workspace_det3(s,0,8,12))));
    bp_bw_store_interval(s,16,bp_bw_sub(bp_bw_load_interval(s,16),bp_bw_mul(bp_bw_load_interval(s,3),bp_bw_workspace_det3(s,4,8,12))));
    return bp_bw_sign(bp_bw_load_interval(s,16));
}
BPBW_HD uint16_t bp_bw_vertex(const BiprismBwTet& t,uint32_t i) {
    return i==0?t.a:(i==1?t.b:(i==2?t.c:t.d));
}
BPBW_HD bool bp_bw_contains(const BiprismBwTet& t,uint32_t a) {
    return t.a==a||t.b==a||t.c==a||t.d==a;
}
BPBW_HD void bp_bw_face(const BiprismBwTet& t,uint32_t f,uint16_t& a,uint16_t& b,uint16_t& c) {
    if(f==0){a=t.b;b=t.c;c=t.d;}else if(f==1){a=t.a;b=t.d;c=t.c;}
    else if(f==2){a=t.a;b=t.b;c=t.d;}else{a=t.a;b=t.c;c=t.b;}
}
BPBW_HD uint32_t bp_bw_edge(uint32_t a,uint32_t b) {
    return a<b?a*BPBW_VERTICES+b:b*BPBW_VERTICES+a;
}
BPBW_HD uint32_t bp_bw_face_key(uint16_t a,uint16_t b,uint16_t c) {
    uint16_t x;if(a>b){x=a;a=b;b=x;}if(b>c){x=b;b=c;c=x;}if(a>b){x=a;a=b;b=x;}
    return (uint32_t(a)<<18)|(uint32_t(b)<<9)|uint32_t(c);
}
BPBW_HD void bp_bw_clear_faces(BiprismBwScratchFfi& s) {
    for(uint32_t i=0;i<BPBW_FACE_SLOTS;++i)s.faces[i]={BPBW_NONE,0,BPBW_NONE,BPBW_NONE};
}
BPBW_HD uint32_t bp_bw_face_slot(BiprismBwScratchFfi& s,uint32_t key,bool create) {
    uint32_t slot=(key*2654435761u)&(BPBW_FACE_SLOTS-1u);
    for(uint32_t n=0;n<BPBW_FACE_SLOTS;++n){
        if(s.faces[slot].key==key)return slot;
        if(s.faces[slot].key==BPBW_NONE){if(create)s.faces[slot].key=key;return create?slot:BPBW_NONE;}
        slot=(slot+1u)&(BPBW_FACE_SLOTS-1u);
    }return BPBW_NONE;
}
BPBW_HD uint32_t bp_bw_fail(BiprismBwStatusFfi& r,uint32_t error,uint32_t t=BPBW_NONE,uint32_t v=BPBW_NONE) {
    r.error=error;r.diagnostic_tet=t;r.diagnostic_vertex=v;
    if(error==BPBW_UNCERTAIN_PREDICATE)++r.uncertain_predicates;
    return error;
}
BPBW_HD double bp_bw_radius2(const BiprismBwScratchFfi& s,const BiprismBwTet& t) {
    const BiprismBwPoint p=s.points[t.a],q=s.points[t.b],r=s.points[t.c],u=s.points[t.d];
    const double ax=q.x-p.x,ay=q.y-p.y,az=q.z-p.z;
    const double bx=r.x-p.x,by=r.y-p.y,bz=r.z-p.z;
    const double cx=u.x-p.x,cy=u.y-p.y,cz=u.z-p.z;
    const double ca=.5*(ax*ax+ay*ay+az*az),cb=.5*(bx*bx+by*by+bz*bz),cc=.5*(cx*cx+cy*cy+cz*cz);
    const double det=ax*(by*cz-bz*cy)-ay*(bx*cz-bz*cx)+az*(bx*cy-by*cx);
    const double x=(ca*(by*cz-bz*cy)+cb*(cy*az-cz*ay)+cc*(ay*bz-az*by))/det;
    const double y=(ca*(bz*cx-bx*cz)+cb*(cz*ax-cx*az)+cc*(az*bx-ax*bz))/det;
    const double z=(ca*(bx*cy-by*cx)+cb*(cx*ay-cy*ax)+cc*(ax*by-ay*bx))/det;
    return x*x+y*y+z*z;
}
BPBW_HD void bp_bw_boundary_change(BiprismBwScratchFfi& s,uint32_t key,int delta) {
    const uint32_t a=(key>>18)&511u,b=(key>>9)&511u,c=key&511u;
    if(delta>0){++s.boundary_vertex_degree[a];++s.boundary_vertex_degree[b];++s.boundary_vertex_degree[c];
        ++s.boundary_edge_degree[bp_bw_edge(a,b)];++s.boundary_edge_degree[bp_bw_edge(a,c)];++s.boundary_edge_degree[bp_bw_edge(b,c)];}
    else{--s.boundary_vertex_degree[a];--s.boundary_vertex_degree[b];--s.boundary_vertex_degree[c];
        --s.boundary_edge_degree[bp_bw_edge(a,b)];--s.boundary_edge_degree[bp_bw_edge(a,c)];--s.boundary_edge_degree[bp_bw_edge(b,c)];}
}
BPBW_HD uint32_t bp_bw_build_incidence(BiprismBwScratchFfi& s,BiprismBwStatusFfi& r,uint32_t n) {
    bp_bw_clear_faces(s);
    for(uint32_t v=0;v<256;++v){s.vertex_degree[v]=0;s.boundary_vertex_degree[v]=0;}
    for(uint32_t e=0;e<65536;++e){s.edge_degree[e]=0;s.boundary_edge_degree[e]=0;}
    for(uint32_t i=0;i<n;++i){
        if(!s.active[i])continue;const BiprismBwTet t=s.tetrahedra[0][i];
        for(uint32_t k=0;k<4;++k){++s.vertex_degree[bp_bw_vertex(t,k)];s.adjacency[i][k]=BPBW_NONE;}
        for(uint32_t a=0;a<4;++a)for(uint32_t b=a+1;b<4;++b)++s.edge_degree[bp_bw_edge(bp_bw_vertex(t,a),bp_bw_vertex(t,b))];
        for(uint32_t f=0;f<4;++f){uint16_t a,b,c;bp_bw_face(t,f,a,b,c);const uint32_t slot=bp_bw_face_slot(s,bp_bw_face_key(a,b,c),true);
            if(slot==BPBW_NONE)return bp_bw_fail(r,BPBW_FACE_CAPACITY_EXCEEDED,i);
            BiprismBwFace& z=s.faces[slot];if(z.count==0){z.count=1;z.owner0=(i<<2)|f;}
            else if(z.count==1){z.count=2;z.owner1=(i<<2)|f;s.adjacency[i][f]=z.owner0>>2;s.adjacency[z.owner0>>2][z.owner0&3]=i;}
            else return bp_bw_fail(r,BPBW_INVALID_COMPLEX,i);
        }
    }
    for(uint32_t f=0;f<BPBW_FACE_SLOTS;++f)if(s.faces[f].count==1)bp_bw_boundary_change(s,s.faces[f].key,1);
    return BPBW_OK;
}
BPBW_HD bool bp_bw_star_connected(BiprismBwScratchFfi& s,uint32_t n,uint32_t a,uint32_t b) {
    uint32_t first=BPBW_NONE,count=0;
    for(uint32_t i=0;i<n;++i){s.seen[i]=0;if(!s.active[i])continue;
        const BiprismBwTet t=s.tetrahedra[0][i];
        if((a==BPBW_NONE||bp_bw_contains(t,a))&&(b==BPBW_NONE||bp_bw_contains(t,b))){first=i;++count;}}
    if(first==BPBW_NONE)return false;
    uint32_t head=0,tail=1;s.queue[0]=first;s.seen[first]=1;
    while(head<tail){const uint32_t t=s.queue[head++];for(uint32_t f=0;f<4;++f){const uint32_t j=s.adjacency[t][f];
        if(j==BPBW_NONE||!s.active[j]||s.seen[j])continue;
        const BiprismBwTet z=s.tetrahedra[0][j];if((a==BPBW_NONE||bp_bw_contains(z,a))&&(b==BPBW_NONE||bp_bw_contains(z,b))){s.seen[j]=1;s.queue[tail++]=j;}}}
    return tail==count;
}
// Full simplicial link checks, using dedicated reusable global scratch. Edge
// links must be connected paths/cycles. Every vertex link must be a connected
// disk/sphere, including its local-manifold and boundary-cycle conditions.
BPBW_HD uint32_t bp_bw_validate_links(BiprismBwScratchFfi& s,BiprismBwStatusFfi& r,uint32_t n) {
    if(!bp_bw_star_connected(s,n,BPBW_NONE,BPBW_NONE))return bp_bw_fail(r,BPBW_INVALID_COMPLEX);
    for(uint32_t a=0;a<256;++a)for(uint32_t b=a+1;b<256;++b){if(!s.edge_degree[bp_bw_edge(a,b)])continue;
        for(uint32_t v=0;v<256;++v)s.link_degree[v]=0;
        for(uint32_t i=0;i<n;++i){if(!s.active[i])continue;const BiprismBwTet t=s.tetrahedra[0][i];
            if(!bp_bw_contains(t,a)||!bp_bw_contains(t,b))continue;
            for(uint32_t k=0;k<4;++k){const uint32_t v=bp_bw_vertex(t,k);if(v!=a&&v!=b)++s.link_degree[v];}}
        uint32_t ends=0;for(uint32_t v=0;v<256;++v){const uint32_t d=s.link_degree[v];if(d==1)++ends;else if(d>2)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,BPBW_NONE,a);}
        if((ends!=0&&ends!=2)||!bp_bw_star_connected(s,n,a,b))return bp_bw_fail(r,BPBW_INVALID_COMPLEX,BPBW_NONE,a);
    }
    for(uint32_t v=0;v<256;++v){
        if(!s.vertex_degree[v]||!bp_bw_star_connected(s,n,v,BPBW_NONE))return bp_bw_fail(r,BPBW_INVALID_COMPLEX,BPBW_NONE,v);
        for(uint32_t e=0;e<65536;++e)s.link_edges[e]=0;
        for(uint32_t a=0;a<256;++a){s.link_present[a]=0;s.link_degree[a]=0;}
        uint32_t nf=0;
        for(uint32_t i=0;i<n;++i){if(!s.active[i])continue;const BiprismBwTet t=s.tetrahedra[0][i];if(!bp_bw_contains(t,v))continue;++nf;
            for(uint32_t ka=0;ka<4;++ka){const uint32_t a=bp_bw_vertex(t,ka);if(a==v)continue;s.link_present[a]=1;
                for(uint32_t kb=ka+1;kb<4;++kb){const uint32_t b=bp_bw_vertex(t,kb);if(b!=v)++s.link_edges[bp_bw_edge(a,b)];}}}
        uint32_t nv=0,ne=0,nb=0,first=BPBW_NONE;
        for(uint32_t a=0;a<256;++a){nv+=s.link_present[a];for(uint32_t b=a+1;b<256;++b){const uint32_t c=s.link_edges[bp_bw_edge(a,b)];
            if(c>2)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,BPBW_NONE,v);if(c)++ne;
            if(c==1){++nb;++s.link_degree[a];++s.link_degree[b];first=a;}}}
        if(nb==0){if(int(nv)-int(ne)+int(nf)!=2)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,BPBW_NONE,v);}
        else{if(int(nv)-int(ne)+int(nf)!=1)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,BPBW_NONE,v);
            uint32_t boundary_vertices=0;for(uint32_t a=0;a<256;++a){s.seen[a]=0;if(s.link_degree[a]){if(s.link_degree[a]!=2)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,BPBW_NONE,v);++boundary_vertices;}}
            uint32_t head=0,tail=1;s.queue[0]=first;s.seen[first]=1;
            while(head<tail){const uint32_t a=s.queue[head++];for(uint32_t b=0;b<256;++b)if(a!=b&&!s.seen[b]&&s.link_edges[bp_bw_edge(a,b)]==1){s.seen[b]=1;s.queue[tail++]=b;}}
            if(tail!=boundary_vertices)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,BPBW_NONE,v);
        }
    }
    return BPBW_OK;
}
BPBW_HD uint32_t bp_bw_validate_geometry(BiprismBwScratchFfi& s,BiprismBwStatusFfi& r,uint32_t n,bool hull) {
    for(uint32_t i=0;i<n;++i){if(!s.active[i])continue;const BiprismBwTet t=s.tetrahedra[0][i];
        if(bp_bw_orientation(s,t.a,t.b,t.c,t.d)!=1)return bp_bw_fail(r,BPBW_UNCERTAIN_PREDICATE,i);
        for(uint32_t v=0;v<256;++v){if(bp_bw_contains(t,v))continue;const int sign=bp_bw_insphere(s,t.a,t.b,t.c,t.d,v);
            if(sign==0)return bp_bw_fail(r,BPBW_UNCERTAIN_PREDICATE,i,v);if(sign<0)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,i,v);}
        if(hull)for(uint32_t f=0;f<4;++f){if(s.adjacency[i][f]!=BPBW_NONE)continue;uint16_t a,b,c;bp_bw_face(t,f,a,b,c);
            for(uint32_t v=0;v<256;++v){if(v==a||v==b||v==c)continue;const int sign=bp_bw_orientation(s,a,b,c,v);
                if(sign==0)return bp_bw_fail(r,BPBW_UNCERTAIN_PREDICATE,i,v);if(sign>0)return bp_bw_fail(r,BPBW_HULL_NOT_CLOSED,i,v);}}
    }return BPBW_OK;
}
BPBW_HD bool bp_bw_covered_vertex(uint32_t face_mask,uint32_t vertex_index) {
    return (face_mask & ~(1u<<vertex_index))!=0;
}
BPBW_HD bool bp_bw_covered_edge(uint32_t face_mask,uint32_t a,uint32_t b) {
    return (face_mask & ~((1u<<a)|(1u<<b)))!=0;
}
BPBW_HD bool bp_bw_can_peel(const BiprismBwScratchFfi& s,uint32_t i) {
    const BiprismBwTet t=s.tetrahedra[0][i];uint32_t shared=0,exposed=0;
    for(uint32_t f=0;f<4;++f){if(s.adjacency[i][f]==BPBW_NONE)exposed|=1u<<f;else shared|=1u<<f;}
    if(!shared||!exposed)return false;
    for(uint32_t a=0;a<4;++a){const uint32_t v=bp_bw_vertex(t,a);
        if(s.vertex_degree[v]<=1)return false;
        if(s.vertex_degree[v]>1&&!bp_bw_covered_vertex(shared,a))return false;
        if(s.boundary_vertex_degree[v]>0&&!bp_bw_covered_vertex(exposed,a))return false;
        for(uint32_t b=a+1;b<4;++b){const uint32_t e=bp_bw_edge(v,bp_bw_vertex(t,b));
            if(s.edge_degree[e]>1&&!bp_bw_covered_edge(shared,a,b))return false;
            if(s.boundary_edge_degree[e]>0&&!bp_bw_covered_edge(exposed,a,b))return false;}}
    return true;
}
BPBW_HD uint32_t bp_bw_peel(BiprismBwScratchFfi& s,BiprismBwStatusFfi& r,uint32_t n,uint32_t target,uint32_t max_edges) {
    uint32_t live=n,edges=0;for(uint32_t e=0;e<65536;++e)if(s.edge_degree[e])++edges;
    r.unpeeled_tets=n;r.unpeeled_edges=edges;
    for(uint32_t i=0;i<n;++i){s.order[i]=i;s.tetrahedra[0][i].radius2=bp_bw_radius2(s,s.tetrahedra[0][i]);
        if(!isfinite(s.tetrahedra[0][i].radius2))return bp_bw_fail(r,BPBW_NONFINITE,i);}
    // Stable descending radius with original scratch ID as the secondary key.
    for(uint32_t i=1;i<n;++i){const uint32_t key=s.order[i];uint32_t j=i;
        while(j>0&&s.tetrahedra[0][s.order[j-1]].radius2<s.tetrahedra[0][key].radius2){s.order[j]=s.order[j-1];--j;}s.order[j]=key;}
    while(live>target||edges>max_edges){
        // Refuse before starting another complete ordered scan at the ceiling.
        if(r.peel_passes>=BPBW_MAX_PEEL_PASSES)return bp_bw_fail(r,BPBW_PEEL_BUDGET_UNREACHABLE);
        bool progress=false;++r.peel_passes;
        for(uint32_t j=0;j<n;++j){const uint32_t i=s.order[j];if(!s.active[i]||!bp_bw_can_peel(s,i))continue;
            const BiprismBwTet t=s.tetrahedra[0][i];s.active[i]=0;--live;++r.removed_tets;progress=true;
            for(uint32_t a=0;a<4;++a){--s.vertex_degree[bp_bw_vertex(t,a)];for(uint32_t b=a+1;b<4;++b){const uint32_t e=bp_bw_edge(bp_bw_vertex(t,a),bp_bw_vertex(t,b));if(--s.edge_degree[e]==0)--edges;}}
            for(uint32_t f=0;f<4;++f){uint16_t a,b,c;bp_bw_face(t,f,a,b,c);const uint32_t slot=bp_bw_face_slot(s,bp_bw_face_key(a,b,c),false);
                if(slot==BPBW_NONE||s.faces[slot].count==0)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,i);
                BiprismBwFace& z=s.faces[slot];const uint32_t opp=s.adjacency[i][f];
                if(z.count==1){bp_bw_boundary_change(s,z.key,-1);z.count=0;}
                else{bp_bw_boundary_change(s,z.key,1);z.count=1;z.owner0=(z.owner0>>2)==i?z.owner1:z.owner0;z.owner1=BPBW_NONE;
                    if(opp==BPBW_NONE)return bp_bw_fail(r,BPBW_INVALID_COMPLEX,i);
                    for(uint32_t of=0;of<4;++of)if(s.adjacency[opp][of]==i)s.adjacency[opp][of]=BPBW_NONE;}
                s.adjacency[i][f]=BPBW_NONE;
            }
            if(live<=target&&edges<=max_edges)break;
        }
        if(!progress)return bp_bw_fail(r,BPBW_PEEL_BUDGET_UNREACHABLE);
    }
    r.final_tets=live;r.final_edges=edges;return BPBW_OK;
}
// Called by one owner thread. Only dedicated scratch/status change until success.
BPBW_HD uint32_t bp_bw_construct(BiprismBwScratchFfi& s,BiprismBwStatusFfi& r,uint32_t target=960u,uint32_t max_edges=1536u) {
    const uint32_t leaf_identity=r.leaf_id;r={};r.abi_magic=BPBW_STATUS_MAGIC;r.leaf_id=leaf_identity;r.diagnostic_tet=BPBW_NONE;r.diagnostic_vertex=BPBW_NONE;
    if(target==0||target>1024||max_edges==0||max_edges>1536)return bp_bw_fail(r,BPBW_BAD_INPUT);
    BiprismBwPoint center={0,0,0};
    for(uint32_t v=0;v<256;++v){const BiprismBwPoint p=s.points[v];if(!isfinite(p.x)||!isfinite(p.y)||!isfinite(p.z))return bp_bw_fail(r,BPBW_BAD_INPUT,BPBW_NONE,v);center.x+=p.x/256.0;center.y+=p.y/256.0;center.z+=p.z/256.0;}
    double extent=0;for(uint32_t v=0;v<256;++v){const BiprismBwPoint p=s.points[v];extent=fmax(extent,fmax(fabs(p.x-center.x),fmax(fabs(p.y-center.y),fabs(p.z-center.z))));}
    if(!(extent>0)||!isfinite(extent))return bp_bw_fail(r,BPBW_BAD_INPUT);
    uint32_t real_n=0;
    for(uint32_t attempt=0;attempt<4;++attempt){
        r.error=0;r.diagnostic_tet=BPBW_NONE;r.diagnostic_vertex=BPBW_NONE;r.super_attempt=attempt;r.phase=1;double scale=64.0;for(uint32_t k=0;k<attempt;++k)scale*=4.0;
        const double span=extent*scale;
        s.points[256]={center.x+3*span,center.y,center.z-span};s.points[257]={center.x-3*span,center.y+3*span,center.z-span};
        s.points[258]={center.x-3*span,center.y-3*span,center.z-span};s.points[259]={center.x,center.y,center.z+4*span};
        BiprismBwTet initial={256,257,258,259,0};int ori=bp_bw_orientation(s,256,257,258,259);
        if(!ori)return bp_bw_fail(r,BPBW_UNCERTAIN_PREDICATE);if(ori<0){initial.b=258;initial.c=257;}s.tetrahedra[0][0]=initial;
        uint32_t n=1,bank=0;
        for(uint32_t v=0;v<256;++v){
            bp_bw_clear_faces(s);uint32_t bad_n=0;
            for(uint32_t i=0;i<n;++i){const BiprismBwTet t=s.tetrahedra[bank][i];const int sign=bp_bw_insphere(s,t.a,t.b,t.c,t.d,v);
                if(!sign)return bp_bw_fail(r,BPBW_UNCERTAIN_PREDICATE,i,v);s.bad[i]=sign<0;if(sign>=0)continue;++bad_n;
                for(uint32_t f=0;f<4;++f){uint16_t a,b,c;bp_bw_face(t,f,a,b,c);const uint32_t slot=bp_bw_face_slot(s,bp_bw_face_key(a,b,c),true);
                    if(slot==BPBW_NONE)return bp_bw_fail(r,BPBW_FACE_CAPACITY_EXCEEDED,i,v);BiprismBwFace& z=s.faces[slot];
                    if(z.count==0){z.count=1;z.owner0=(i<<2)|f;}else if(z.count==1){z.count=2;z.owner1=(i<<2)|f;}else return bp_bw_fail(r,BPBW_INVALID_COMPLEX,i,v);}}
            if(!bad_n)return bp_bw_fail(r,BPBW_EMPTY_CAVITY,BPBW_NONE,v);
            uint32_t boundary=0;for(uint32_t f=0;f<BPBW_FACE_SLOTS;++f)if(s.faces[f].count==1)++boundary;
            if(n-bad_n+boundary>BPBW_TET_CAPACITY)return bp_bw_fail(r,BPBW_TET_CAPACITY_EXCEEDED,BPBW_NONE,v);
            uint32_t next=0;for(uint32_t i=0;i<n;++i)if(!s.bad[i])s.tetrahedra[1-bank][next++]=s.tetrahedra[bank][i];
            for(uint32_t f=0;f<BPBW_FACE_SLOTS;++f){if(s.faces[f].count!=1)continue;const uint32_t owner=s.faces[f].owner0;uint16_t a,b,c;bp_bw_face(s.tetrahedra[bank][owner>>2],owner&3,a,b,c);
                const int sign=bp_bw_orientation(s,a,b,c,v);if(!sign)return bp_bw_fail(r,BPBW_UNCERTAIN_PREDICATE,owner>>2,v);
                if(sign<0){const uint16_t tmp=b;b=c;c=tmp;}s.tetrahedra[1-bank][next++]={a,b,c,uint16_t(v),0};}
            n=next;bank=1-bank;r.inserted_vertices=v+1;if(n>r.peak_tets)r.peak_tets=n;
        }
        real_n=0;for(uint32_t i=0;i<n;++i){const BiprismBwTet t=s.tetrahedra[bank][i];if(t.a<256&&t.b<256&&t.c<256&&t.d<256)s.tetrahedra[1-bank][real_n++]=t;}
        if(1-bank!=0)for(uint32_t i=0;i<real_n;++i)s.tetrahedra[0][i]=s.tetrahedra[1-bank][i];
        for(uint32_t i=0;i<real_n;++i)s.active[i]=1;
        r.phase=2;if(bp_bw_build_incidence(s,r,real_n))return r.error;
        if(bp_bw_validate_geometry(s,r,real_n,true)==BPBW_HULL_NOT_CLOSED){if(attempt<3)continue;return r.error;}
        if(r.error)return r.error;
        if(bp_bw_validate_links(s,r,real_n))return r.error;
        break;
    }
    r.phase=3;if(bp_bw_peel(s,r,real_n,target,max_edges))return r.error;
    r.phase=4;if(bp_bw_validate_links(s,r,real_n))return r.error;
    if(r.final_tets>1024||r.final_edges>1536)return bp_bw_fail(r,BPBW_EXPORT_CAPACITY_EXCEEDED);
    uint32_t exported=0;for(uint32_t i=0;i<real_n;++i)s.remap[i]=s.active[i]?exported++:BPBW_NONE;
    // Final selected tetrahedra are a subset of the certified initial empty-sphere
    // complex. Their geometry is unchanged by the combinatorial removal stage.
    r.phase=5;r.error=BPBW_OK;return BPBW_OK;
}

#if !defined(BIPRISM_BW_CPU_REFERENCE)
#include "nhs_biprism_simplicial_tile.cuh"

// ABI: coords pointer table[leaf_count], selected IDs[leaf_count][256]. Input
// buffers and scratch must be nonoverlapping and exclusively owned through
// completion. No output publication occurs for a refused leaf. The host must
// admit all leaf statuses before Phase2/v10; this is not a grid transaction.
extern "C" __global__ __launch_bounds__(256,2)
void nhs_seed_bowyer_watson_256_v1(
    const float* const* __restrict__ d_coords_by_leaf,
    const float* __restrict__ d_charges,
    const uint8_t* __restrict__ d_elem_classes,
    const uint32_t* __restrict__ d_selected_atom_ids,
    uint32_t n_atoms,
    ContactShellSimplicialStateFfi* __restrict__ d_tiles,
    BiprismBwScratchFfi* __restrict__ d_scratch,
    BiprismBwStatusFfi* __restrict__ d_status,
    uint32_t leaf_count,
    uint32_t target_tets
) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=leaf_count||blockDim.x!=256||!d_status)return;
    BiprismBwStatusFfi& r=d_status[leaf];
    if(tid==0){r={};r.abi_magic=BPBW_STATUS_MAGIC;r.leaf_id=leaf;r.diagnostic_tet=BPBW_NONE;r.diagnostic_vertex=BPBW_NONE;
        if(!d_coords_by_leaf||!d_charges||!d_elem_classes||!d_selected_atom_ids||!d_tiles||!d_scratch||n_atoms==0||n_atoms>0x01000000u||leaf_count==0||leaf_count>8)r.error=BPBW_BAD_INPUT;
        else if(!d_coords_by_leaf[leaf])r.error=BPBW_BAD_INPUT;}
    __syncthreads();if(r.error)return;
    BiprismBwScratchFfi& s=d_scratch[leaf];const uint32_t atom=d_selected_atom_ids[leaf*256u+tid];
    s.selected_ids[tid]=atom;
    if(atom>=n_atoms||d_elem_classes[atom]>=8u)atomicCAS(&r.error,0u,uint32_t(BPBW_BAD_INPUT));
    else{const float* xyz=d_coords_by_leaf[leaf];const float x=xyz[atom*3u],y=xyz[atom*3u+1u],z=xyz[atom*3u+2u];
        s.points[tid]={double(x),double(y),double(z)};if(!isfinite(x)||!isfinite(y)||!isfinite(z)||!isfinite(d_charges[atom]))atomicCAS(&r.error,0u,uint32_t(BPBW_BAD_INPUT));}
    __syncthreads();if(r.error)return;
    if(tid==0){for(uint32_t a=0;a<256&&r.error==0;++a)for(uint32_t b=0;b<a;++b)if(s.selected_ids[a]==s.selected_ids[b]){r.error=BPBW_BAD_INPUT;r.diagnostic_vertex=a;break;}
        if(r.error==0)bp_bw_construct(s,r,target_tets,TILE_MAX_EDGES);}
    __syncthreads();if(r.error)return;
    ContactShellSimplicialStateFfi& tile=d_tiles[leaf];
    // Assign edge IDs and compact topology before any parallel state initializer.
    if(tid==0){uint32_t edge=0;for(uint32_t a=0;a<256;++a)for(uint32_t b=a+1;b<256;++b){const uint32_t e=bp_bw_edge(a,b);if(s.edge_degree[e]){s.edge_degree[e]=edge+1;tile.edge_u8[edge][0]=uint8_t(a);tile.edge_u8[edge][1]=uint8_t(b);++edge;}}
        for(uint32_t i=0;i<r.unpeeled_tets;++i){if(!s.active[i])continue;const uint32_t t=s.remap[i];const BiprismBwTet z=s.tetrahedra[0][i];
            for(uint32_t f=0;f<4;++f){tile.tet_v[t][f]=uint8_t(bp_bw_vertex(z,f));const uint32_t j=s.adjacency[i][f];tile.tet_opp_tet[t][f]=j==BPBW_NONE?0xffffu:uint16_t(s.remap[j]);}
            uint32_t k=0;for(uint32_t a=0;a<4;++a)for(uint32_t b=a+1;b<4;++b)tile.tet_edges[t][k++]=uint16_t(s.edge_degree[bp_bw_edge(bp_bw_vertex(z,a),bp_bw_vertex(z,b))]-1);}
    }
    const su2_mat identity={{1,0},{0,0},{0,0},{1,0}};
    tile.global_atom_idx[tid]=PACK_ATOM_ELEM(atom,d_elem_classes[atom]);
    const BiprismBwPoint p=s.points[tid];const float q=d_charges[atom];tile.ref_pos_q[tid]=make_float4(float(p.x),float(p.y),float(p.z),q);
    tile.phi_dec[tid]=q*0.07957747f;tile.star0_vol[tid]=1;tile.psi_vertex[tid][0]={1,0};tile.psi_vertex[tid][1]={0,0};tile.v_leaf_vertex[tid]=identity;
    for(uint32_t e=tid;e<r.final_edges;e+=256){tile.u_spatial_edge[e]=identity;tile.star1_wt[e]=0;tile.grad_star1_u[e]={0,0,0};tile.s_ovlp_edges[e]=0;tile.grad_s_edges[e]={0,0,0};
        tile.h_cov_edges[e][0]=0;tile.h_cov_edges[e][1]=0;tile.h_cov_edges[e][2]=0;tile.h_cov_edges[e][3]=0;
        tile.grad_h_diag_u[e][0]={0,0,0};tile.grad_h_diag_u[e][1]={0,0,0};tile.grad_h_cpl_u[e]={0,0,0};}
    __syncthreads();
    if(tid==0){tile.num_active_tets=r.final_tets;tile.num_active_edges=r.final_edges;tile.pachner_flips_step=0;tile.active_pocket_id=s.selected_ids[0];__threadfence();r.published=1;}
}

// Explicit precision-profile export. The topology/predicate/peeling algorithm is
// unchanged; its FP64 points now retain the authoritative current hi+lo state.
extern "C" __global__ __launch_bounds__(256,2)
void nhs_seed_bowyer_watson_256_compensated_v1(
    const float* const* __restrict__ d_coords_by_leaf,
    const float* __restrict__ d_charges,
    const uint8_t* __restrict__ d_elem_classes,
    const uint32_t* __restrict__ d_selected_atom_ids,
    uint32_t n_atoms,
    ContactShellSimplicialStateFfi* __restrict__ d_tiles,
    BiprismBwScratchFfi* __restrict__ d_scratch,
    BiprismBwStatusFfi* __restrict__ d_status,
    uint32_t leaf_count,
    uint32_t target_tets,
    const BiprismCoordinateOwnerFfi* __restrict__ d_coordinate_owners
) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=leaf_count||blockDim.x!=256||!d_status)return;
    BiprismBwStatusFfi& r=d_status[leaf];
    if(tid==0){r={};r.abi_magic=BPBW_STATUS_MAGIC;r.leaf_id=leaf;r.diagnostic_tet=BPBW_NONE;r.diagnostic_vertex=BPBW_NONE;
        if(!d_coords_by_leaf||!d_charges||!d_elem_classes||!d_selected_atom_ids||!d_tiles||!d_scratch||n_atoms==0||n_atoms>0x01000000u||leaf_count==0||leaf_count>8)r.error=BPBW_BAD_INPUT;
        else if(!d_coords_by_leaf[leaf]||!d_coordinate_owners||(reinterpret_cast<uintptr_t>(d_coordinate_owners)&15u)!=0u)r.error=BPBW_BAD_INPUT;
        else {const auto* owner=&d_coordinate_owners[leaf];
            if(!bp_coord_owner_valid(owner,leaf,n_atoms,owner->generation,BP_COORD_ATOMIC_BOHR,
                BP_COORD_ALL_INITIALIZED,d_coords_by_leaf[leaf]))r.error=BPBW_BAD_INPUT;}}
    __syncthreads();if(r.error)return;
    BiprismBwScratchFfi& s=d_scratch[leaf];const uint32_t atom=d_selected_atom_ids[leaf*256u+tid];
    s.selected_ids[tid]=atom;
    if(atom>=n_atoms||d_elem_classes[atom]>=8u)atomicCAS(&r.error,0u,uint32_t(BPBW_BAD_INPUT));
    else{const auto& owner=d_coordinate_owners[leaf];
        const auto current=bp_coord_load(owner,BiprismCoordinatePlane::Current,atom);
        const auto reference=bp_coord_load(owner,BiprismCoordinatePlane::MeshReference,atom);
        const auto same=bp_coord_difference(current,reference),x=bp_coord_value(current);
        s.points[tid]={x.x,x.y,x.z};
        if(!bp_coord_finite(current)||!bp_coord_finite(reference)||same.x!=0.||same.y!=0.||same.z!=0.||!isfinite(d_charges[atom]))
            atomicCAS(&r.error,0u,uint32_t(BPBW_BAD_INPUT));}
    __syncthreads();if(r.error)return;
    if(tid==0){for(uint32_t a=0;a<256&&r.error==0;++a)for(uint32_t b=0;b<a;++b)if(s.selected_ids[a]==s.selected_ids[b]){r.error=BPBW_BAD_INPUT;r.diagnostic_vertex=a;break;}
        if(r.error==0)bp_bw_construct(s,r,target_tets,TILE_MAX_EDGES);}
    __syncthreads();if(r.error)return;
    ContactShellSimplicialStateFfi& tile=d_tiles[leaf];
    // Assign edge IDs and compact topology before any parallel state initializer.
    if(tid==0){uint32_t edge=0;for(uint32_t a=0;a<256;++a)for(uint32_t b=a+1;b<256;++b){const uint32_t e=bp_bw_edge(a,b);if(s.edge_degree[e]){s.edge_degree[e]=edge+1;tile.edge_u8[edge][0]=uint8_t(a);tile.edge_u8[edge][1]=uint8_t(b);++edge;}}
        for(uint32_t i=0;i<r.unpeeled_tets;++i){if(!s.active[i])continue;const uint32_t t=s.remap[i];const BiprismBwTet z=s.tetrahedra[0][i];
            for(uint32_t f=0;f<4;++f){tile.tet_v[t][f]=uint8_t(bp_bw_vertex(z,f));const uint32_t j=s.adjacency[i][f];tile.tet_opp_tet[t][f]=j==BPBW_NONE?0xffffu:uint16_t(s.remap[j]);}
            uint32_t k=0;for(uint32_t a=0;a<4;++a)for(uint32_t b=a+1;b<4;++b)tile.tet_edges[t][k++]=uint16_t(s.edge_degree[bp_bw_edge(bp_bw_vertex(z,a),bp_bw_vertex(z,b))]-1);}
    }
    const su2_mat identity={{1,0},{0,0},{0,0},{1,0}};
    tile.global_atom_idx[tid]=PACK_ATOM_ELEM(atom,d_elem_classes[atom]);
    // The dedicated full-N mesh reference is host-owned and immutable until
    // reseed. The tile retains only its high projection; every compensated
    // geometry consumer joins the same owner's reference low plane.
    const auto reference=bp_coord_load(d_coordinate_owners[leaf],BiprismCoordinatePlane::MeshReference,atom);
    const float q=d_charges[atom];tile.ref_pos_q[tid]=make_float4(reference.x.hi,reference.y.hi,reference.z.hi,q);
    tile.phi_dec[tid]=q*0.07957747f;tile.star0_vol[tid]=1;tile.psi_vertex[tid][0]={1,0};tile.psi_vertex[tid][1]={0,0};tile.v_leaf_vertex[tid]=identity;
    for(uint32_t e=tid;e<r.final_edges;e+=256){tile.u_spatial_edge[e]=identity;tile.star1_wt[e]=0;tile.grad_star1_u[e]={0,0,0};tile.s_ovlp_edges[e]=0;tile.grad_s_edges[e]={0,0,0};
        tile.h_cov_edges[e][0]=0;tile.h_cov_edges[e][1]=0;tile.h_cov_edges[e][2]=0;tile.h_cov_edges[e][3]=0;
        tile.grad_h_diag_u[e][0]={0,0,0};tile.grad_h_diag_u[e][1]={0,0,0};tile.grad_h_cpl_u[e]={0,0,0};}
    __syncthreads();
    if(tid==0){tile.num_active_tets=r.final_tets;tile.num_active_edges=r.final_edges;tile.pachner_flips_step=0;tile.active_pocket_id=s.selected_ids[0];__threadfence();r.published=1;}
}
#endif
#undef BPBW_HD

#include "nhs_biprism_potential_precision.cuh"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
static unsigned checks=0;
static void require(bool yes,const char* name){++checks;if(!yes){std::fprintf(stderr,"FAIL %s\n",name);std::exit(1);}}
int main(){
    alignas(128) std::array<double,256> accepted{},prepared{};
    alignas(128) unsigned char tile[128]{};
    BiprismCoordinateOwnerFfi coordinate{};coordinate.abi_magic=BP_COORDINATE_MAGIC;
    coordinate.abi_version=BP_COORDINATE_VERSION;coordinate.leaf_id=3;coordinate.generation=17;coordinate.unit_frame=BP_COORD_ATOMIC_BOHR;
    BiprismPotentialOwnerFfi owner{};owner.abi_magic=BP_POTENTIAL_MAGIC;owner.version=1;owner.leaf_id=3;
    owner.allocation_generation=17;owner.topology_generation=29;owner.tile_address=uint64_t(reinterpret_cast<uintptr_t>(tile));
    owner.coordinate_owner_address=uint64_t(reinterpret_cast<uintptr_t>(&coordinate));
    owner.accepted_phi=accepted.data();owner.prepared_phi=prepared.data();
    auto valid=[&](const BiprismPotentialOwnerFfi& o){return bp_potential_owner_valid(&o,3,tile,&coordinate,29);};
    require(valid(owner)&&bp_potential_cold_valid(owner),"cold allocation identity");
    require(!bp_potential_input_valid(owner,0,72,true),"cold never admitted to propagation");
    for(unsigned i=0;i<256;++i)accepted[i]=1.+double(i)/512.+std::ldexp(1.,-40);
    bp_potential_publish_cold(owner,0,71);
    require(bp_potential_accepted_valid(owner,0,71),"cold initialization retains configured clockN");
    require(bp_potential_prepare_valid(owner,0,72),"next step operator preparation");
    require(!bp_potential_prepare_valid(owner,4,72)&&!bp_potential_prepare_valid(owner,0,71),"stale count/step refuses");
    require(!bp_potential_input_valid(owner,0,72,true)&&bp_potential_input_valid(owner,0,72,false),"explicit diagnostic unchanged operator only");
    const auto old_accepted=accepted;const auto old_metadata=owner;
    for(unsigned i=0;i<256;++i)prepared[i]=accepted[i]+std::ldexp(1.,-39);
    bp_potential_publish_prepared(owner,0,72);
    require(std::memcmp(accepted.data(),old_accepted.data(),2048)==0,"Phase2 leaves accepted bank unchanged");
    require(owner.accepted_step==old_metadata.accepted_step&&owner.accepted_committed_substeps==old_metadata.accepted_committed_substeps
        &&owner.accepted_operator_generation==old_metadata.accepted_operator_generation,"Phase2 leaves accepted identity unchanged");
    require(bp_potential_prepared_valid(owner,0,72)&&bp_potential_input_valid(owner,0,72,true),"prepared current operator admitted");
    require(bp_potential_input_phi(owner)==prepared.data(),"prepared fullbank selected");
    require(!bp_potential_prepare_valid(owner,0,72),"prepared operator cannot be silently overwritten");
    for(unsigned i=0;i<256;++i){
        require(bp_potential_projection_value_valid(owner,i,float(prepared[i])),"finite exact RN32 projection");
        require(!bp_potential_projection_value_valid(owner,i,std::nextafter(float(prepared[i]),INFINITY)),"altered projection refuses");
    }
    const auto prepared_metadata=owner;
    // A rejected main has no publication call and cannot change the banks.
    require(!bp_potential_input_valid(owner,0,73,true)&&owner.accepted_step==71,"wrong attempted body preserves accepted clock");
    accepted=prepared;bp_potential_publish_main_commit(owner,4,72);
    require(bp_potential_accepted_valid(owner,4,72)&&bp_potential_input_phi(owner)==accepted.data(),"main fullbank commit publishes once");
    require(owner.accepted_operator_generation==prepared_metadata.prepared_operator_generation,"prepared operator generation committed");
    require(!bp_potential_input_valid(owner,4,72,true)&&bp_potential_prepare_valid(owner,4,73),"replay refused, next preparation allowed");
    require(accepted[0]!=double(float(accepted[0])),"accepted low potential bits retained");
    auto bad=owner;bad.prepared_phi=accepted.data();require(!valid(bad),"bank alias refused");
    bad=owner;bad.prepared_phi=accepted.data()+1;require(!valid(bad),"partial bank overlap refused");
    bad=owner;++bad.allocation_generation;require(!valid(bad),"generation mismatch refused");
    bad=owner;++bad.topology_generation;require(!valid(bad),"topology mismatch refused");
    bad=owner;bad.tile_address+=128;require(!valid(bad),"foreign tile refused");
    bad=owner;bad.coordinate_owner_address+=96;require(!valid(bad),"foreign coordinate owner refused");
    bad=owner;++bad.version;require(!valid(bad),"unknown version refused");
    bad=owner;bad.state=3;require(!valid(bad),"unknown lifecycle refused");
    bad=owner;bad.reserved0=1;require(!valid(bad),"reserved word refused");
    bad=owner;bad.reserved1=1;require(!valid(bad),"reserved tail refused");
    bad=owner;bad.projection_operator_generation++;require(!bp_potential_accepted_valid(bad,4,72),"stale projection generation refused");
    accepted[0]=INFINITY;require(!bp_potential_projection_value_valid(owner,0,INFINITY),"nonfinite full potential refused");
    std::printf("PASS %u actual potential-owner CPU lifecycle/identity/projection controls; no native execution claim\n",checks);
}

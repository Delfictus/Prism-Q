#include "nhs_biprism_motion_ownership.cuh"
#include <cassert>
#include <cstdio>
struct Cluster { int central_atom; int hydrogen_atoms[3]; int n_hydrogens; };
int main() {
    unsigned checks=0;
    for(unsigned x=0;x<256;++x) {
        assert(prism_biprism_ownership_valid_v1(uint8_t(x)) == (x==0||x==1||x==3)); ++checks;
    }
    Cluster a{0,{1,2,-1},2};
    uint8_t m[4]={0,0,0,0};
    assert(prism_biprism_constraint_action_v1(m,4,a)==0); ++checks;
    m[0]=1; assert(prism_biprism_constraint_action_v1(m,4,a)==0); ++checks;
    m[1]=1; m[2]=1; assert(prism_biprism_constraint_action_v1(m,4,a)==0); ++checks;
    m[0]=3; assert(prism_biprism_constraint_action_v1(m,4,a)==-1); ++checks;
    m[1]=3; m[2]=3; assert(prism_biprism_constraint_action_v1(m,4,a)==1); ++checks;
    // A second cluster sharing atom2 forces exclusion closure through atom3.
    Cluster b{2,{3,-1,-1},1};
    assert(prism_biprism_constraint_action_v1(m,4,b)==-1); ++checks;
    m[3]=3; assert(prism_biprism_constraint_action_v1(m,4,b)==1); ++checks;
    m[1]=2; assert(prism_biprism_constraint_action_v1(m,4,a)==-1); ++checks;
    m[1]=4; assert(prism_biprism_constraint_action_v1(m,4,a)==-1); ++checks;
    m[1]=3; assert(prism_biprism_constraint_action_v1(nullptr,4,a)==-1); ++checks;
    assert(prism_biprism_constraint_action_v1(m,0,a)==-1); ++checks;
    a.central_atom=4; assert(prism_biprism_constraint_action_v1(m,4,a)==-1); ++checks;
    a.central_atom=0; a.hydrogen_atoms[0]=4;
    assert(prism_biprism_constraint_action_v1(m,4,a)==-1); ++checks;
    a.hydrogen_atoms[0]=1; a.n_hydrogens=4;
    assert(prism_biprism_constraint_action_v1(m,4,a)==-1); ++checks;
    assert(!prism_biprism_external_motion_v1(0)); ++checks;
    assert(prism_biprism_external_motion_v1(1)); ++checks;
    assert(prism_biprism_external_motion_v1(3)); ++checks;
    std::printf("ownership_controls_passed=%u gpu_execution=0\n",checks);
}

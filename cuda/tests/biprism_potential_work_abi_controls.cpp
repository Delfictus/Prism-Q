// CPU controls of the actual BPW5/BME6 ABI and pure energy helpers. This does
// not execute CUDA or establish collective publication/trajectory acceptance.
#define BIPRISM_PHASE2_WORK_ABI_ONLY
#define BIPRISM_MIXED_ENERGY_ABI_ONLY
#include "nhs_biprism_phase2_work.cuh"
#include "nhs_biprism_mixed_energy.cuh"
#include <cstdio>
#include <cstring>
#include <stdexcept>

static void require(bool value,const char* label) {
    if(!value)throw std::runtime_error(label);
}
int main() {
    try {
        static_assert(sizeof(BiprismPhase2VertexWorkFfi)==256);
        static_assert(offsetof(BiprismPhase2VertexWorkFfi,phi_entry)==216);
        static_assert(sizeof(BiprismPhase2WorkLedgerFfi)==240);
        static_assert(offsetof(BiprismPhase2WorkLedgerFfi,entry_refinement_delta)==232);
        static_assert(sizeof(BiprismMixedVertexFfi)==336);
        static_assert(offsetof(BiprismMixedVertexFfi,phi_entry)==240);
        static_assert(offsetof(BiprismMixedVertexFfi,metric_f64)==256);
        const double low=0x1p-40;
        BiprismPhase2VertexWorkFfi u={},v={};
        u.phi_entry=.75;u.phi_before=.75+low;u.phi_same_halo=.5+low;u.phi_after=.5+2.*low;
        v.phi_entry=.25;v.phi_before=.25;v.phi_same_halo=.25;v.phi_after=.25+low;
        BiprismPhase2EdgeWorkFfi e={};e.live_before=2.f;e.live_after=4.f;
        biprism_phase2_edge_energy(e,u,v);
        require(e.energy_before==-((.5+low)*(.5+low)),"FP64 old functional was rounded");
        require(e.energy_after==-2.*((.25+low)*(.25+low)),"FP64 new functional was rounded");
        require(e.operator_cross_work==-(.5+low)*(.25+low),"FP64 operator cross work was rounded");
        require(e.energy_before!=-.25,"sub-FP32 potential difference was discarded");
        require(float(u.phi_before)==float(u.phi_entry),"control did not isolate a sub-FP32 difference");
        const auto bytes=u;BiprismPhase2VertexWorkFfi copy={};std::memcpy(&copy,&bytes,sizeof(copy));
        require(copy.phi_before==u.phi_before&&copy.phi_entry==u.phi_entry,"BPW5 full potential storage");
        // Exact dyadic stationary two-node operator change with fixed halo.
        u={};v={};u.charge=1.;u.phi_before=.75;u.phi_same_halo=.5;u.phi_after=.625;
        v.phi_before=.25;v.phi_same_halo=.25;v.phi_after=.375;
        biprism_phase2_edge_energy(e,u,v);
        BiprismPhase2WorkLedgerFfi ledger={};auto* t=ledger.totals;
        t[0]=u.charge*u.phi_before+e.energy_before;
        t[1]=u.charge*u.phi_same_halo+e.energy_same_halo;
        t[2]=u.charge*u.phi_after+e.energy_after;
        t[4]=e.operator_cross_work;
        const double flux_m=4.*(v.phi_same_halo-u.phi_same_halo);
        const double flux_1=4.*(v.phi_after-u.phi_after);
        t[7]=-.5*(flux_m+flux_1)*(v.phi_after-v.phi_same_halo);
        biprism_phase2_finish_totals(t);
        require(biprism_phase2_identities_admitted(t)&&t[13]==0.&&t[14]==0.,"fixed-halo/operator identities");
        const double declared=t[12];ledger.entry_refinement_delta=.125;
        biprism_phase2_finish_totals(t);
        require(t[12]==declared,"refinement diagnostic became physical work");
        t[13]=2e-6;
        require(!biprism_phase2_identities_admitted(t),"unchanged identity gate was relaxed");
        BiprismMixedVertexFfi m={};m.phi_entry=.75;m.phi=.75+low;
        require(m.phi!=m.phi_entry&&float(m.phi)==float(m.phi_entry),"BME6 full potential storage");
        BiprismMixedSampleFfi sample={};sample.values[0]=3.;sample.values[1]=2.;sample.values[2]=1.;
        sample.values[3]=.5;sample.values[4]=1.;sample.values[6]=.75;sample.values[19]=.125;
        biprism_mixed_energy_values(sample,.25);
        const double energy=sample.values[11],correction=sample.values[10];
        sample.values[19]=-3.;biprism_mixed_energy_values(sample,.25);
        require(sample.values[11]==energy&&sample.values[10]==correction,"mixed refinement diagnostic received credit");
        require(!biprism_mixed_energy_admitted(1.000001e-6,0.),"mixed body threshold was relaxed");
        require(!biprism_mixed_energy_admitted(0.,1.000001e-6),"mixed run threshold was relaxed");
        require(biprism_mixed_energy_admitted(1e-6,-1e-6),"unchanged mixed energy boundary");
        std::puts("actual_BPW5_BME6_ABI_and_pure_work_controls=passed CUDA_execution=0 trajectory_acceptance=0");
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}

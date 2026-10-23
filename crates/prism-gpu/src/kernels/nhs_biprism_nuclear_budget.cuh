#ifndef NHS_BIPRISM_NUCLEAR_BUDGET_CUH
#define NHS_BIPRISM_NUCLEAR_BUDGET_CUH
// Private nonlinear trials and full residual evaluations are distinct work.
// Native155 crosses the fixed-domain second fold with contracting actual
// corrections, then exhausts1536 evaluations/252 trials before original h.
// Its retained-anchor full-response reference needs1159+36 more evaluations
// and180+6 trials to return to/close the original equation.3072/512 bounds
// retain that measured work plus independent native action-check headroom.
// These bounds change resource work only; every original physical gate remains.
static constexpr unsigned BIPRISM_MAX_NUCLEAR_TRIALS=512u;
static constexpr unsigned BIPRISM_MAX_NUCLEAR_EVALUATIONS=3072u;
#endif

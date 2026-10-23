#include <cassert>
#include <cstdio>
#include "protocol_state.cuh"

static void check_cadence(int inner) {
    ProtocolState state{};
    state.fused_inner_steps = inner;
    int physical = 0;
    int observations = 0;
    int builds = 0;
    int clears = 0;
    int buffers[2] = {0, 0};
    const int chunks[] = {500, 500, 7};
    for (int chunk : chunks) {
        for (int step = 0; step < chunk; ++step) {
            const int prior_phase = state.coupling_phase;
            const int prior_a = buffers[0], prior_b = buffers[1];
            state.fused_step_counter = prism_protocol_next_observation_counter(
                state.fused_step_counter, state.fused_inner_steps);
            ++physical;
            const bool due = prism_protocol_observation_due(&state);
            assert(due == (physical % inner == 0));
            if (due) {
                state.coupling_phase = 1 - state.coupling_phase;
                ++observations;
                // Prior observation's write side is this observation's read side.
                assert(buffers[state.coupling_phase] == observations - 1);
                buffers[1 - state.coupling_phase] = observations;
                ++builds;
                buffers[state.coupling_phase] = 0;
                ++clears;
            } else {
                assert(state.coupling_phase == prior_phase);
                assert(buffers[0] == prior_a && buffers[1] == prior_b);
            }
            assert(observations == physical / inner);
            assert(builds == observations && clears == observations);
            assert(state.coupling_phase == (observations % 2));
        }
        // No reset or forced observer at either a full chunk or exact tail.
        assert(state.fused_step_counter == physical % inner);
    }
    assert(physical == 1007);
    if (inner > 1) assert(!prism_protocol_observation_due(&state));
    // Existing initialization and replica-reset start the next epoch at zero.
    state.fused_step_counter = 0;
    for (int step = 1; step <= inner; ++step) {
        state.fused_step_counter = prism_protocol_next_observation_counter(
            state.fused_step_counter, inner);
        assert(prism_protocol_observation_due(&state) == (step == inner));
    }
}

int main() {
    static_assert(sizeof(ProtocolState) == 716, "ProtocolState ABI changed");
    check_cadence(1);
    check_cadence(6);
    check_cadence(12);
    std::puts("protocol observation cadence N1/6/12, chunk500+500+tail7, buffers, reset: PASS");
}

#include <cassert>
#include <cmath>
#include <cstdio>

// Execute the actual Director bodies on CPU; this does not qualify CUDA execution.
#define __global__
#define __device__
#define __forceinline__ inline
static const struct { unsigned x; } threadIdx{0}, blockIdx{0};
#include "protocol_director.cu"

static ProtocolState state_for(int ramp_down_steps, bool warm_tail) {
    ProtocolState state{};
    state.total_steps = 1007;
    state.start_temp = 83.0f;
    state.end_temp = 317.0f;
    state.cold_hold_end = 10;
    state.ramp_end = 30;
    state.warm_hold_end = 60;
    state.ramp_down_end = 60 + ramp_down_steps;
    state.gamma_base = 1.0f;
    state.dt = state.base_dt = 0.002f;
    state.adaptive_dt_enabled = 1;
    state.fused_inner_steps = 6;
    state.steering_uv_boost = 1.0f;
    state.steering_flags = 0x8 | (warm_tail ? PRISM_WARM_TAIL_WITHOUT_COOLING_FLAG : 0);
    return state;
}

static float expected(int step, int ramp_down_steps, bool warm_tail) {
    if (step < 10) return 83.0f;
    if (step < 30) return 83.0f + (step - 10) / 20.0f * 234.0f;
    if (step < 60) return 317.0f;
    if (step < 60 + ramp_down_steps) return 317.0f - (step - 60) / float(ramp_down_steps) * 234.0f;
    return warm_tail ? 317.0f : 83.0f;
}

static void check(int ramp_down_steps, bool warm_tail) {
    auto ordinary = state_for(ramp_down_steps, warm_tail);
    auto graph = ordinary;
    const int initial_flags = ordinary.steering_flags;
    for (int step = 1; step <= 1007; ++step) {
        protocol_director(&ordinary);
        protocol_director_graph(&graph, 0, 0);
        const float temperature = expected(step, ramp_down_steps, warm_tail);
        assert(std::fabs(ordinary.current_temperature - temperature) < 0.0001f);
        assert(std::fabs(graph.current_temperature - temperature) < 0.0001f);
        assert(ordinary.current_step == unsigned(step));
        assert(graph.current_step == unsigned(step));
        assert(ordinary.steering_flags == initial_flags);
        assert(graph.steering_flags == initial_flags);
        const bool hold = step < 10 || (step >= 30 && step < 60) || step >= 60 + ramp_down_steps;
        const float dt = hold ? 0.002f : 0.001f;
        assert(ordinary.dt == dt && graph.dt == dt);
    }
}

int main() {
    static_assert(sizeof(ProtocolState) == 716, "ProtocolState ABI changed");
    check(0, true);   // Standard: warm tail, including the exact boundary.
    check(20, false); // Explicit ramp-down and cold return.
    check(0, false);  // Explicit cold-return-only quench.
    std::puts("CPU actual Director bodies: Standard warm tail, cooling, quench, clocks, dt, flag preservation: PASS");
}

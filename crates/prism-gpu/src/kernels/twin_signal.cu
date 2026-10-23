// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN: Device-side stream completion signaling
//
// Trivial kernels appended to each physics stream's per-step sequence.
// The persistent coupling kernel spin-waits on these flags to know
// when both streams have finished their physics + detection passes.
//
// Protocol:
//   Physics stream A: ... → ladd_*(A) → signal_stream_done(flag_a)
//   Physics stream B: ... → ladd_*(B) → signal_stream_done(flag_b)
//   Coupling kernel:  while (!flag_a || !flag_b) { __nanosleep(100); }
//                     ... ring buffer exchange ...
//                     clear_signals(flag_a, flag_b)
//
// Flag semantics:
//   0 = stream is still running (or has been cleared for next step)
//   step_number = stream has completed this step
//
// Using step numbers instead of 0/1 prevents ABA problems where the
// coupling kernel might read a stale "done" signal from the previous step.
// ═══════════════════════════════════════════════════════════════════════

// Signal that a physics stream has completed its step.
// Launched with 1 thread on the physics stream AFTER all detection kernels.
// The coupling kernel reads this flag to know the step is done.
extern "C" __global__ void signal_stream_done(
    volatile unsigned int* flag,     // [1] — atomically set to step_number
    unsigned int step_number         // current simulation step (1-indexed)
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicExch((unsigned int*)flag, step_number);
        // Memory fence ensures the flag write is globally visible
        // before any subsequent kernel on this stream reads it.
        __threadfence_system();
    }
}

// Clear both stream flags after the coupling kernel has processed them.
// Launched by the coupling kernel (or from host if host-mediated).
extern "C" __global__ void clear_signals(
    volatile unsigned int* flag_a,   // [1]
    volatile unsigned int* flag_b    // [1]
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicExch((unsigned int*)flag_a, 0);
        atomicExch((unsigned int*)flag_b, 0);
        __threadfence_system();
    }
}

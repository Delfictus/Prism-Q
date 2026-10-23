// Druggability scoring kernel (weighted sum of pocket components)
extern "C" __global__ void druggability_score_kernel(
    const float* volume,
    const float* hydrophobicity,
    const float* enclosure,
    const float* depth,
    const float* hbond,
    const float* flexibility,
    const float* conservation,
    const float* topology,
    const float* weights,
    int num_pockets,
    float* out_score
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pockets) {
        float wv = weights[0];
        float wh = weights[1];
        float we = weights[2];
        float wd = weights[3];
        float whb = weights[4];
        float wf = weights[5];
        float wc = weights[6];
        float wt = weights[7];
        out_score[idx] = wv * volume[idx]
            + wh * hydrophobicity[idx]
            + we * enclosure[idx]
            + wd * depth[idx]
            + whb * hbond[idx]
            + wf * flexibility[idx]
            + wc * conservation[idx]
            + wt * topology[idx];
    }
}

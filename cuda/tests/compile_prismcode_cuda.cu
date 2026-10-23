#include "prismcode_cuda.h"

#include <stdio.h>

int main(void) {
    if (prismcode_cuda_abi_version() != PRISMCODE_CUDA_ABI_V0_1) {
        fprintf(stderr, "ABI version mismatch\n");
        return 2;
    }
    if (prismcode_cuda_isa_version() != PRISMCODE_CUDA_ISA_V0_1) {
        fprintf(stderr, "ISA version mismatch\n");
        return 3;
    }
    if (prismcode_cuda_link_probe() != PRISMCODE_CUDA_LINK_SENTINEL) {
        fprintf(stderr, "link probe mismatch\n");
        return 4;
    }
    if (sizeof(PrismcodeInstruction) != 64u ||
        sizeof(PrismcodeMemoryRow) != 112u ||
        sizeof(PrismcodeEventRecord) != 56u) {
        fprintf(stderr, "layout mismatch\n");
        return 5;
    }
    return 0;
}

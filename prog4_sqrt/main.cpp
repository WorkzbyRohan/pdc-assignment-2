#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>

#include "CycleTimer.h"
#include "sqrt_ispc.h"

using namespace ispc;

extern void sqrtSerial(int N, float startGuess, float* values, float* output);

static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
            break;
        }
    }
}

static void printUsage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -m  --mode <best|worst|default>  Select input workload pattern\n");
    printf("  -?  --help                       Display this help message\n");
}

int main(int argc, char** argv) {

    const unsigned int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;
    const int VECTOR_WIDTH = 8; // AVX2 256-bit single precision floats

    enum WorkloadMode { MODE_DEFAULT, MODE_BEST, MODE_WORST };
    WorkloadMode mode = MODE_DEFAULT;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mode") == 0) {
            if (i + 1 < argc) {
                if (strcmp(argv[i+1], "best") == 0) mode = MODE_BEST;
                else if (strcmp(argv[i+1], "worst") == 0) mode = MODE_WORST;
                else mode = MODE_DEFAULT;
                i++;
            }
        } else if (strcmp(argv[i], "-?") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    float* values = new float[N];
    float* output = new float[N];
    float* gold = new float[N];

    // Populate array based on selected mode
    if (mode == MODE_BEST) {
        printf("--- Running BEST-CASE Workload (Uniform Heavy: 2.99f) ---\n");
        for (unsigned int i = 0; i < N; i++) {
            values[i] = 2.99f; // Maximum iterations, 100% vector lane utilization
        }
    } else if (mode == MODE_WORST) {
        printf("--- Running WORST-CASE Workload (Pathological Divergent) ---\n");
        for (unsigned int i = 0; i < N; i++) {
            if (i % VECTOR_WIDTH == 0) {
                values[i] = 2.99f; // 1 lane heavy (max iterations)
            } else {
                values[i] = 1.0f;  // 7 lanes light (instant convergence)
            }
        }
    } else {
        printf("--- Running DEFAULT Workload (Random [0.001f, 3.001f]) ---\n");
        for (unsigned int i = 0; i < N; i++) {
            values[i] = .001f + 3.0f * static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Generate gold version to check results
    for (unsigned int i = 0; i < N; i++)
        gold[i] = sqrt(values[i]);

    //
    // Run the serial implementation 3 times, reporting the minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    //
    // ISPC implementation
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    // Clear buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, gold);

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);

    delete [] values;
    delete [] output;
    delete [] gold;

    return 0;
}

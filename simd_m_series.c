#include <arm_neon.h>
#include <stdio.h>

int main() {
    /* Initialize vectors using NEON intrinsics */
    float evens2_data[4] = {16.0, 14.0, 12.0, 10.0};
    float32_t odds1_data[4] = {7.0, 5.0, 3.0, 1.0};
    // float and float32_t are interchangable

    /* Load data into NEON vectors */
    float32x4_t evens2 = vld1q_f32(&evens2_data[0]);
    float32x4_t odds1 = vld1q_f32(odds1_data);

    /* Compute the difference between the two vectors */
    float32x4_t result1 = vsubq_f32(evens2, odds1);

    /* Display the elements of the result vectors */
    float results[4];
    vst1q_f32(results, result1);

    printf("%f %f %f %f\n",
           results[0], results[1], results[2], results[3]);
    return 0;
}

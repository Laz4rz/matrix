#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef struct {
    int depth;
    int rows;
    int cols;
    int length;
    float *data;
} Matrix;

void allocate_matrix_zeros(Matrix *m, int depth, int rows, int cols) {
    m->depth = depth;
    m->rows = rows;
    m->cols = cols;
    m->length = depth * rows * cols;
    m->data = (float *)calloc(depth * rows * cols, sizeof(float));
    if (m->data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);  // Or handle the error appropriately
    }
}

void allocate_matrix_random(Matrix *m, int depth, int rows, int cols) {
    m->depth = depth;
    m->rows = rows;
    m->cols = cols;
    m->length = depth * rows * cols;
    m->data = (float *)calloc(depth * rows * cols,  sizeof(float));

    if (m->data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);  // Or handle the error appropriately
    }

    srand(time(NULL));
    for (int i = 0; i < depth * rows * cols; i++) {
        m->data[i] = (float)rand() / RAND_MAX;
    }
}

void allocate_matrix_consecutive(Matrix *m, int depth, int rows, int cols) {
    m->depth = depth;
    m->rows = rows;
    m->cols = cols;
    m->length = depth * rows * cols;
    m->data = (float *)calloc(depth * rows * cols, sizeof(float));

    if (m->data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);  // Or handle the error appropriately
    }
    
    for (int i = 0; i < m->length; i++) {
        m->data[i] = i;
    }
}

void free_matrix(Matrix *m) {
    free(m->data);
}

int strided_index(Matrix *m, int d, int r, int c) {
    return d * m->rows * m->cols + r * m->cols + c;
}

float get(Matrix *m, int d, int r, int c) {
    return m->data[strided_index(m, d, r, c)];
}

void set(Matrix *m, int d, int r, int c, float value) {
    m->data[strided_index(m, d, r, c)] = value;
}

// avoiding get and set due to significant function overhead
void transpose(Matrix *m, Matrix *dst) {
    for (int d = 0; d < m->depth; d++) {
        for (int r = 0; r < m->rows; r++) {
            for (int c = 0; c < m->cols; c++) {
                dst->data[d * dst->rows * dst->cols + r * dst->cols + c] = m->data[d * m->rows * m->cols + r * m->cols + c];
            }
        }
    }
}

void transpose_inplace(Matrix *m) {
    for (int d = 0; d < m->depth; d++) {
        for (int r = 0; r < m->rows; r++) {
            for (int c = 0; c < r; c++) {
                float temp = m->data[d * m->rows * m->cols + r * m->cols + c];
                m->data[d * m->rows * m->cols + r * m->cols + c] = m->data[d * m->rows * m->cols + c * m->cols + r];
                m->data[d * m->rows * m->cols + c * m->cols + r] = temp;
            }
        }
    }
}

void print_matrix(Matrix *m) {
    for (int d = 0; d < m->depth; d++) {
        for (int r = 0; r < m->rows; r++) {
            for (int c = 0; c < m->cols; c++) {
                printf("%f ", get(m, d, r, c));
            }
            printf("\n");
        }
        printf("\n");
    }
}

void matmul(Matrix *a, Matrix *b, Matrix *res) {
    for (int d = 0; d < a->depth; d++) {
        for (int r = 0; r < a->rows; r++) {
            for (int c = 0; c < b->cols; c++) {
                
                float temp = 0.0f;
                for (int i = 0; i < a->cols; i++) {
                    temp += get(a, d, r, i) * get(b, d, i, c);
                }
                // printf("temp: %f\n", temp);
                set(res, d, r, c, temp);

            }
        }
    }
}

void matmul_transpose(Matrix *a, Matrix *b, Matrix *res) {
    transpose_inplace(b);
    for (int d = 0; d < a->depth; d++) {
        for (int r = 0; r < a->rows; r++) {
            for (int c = 0; c < b->cols; c++) {
                
                float temp = 0.0f;
                for (int i = 0; i < a->cols; i++) {
                    temp += a->data[d * a->rows * a->cols + r * a->cols + i] * b->data[d * b->rows * b->cols + c * b->cols + i];
                }
                res -> data[d * res->rows * res->cols + r * res->cols + c] = temp;
            }
        }
    }
}


int main() {
    struct timespec start, end;
    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iterations = 10;

    // srand(time(NULL));

    printf("m,n,k,time,flops\n");

    for (int i = 0; i < num_sizes; i++) {
        int M = sizes[i][0];
        int N = sizes[i][1];
        int K = sizes[i][2];

        Matrix A;
        Matrix B;
        Matrix C;

        allocate_matrix_random(&A, 1, M, N);
        allocate_matrix_random(&B, 1, N, K);
        allocate_matrix_zeros(&C, 1, M, K);

        for (int iter = 0; iter < num_iterations; iter++) {
            clock_gettime(CLOCK_MONOTONIC, &start);

            matmul_transpose(&A, &B, &C);

            clock_gettime(CLOCK_MONOTONIC, &end); 

            double time_taken = (end.tv_sec - start.tv_sec) + 
                       (end.tv_nsec - start.tv_nsec) / 1e9;
            double flops = 2.0 * M * N * K;
            double flops_per_second = flops / time_taken / 1e9;  // Convert to gigaflops

            printf("%d,%d,%d,%.6f,%.2f\n", M, N, K, time_taken, flops_per_second);
        }

        free_matrix(&A);
        free_matrix(&B);
        free_matrix(&C);
    }

    return 0;
}

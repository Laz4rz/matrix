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


int main() {
    // Matrix m;
    // allocate_matrix_random(&m, 1, 2, 2);

    // print_matrix(&m);
    // set(&m, 0, 0, 0, 1.0);
    // print_matrix(&m);

    // Matrix n;
    // Matrix res;
    // allocate_matrix_consecutive(&n, 1, 2, 2);
    // allocate_matrix_zeros(&res, 1, 2, 2);

    // matmul(&n, &n, &res);

    // print_matrix(&n);
    // print_matrix(&res);

    // Matrix o;
    // Matrix res2;
    // allocate_matrix_consecutive(&o, 1, 2, 1);
    // allocate_matrix_zeros(&res2, 1, 2, 3);

    // matmul(&n, &o, &res2);

    // print_matrix(&n);
    // print_matrix(&o);
    // print_matrix(&res2);

    // Matrix p;
    // Matrix res3;
    // allocate_matrix_consecutive(&p, 1, 1, 2);
    // allocate_matrix_zeros(&res3, 1, 3, 3);

    // matmul(&p, &o, &res3);

    // print_matrix(&p);
    // print_matrix(&o);
    // print_matrix(&res3);

    // free_matrix(&res3);
    // free_matrix(&p);
    // free_matrix(&m);
    // free_matrix(&n);
    // free_matrix(&o);
    // free_matrix(&res);
    // free_matrix(&res2);
    
    struct timespec start, end;
    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iterations = 1;

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

            matmul(&A, &B, &C);

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

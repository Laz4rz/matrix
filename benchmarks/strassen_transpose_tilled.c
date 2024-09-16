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
        exit(1);
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
        exit(1); 
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
        exit(1); 
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
                    temp += a->data[d * a->rows * a->cols + r * a->cols + i] * b->data[d * b->rows * b->cols + i * b->cols + c];
                }
                // printf("temp: %f\n", temp);
                set(res, d, r, c, temp);

            }
        }
    }
}

void add(Matrix *a, Matrix *b, Matrix *res) {
    for (int d = 0; d < a->depth; d++) {
        for (int r = 0; r < a->rows; r++) {
            for (int c = 0; c < a->cols; c++) {
                res->data[d * res->rows * res->cols + r * res->cols + c] = a->data[d * a->rows * a->cols + r * a->cols + c] + b->data[d * b->rows * b->cols + r * b->cols + c];
            }
        }
    }
}

void sub(Matrix *a, Matrix *b, Matrix *res) {
    for (int d = 0; d < a->depth; d++) {
        for (int r = 0; r < a->rows; r++) {
            for (int c = 0; c < a->cols; c++) {
                res->data[d * res->rows * res->cols + r * res->cols + c] = a->data[d * a->rows * a->cols + r * a->cols + c] - b->data[d * b->rows * b->cols + r * b->cols + c];
            }
        }
    }
}

void split(Matrix *m, Matrix *a11, Matrix *a12, Matrix *a21, Matrix *a22) {
    int r = m->rows / 2;
    int c = m->cols / 2;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            a11->data[0 * a11->rows * a11->cols + i * a11->cols + j] = m->data[0 * m->rows * m->cols + i * m->cols + j];
            a12->data[0 * a12->rows * a12->cols + i * a12->cols + j] = m->data[0 * m->rows * m->cols + i * m->cols + j + c];
            a21->data[0 * a21->rows * a21->cols + i * a21->cols + j] = m->data[0 * m->rows * m->cols + i + r * m->cols + j];
            a22->data[0 * a22->rows * a22->cols + i * a22->cols + j] = m->data[0 * m->rows * m->cols + i + r * m->cols + j + c];
        }
    }
}

void combine(Matrix *a11, Matrix *a12, Matrix *a21, Matrix *a22, Matrix *m) {
    int r = m->rows / 2;
    int c = m->cols / 2;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            m->data[0 * m->rows * m->cols + i * m->cols + j] = a11->data[0 * a11->rows * a11->cols + i * a11->cols + j];
            m->data[0 * m->rows * m->cols + i * m->cols + j + c] = a12->data[0 * a12->rows * a12->cols + i * a12->cols + j];
            m->data[0 * m->rows * m->cols + i + r * m->cols + j] = a21->data[0 * a21->rows * a21->cols + i * a21->cols + j];
            m->data[0 * m->rows * m->cols + i + r * m->cols + j + c] = a22->data[0 * a22->rows * a22->cols + i * a22->cols + j];
        }
    }
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

void zero_matrix(Matrix *m) {
    for (int i = 0; i < m->length; i++) {
        m->data[i] = 0.0f;
    }
}

// setting the tille size to 32, as Ryzen 3600 could hold ~52x52, but we need a power of 2
// and the closest power of 2 is 32
void matmul_transpose_tiled(Matrix *a, Matrix *b, Matrix *res, int tile_size) {
    transpose_inplace(b);  
    zero_matrix(res);      

    if (tile_size > a->rows || tile_size > b->cols) {
        tile_size = a->rows;
    }

    for (int d = 0; d < a->depth; d++) {
        for (int r = 0; r < a->rows; r += tile_size) {
            for (int c = 0; c < b->cols; c += tile_size) {
                for (int rr = r; rr < r + tile_size; rr++) {      
                    for (int cc = c; cc < c + tile_size; cc++) {  
                        float sum = 0.0f;
                        for (int ii = 0; ii < tile_size; ii++) {   
                            sum += a->data[d * a->rows * a->cols + rr * a->cols + ii] * 
                                   b->data[d * b->rows * b->cols + cc * b->cols + ii];
                        }
                        
                        res->data[d * res->rows * res->cols + rr * res->cols + cc] += sum;
                    }
                }
            }
        }
    }
}

void strassens(Matrix *a, Matrix *b, Matrix *res) {
    if (a->rows <= 64) {
        matmul_transpose_tiled(a, b, res, 32);
        return;
    }

    Matrix a11, a12, a21, a22;
    Matrix b11, b12, b21, b22;
    Matrix p1, p2, p3, p4, p5, p6, p7;
    Matrix c11, c12, c21, c22;
    Matrix temp1, temp2;
    allocate_matrix_zeros(&a11, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&a12, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&a21, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&a22, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&b11, 1, b->rows / 2, b->cols / 2);
    allocate_matrix_zeros(&b12, 1, b->rows / 2, b->cols / 2);
    allocate_matrix_zeros(&b21, 1, b->rows / 2, b->cols / 2);
    allocate_matrix_zeros(&b22, 1, b->rows / 2, b->cols / 2);
    allocate_matrix_zeros(&p1, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&p2, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&p3, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&p4, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&p5, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&p6, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&p7, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&c11, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&c12, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&c21, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&c22, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&temp1, 1, a->rows / 2, a->cols / 2);
    allocate_matrix_zeros(&temp2, 1, a->rows / 2, a->cols / 2);

    split(a, &a11, &a12, &a21, &a22);
    split(b, &b11, &b12, &b21, &b22);

    sub(&b12, &b22, &temp1);  // B12 - B22
    strassens(&a11, &temp1, &p1);   // P1 = A11 * (B12 - B22)

    add(&a11, &a12, &temp1);       // A11 + A12
    strassens(&temp1, &b22, &p2);   // P2 = (A11 + A12) * B22

    add(&a21, &a22, &temp1);       // A21 + A22
    strassens(&temp1, &b11, &p3);   // P3 = (A21 + A22) * B11

    sub(&b21, &b11, &temp1);  // B21 - B11
    strassens(&a22, &temp1, &p4);   // P4 = A22 * (B21 - B11)

    add(&a11, &a22, &temp1);       // A11 + A22
    add(&b11, &b22, &temp2);       // B11 + B22
    strassens(&temp1, &temp2, &p5); // P5 = (A11 + A22) * (B11 + B22)

    sub(&a12, &a22, &temp1);  // A12 - A22
    add(&b21, &b22, &temp2);       // B21 + B22
    strassens(&temp1, &temp2, &p6); // P6 = (A12 - A22) * (B21 + B22)

    sub(&a11, &a21, &temp1);  // A11 - A21
    add(&b11, &b12, &temp2);       // B11 + B12
    strassens(&temp1, &temp2, &p7); // P7 = (A11 - A21) * (B11 + B12)

    // Calculate the result submatrices
    add(&p5, &p4, &temp1);          // P5 + P4
    sub(&temp1, &p2, &temp2);  // P5 + P4 - P2
    add(&temp2, &p6, &c11);         // C11 = P5 + P4 - P2 + P6

    add(&p1, &p2, &c12);            // C12 = P1 + P2
    add(&p3, &p4, &c21);            // C21 = P3 + P4

    add(&p1, &p5, &temp1);          // P1 + P5
    sub(&temp1, &p3, &temp2);  // P1 + P5 - P3
    sub(&temp2, &p7, &c22);    // C22 = P1 + P5 - P3 - P7

    // Combine the result submatrices into the result matrix
    combine(res, &c11, &c12, &c21, &c22);

    // Free allocated memory for intermediate matrices
    free_matrix(&a11); free_matrix(&a12); free_matrix(&a21); free_matrix(&a22);
    free_matrix(&b11); free_matrix(&b12); free_matrix(&b21); free_matrix(&b22);
    free_matrix(&p1); free_matrix(&p2); free_matrix(&p3); free_matrix(&p4);
    free_matrix(&p5); free_matrix(&p6); free_matrix(&p7);
    free_matrix(&temp1); free_matrix(&temp2);
    free_matrix(&c11); free_matrix(&c12); free_matrix(&c21); free_matrix(&c22);
}


int main() {
    struct timespec start, end;
    int sizes[][3] = {{128, 128, 128}, {512, 512, 512}, {1024, 1024, 1024}};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iterations = 5;

    // srand(time(NULL));

    printf("m, n, k, time, GFLOPS\n");

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

            strassens(&A, &B, &C);

            clock_gettime(CLOCK_MONOTONIC, &end); 

            double time_taken = (end.tv_sec - start.tv_sec) + 
                       (end.tv_nsec - start.tv_nsec) / 1e9;
            double flops = 2.0 * M * N * K;
            double flops_per_second = flops / time_taken / 1e9;  // Convert to gigaflops

            printf("%d, %d, %d, %.6f, %.2f GLOPS\n", M, N, K, time_taken, flops_per_second);
        }

        free_matrix(&A);
        free_matrix(&B);
        free_matrix(&C);
    }

    return 0;
}

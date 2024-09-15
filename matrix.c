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
    Matrix m;
    allocate_matrix_random(&m, 1, 2, 2);

    print_matrix(&m);
    set(&m, 0, 0, 0, 1.0);
    print_matrix(&m);

    Matrix n;
    Matrix res;
    allocate_matrix_consecutive(&n, 1, 2, 2);
    allocate_matrix_zeros(&res, 1, 2, 2);

    matmul(&n, &n, &res);

    print_matrix(&n);
    print_matrix(&res);

    Matrix o;
    Matrix res2;
    allocate_matrix_consecutive(&o, 1, 2, 1);
    allocate_matrix_zeros(&res2, 1, 2, 3);

    matmul(&n, &o, &res2);

    print_matrix(&n);
    print_matrix(&o);
    print_matrix(&res2);

    Matrix p;
    Matrix res3;
    allocate_matrix_consecutive(&p, 1, 1, 2);
    allocate_matrix_zeros(&res3, 1, 3, 3);

    matmul(&p, &o, &res3);

    print_matrix(&p);
    print_matrix(&o);
    print_matrix(&res3);

    Matrix a, b;
    allocate_matrix_consecutive(&a, 1, 2, 2);
    allocate_matrix_consecutive(&b, 1, 2, 2);
    matmul_transpose(&a, &b, &res);

    print_matrix(&a);
    print_matrix(&b);
    print_matrix(&res);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&res3);
    free_matrix(&p);
    free_matrix(&m);
    free_matrix(&n);
    free_matrix(&o);
    free_matrix(&res);
    free_matrix(&res2);
}

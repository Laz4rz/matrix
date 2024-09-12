#include <stdio.h>
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
}

void allocate_matrix_random(Matrix *m, int depth, int rows, int cols) {
    m->depth = depth;
    m->rows = rows;
    m->cols = cols;
    m->length = depth * rows * cols;
    m->data = (float *)calloc(depth * rows * cols,  sizeof(float));

    srand(0);
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
                printf("temp: %f\n", temp);
                set(res, d, r, c, temp);

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
    allocate_matrix_consecutive(&o, 1, 2, 3);
    allocate_matrix_zeros(&res2, 1, 2, 3);

    matmul(&n, &o, &res2);

    print_matrix(&n);
    print_matrix(&o);
    print_matrix(&res2);

    free_matrix(&m);
    free_matrix(&n);
    free_matrix(&o);
    free_matrix(&res);
    free_matrix(&res2);
    return 0;
}

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

void add(Matrix *a, Matrix *b, Matrix *res) {
    for (int d = 0; d < a->depth; d++) {
        for (int r = 0; r < a->rows; r++) {
            for (int c = 0; c < a->cols; c++) {
                set(res, d, r, c, get(a, d, r, c) + get(b, d, r, c));
            }
        }
    }
}

void sub(Matrix *a, Matrix *b, Matrix *res) {
    for (int d = 0; d < a->depth; d++) {
        for (int r = 0; r < a->rows; r++) {
            for (int c = 0; c < a->cols; c++) {
                set(res, d, r, c, get(a, d, r, c) - get(b, d, r, c));
            }
        }
    }
}

void split(Matrix *m, Matrix *a11, Matrix *a12, Matrix *a21, Matrix *a22) {
    int r = m->rows / 2;
    int c = m->cols / 2;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            set(a11, 0, i, j, get(m, 0, i, j));
            set(a12, 0, i, j, get(m, 0, i, j + c));
            set(a21, 0, i, j, get(m, 0, i + r, j));
            set(a22, 0, i, j, get(m, 0, i + r, j + c));
        }
    }
}

void combine(Matrix *a11, Matrix *a12, Matrix *a21, Matrix *a22, Matrix *m) {
    int r = m->rows / 2;
    int c = m->cols / 2;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            set(m, 0, i, j, get(a11, 0, i, j));
            set(m, 0, i, j + c, get(a12, 0, i, j));
            set(m, 0, i + r, j, get(a21, 0, i, j));
            set(m, 0, i + r, j + c, get(a22, 0, i, j));
        }
    }
}

void strassens(Matrix *a, Matrix *b, Matrix *res) {
    if (a->rows == 1) {
        set(res, 0, 0, 0, get(a, 0, 0, 0) * get(b, 0, 0, 0));
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
    Matrix m;
    allocate_matrix_consecutive(&m, 1, 4, 4);

    Matrix n;
    Matrix res;
    allocate_matrix_consecutive(&n, 1, 4, 4);
    allocate_matrix_zeros(&res, 1, 4, 4);

    matmul(&n, &n, &res);
    Matrix a11, a12, a21, a22;
    allocate_matrix_consecutive(&a11, 1, 2, 2);
    allocate_matrix_consecutive(&a12, 1, 2, 2);
    allocate_matrix_consecutive(&a21, 1, 2, 2);
    allocate_matrix_consecutive(&a22, 1, 2, 2);

    split(&m, &a11, &a12, &a21, &a22);

    print_matrix(&m);
    print_matrix(&a11);
    print_matrix(&a12);
    print_matrix(&a21);
    print_matrix(&a22);

    Matrix m2;
    allocate_matrix_consecutive(&m2, 1, 4, 4);

    combine(&a11, &a12, &a21, &a22, &m2);

    print_matrix(&m2);

    strassens(&m, &m2, &res);

    print_matrix(&res);

    free_matrix(&m);
    free_matrix(&n);
    free_matrix(&res);
    free_matrix(&a11);
    free_matrix(&a12);
    free_matrix(&a21);
    free_matrix(&a22);
    free_matrix(&m2);
}

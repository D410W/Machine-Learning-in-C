#ifndef TENSORS_LIBRARY
#define TENSORS_LIBRARY

#include <stdlib.h>
#include <stdbool.h>

#define SHORT_TYPES
#include "arena.h"
#define RANDOM_IMPLEMENTATION
#include "random.h"

typedef struct {
  size_t rows, cols;
  f32* data;
} Matrix;

Matrix* mat_create(ArenaAlloc* arena, size_t rows, size_t cols);
bool mat_load(Matrix* matrix, char* file_name, size_t* bytes_read); // file is expected to be binary with f32 numbers.
bool mat_copy(Matrix* dst, Matrix* src);
bool mat_copy_section(Matrix* dst, Matrix* src, size_t start);
void mat_clear(Matrix* matrix);
void mat_fill(Matrix* matrix, f32 value);
void mat_fill_rand(Matrix* matrix, f32 lower, f32 upper);
void mat_scale(Matrix* matrix, f32 scale);
f32 mat_sum(Matrix* matrix);
size_t mat_argmax(Matrix* matrix);

void mat_draw(FILE* stream, Matrix *mat);

bool mat_add(Matrix* out, const Matrix* a, const Matrix* b);
bool mat_sub(Matrix* out, const Matrix* a, const Matrix* b);
bool mat_mul(Matrix* out, const Matrix* a, const Matrix* b, bool zero_out, bool transpose_a, bool transpose_b);

bool mat_relu(Matrix* out, const Matrix* in); // MAX(0.0, x)
bool mat_softmax(Matrix* out, const Matrix* in); // turns input vector into probab. distr.
bool mat_cross_entropy(Matrix* out, const Matrix* p, const Matrix* q); // cost function

bool mat_relu_add_grad(Matrix* out, const Matrix* in, const Matrix* gradient);
bool mat_softmax_add_grad(Matrix* out, const Matrix* softmax_out, const Matrix* gradient);
bool mat_cross_entropy_add_grad(Matrix* p_gradient, Matrix* q_gradient, const Matrix* p, const Matrix* q, const Matrix* gradient);

#ifdef TENSORS_IMPLEMENTATION
#undef TENSORS_IMPLEMENTATION

Matrix* mat_create(ArenaAlloc* arena, size_t rows, size_t cols) {
  Matrix* mat = PUSH_STRUCT(arena, Matrix);
  if (mat == NULL) return NULL;
  
  mat->rows = rows;
  mat->cols = cols;
  mat->data = PUSH_ARRAY(arena, f32, rows * cols);
  
  if (mat->data == NULL) {
    arena_pop(arena, sizeof(Matrix));
    return NULL;
  }
  
  return mat;
}

#if defined(_WIN32)

bool mat_load(Matrix* matrix, char* file_name, size_t* bytes_read) { // file is expected to be binary with f32 numbers.
  abort(); // TODO: implement this for windows
  return 0;
}

#elif defined(__linux__) || defined(__gnu_linux__)

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

bool mat_load(Matrix* matrix, char* file_name, size_t* bytes_read) { // file is expected to be binary with f32 numbers.
  int file_descriptor = open(file_name, O_RDONLY);
  if (file_descriptor == -1) {
    fprintf(stderr, "Error: Failed to open file '%s'.\n", file_name);
    if (bytes_read != NULL) *bytes_read = 0;
    return false;
  }
  
  struct stat fstatus;
  if (fstat(file_descriptor, &fstatus) == -1) {
    fprintf(stderr, "Error: Failed to get file info from '%s'.\n", file_name);
    if (bytes_read != NULL) *bytes_read = 0;
    return false;
  }
  
  if (fstatus.st_size == 0) {
    fprintf(stderr, "Warning: file '%s' is empty.\n", file_name);
    if (bytes_read != NULL) *bytes_read = 0;
    return false;
  }
  
  f32* loaded_file = mmap(NULL, fstatus.st_size, PROT_READ, MAP_PRIVATE, file_descriptor, 0);
  if (loaded_file == MAP_FAILED) {
    close(file_descriptor);
    fprintf(stderr, "Error: Failed to load file '%s' to memory.\n", file_name);
    if (bytes_read != NULL) *bytes_read = 0;
    return false;
  }
  
  size_t length = MIN(matrix->rows * matrix->cols * sizeof(f32), (size_t)fstatus.st_size);
  memcpy(matrix->data, loaded_file, length);
  
  munmap(loaded_file, fstatus.st_size);
  close(file_descriptor);
  
  if (bytes_read != NULL) *bytes_read = length;
  return true;
}

#else
    #error "Platform not supported"
#endif // platform specific

bool mat_copy(Matrix* dst, Matrix* src) {
  if (dst->rows != src->rows || dst->cols != src->cols) return false;
  
  memcpy(dst->data, src->data, sizeof(f32) * src->rows * src->cols);
  
  return true;
}

bool mat_copy_section(Matrix* dst, Matrix* src, size_t start) {
  size_t src_size = src->rows * src->cols;
  size_t dst_size = dst->rows * dst->cols;

  if (dst_size + start > src_size) return false;
  
  memcpy(dst->data, src->data + start, dst_size * sizeof(f32));
  
  return true;
}

void mat_clear(Matrix* matrix) {
  memset(matrix->data, 0, sizeof(f32) * matrix->rows * matrix->cols);
}

void mat_fill(Matrix* matrix, f32 value) {
  size_t size = matrix->rows * matrix->cols;
  
  for (size_t i = 0; i < size; ++i) {
    matrix->data[i] = value;
  }
}

void mat_fill_rand(Matrix* matrix, f32 lower, f32 upper) {
  size_t size = matrix->rows * matrix->cols;
  
  for (size_t i = 0; i < size; ++i) {
    matrix->data[i] = rng_fgen() * (upper - lower) + lower;
  }
}

void mat_scale(Matrix* matrix, f32 scale) {
  size_t size = matrix->rows * matrix->cols;
  
  for (size_t i = 0; i < size; ++i) {
    matrix->data[i] *= scale;
  }
}

f32 mat_sum(Matrix* matrix) {
  size_t size = matrix->rows * matrix->cols;
  
  f32 sum = 0;
  for (size_t i = 0; i < size; ++i) {
    sum += matrix->data[i];
  }
  
  return sum;
}

size_t mat_argmax(Matrix* matrix) {
  size_t size = matrix->rows * matrix->cols;
  
  size_t max_i = 0;
  for (size_t i = 0; i < size; ++i) {
    if (matrix->data[i] > matrix->data[max_i]) {
      max_i = i;
    }
  }
  
  return max_i;
}

void mat_draw(FILE* stream, Matrix *matrix) {
  for (size_t i = 0; i < matrix->rows; ++i) {
    for (size_t j = 0; j < matrix->cols; ++j) {
      fprintf(stream, "\033[48;5;%dm  ", 232 + (u16)(matrix->data[j + i * matrix->cols] * 23.0));
    }
    fprintf(stream, "\n");
  }
  
  fprintf(stream, "\x1b[0m");
}

bool mat_add(Matrix* out, const Matrix* a, const Matrix* b) {
  if (out->rows != a->rows || out->cols != a->cols) return false;
  if (a->rows != b->rows || a->cols != b->cols) return false;
  
  size_t size = a->rows * a->cols;
  for (size_t i = 0; i < size; ++i) {
    out->data[i] = a->data[i] + b->data[i];
  }
  
  return true;
}

bool mat_sub(Matrix* out, const Matrix* a, const Matrix* b) {
  if (out->rows != a->rows || out->cols != a->cols) return false;
  if (a->rows != b->rows || a->cols != b->cols) return false;
  
  size_t size = a->rows * a->cols;
  for (size_t i = 0; i < size; ++i) {
    out->data[i] = a->data[i] - b->data[i];
  }
  
  return true;
}

void _mat_mul_nn(Matrix* out, const Matrix* a, const Matrix* b) {
  for (size_t r = 0; r < out->rows; ++r) {
    for (size_t i = 0; i < a->cols; ++i) {
      f32 va = a->data[i + r * a->cols];
    
      for (size_t c = 0; c < out->cols; ++c) {
        out->data[c + r * out->cols] += va * b->data[c + i * b->cols]; // calculating out[r, c] = a[r, i] * b[i, c]
      }                                                    // in alternate order, using [y, x]
    }    
  }
}
void _mat_mul_nt(Matrix* out, const Matrix* a, const Matrix* b) {
  for (size_t r = 0; r < out->rows; ++r) {
    for (size_t c = 0; c < out->cols; ++c) {
      for (size_t i = 0; i < a->cols; ++i) {
        out->data[c + r * out->cols] += a->data[i + r * a->cols] * b->data[i + c * b->cols]; // calculating out[r, c] = a[r, i] * b[i, c]
      }                                                    // in alternate order, using [y, x]
    }    
  }
}
void _mat_mul_tn(Matrix* out, const Matrix* a, const Matrix* b) {
  for (size_t r = 0; r < out->rows; ++r) {
    for (size_t i = 0; i < a->rows; ++i) {
      f32 va = a->data[r + i * a->cols];
    
      for (size_t c = 0; c < out->cols; ++c) {
        out->data[c + r * out->cols] += va * b->data[c + i * b->cols]; // calculating out[r, c] = a[r, i] * b[i, c]
      }                                                    // in alternate order, using [y, x]
    }    
  }
}
void _mat_mul_tt(Matrix* out, const Matrix* a, const Matrix* b) {
  for (size_t r = 0; r < out->rows; ++r) {
    for (size_t i = 0; i < a->rows; ++i) {
      f32 va = a->data[r + i * a->cols];
      
      for (size_t c = 0; c < out->cols; ++c) {
        out->data[c + r * out->cols] += va * b->data[i + c * b->cols]; // calculating out[r, c] = a[r, i] * b[i, c]
      }                                                    // in alternate order, using [y, x]
    }    
  }
}

bool mat_mul(
  Matrix* out, const Matrix* a, const Matrix* b, 
  bool zero_out, bool transpose_a, bool transpose_b
) {
  size_t a_rows = transpose_a ? a->cols : a->rows;
  size_t a_cols = transpose_a ? a->rows : a->cols;
  size_t b_rows = transpose_b ? b->cols : b->rows;
  size_t b_cols = transpose_b ? b->rows : b->cols;
  
  if (a_cols != b_rows) return false;
  if (out->rows != a_rows || out->cols != b_cols) return false;
  
  if (zero_out) mat_clear(out);
  
  u8 transpose = (transpose_a << 1) | transpose_b;
  
  switch (transpose) {
    case 0b00: { _mat_mul_nn(out, a, b); } break;
    case 0b01: { _mat_mul_nt(out, a, b); } break;
    case 0b10: { _mat_mul_tn(out, a, b); } break;
    case 0b11: { _mat_mul_tt(out, a, b); } break;
  }
  
  return true;
}

bool mat_relu(Matrix* out, const Matrix* in) { // MAX(0.0, x)
  if (out->rows != in->rows || out->cols != in->cols) return false;
  
  size_t size = in->cols * in->rows;
  for (size_t i = 0; i < size; ++i) {
    out->data[i] = MAX(in->data[i], 0.0f);
  }
  
  return true;
}

bool mat_softmax(Matrix* out, const Matrix* in) { // turns input vector into probab. distr.
  if (out->rows != in->rows || out->cols != in->cols) return false;
  f32 sum = 0.0;
  
  size_t size = in->cols * in->rows;
  for (size_t i = 0; i < size; ++i) {
    out->data[i] = expf(in->data[i]);
    sum += expf(in->data[i]);
  }
  
  mat_scale(out, 1.0f / sum);
  
  return true;
}

bool mat_cross_entropy(Matrix* out, const Matrix* p, const Matrix* q) { // cost function
  if (p->rows != q->rows || p->cols != q->cols) return false;
  if (out->rows != p->rows || out->cols != p->cols) return false;

  size_t size = p->cols * p->rows;
  for (size_t i = 0; i < size; ++i) {
    out->data[i] = p->data[i] == 0.0f ? 0.0f : p->data[i] * -logf(q->data[i] + 1e-7f); // might not need the negative here
  }
  
  return true;  
}

bool mat_relu_add_grad(Matrix* output, const Matrix* input, const Matrix* gradient) {
  if (output->rows != input->rows || output->cols != input->cols) return false;
  if (output->rows != gradient->rows || output->cols != gradient->cols) return false;
  
  size_t size = input->cols * input->rows;
  for (size_t i = 0; i < size; ++i) {
    output->data[i] += input->data[i] > 0.0f ? gradient->data[i] : 0.0f;
  }
  
  return true;
}

bool mat_softmax_add_grad(Matrix* output, const Matrix* softmax_out, const Matrix* gradient) {
  if (softmax_out->rows != 1 && softmax_out->cols != 1) return false;
  
  ArenaAllocTemp scratch = arena_scratch_begin(0);
  
  size_t size = MAX(softmax_out->rows, softmax_out->cols);
  Matrix* jacobian = mat_create(scratch.arena, size, size);
  
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      jacobian->data[j + i * size] = softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
    }
  }
  
  mat_mul(output, jacobian, gradient, 0, 0, 0);
  
  arena_scratch_end(scratch);
  
  return true;
}

bool mat_cross_entropy_add_grad(Matrix* p_gradient, Matrix* q_gradient, const Matrix* p, const Matrix* q, const Matrix* gradient) {
  if (p->rows != q->rows || p->cols != q->cols) return false;
  
  size_t size = p->cols * p->rows;
  
  if (p_gradient != NULL) {
    if (p_gradient->rows != p->rows || p_gradient->cols != p->cols) return false;
    
    for (size_t i = 0; i < size; ++i) {
      p_gradient->data[i] += -logf(q->data[i] + 1e-7f) * gradient->data[i];
    }
  }

  if (q_gradient != NULL) {
    if (q_gradient->rows != q->rows || q_gradient->cols != q->cols) return false;
    
    for (size_t i = 0; i < size; ++i) {
      q_gradient->data[i] += -p->data[i] / (q->data[i] + 1e-7f) * gradient->data[i];
    }
  }

  
  return true; 
}

#endif // TENSORS_IMPLEMENTATION

#endif // TENSORS_LIBRARY

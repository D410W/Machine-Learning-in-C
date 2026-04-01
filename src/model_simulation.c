#include <stdio.h>

#define SHORT_TYPES
#define ARENA_IMPLEMENTATION
#include "core/arena.h"
#define RANDOM_IMPLEMENTATION
#include "core/random.h"
#define TENSORS_IMPLEMENTATION
#include "core/tensors.h"
#define ML_MODEL_IMPLEMENTATION
#include "core/ml_model.h"

void create_mnist_model(ArenaAlloc* arena, ModelContext* model) {
  ModelVar* input = mv_create(arena, model, 784, 1, MV_FLAG_INPUT); // 28 * 28 = 784
  
  ModelVar* w0 = mv_create(arena, model, 64, 784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* w1 = mv_create(arena, model, 64, 64, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* w2 = mv_create(arena, model, 16, 64, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* w3 = mv_create(arena, model, 10, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  
  f32 bound0 = sqrtf(6.0f / (784.0f + 64.0f));
  f32 bound1 = sqrtf(6.0f / (64.0f + 64.0f));
  f32 bound2 = sqrtf(6.0f / (64.0f + 16.0f));
  f32 bound3 = sqrtf(6.0f / (16.0f + 10.0f));
  mat_fill_rand(w0->value, -bound0, bound0);
  mat_fill_rand(w1->value, -bound1, bound1);
  mat_fill_rand(w2->value, -bound2, bound2);
  mat_fill_rand(w3->value, -bound3, bound3);
    
  ModelVar* b0 = mv_create(arena, model, 64, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* b1 = mv_create(arena, model, 64, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* b2 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* b3 = mv_create(arena, model, 10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

  ModelVar* z0_a = mv_matmul(arena, model, w0, input, MV_FLAG_NONE);
  ModelVar* z0_b = mv_add(arena, model, z0_a, b0, MV_FLAG_NONE);
  ModelVar* a0 = mv_relu(arena, model, z0_b, MV_FLAG_NONE);
  
  ModelVar* z1_a = mv_matmul(arena, model, w1, a0, MV_FLAG_NONE);
  ModelVar* z1_b = mv_add(arena, model, z1_a, b1, MV_FLAG_NONE);
  ModelVar* z1_c = mv_relu(arena, model, z1_b, MV_FLAG_NONE);
  ModelVar* a1 = mv_add(arena, model, z1_c, a0, MV_FLAG_NONE);
  
  ModelVar* z2_a = mv_matmul(arena, model, w2, a1, MV_FLAG_NONE);
  ModelVar* z2_b = mv_add(arena, model, z2_a, b2, MV_FLAG_NONE);
  ModelVar* a2 = mv_relu(arena, model, z2_b, MV_FLAG_NONE);
  
  ModelVar* z3_a = mv_matmul(arena, model, w3, a2, MV_FLAG_NONE);
  ModelVar* z3_b = mv_add(arena, model, z3_a, b3, MV_FLAG_NONE);
  ModelVar* output = mv_softmax(arena, model, z3_b, MV_FLAG_OUTPUT);
  
  ModelVar* desired_output = mv_create(arena, model, 10, 1, MV_FLAG_DESIRED_OUTPUT);
  ModelVar* cost = mv_cross_entropy(arena, model, desired_output, output, MV_FLAG_COST);
}

bool mat_from_bmp_mnist(ArenaAlloc* arena, Matrix* matrix, char* file_name) {
  if (matrix->rows != 28 || matrix->cols != 28) return false;
  
  FILE* f = fopen(file_name, "rb");
  if (f == NULL) return false;
  
  ArenaAllocTemp scratch = arena_temp_begin(arena);
  
  u8 file_header[14];
  if (fread(file_header, sizeof(u8), 14, f) != 14) goto error; 
  if (file_header[0] != 'B' || file_header[1] != 'M') goto error;
  
  size_t pixels_offset = *(u32*)(file_header + 10);
  
  u8 info_header[40];
  if (fread(info_header, 1, 40, f) != 40) goto error;
  
  i32 width  = *(i32*)(info_header + 4);
  i32 height = *(i32*)(info_header + 8);
  u16 bpp    = *(u16*)(info_header + 14); // bits per pixel
  
  if (width != 28 || abs(height) != 28) goto error;
  
  fseek(f, pixels_offset, SEEK_SET);
  
  i16 bytes_per_pixel = bpp / 8;
  i16 row_stride = (width * bytes_per_pixel + 3) & ~3;
  size_t data_size = row_stride * abs(height);
  
  u8* pixels = PUSH_ARRAY_NZ(scratch.arena, u8, data_size);
  if (fread(pixels, sizeof(u8), data_size, f) != data_size) goto error;
  
  fclose(f);
  
  for (size_t j = 0; j < 28; ++j) {
    for (size_t i = 0; i < 28; ++i) {
      u16 red = pixels[j * row_stride + bytes_per_pixel * i];
      // u16 green = pixels[j * row_stride + bytes_per_pixel * i + 1];
      // u16 blue = pixels[j * row_stride + bytes_per_pixel * i + 2];
      
      f32 average = (f32)(red) / 255.0f;
      matrix->data[(27 - j) * 28 + i] = average;
    }
  }
  
  arena_temp_end(scratch);
  return true;
  
error:
  if (f) fclose(f);
  arena_temp_end(scratch);
  return false;
}

int main() {
  ArenaAlloc* arena = arena_create(MiB(1500));
  
  size_t sys_entropy[2];
  platform_get_entropy(&sys_entropy, sizeof(size_t) * 2);
  rng_seed(sys_entropy[0], sys_entropy[1]);
  
  // RNGState* rng = arena_push(arena, sizeof(RNGState), false);
  // u64 seeds[2] = {};
  // platform_get_entropy(seeds, sizeof(u64) * 2);
  // rng_seed_r(rng, seeds[0], seeds[1]);
    
  Matrix* train_images = mat_create(arena, 60000, 784);
  Matrix* test_images = mat_create(arena, 10000, 784);

  Matrix* train_labels = mat_create(arena, 60000, 10);
  Matrix* test_labels = mat_create(arena, 10000, 10);
  
  
  if (train_images == NULL || test_images == NULL ||
      train_labels == NULL || test_labels == NULL) {
    fprintf(stderr, "Not enough memory.\n"); arena_destroy(arena); return 0;
  }
  
  if (!mat_load(train_images, "datasets/train_images", NULL)) {
    printf("Could not load 'datasets/train_images' matrix.\n"); arena_destroy(arena); return 0;
  }
  if (!mat_load(test_images, "datasets/test_images", NULL)) {
    printf("Could not load 'datasets/test_images' matrix.\n"); arena_destroy(arena); return 0;
  }

  {
    ArenaAllocTemp scratch = arena_temp_begin(arena);
    
    Matrix* train_labels_file = mat_create(scratch.arena, 60000, 1);
    Matrix* test_labels_file = mat_create(scratch.arena, 10000, 1);
    
    if (train_labels_file == NULL || test_labels_file == NULL) {
      fprintf(stderr, "Not enough memory.\n");
      arena_destroy(arena);
      return 0;
    }
    
    if (!mat_load(train_labels_file, "datasets/train_labels", NULL)) {
      printf("Could not load 'datasets/train_labels' matrix.\n"); arena_destroy(arena); return 0; }
    if (!mat_load(test_labels_file, "datasets/test_labels", NULL)) {
      printf("Could not load 'datasets/test_labels' matrix.\n"); arena_destroy(arena); return 0; }
    
    for (size_t i = 0; i < 60000; ++i) {
      size_t num = roundf(train_labels_file->data[i]);
      train_labels->data[i * 10 + num] = 1.0;
    }
    
    for (size_t i = 0; i < 10000; ++i) {
      size_t num = roundf(test_labels_file->data[i]);
      test_labels->data[i * 10 + num] = 1.0;
    }
    
    arena_temp_end(scratch);
  }
  
  ModelContext* model = model_create(arena);
  create_mnist_model(arena, model);
  model_compile(arena, model);
  
  Matrix* image1 = mat_create(arena, 28, 28);
  Matrix* image2 = mat_create(arena, 28, 28);
  
  ModelTrainingDesc train_desc = {
    .train_input = train_images,
    .train_output = train_labels,
    .test_input = test_images,
    .test_output = test_labels,
    
    .epochs = 1,
    .batch_size = 50,
    .learning_rate = 0.01f,
  };
  
  model_train(model, &train_desc);
  
  // mat_copy_section(model->input->value, image, 0);
  // model_feedforward(model);
  // printf("Post-training output: ");
  // for (size_t i = 0; i < 10; ++i) {
  //   printf("%f ", model->output->value->data[i]);
  // }
  
  const size_t max_chars = 100;
  char input[max_chars];
  while (true) {
    fgets(input, max_chars, stdin);
    if (!strcmp(input, "exit\n")) break;
    
    mat_from_bmp_mnist(arena, image1, "number.bmp");
    mat_fill(image2, 0.0f);
    
    {
      i64 min_x = 28;
      i64 min_y = 28;
      i64 max_x = 0;
      i64 max_y = 0;
      
      for (size_t i = 0; i < 784; ++i) {
        float value = image1->data[i];
        i64 x = i % 28;
        i64 y = i / 28;
        
        if (value >= 0.3) {
          min_x = MIN(min_x, x);
          min_y = MIN(min_y, y);
          max_x = MAX(max_x, x);
          max_y = MAX(max_y, y);
        }
      }
      
      i64 space_x = (28 - (max_x - min_x)) / 2;
      i64 space_y = (28 - (max_y - min_y)) / 2;
      i64 diff_x = space_x - min_x - 1;
      i64 diff_y = space_y - min_y - 1;
      
      for (i64 i = 0; i < 28; ++i) {
        for (i64 j = 0; j < 28; ++j) {
          if (i + diff_x < 0 || i + diff_x > 27) continue;
          if (j + diff_y < 0 || j + diff_y > 27) continue;
          
          image2->data[i + diff_x + (j + diff_y) * 28] = image1->data[i + j * 28];
        }
      }
    }
    
    mat_draw(stdout, image2);
    
    mat_copy_section(model->input->value, image2, 0);
    model_feedforward(model);
    printf("Probability output: ");
    size_t highest = 0;
    for (size_t i = 0; i < 10; ++i) {
      printf("%f ", model->output->value->data[i]);
      if (model->output->value->data[i] > model->output->value->data[highest])
        highest = i;
    }
    printf("\nModel's prediction: %ld\n", highest);
  }
  
  arena_destroy(arena);
  return 0;
}

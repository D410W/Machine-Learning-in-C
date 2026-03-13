#include <stdio.h>

#define SHORT_TYPES
#define ARENA_IMPLEMENTATION
#include "arena.h"
#define RANDOM_IMPLEMENTATION
#include "random.h"
#define TENSORS_IMPLEMENTATION
#include "tensors.h"
#define ML_MODEL_IMPLEMENTATION
#include "ml_model.h"

void create_mnist_model(ArenaAlloc* arena, ModelContext* model) {
  ModelVar* input = mv_create(arena, model, 784, 1, MV_FLAG_INPUT);
  
  ModelVar* w0 = mv_create(arena, model, 16, 784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* w1 = mv_create(arena, model, 16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* w2 = mv_create(arena, model, 10, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  
  f32 bound0 = sqrtf(6.0f / (784.0f + 16.0f));
  f32 bound1 = sqrtf(6.0f / (16.0f + 16.0f));
  f32 bound2 = sqrtf(6.0f / (16.0f + 10.0f));
  mat_fill_rand(w0->value, -bound0, bound0);
  mat_fill_rand(w1->value, -bound1, bound1);
  mat_fill_rand(w2->value, -bound2, bound2);
  
  ModelVar* b0 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* b1 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  ModelVar* b2 = mv_create(arena, model, 10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

  ModelVar* z0_a = mv_matmul(arena, model, w0, input, MV_FLAG_NONE);
  ModelVar* z0_b = mv_add(arena, model, z0_a, b0, MV_FLAG_NONE);
  ModelVar* a0 = mv_relu(arena, model, z0_b, MV_FLAG_NONE);
  
  ModelVar* z1_a = mv_matmul(arena, model, w1, a0, MV_FLAG_NONE);
  ModelVar* z1_b = mv_add(arena, model, z1_a, b1, MV_FLAG_NONE);
  ModelVar* z1_c = mv_relu(arena, model, z1_b, MV_FLAG_NONE);
  ModelVar* a1 = mv_add(arena, model, z1_c, a0, MV_FLAG_NONE);
  
  ModelVar* z2_a = mv_matmul(arena, model, w2, a1, MV_FLAG_NONE);
  ModelVar* z2_b = mv_add(arena, model, z2_a, b2, MV_FLAG_NONE);
  ModelVar* output = mv_softmax(arena, model, z2_b, MV_FLAG_OUTPUT);
  
  ModelVar* desired_output = mv_create(arena, model, 10, 1, MV_FLAG_DESIRED_OUTPUT);
  ModelVar* cost = mv_cross_entropy(arena, model, desired_output, output, MV_FLAG_COST);
}

int main() { // int argc, char **argv) {
  
  ArenaAlloc* arena = arena_create(MiB(500));
  
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
  
  if (!mat_load(train_images, "train_images", NULL)) {
    printf("Could not load 'train_images' matrix.\n"); arena_destroy(arena); return 0;
  }
  if (!mat_load(test_images, "test_images", NULL)) {
    printf("Could not load 'test_images' matrix.\n"); arena_destroy(arena); return 0;
  }

  {
    Matrix* train_labels_file = mat_create(arena, 60000, 1);
    Matrix* test_labels_file = mat_create(arena, 10000, 1);
    
    if (train_labels_file == NULL || test_labels_file == NULL) {
      fprintf(stderr, "Not enough memory.\n");
      arena_destroy(arena);
      return 0;
    }
    
    if (!mat_load(train_labels_file, "train_labels", NULL)) {
      printf("Could not load 'train_labels' matrix.\n"); arena_destroy(arena); return 0; }
    if (!mat_load(test_labels_file, "test_labels", NULL)) {
      printf("Could not load 'test_labels' matrix.\n"); arena_destroy(arena); return 0; }
    
    for (size_t i = 0; i < 60000; ++i) {
      size_t num = roundf(train_labels_file->data[i]);
      train_labels->data[i * 10 + num] = 1.0;
    }
    
    for (size_t i = 0; i < 10000; ++i) {
      size_t num = roundf(test_labels_file->data[i]);
      test_labels->data[i * 10 + num] = 1.0;
    }

  }
  
  ModelContext* model = model_create(arena);
  create_mnist_model(arena, model);
  model_compile(arena, model);
  
  size_t image_idx = 3;
  
  Matrix* image = mat_create(arena, 28, 28);
  mat_copy_section(image, train_images, image_idx * 784);
  
  printf("Expected output: ");
  for (size_t i = 0; i < 10; ++i) {
    printf("%.0f ", train_labels->data[i + image_idx * 10]);
  }
  printf("\n");
  mat_draw(stdout, image);
  
  mat_copy_section(model->input->value, image, 0);
  model_feedforward(model);
  printf("Pre-training output: ");
  for (size_t i = 0; i < 10; ++i) {
    printf("%f ", model->output->value->data[i]);
  }
  printf("\n");
  
  ModelTrainingDesc train_desc = {
    .train_input = train_images,
    .train_output = train_labels,
    .test_input = test_images,
    .test_output = test_labels,
    
    .epochs = 3,
    .batch_size = 50,
    .learning_rate = 0.01f,
  };
  
  model_train(model, &train_desc);
  
  mat_copy_section(model->input->value, image, 0);
  model_feedforward(model);
  printf("Post-training output: ");
  for (size_t i = 0; i < 10; ++i) {
    printf("%f ", model->output->value->data[i]);
  }
  printf("\n");
    
  arena_destroy(arena);
  
  return 0;
}

#include <stdio.h>

#define SHORT_TYPES
#define ARENA_IMPLEMENTATION
#include "arena.h"
#define RANDOM_IMPLEMENTATION
#include "random.h"
#define TENSORS_IMPLEMENTATION
#include "tensors.h"
// #define ML_MODEL_IMPLEMENTATION
// #include "ml_model.h"

int main() { // int argc, char **argv) {
  
  ArenaAlloc* arena = arena_create(MiB(400));
  
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
    fprintf(stderr, "Not enough memory.\n");
    arena_destroy(arena);
    return 0;
  }
  
  size_t bytes_read = 0;
  
  if (!mat_load(train_images, "train_images", &bytes_read)) {
    printf("Could not load 'train_images' matrix.\n"); arena_destroy(arena); return 0;
  }
  if (!mat_load(test_images, "train_images", &bytes_read)) {
    printf("Could not load 'test_images' matrix.\n"); arena_destroy(arena); return 0;
  }

  {
    Matrix* train_labels_file = mat_create(arena, 60000, 1);
    Matrix* test_labels_file = mat_create(arena, 10000, 1);
    
    if (train_labels == NULL || test_labels == NULL) {
      fprintf(stderr, "Not enough memory.\n");
      arena_destroy(arena);
      return 0;
    }
    
    if (!mat_load(train_labels_file, "train_labels", &bytes_read)) {
      printf("Could not load 'train_labels' matrix.\n"); arena_destroy(arena); return 0;
    }
    if (!mat_load(test_labels_file, "test_labels", &bytes_read)) {
      printf("Could not load 'test_labels' matrix.\n"); arena_destroy(arena); return 0;
    }
    
    for (size_t i = 0; i < 60000; ++i) {
      size_t num = train_labels_file->data[i];
      train_labels->data[i * 10 + num] = 1.0;
    }
    
    for (size_t i = 0; i < 10000; ++i) {
      size_t num = test_labels_file->data[i];
      test_labels->data[i * 10 + num] = 1.0;
    }

  }
  
  size_t image_idx = 3;
  
  Matrix* image = mat_create(arena, 28, 28);
  mat_copy_section(image, train_images, image_idx * 784);
  
  for (size_t i = 0; i < 10; ++i) {
    printf("%.0f ", train_labels->data[i + image_idx * 10]);
  }
  mat_draw(stdout, image);
  
  arena_destroy(arena);
  
  return 0;
}

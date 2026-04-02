#include <stdio.h>

#define SHORT_TYPES
#define ARENA_IMPLEMENTATION
#include "core/arena.h"

#define TENSORS_IMPLEMENTATION
#include "core/tensors.h"

typedef struct {
  bool help;
  
  bool draw_mat;
  char* data_path;
  char* labels_path;
} Arguments;

bool add_arg(Arguments *args, int argc, char** argv, int *i) {
  if (!strcmp(argv[*i], "--draw")) {
    args->draw_mat = true;
  } else if (!strcmp(argv[*i], "--data")) {
    if (++(*i) == argc) {
      fprintf(stderr, "[ERROR] missing path after '--data'.\n");
      return false;
    }
    
    args->data_path = argv[*i];
  } else if (!strcmp(argv[*i], "--labels")) {
    if (++(*i) == argc) {
      fprintf(stderr, "[ERROR] missing path after '--labels'.\n");
      return false;
    }
    
    args->labels_path = argv[*i];
  } else if (!strcmp(argv[*i], "--help")) {
    args->help = true;
    printf(""
    "  Usage:\n"
    "    --help          : Show this text.\n"
    "    --data <path>   : Select the dataset file.\n"
    "    --labels <path> : Select the labels file.\n"
    "    --draw          : Draw the data as a grayscale image.\n"
    );
    return false;
  } else {
    fprintf(stderr, "[ERROR] unrecognized command. Try '--help'.\n");
    return false;
  }
  
  return true;
}

bool load_args(Arguments *args, int argc, char** argv) {
  if (argc == 1) {
    printf("  Use '--help' to learn about this program.\n");
  }
  for (int i = 1; i < argc; ++i) {
    if (!add_arg(args, argc, argv, &i)) return false;
  }
  
  bool success = true;
  if (args->data_path == NULL) {
    fprintf(stderr, "[ERROR] No data path given.\n");
    success = false;
  }
  if (args->labels_path == NULL) {
    fprintf(stderr, "[ERROR] No labels path given.\n");
    success = false;
  }
  
  return success;
}

int main(int argc, char** argv) {
  Arguments args = {0};
  
  if (!load_args(&args, argc, argv) && !args.help) return 1;
  if (args.help) return 0;
  
  ArenaAlloc* arena = arena_create(MiB(500));
  
  // loading 28x28 float dataset of given length (currently, length = 60'000)
  Matrix* data = mat_create(arena, 60000, 784);
  Matrix* labels = mat_create(arena, 60000, 10);
  
  if (data == NULL || labels == NULL) {
    fprintf(stderr, "Not enough memory.\n"); arena_destroy(arena); return 1;
  }
  
  if (!mat_load(data, args.data_path, NULL)) {
    fprintf(stderr, "[ERROR] Could not load '%s' file.\n", args.data_path); arena_destroy(arena); return 1;
  }
  
  {
    ArenaAllocTemp scratch = arena_temp_begin(arena);
    
    Matrix* labels_file = mat_create(scratch.arena, 60000, 1);
    
    if (labels_file == NULL) {
      fprintf(stderr, "Not enough memory.\n");
      arena_destroy(arena);
      return 1;
    }
    
    if (!mat_load(labels_file, args.labels_path, NULL)) {
      fprintf(stderr, "[ERROR] Could not load '%s' file.\n", args.labels_path); arena_destroy(arena); return 1;
    }
    
    for (size_t i = 0; i < 60000; ++i) {
      size_t num = roundf(labels_file->data[i]);
      labels->data[i * 10 + num] = 1.0;
    }
    
    arena_temp_end(scratch);
  }
  
  if (args.draw_mat) { // drawing the average for each number in the dataset
    Matrix* temp = mat_create(arena, 28, 28);
    Matrix** averages = PUSH_ARRAY(arena, Matrix*, 10);
    for (size_t i = 0; i < 10; ++i) averages[i] = mat_create(arena, 28, 28);
    
    for (size_t i = 0; i < 60'000; ++i) {
      size_t num = -1;
      for (size_t j = 0; j < 10; ++j) {
        if (labels->data[i * 10 + j] != 0.0) {
          num = j;
          break;
        }
      }
      
      mat_copy_section(temp, data, 784 * i);
      mat_add(averages[num], averages[num], temp);
    }
    
    
    for (size_t i = 0; i < 10; ++i) {
      f32 max_arg = averages[i]->data[mat_argmax(averages[i])];
      mat_scale(averages[i], 1.0 / max_arg);
      
      mat_draw(stdout, averages[i]);
      printf("\n");
    }
  }

  return 0;
}

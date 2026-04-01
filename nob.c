#define NOB_IMPLEMENTATION
#include "nob.h"

#define BUILD_FOLDER "build/"
#define SRC_FOLDER   "src/"

#define MAX_EXTRA_ARGS 20
#define append_extra_arg(targets_arr, idx, arg_s) do { \
  (targets_arr)[(idx)].extra_args[(targets_arr)[(idx)].extra_args_size++] = (arg_s); \
} while (0)

int main(int argc, char **argv) {
  NOB_GO_REBUILD_URSELF(argc, argv);
  
  if (!nob_mkdir_if_not_exists(BUILD_FOLDER)) return 1;
  
  Nob_Cmd cmd = {0};
  Nob_Procs procs = {0};
  
  static struct {
    const char* src_path;
    const char* bin_path;
    const char* extra_args[MAX_EXTRA_ARGS];
    size_t extra_args_size;
  } targets[] = {
    { .src_path = BUILD_FOLDER"model_simulation", .bin_path = SRC_FOLDER"model_simulation.c",
    .extra_args_size = 0 },
    { .src_path = BUILD_FOLDER"dataset_analysis", .bin_path = SRC_FOLDER"dataset_analysis.c",
    .extra_args_size = 0 },
  };
  
  append_extra_arg(targets, 0, "-g");
  
#if defined(_WIN32)
  append_extra_arg(targets, 0, "-lBcrypt");
#elif defined(__linux__) || defined(__gnu_linux__)
  // nob_cmd_append(&cmd, "");
#else
    #error "Platform not supported"
#endif // platform specific
  
  for (size_t t = 0; t < ARRAY_LEN(targets); ++t) {
    for (size_t i = 1; i < argc && i < MAX_EXTRA_ARGS; ++i) {
      append_extra_arg(targets, t, argv[i]);
    }
    append_extra_arg(targets, t, "-lm");
  }
  
  for (size_t t = 0; t < ARRAY_LEN(targets); ++t) {  
    nob_cc(&cmd);
    nob_cc_flags(&cmd);
    nob_cc_output(&cmd, targets[t].src_path);
    nob_cc_inputs(&cmd, targets[t].bin_path);
    
    for (size_t i = 0; i < targets[t].extra_args_size; ++i)
      nob_cmd_append(&cmd, targets[t].extra_args[i]);
    
    if (!nob_cmd_run(&cmd, .async = &procs)) return 1;
  }
  
  if (!procs_flush(&procs)) return 1;
  
  return 0;
}

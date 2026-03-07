#include <stdio.h>

#define SHORT_TYPES
#define ARENA_IMPLEMENTATION
#define RANDOM_IMPLEMENTATION
#include "arena.h"
#include "random.h"

int main() { // int argc, char **argv) {
  
  ArenaAlloc* arena = arena_create(MiB(1));
  
  RNGState* rng = arena_push(arena, sizeof(RNGState), false);
  u64 seeds[2] = {};
  platform_get_entropy(seeds, sizeof(u64) * 2);
  rng_seed_r(rng, seeds[0], seeds[1]);
  
  // i32 counter = 0;
  f32 fl;
  
  for (int i = 0; i < 10; ++i) {
    fl = rng_fnorm_gen_r(rng, 10.0f, 0.0f);
    printf("%f\n", fl);
    // if (fl <= 0.0f) ++counter;
  }
  // printf("Number of samples <= 0.0: %d\n", counter);
  // printf("Number of samples > 0.0:  %d\n", 1000 - counter);

  arena_destroy(arena);
  
  return 0;
}

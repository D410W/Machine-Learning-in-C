#ifndef ARENA_LIBRARY
#define ARENA_LIBRARY

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef SHORT_TYPES
  typedef int8_t i8;
  typedef int16_t i16;
  typedef int32_t i32;
  typedef int64_t i64;
  typedef uint8_t u8;
  typedef uint16_t u16;
  typedef uint32_t u32;
  typedef uint64_t u64;
  typedef float f32;
  typedef double f64;
#else
  #define i8 int8_t
  #define i16 int16_t
  #define i32 int32_t
  #define i64 int64_t
  #define u8 uint8_t
  #define u16 uint16_t
  #define u32 uint32_t
  #define u64 uint64_t
  #define f32 float
  #define f64 double
#endif // SHORT_TYPES

#define KiB(n) ((size_t)(n) << 10)
#define MiB(n) ((size_t)(n) << 20)
#define GiB(n) ((size_t)(n) << 30)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define ARENA_BASE_POS (sizeof(ArenaAlloc))
#define ARENA_ALIGN (sizeof(void*))
#define ALIGN_UP_POW2(n, p) (((size_t)(n) + (size_t)(p) - 1) & (~((size_t)(p) - 1)))

#define PUSH_STRUCT(arena, T) (T*)arena_push((arena), sizeof(T), true)
#define PUSH_STRUCT_NZ(arena, T) (T*)arena_push((arena), sizeof(T), false)
#define PUSH_ARRAY(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), true)
#define PUSH_ARRAY_NZ(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), false)

typedef struct {
  size_t capacity;
  size_t position;
} ArenaAlloc;

typedef struct {
  ArenaAlloc* arena;
  size_t position;
} ArenaAllocTemp;

ArenaAlloc* arena_create(size_t capacity);
void arena_destroy(ArenaAlloc* arena);
void* arena_push(ArenaAlloc* arena, size_t size, bool zero_out);
void arena_pop(ArenaAlloc* arena, size_t size);
void arena_pop_to(ArenaAlloc* arena, size_t position);
void arena_clear(ArenaAlloc* arena);


ArenaAllocTemp arena_temp_begin(ArenaAlloc* arena);
void arena_temp_end(ArenaAllocTemp);
ArenaAllocTemp arena_scratch_begin(size_t min_arena_capacity); // min_arena_capacity defaults to 16Mb
void arena_scratch_end(ArenaAllocTemp scratch);

#ifdef ARENA_IMPLEMENTATION
#undef ARENA_IMPLEMENTATION

ArenaAlloc* arena_create(size_t capacity) {
  ArenaAlloc* arena = (ArenaAlloc*)malloc(capacity);
  arena->capacity = capacity;
  arena->position = ARENA_BASE_POS;
  
  return arena;
}

void arena_destroy(ArenaAlloc* arena) {
  free(arena);
}

void* arena_push(ArenaAlloc* arena, size_t size, bool zero_out) {
  size_t position_aligned = ALIGN_UP_POW2(arena->position, ARENA_ALIGN);
  size_t new_position = position_aligned + size;
  
  if (new_position > arena->capacity) { return NULL; }
  
  arena->position = new_position;
  
  u8* out = (u8*)arena + position_aligned;
  
  if (zero_out) {
    memset(out, 0, size);
  }
  
  return out;
}
void arena_pop(ArenaAlloc* arena, size_t size){
  size = MIN(size, arena->position - ARENA_BASE_POS);
  arena->position -= size;
}
void arena_pop_to(ArenaAlloc* arena, size_t position) {
  u64 size = (position < arena->position) ? arena->position - position : 0;
  arena_pop(arena, size);
}
void arena_clear(ArenaAlloc* arena) {
  arena_pop_to(arena, ARENA_BASE_POS);
}

ArenaAllocTemp arena_temp_begin(ArenaAlloc* arena) {
  if (arena == NULL) return (ArenaAllocTemp) {};
  
  return (ArenaAllocTemp) {
    .arena = arena,
    .position = arena->position,
  };
}
void arena_temp_end(ArenaAllocTemp temp) {
  if (temp.arena != NULL)
    arena_pop_to(temp.arena, temp.position);
}

static __thread ArenaAlloc* _scratch_arena = NULL;
void _scratch_arena_destroy() {
  arena_destroy(_scratch_arena);
}

ArenaAllocTemp arena_scratch_begin(size_t min_arena_capacity) { // min_arena_capacity defaults to 16Mb
  if (_scratch_arena == NULL) {
    _scratch_arena = arena_create(MIN(MiB(16), min_arena_capacity));
    atexit( _scratch_arena_destroy );
  } else if (_scratch_arena->capacity < min_arena_capacity) {
    arena_destroy(_scratch_arena);
    _scratch_arena = arena_create(min_arena_capacity);
  }
  return arena_temp_begin(_scratch_arena);
}

void arena_scratch_end(ArenaAllocTemp scratch) {
  arena_temp_end(scratch);
}

#endif // ARENA_IMPLEMENTATION

#ifndef SHORT_TYPES
  #undef i8
  #undef i16
  #undef i32
  #undef i64
  #undef u8
  #undef u16
  #undef u32
  #undef u64
  #undef f32
  #undef f64
#else
  #undef SHORT_TYPES
#endif // SHORT_TYPES

#endif // ARENA_LIBRARY

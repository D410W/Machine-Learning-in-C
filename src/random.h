// Based on the PGC Random Number Generator, Minimal C Edition (https://github.com/imneme/pcg-c-basic)
// Licensed under Apache License 2.0
#ifndef RANDOM_LIBRARY
#define RANDOM_LIBRARY

#include <stdint.h>
#include <math.h>

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

#define PI 3.14159265358979323846264f

typedef struct {
  u64 state;
  u64 increment;
  
  f32 prev_norm; // other normally distributed number generated when calling 
} RNGState; // (Pseudo) Random Number Generator State

void rng_seed_r(RNGState* rng, u64 init_state, u64 init_seq);
void rng_seed(u64 init_state, u64 init_seq);
u32 rng_gen_r(RNGState* rng);
u32 rng_gen(void);
f32 rng_fgen_r(RNGState* rng);
f32 rng_fgen(void);
f32 rng_fnorm_gen_r(RNGState* rng, f32 deviation, f32 mean);
f32 rng_fnorm_gen(f32 deviation, f32 mean);
void platform_get_entropy(void* data, size_t size);

#ifdef RANDOM_IMPLEMENTATION
#undef RANDOM_IMPLEMENTATION

static RNGState global_rng_state= {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL, NAN};

void rng_seed_r(RNGState* rng, u64 init_state, u64 init_seq) {
  rng->prev_norm = NAN;
  rng->state = 0U;
  rng->increment = (init_seq << 1u) | 1u;
  rng_gen_r(rng);
  rng->state += init_state;
  rng_gen_r(rng);
}

void rng_seed(u64 init_state, u64 init_seq) {
  rng_seed_r(&global_rng_state, init_state, init_seq);
}

u32 rng_gen_r(RNGState* rng) {
  u64 oldstate = rng->state;
  rng->state = oldstate * 6364136223846793005ULL + rng->increment;
  u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  u32 rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

u32 rng_gen(void) {
  return rng_gen_r(&global_rng_state);
}

f32 rng_fgen_r(RNGState* rng) {
  return (f32)rng_gen_r(rng) / (f32)UINT32_MAX;
}

f32 rng_fgen(void) {
  return rng_gen_r(&global_rng_state);
}

f32 rng_fnorm_gen_r(RNGState* rng, f32 deviation, f32 mean) {
  if (!isnan(rng->prev_norm)) {
    f32 out = rng->prev_norm;
    rng->prev_norm = NAN;
    return out;
  }
  
  f32 u1, u2;
  
  do {
    u1 = rng_fgen_r(rng);
  } while (u1 == 0);
  
  u2 = rng_fgen_r(rng);
  
  f32 mag = deviation * sqrtf(-2.0f * logf(u1));
  f32 z0  = mag * cosf(2.0f * PI * u2) + mean;
  f32 z1  = mag * sinf(2.0f * PI * u2) + mean;
  
  rng->prev_norm = z0;
  return z1;
}

f32 rng_fnorm_gen(f32 deviation, f32 mean) {
  return rng_fnorm_gen_r(&global_rng_state, deviation, mean);
}

#if defined(_WIN32)

#include <windows.h>
#include <bcrypt.h>

void platform_get_entropy(void* data, size_t size) {
  BCryptGenRandom(NULL, data, size, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}

#elif defined(__linux__) || defined(__gnu_linux__)

#include <unistd.h>

void platform_get_entropy(void* data, size_t size) {
   getentropy(data, size);
}

#else
    #error "Platform not supported"
#endif

#endif // RANDOM_IMPLEMENTATION

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

#endif // RANDOM_LIBRARY

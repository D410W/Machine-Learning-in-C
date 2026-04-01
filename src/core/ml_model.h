#ifndef ML_MODEL_LIBRARY
#define ML_MODEL_LIBRARY

#include "arena.h"
#include "tensors.h"

typedef enum {
  MV_FLAG_NONE = 0,
  
  MV_FLAG_REQUIRES_GRAD  = (1 << 0),
  MV_FLAG_PARAMETER      = (1 << 1),
  MV_FLAG_INPUT          = (1 << 2),
  MV_FLAG_OUTPUT         = (1 << 3),
  MV_FLAG_DESIRED_OUTPUT = (1 << 4),
  MV_FLAG_COST           = (1 << 5),  
} ModelVarFLags;

typedef enum {
  MV_OP_NULL,
  MV_OP_CREATE,
  
  _MV_OP_UNARY_START,
  
  MV_OP_RELU,
  MV_OP_SOFTMAX,
  
  _MV_OP_BINARY_START,
  
  MV_OP_ADD,
  MV_OP_SUB,
  MV_OP_MATMUL,
  MV_OP_CROSS_ENTROPY,
} ModelVarOp;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_OP_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

typedef struct ModelVar {
  size_t index;
  u32 flags;
  
  Matrix* value;
  Matrix* gradient;
  
  ModelVarOp op;
  struct ModelVar* inputs[MODEL_VAR_MAX_INPUTS];
} ModelVar;

typedef struct {
  size_t size;
  ModelVar** vars;
} ModelProgram;

typedef struct {
  size_t num_vars;
  
  ModelVar* input;
  ModelVar* output;
  ModelVar* desired_output;
  ModelVar* cost;
  
  ModelProgram forward_program;
  ModelProgram cost_program;
} ModelContext;

typedef struct {
  Matrix* train_input;
  Matrix* train_output;
  Matrix* test_input;
  Matrix* test_output;
  
  size_t epochs;
  size_t batch_size;
  f32 learning_rate;
} ModelTrainingDesc;

ModelVar* mv_create(ArenaAlloc* arena, ModelContext* model, size_t rows, size_t cols, u32 flags);
ModelVar* mv_add(ArenaAlloc* arena, ModelContext* model, ModelVar* input_a, ModelVar* input_b, u32 flags);
ModelVar* mv_sub(ArenaAlloc* arena, ModelContext* model, ModelVar* input_a, ModelVar* input_b, u32 flags);
ModelVar* mv_matmul(ArenaAlloc* arena, ModelContext* model, ModelVar* input_a, ModelVar* input_b, u32 flags);

ModelVar* mv_relu(ArenaAlloc* arena, ModelContext* model, ModelVar* input, u32 flags);
ModelVar* mv_softmax(ArenaAlloc* arena, ModelContext* model, ModelVar* input, u32 flags);
ModelVar* mv_cross_entropy(ArenaAlloc* arena, ModelContext* model, ModelVar* p, ModelVar* q, u32 flags);

ModelProgram model_program_create(ArenaAlloc* arena, ModelContext* model, ModelVar* output);
void model_program_compute(ModelProgram* program); // forward pass
void model_program_compute_grads(ModelProgram* program); // backwards pass

ModelContext* model_create(ArenaAlloc* arena);
void model_compile(ArenaAlloc* arena, ModelContext* model);
void model_feedforward(ModelContext* model);
void model_train(ModelContext* model, const ModelTrainingDesc* training_desc);

#ifdef ML_MODEL_IMPLEMENTATION
#undef ML_MODEL_IMPLEMENTATION

ModelVar* mv_create(ArenaAlloc* arena, ModelContext* model, size_t rows, size_t cols, u32 flags) {
  ModelVar* out = PUSH_STRUCT(arena, ModelVar);
  
  if (out == NULL) return NULL;
  
  out->index = model->num_vars++;
  out->op = MV_OP_CREATE;
  out->flags = flags;
  out->value = mat_create(arena, rows, cols);
  
  if (out->value == NULL) {
    arena_pop(arena, sizeof(ModelVar));
    return NULL;
  }
  
  if (flags & MV_FLAG_REQUIRES_GRAD) {
    out->gradient = mat_create(arena, rows, cols);
    
    if (out->gradient == NULL) {
      arena_pop(arena, sizeof(Matrix) + sizeof(f32) * rows * cols);
      arena_pop(arena, sizeof(ModelVar));
      return NULL;
    }
  }
  
  if (flags & MV_FLAG_INPUT) { model->input = out; }
  if (flags & MV_FLAG_OUTPUT) { model->output = out; }
  if (flags & MV_FLAG_DESIRED_OUTPUT) { model->desired_output = out; }
  if (flags & MV_FLAG_COST) { model->cost = out; }

  return out;
}

ModelVar* _mv_unary_impl(ArenaAlloc* arena, ModelContext* model, ModelVar* input, size_t rows, size_t cols, ModelVarOp op, u32 flags) {
  if (input->flags & MV_FLAG_REQUIRES_GRAD) {
    flags |= MV_FLAG_REQUIRES_GRAD;
  }
  
  ModelVar* out = mv_create(arena, model, rows, cols, flags);
  if (out == NULL) return NULL;
  
  out->op = op;
  out->inputs[0] = input;
  
  return out;
}

ModelVar* _mv_binary_impl(ArenaAlloc* arena, ModelContext* model, ModelVar* input_a, ModelVar* input_b, size_t rows, size_t cols, ModelVarOp op, u32 flags) {
  if ((input_a->flags | input_b->flags) & MV_FLAG_REQUIRES_GRAD) {
    flags |= MV_FLAG_REQUIRES_GRAD;
  }
  
  ModelVar* out = mv_create(arena, model, rows, cols, flags);
  if (out == NULL) return NULL;
  
  out->op = op;
  out->inputs[0] = input_a;
  out->inputs[1] = input_b;
  
  return out;
}

ModelVar* mv_add(ArenaAlloc* arena, ModelContext* model, ModelVar* input_a, ModelVar* input_b, u32 flags) {
  if (input_a->value->rows != input_b->value->rows || input_a->value->cols != input_b->value->cols)
    return NULL;
    
  return _mv_binary_impl(arena, model, input_a, input_b, input_a->value->rows, input_a->value->cols, MV_OP_ADD, flags);
}

ModelVar* mv_sub(ArenaAlloc* arena, ModelContext* model, ModelVar* input_a, ModelVar* input_b, u32 flags) {
  if (input_a->value->rows != input_b->value->rows || input_a->value->cols != input_b->value->cols)
    return NULL;
    
  return _mv_binary_impl(arena, model, input_a, input_b, input_a->value->rows, input_a->value->cols, MV_OP_SUB, flags);
}

ModelVar* mv_matmul(ArenaAlloc* arena, ModelContext* model, ModelVar* input_a, ModelVar* input_b, u32 flags) {
  if (input_a->value->cols != input_b->value->rows) return NULL;

  return _mv_binary_impl(arena, model, input_a, input_b, input_a->value->rows, input_b->value->cols, MV_OP_MATMUL, flags);
}

ModelVar* mv_relu(ArenaAlloc* arena, ModelContext* model, ModelVar* input, u32 flags) {
  return _mv_unary_impl(arena, model, input, input->value->rows, input->value->cols, MV_OP_RELU, flags);
}

ModelVar* mv_softmax(ArenaAlloc* arena, ModelContext* model, ModelVar* input, u32 flags) {
  return _mv_unary_impl(arena, model, input, input->value->rows, input->value->cols, MV_OP_SOFTMAX, flags);
}

ModelVar* mv_cross_entropy(ArenaAlloc* arena, ModelContext* model, ModelVar* p, ModelVar* q, u32 flags) {
  if (p->value->rows != q->value->rows || p->value->cols != q->value->cols)
    return NULL;
    
  return _mv_binary_impl(arena, model, p, q, p->value->rows, p->value->cols, MV_OP_CROSS_ENTROPY, flags);
}

ModelProgram model_program_create(ArenaAlloc* arena, ModelContext* model, ModelVar* output_var) {
  ArenaAllocTemp scratch = arena_scratch_begin(MiB(64)); 
  
  bool* visited = PUSH_ARRAY(scratch.arena, bool, model->num_vars);
  
  size_t stack_size = 0;
  size_t out_size = 0;
  ModelVar** stack = PUSH_ARRAY(scratch.arena, ModelVar*, model->num_vars);
  ModelVar** out = PUSH_ARRAY(scratch.arena, ModelVar*, model->num_vars);
  
  if (stack == NULL) {
    printf("null encountered\n");
  }
  
  stack[stack_size++] = output_var;
  
  while (stack_size > 0) {
    ModelVar* current = stack[stack_size - 1];
    
    if (current->index >= model->num_vars) { continue; } // shouldn't be possible
    
    if (visited[current->index]) {
      --stack_size; // taking it out
      out[out_size++] = current;
      continue;
    }
    
    visited[current->index] = true;
    
    size_t num_inputs = MV_OP_NUM_INPUTS(current->op);
    for (size_t i = 0; i < num_inputs; ++i) {
      ModelVar* input = current->inputs[i];
      
      if (input->index >= model->num_vars || visited[input->index]) { continue; }
      
      stack[stack_size++] = input;
    }
  }
  
  ModelProgram program = {
    .size = out_size,
    .vars = PUSH_ARRAY_NZ(arena, ModelVar*, out_size),
  };
  memcpy(program.vars, out, sizeof(ModelVar*) * out_size);
  
  arena_temp_end(scratch);
  return program;
}

void model_program_compute(ModelProgram* program) { // forward pass
  for (size_t i = 0; i < program->size; ++i) {
    ModelVar* current = program->vars[i];
    
    ModelVar* a = current->inputs[0];
    ModelVar* b = current->inputs[1];
    
    switch (current->op) {
      case MV_OP_NULL:
      case MV_OP_CREATE: break;
      
      case _MV_OP_UNARY_START: break;
      
      case MV_OP_RELU: { mat_relu(current->value, a->value); } break;
      case MV_OP_SOFTMAX: { mat_softmax(current->value, a->value); } break;
      
      case _MV_OP_BINARY_START: break;
      
      case MV_OP_ADD: { mat_add(current->value, a->value, b->value); } break;
      case MV_OP_SUB: { mat_sub(current->value, a->value, b->value); } break;
      case MV_OP_MATMUL: { mat_mul(current->value, a->value, b->value, 1, 0, 0); } break;
      case MV_OP_CROSS_ENTROPY: { mat_cross_entropy(current->value, a->value, b->value); } break;      
    }
  }
}

void model_program_compute_grads(ModelProgram* program) { // backwards pass
  for (size_t i = 0; i < program->size; ++i) {
    ModelVar* current = program->vars[i];
    
    if ((current->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) continue;
    if (current->flags & MV_FLAG_PARAMETER) continue;
    
    mat_clear(current->gradient);
  }
  
  mat_fill(program->vars[program->size - 1]->gradient, 1.0f);
  
  for (size_t i = program->size; i > 0; --i) {
    ModelVar* current = program->vars[i - 1];
    // printf("i = %ld\n", i);
    ModelVar* a = current->inputs[0];
    ModelVar* b = current->inputs[1];
    
    size_t num_inputs = MV_OP_NUM_INPUTS(current->op);
    
    if (!(current->flags & MV_FLAG_REQUIRES_GRAD)) continue;
    if (num_inputs == 1 && !(a->flags & MV_FLAG_REQUIRES_GRAD)) continue;
    if (num_inputs == 2 && !(a->flags & MV_FLAG_REQUIRES_GRAD)
                        && !(b->flags & MV_FLAG_REQUIRES_GRAD)) continue;
    
    switch (current->op) {
      case MV_OP_NULL:
      case MV_OP_CREATE: break;
      
      case _MV_OP_UNARY_START: break;
      
      case MV_OP_RELU: {
        mat_relu_add_grad(a->gradient, a->value, current->gradient);
      } break;
      case MV_OP_SOFTMAX: {
        mat_softmax_add_grad(a->gradient, current->value, current->gradient);
      } break;
      
      case _MV_OP_BINARY_START: break;
      
      case MV_OP_ADD: {
        if (a->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_add(a->gradient, a->gradient, current->gradient); // adding the partial derivative dc/da i think
        }
        if (b->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_add(b->gradient, b->gradient, current->gradient);
        }
      } break;
      case MV_OP_SUB: {
        if (a->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_add(a->gradient, a->gradient, current->gradient); // adding the partial derivative dc/da i think
        }
        if (b->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_sub(b->gradient, b->gradient, current->gradient);
        }
      } break;
      case MV_OP_MATMUL: {
        if (a->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_mul(a->gradient, current->gradient, b->value, 0, 0, 1);
        }
        if (b->flags & MV_FLAG_REQUIRES_GRAD) {
          mat_mul(b->gradient, a->value, current->gradient, 0, 1, 0);
        }
      } break;
      case MV_OP_CROSS_ENTROPY: {
        ModelVar* p = a;
        ModelVar* q = b;
        
        mat_cross_entropy_add_grad(p->gradient, q->gradient, p->value, q->value, current->gradient);
      } break;
    }
  }
}

ModelContext* model_create(ArenaAlloc* arena) {
  ModelContext* model = PUSH_STRUCT(arena, ModelContext);
  return model;
}

void model_compile(ArenaAlloc* arena, ModelContext* model) {
  if (model->output != NULL) {
    model->forward_program = model_program_create(arena, model, model->output);
  }
  
  if (model->cost != NULL) {
    model->cost_program = model_program_create(arena, model, model->cost);
  }
}

void model_feedforward(ModelContext* model) {
  model_program_compute(&model->forward_program);
}

void model_train(ModelContext* model, const ModelTrainingDesc* training_desc) {
  Matrix* train_input = training_desc->train_input;
  Matrix* train_output = training_desc->train_output;
  Matrix* test_input = training_desc->test_input;
  Matrix* test_output = training_desc->test_output;
  
  size_t num_examples = train_input->rows;
  size_t num_tests = test_input->rows;
  size_t input_size = train_input->cols;
  size_t output_size = train_output->cols;
  
  size_t num_batches = num_examples / training_desc->batch_size;
  
  ArenaAllocTemp scratch = arena_scratch_begin(0);
  
  size_t* training_order = PUSH_ARRAY_NZ(scratch.arena, size_t, num_examples);
  for (size_t i = 0; i < num_examples; ++i) training_order[i] = i;
  
  for (size_t epoch = 0; epoch < training_desc->epochs; ++epoch) {
    for (size_t i = 0; i < num_examples; ++i) {
      size_t a = rng_gen() % num_examples;
      size_t b = rng_gen() % num_examples;
      
      size_t tmp = training_order[b];
      training_order[b] = training_order[a];
      training_order[a] = tmp;
    }
    
    for (size_t batch = 0; batch < num_batches; ++batch) {
      for (size_t i = 0; i < model->cost_program.size; ++i) {
        ModelVar* current = model->cost_program.vars[i];
        
        if (current->flags & MV_FLAG_PARAMETER) {
          mat_clear(current->gradient);
        }
      }
      
      f32 average_cost = 0.0f;
      for (size_t i = 0; i < training_desc->batch_size; ++i) {
        size_t order_index = batch * training_desc->batch_size + i;
        size_t index = training_order[order_index];
        
        mat_copy_section(model->input->value, train_input, index * input_size);
        mat_copy_section(model->desired_output->value, train_output, index * output_size);
        
        model_program_compute(&model->cost_program);
        model_program_compute_grads(&model->cost_program);
        
        average_cost = mat_sum(model->cost->value);
      }
      average_cost /= (f32)training_desc->batch_size;
      
      for (size_t i = 0; i < model->cost_program.size; ++i) {
        ModelVar* current = model->cost_program.vars[i];
        
        if ((current->flags & MV_FLAG_PARAMETER) != MV_FLAG_PARAMETER) {
          continue;
        }
        
        mat_scale(current->gradient, training_desc->learning_rate / training_desc->batch_size);
        mat_sub(current->value, current->value, current->gradient);
      }
      printf("Epoch %2ld / %2ld, Batch %4ld / %4ld, Average Cost: %.4f\r",
            epoch + 1, training_desc->epochs,
            batch + 1, num_batches, average_cost
      );
    }
    printf("\n");
    
    size_t num_correct = 0;
    f32 average_cost = 0.0f;
    for (size_t i = 0; i < num_tests; ++i) {
      mat_copy_section(model->input->value, test_input, i * input_size);
      mat_copy_section(model->desired_output->value, test_output, i * output_size);
      
      model_program_compute(&model->cost_program);
      
      average_cost += mat_sum(model->cost->value);
      num_correct += mat_argmax(model->output->value) == mat_argmax(model->desired_output->value);
    }
    average_cost /= (f32)num_tests;
    printf("Testing completed. Accuracy: %5ld / %ld (%.1f%%), Average cost: %.4f\n",
           num_correct, num_tests, (f32)num_correct / (f32)num_tests * 100.0f, average_cost);
  } 
  
  arena_scratch_end(scratch);
}

#endif // ML_MODEL_IMPLEMENTATION

#endif // ML_MODEL_LIBRARY

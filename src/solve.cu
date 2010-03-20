#include "interfaces/solve.h"
#include "interfaces/dispatch.h"
#include "heuristic.h"
#include "dpll.h"


// sets the number of threads

#ifndef THREAD_NUM
#define THREAD_NUM 64
#warning should define THREAD_NUM !
#endif

/*
* we define here functions for preparing the solve, ie getting the GPU memory in the desired state
* we also define the solving threads kernels here
*/

/*
* chooses immutable vars and sets them differently for each thread
*/
__host__ void
prepare_presets( atom_t * formula, atom_t * clauses_index, value_t * vars, int clause_n, int var_n, int thread_n, value_t * all_vars)
{
  value_t *all_vars = calloc(thread_n * (var_n+1), sizeof(value_t));

#ifdef DEBUG
  printf("sorts vars by value\n");
#endif
  
  int sorted_vars[var_n+1];
  sort_vars_by_value( formula, clauses_index, all_vars, sorted_vars, clause_n, var_n );

  // sets immutable vars (differently for each thread...)
#ifdef DEBUG
  printf("chooses immutable vars and sets them\n");
#endif
  set_immutable_vars( all_vars, sorted_vars, var_n, THREAD_NUM );
}

/*
* this function transfers the structures to the gpu global memory
*/
__host__ void
prepare_gpu_memory( atom_t * formula, __device__ atom_t * formula_d, atom_t * clauses_index, __device__ atom_t * clauses_index_d, value_t * vars_affectations, __device__ value_t * vars_affectations_d, int clause_n, int var_n, truth_t * answers, __device__ truth_t * answers_d, int thread_n)
{
    // first, we allocate the meomry on the device
    size_t fomrmula_size = (clauses_index[clause_n] - clauses_index[0]+ 1) * sizeof(atom_t);
    size_t clauses_index_size = clause_n * sizeof(atom_t);
    size_t vars_size = thread_n * (var_n + 1) * sizeof( value_t);
    size_t ans_size = thread_n * sizeof( truth_t);
    cudaMalloc( (void **) &formula_d, fomrmula_size);
    cudaMalloc( (void **) &clauses_index_d, clauses_index_size);
    cudaMalloc( (void **) &vars_affectations_d, vars_size);
    cudaMalloc( (void **) &answers_d, ans_size);
    // now we transfer data to the device
    cudaMemcpy( formula_d, formula, fomrmula_size, cudaMemcpyHostToDevice);
    cudaMemcpy( clauses_index_d, clauses_index, clauses_index_size, cudaMemcpyHostToDevice);
    cudaMemcpy( vars_affectations_d, vars_affectations, vars_size, cudaMemcpyHostToDevice);
    cudaMemcpy( answers_d, answers, ans_size, cudaMemcpyHostToDevice);

}


/*
* this function is a thread solving one of the 2^k variable tries.
* it uses its thread id to recover which variables were chosen, and their affectations
*/
__global__ void
cuda_solve ( atom_t * formula, atom_t * clause_index, value_t * vars_affectations, int clause_n, int var_n, truth_t * answers, int thread_n)
{
    int id = threadIdx.x;
    assert( id < thread_n);
    // there exists a global GPU memory array storing vars value for each thread
    // we retrieve it
    value_t * vars_in_global = vars_affectations[id];
    // now we copy this data into shared memory
    __shared__ value_t * vars;
    size_t size = var_n * sizeof( value_t);
    cudaMalloc( (void **) & vars, size);
    cudaMemcpy( vars, vars_in_global, size, cudaMemcpyDeviceToDevice);

    truth_t result = solve_thread( formula, clause_index, vars, clause_n, var_n);

    answers[id] = result;

    // now we free our shared memory
    cudaFree( vars);

    return;
}

success_t
solve ( atom_t *formula, atom_t* clauses_index, int clause_n, int var_n )
{
#ifdef DEBUG
  printf("uses %d threads on cuda\n", THREAD_NUM);
#endif

  value_t * vars_affectations;
  // we select and preset k variablees, where 2^k = THREAD_NUM
  prepare_presets( formula, clauses_index, vars, clause_n, var_n, THREAD_NUM, vars_affectations);
  
  __device__ atom_t * formula_d;
  __device__ atom_t * clauses_index_d;
  __device__ value_t * vars_affectations_d;
  __device__ truth_t * answers_d;
  truth_t * answers = malloc ( THREAD_NUM * sizeof(truth_t));

  // transfering all data to the gpu global memory
  prepare_gpu_memory( formula, formula_d, clauses_index, clauses_index_d, vars_affectations, vars_affectations_d, clause_n, var_n, answers, answers_d, THREAD_NUM);

  // now we call the cuda kernel to solve each instance
  cuda_solve<<<THREAD_NUM,1>>> ( formula_d, clause_index_d, vars_affectations_d, clause_n, var_n, answers_d, THREAD_NUM);

  cudaMemcpy( answers, answers_d, THREAD_NUM * sizeof(truth_t), cudaMemcpyDeviceToHost);

  truth_t answer = FALSE;

  // free the now useless arrays
  cudaFree(formula_d);
  cudaFree(clauses_index_d);
  cudaFree(vars_affectations_d);
  cudaFree(answers_d);
  free(vars_affectations);
  // find out wether one thread found the formula to be satisfiable
  for ( int i = 0; i < THREAD_NUM; ++i)
    answer = answer || answers[i];
  free(answers);
  if ( answer == TRUE )
    return SUCCESS;
  else
    return FAILURE;
}

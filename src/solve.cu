#include "interfaces/dispatch.h"
#include "heuristic.h"
#include "heuristic.c"
#include "dpll_while.c"
#include <stdlib.h>
#include <errno.h>


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
* global variables for device memory, as it seems to be required for compilation
*/
__device__ atom_t * formula_d;
__device__ atom_t * clauses_index_d;
__device__ value_t * vars_affectations_d;
__device__ truth_t * answer_d;
__device__ satisfied_t * satisfied_clauses_d;

/*
* chooses immutable vars and sets them differently for each thread
*/
__host__ void
prepare_presets( atom_t * formula, atom_t * clauses_index, int clause_n, int var_n, int thread_n, value_t * all_vars)
{

#ifdef DEBUG
  printf("sorts vars by value\n");
#endif
  
  int sorted_vars[var_n+1];
  sort_vars_by_value( formula, clauses_index, all_vars, sorted_vars, clause_n, var_n );

  // sets immutable vars (differently for each thread...)
#ifdef DEBUG
  printf("chooses immutable vars and sets them\n");
#endif
  set_immutable_vars( all_vars, sorted_vars, var_n, thread_n);
}

/*
* this function transfers the structures to the gpu global memory
*/
__host__ void
prepare_gpu_memory( atom_t * formula,  atom_t * formula_d, atom_t * clauses_index,  atom_t * clauses_index_d, value_t * vars_affectations,  value_t * vars_affectations_d, int clause_n, int var_n, truth_t * answer,  truth_t * answer_d, int thread_n, satisfied_t * satisfied_clauses, satisfied_t * satisfied_clauses_d)
{
    // first, we allocate the meomry on the device
    size_t formula_size = (clauses_index[clause_n] - clauses_index[0]+ 1) * sizeof(atom_t);
    size_t clauses_index_size = clause_n * sizeof(atom_t);
    size_t vars_size = thread_n * (var_n + 1) * sizeof( value_t);
    size_t ans_size = sizeof( truth_t);
    size_t satis_size = thread_n * clause_n * sizeof( satisfied_t);
    cudaMalloc( (void **) &formula_d, formula_size);
    cudaMalloc( (void **) &clauses_index_d, clauses_index_size);
    cudaMalloc( (void **) &vars_affectations_d, vars_size);
    cudaMalloc( (void **) &answer_d, ans_size);
    cudaMalloc( (void **) &satisfied_clauses_d, satis_size);
    if ( formula_d == NULL)
    {
      printf("dtc\n");
      exit(-1);
    }
    // now we transfer data to the device
    cudaMemcpy( formula_d, formula, formula_size, cudaMemcpyHostToDevice);
    cudaMemcpy( clauses_index_d, clauses_index, clauses_index_size, cudaMemcpyHostToDevice);
    cudaMemcpy( vars_affectations_d, vars_affectations, vars_size, cudaMemcpyHostToDevice);
    cudaMemcpy( answer_d, answer, ans_size, cudaMemcpyHostToDevice);
    cudaMemcpy( satisfied_clauses_d, satisfied_clauses, satis_size, cudaMemcpyHostToDevice);

}


/*
* this function is a thread solving one of the 2^k variable tries.
* it uses its thread id to recover which variables were chosen, and their affectations
*/
__global__ void
cuda_solve ( atom_t * formula, atom_t * clause_index, value_t * vars_affectations, int clause_n, int var_n, truth_t * answer, int thread_n, satisfied_t * satisfied_clauses)
{
    int block_id = blockIdx.x * blockDim.x;
    int id_in_block = threadIdx.x;
    //extern __shared__ value_t vars_in_global[];
    // we now longer use this
    // TODO : verify this affectation is correct
    //for ( int i = 1; i <= var_n; ++i) 
    //{
    //  vars_in_global[id_in_block*(var_n+1) +i ] = vars_affectations[(8*block_id+id_in_block)*(var_n +1) +i];
    //}
    satisfied_t * threads_satisfied_clauses;
    threads_satisfied_clauses = &satisfied_clauses_d[(block_id + id_in_block) * clause_n];
    // now  we sync threads to ensure we're in a consistent state
    __syncthreads();
    // TODO : verify this affectation is correct
    value_t * vars = vars_affectations_d + ((block_id + id_in_block) * (var_n + 1));
    if (  (block_id + id_in_block ) >= thread_n)
      exit(-2);
    // now call the solver
    success_t result = solve_thread( formula_d, clauses_index_d, vars_affectations_d, clause_n, var_n, threads_satisfied_clauses);
    // store our result
    // TODO : notify other threads if we found the formula to be satisfiable
    if (result == SUCCESS && *answer != TRUE) // no thread found the formula satisfiable before
      *answer = TRUE;
    return;
}

extern "C"
success_t
solve ( atom_t *formula, atom_t* clauses_index, int clause_n, int var_n, int thread_n )
{
#ifdef DEBUG
  printf("uses %d threads on cuda\n", thread_n);
#endif

  value_t * vars_affectations;
  satisfied_t * satisfied_clauses;
  // we select and preset k variablees, where 2^k = thread_n
  //vars_affectations = (value_t *) calloc(thread_n * (var_n+1), sizeof(value_t));
  vars_affectations = (value_t *) calloc(thread_n * (var_n+1) , sizeof(value_t));
  if ( vars_affectations == NULL)
  {
    perror(" calloc failed");
    exit(-1);
  }
  satisfied_clauses = (satisfied_t *) calloc( thread_n * clause_n , sizeof(satisfied_t));
  if ( satisfied_clauses == NULL)
  {
    perror("calloc failed");
    exit(-1);
  }
  prepare_presets( formula, clauses_index, clause_n, var_n, thread_n, vars_affectations);
  
  truth_t * answer = (truth_t *) malloc(sizeof(truth_t));
  *answer = FALSE;

  // apparently, can't decide this size at execution..
  //size_t shared_mem_size = 8 * (var_n +1) * sizeof( value_t);
  //size_t shared_mem_size = 8 * 128 * sizeof( value_t);
  // so no more than 128 variables...

  // transfering all data to the gpu global memory
  prepare_gpu_memory( formula, formula_d, clauses_index, clauses_index_d, vars_affectations, vars_affectations_d, clause_n, var_n, answer, answer_d, thread_n, satisfied_clauses, satisfied_clauses_d);

  // now we call the cuda kernel to solve each instance
  dim3 dimGrid(thread_n/8);
  dim3 dimBlock(8);
  cuda_solve<<<dimGrid, dimBlock>>> ( formula, clauses_index, vars_affectations, clause_n, var_n, answer, thread_n, satisfied_clauses);

  cudaThreadSynchronize();

  cudaMemcpy( answer, answer_d, sizeof(truth_t), cudaMemcpyDeviceToHost);


  // free the now useless arrays
  cudaFree(formula_d);
  cudaFree(clauses_index_d);
  cudaFree(vars_affectations_d);
  cudaFree(answer_d);
  //free(vars_affectations);
  if ( *answer )
    return SUCCESS;
  else
    return FAILURE;
}

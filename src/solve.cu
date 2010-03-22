#include "interfaces/dispatch.h"
#include "heuristic.h"
#include "heuristic.c"
#include "dpll_while.c"


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
__device__ truth_t * answers_d;

/*
* chooses immutable vars and sets them differently for each thread
*/
__host__ void
prepare_presets( atom_t * formula, atom_t * clauses_index, int clause_n, int var_n, int thread_n, value_t * all_vars)
{
  all_vars = (value_t *) calloc(thread_n * (var_n+1), sizeof(value_t));

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
prepare_gpu_memory( atom_t * formula,  atom_t * formula_d, atom_t * clauses_index,  atom_t * clauses_index_d, value_t * vars_affectations,  value_t * vars_affectations_d, int clause_n, int var_n, truth_t * answers,  truth_t * answers_d, int thread_n)
{
    // first, we allocate the meomry on the device
    size_t formula_size = (clauses_index[clause_n] - clauses_index[0]+ 1) * sizeof(atom_t);
    size_t clauses_index_size = clause_n * sizeof(atom_t);
    size_t vars_size = thread_n * (var_n + 1) * sizeof( value_t);
    size_t ans_size = thread_n * sizeof( truth_t);
    cudaMalloc( (void **) &formula_d, formula_size);
    cudaMalloc( (void **) &clauses_index_d, clauses_index_size);
    cudaMalloc( (void **) &vars_affectations_d, vars_size);
    cudaMalloc( (void **) &answers_d, ans_size);
    // now we transfer data to the device
    cudaMemcpy( formula_d, formula, formula_size, cudaMemcpyHostToDevice);
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
    int block_id = blockIdx.x * blockDim.x;
    int id_in_block = threadIdx.x;
    extern __shared__ value_t vars_in_global[];
    // TODO : verify this affectation is correct
    for ( int i = 1; i <= var_n; ++i) 
    {
      vars_in_global[id_in_block*(var_n+1) +i ] = vars_affectations[(block_id+id_in_block)*(var_n +1) +i];
    }
    // now  we sync threads to ensure we're in a consistent state
    __syncthreads();
    // TODO : verify this affectation is correct
    value_t * vars = &vars_in_global[ id_in_block * (var_n+1)];
    // now call the solver
    truth_t result = solve_thread( formula, clause_index, vars, clause_n, var_n);
    // store our result
    // TODO : notify other threads if we found the formula to be satisfiable
    answers[block_id + id_in_block] = result;
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
  prepare_presets( formula, clauses_index, clause_n, var_n, THREAD_NUM, vars_affectations);
  
  truth_t * answers = (truth_t *) malloc ( THREAD_NUM * sizeof(truth_t));

  // apparently, can't decide this size at execution..
  //size_t shared_mem_size = 8 * (var_n +1) * sizeof( value_t);
  size_t shared_mem_size = 8 * 128 * sizeof( value_t);
  // so no more than 128 variables...

  // transfering all data to the gpu global memory
  prepare_gpu_memory( formula, formula_d, clauses_index, clauses_index_d, vars_affectations, vars_affectations_d, clause_n, var_n, answers, answers_d, THREAD_NUM);

  // now we call the cuda kernel to solve each instance
  cuda_solve<<<8,THREAD_NUM/8, shared_mem_size>>> ( formula_d, clauses_index_d, vars_affectations_d, clause_n, var_n, answers_d, THREAD_NUM);

  cudaThreadSynchronize();

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

#include "interfaces/solve.h"
#include "interfaces/dispatch.h"
#include "heuristic.h"

/*
* we define here functions for preparing the solve, ie getting the GPU memory in the desired state
* we also define the solving threads kernels here
*/

/*
* chooses immutable vars and sets them differently for each thread
*/
__host__ void
prepare_presets( atom_t * formula, atom_t * clauses_index, value_t * vars, int clause_n, int var_n, int thread_n)
{
    value_t *all_vars = calloc(THREAD_NUM * (var_n+1), sizeof(value_t));
    value_t sorted_vars[var_n+1];
    choose_immutable_vars( formula, clauses_index, all_vars, sorted_vars, clause_n, var_n );
    set_immutable_vars( all_vars, var_n, THREAD_NUM);
}

/*
* this function transfers the structures to the gpu global memory
*/
__host__ void
prepare_gpu_memory( atom_t * formula, __device__ atom_t * formula_d, atom_t * clauses_index, __device__ atom_t clauses_index_d, value_t * vars_affectations, __device__ value_t ** vars_affectations_d, int clause_n, int var_n, truth_t * answers, int thread_n)
{
    // first, we allocate the meomry on the device
    size_t fomrmula_size = (clauses_index[clause_n] - clauses_index[0]+ 1) * sizeof(atom_t);
    size_t clauses_index_size = clause_n * sizeof(atom_t);
    size_t vars_size = thread_n * (var_n + 1) * sizeof( value_t);
    cudaMalloc( (void **) &formula_d, fomrmula_size);
    cudaMalloc( (void **) &clauses_index_d, clauses_index_size);
    cudaMalloc( (void **) &vars_affectations_d, vars_size);
    // now we transfer data to the device
    cudaMemcpy( formula_d, formula, fomrmula_size, cudaMemcpyHostToDevice);
    cudaMemcpy( clauses_index_d, clauses_index, clauses_index_size, cudaMemcpyHostToDevice);
    cudaMemcpy( vars_affectations_d, vars_affectations, vars_size, cudaMemcpyHostToDevice);

}


/*
* this function is a thread solving one of the 2^k variable tries.
* it uses its thread id to recover which variables were chosen, and their affectations
*/
__global__ void
solve ( atom_t * formula, atom_t * clause_index, value_t * vars_affectations, int clause_n, int var_n, truth_t * answers, int thread_n)
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

    return;

}

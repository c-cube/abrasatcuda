/*
 * This header defines an interface that the main() can use to solve 
 * a problem. It provides only a function, solve(), which purpose
 * is to dispatch solving work on one or many threads (or even on
 * network...) and then gather the results to give them back to 
 * the caller.
 * It uses a solve_thread (module implementing solve.h) as a basic 
 * computing unit.
 */



#ifndef DISPATCH_H
#define DISPATCH_H

#include "../consts.h"
#include "../clause.h"
#include "../vars.h"



/*
 * this function's purpose is to manage to solve the problem.
 * It relies on solve_thread (one or many instances) to do so.
 */

#ifdef CUDA
__global__ void solve( atom_t *formula, atom_t* clauses_index, value_t * vars_affectations, int clause_n, int var_n , truth_t * answers, int thread_n);
#else
// TODO : unifify solve declaration with that of abrasatcuda.h
//int solve( atom_t *formula, atom_t* clauses_index, value_t ** vars_affectations, int clause_n, int var_n, truth_t * answers, int thread_n);
#endif

#ifdef CUDA
/*
* chooses immutable vars and sets them differently for each thread
*/
__host__ void
prepare_presets( atom_t * formula, atom_t * clauses_index, value_t * vars, int clause_n, int var_n, int thread_n);

/*
* this functions makes sure the gpu memory is in a proper state
*/
prepare_gpu_memory( atom_t * formula, __device__ atom_t * formula_d, atom_t * clauses_index, __device__ atom_t clauses_index_d, value_t * vars_affectations, __device__ value_t * vars_affectations_d, int clause_n, int var_n, truth_t * answers, int thread_n);
#endif

    // TODO : create CUDA threads, each with its own [vars] array,
    // and dispatch it in CUDA.
    // TODO : find the k most "interesting" vars, and create 2^k threads
    // with different var affectations.
    // TODO : think of a way to share information between threads (for example,
    // if a thread finds out that a var cannot be true (exhausted possibilities ?)
    // it may tell the other threads to set this var to 0)



#endif


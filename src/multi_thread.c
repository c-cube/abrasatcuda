#include "multi_thread.h"
#include "interfaces/solve.h" // solve_thread


#include <pthread.h> // pthread stuff




















/*
 * this solves the problem on several thread
 */
int solve( atom_t *formula, atom_t* clauses_index, int clause_n, int var_n )
{
    // allocates space for n vars
    value_t vars[var_n];

    // initialization
    for (int i=1; i <= var_n; ++i)
        vars[i] = 0;

    return solve_thread( formula, clauses_index, vars, clause_n, var_n );
}


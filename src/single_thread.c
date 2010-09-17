#include "single_thread.h"
#include "interfaces/solve.h" // solve_thread



/*
 * this solves the problem on a single thread, as a normal
 * C program.
 */
success_t solve( atom_t *formula, atom_t* clauses_index, int clause_n, int var_n, int thread_n )
{
    // allocates space for n vars
    value_t vars[var_n];

    // initialization
    for (int i=1; i <= var_n; ++i)
        vars[i] = 0;

    success_t answer = solve_thread( formula, clauses_index, vars, clause_n, var_n );

    if ( answer == SUCCESS )
        value_print( vars, var_n );


    return answer;
}


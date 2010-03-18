#include "heuristic.h"

// holds the address of interest table
static double **interest_ptr = NULL;


// this compares two vars by comparing their "interest value"
static int 
compare( const void *a, const void *b )
{
    value_t my_a = *((value_t*) a);
    value_t my_b = *((value_t*) b);

    // the function uses this to compare vars
    assert( interest_ptr != NULL );

    if ( (*interest_ptr)[my_a] < (*interest_ptr)[my_b] )
        return -1;
    if ( (*interest_ptr)[my_a] > (*interest_ptr)[my_b] )
        return 1;
    else 
        return 0;
}





void
choose_immutable_vars( atom_t *formula, atom_t *clauses_index, value_t *vars, value_t *sorted_vars, int clause_n, int var_n )
{
    // array for handling marks given to every var
    double interest[var_n+1];
    for (int i=1; i <= var_n; ++i)
        interest[i] = 0.0;

    // computes marks
    for (int i=1; i <= var_n; ++i){
        interest[i] = compute_value( formula, clauses_index, vars[i], clause_n );
#ifdef DEBUG
        printf("var %i is given a mark of %lf\n", interest[i]);
#endif
    }

    // initializes the array sorted_vars
    for (int i=0; i <= var_n; ++i )
        sorted_vars[i] = i;

    // sort the array. It uses compare as a compare function.
    // we start sorting vars from the first one, which is at [sorted_vars]+1
    interest_ptr = &interest;
    qsort( sorted_vars+1, var_n, sizeof(value_t), &compare );

#ifdef DEBUG
    // check if sorted
    int is_sorted = 1;
    for (int i=1; i < var_n; ++i ){
        if ( ! ( interest[sorted_vars[i]] <= interest[sorted_vars[i+1]] ) ){
            is_sorted = 0;
            break;
        }
    }
    assert(is_sorted);
#endif
}

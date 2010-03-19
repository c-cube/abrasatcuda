#include "heuristic.h"
#include "vars.h" // to set immutable vars
#include <stdlib.h> // malloc, free

// utility function to set vars
static inline void to_base_two( int *, int);

// holds the address of interest table
static double *interest_ptr = NULL;


/*
 * This function tries to evaluate how important vars are
 * in the formula, int one pass.
 */
static inline void
compute_values( atom_t *formula, atom_t *clauses_index, value_t *vars, double *interest, int clause_n, int var_n )
{
    
    // number of times vars appear positively or negatively in the formula
    int positive_occur_num[var_n+1];
    int negative_occur_num[var_n+1];

    int name = 0;

    for ( int i = 0; i < clause_n; ++i ){
        
        atom_t *clause = formula + (clauses_index[i]);
        atom_t *clause_end = formula + (clauses_index[i+1]);

        // for each atom in the clause
        for ( atom_t *cur_atom = clause; cur_atom < clause_end; ++cur_atom ){
            name = VARIABLE_NAME(*cur_atom);
            if ( IS_NEGATED(*cur_atom) )
                negative_occur_num[name]++;
            else
                positive_occur_num[name]++;
        }
    }

    /*
     * The heuristic is based on two things :
     *      the number of times a var appears, which increases its importance;
     *      the "equilibrium" of those occurrences, ie a var which appears 
     *          as many times positively and negatively is more likely to
     *          be of interest in both positive and negative value (remember
     *          that we try to know which vars are good to be switched on 
     *          from the beginning)
     * This is motivated by the assumption that the more a var appears, the earlier
     * it reveals whether this choice was good or not.
     */

    for (int i = 1; i <= var_n; ++i){

        // total number of occurrences
        int total_occur_num = positive_occur_num[i] + negative_occur_num[i];

        double value = exp( ((double) (total_occur_num)) / 
            ((double) abs(negative_occur_num - positive_occur_num)+1) );
#ifdef DEBUG
        printf("var %d is given a mark of %lf\n", i, value);
#endif

        interest[i] = value;
    }
}


// this compares two vars by comparing their "interest value"
static int 
compare( const void *a, const void *b )
{
    value_t my_a = *((value_t*) a);
    value_t my_b = *((value_t*) b);

    // the function uses this to compare vars
    assert( interest_ptr != NULL );

    if ( interest_ptr[my_a] < interest_ptr[my_b] )
        return -1;
    if ( interest_ptr[my_a] > interest_ptr[my_b] )
        return 1;
    else 
        return 0;
}




// function explained in heuristic.h
void
sort_vars_by_value( atom_t *formula, atom_t *clauses_index, value_t *vars, value_t *sorted_vars, int clause_n, int var_n )
{
    // array for handling marks given to every var
    double interest[var_n+1];

    // compute marks
    compute_values( formula, clauses_index, vars, interest, clause_n, var_n );
    

    // initializes the array sorted_vars
    for (int i=0; i <= var_n; ++i )
        sorted_vars[i] = i;

    // sort the array. It uses compare as a compare function.
    // we start sorting vars from the first one, which is at [sorted_vars]+1
    interest_ptr = interest;
    qsort( sorted_vars+1, var_n, sizeof(value_t), &compare );

#ifdef DEBUG
    // check if sorted
    int is_sorted = 1;
    for (int i=1; i < var_n; ++i ){
        if ( ! ( interest[sorted_vars[i]] <= interest[sorted_vars[i+1]] ) ){
            printf("var %d (value %lf) not <= than var %d (value %lf)\n",
                sorted_vars[i], interest[sorted_vars[i]],
                sorted_vars[i+1], interest[sorted_vars[i+1]]);
            is_sorted = 0;
            break;
        }
    }
    assert(is_sorted);
#endif
}



// TODO : comment ;)
void
set_immutable_vars( value_t * all_vars, int var_n, int thread_n)
{
  
  int * base_two_decomp = (int *) malloc( sizeof(int) * 32);
  int var_count, var_affected;
  for (int i = 0; i < thread_n; ++i)
  {
    to_base_two( base_two_decomp, i);
    var_count = var_affected = 0;
    while (var_count < var_n)
    {
      if ( IS_IMMUTABLE( all_vars[ i * var_n + var_count] ))
      {
        assert( var_affected < 32);
        if ( base_two_decomp[var_affected] )
          SET_TRUE( all_vars[ i * var_n + var_count]);
        else
          SET_FALSE( all_vars[ i * var_n + var_count]);
        ++var_affected;
      }
      ++var_count;
    }
  }
  free(base_two_decomp);
}

// TODO : comment
static inline void
to_base_two( int * base_2_array, int input)
{
  for (int i = 0; i < 32; ++i)
  {
    base_2_array[i] = input % 2;
    input /= 2; // compiler does optimize this I guess :)
  } 
}

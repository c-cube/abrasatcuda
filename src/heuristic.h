/*
 * this file contains stuff for finding what vars are interesting
 */

#ifndef HEURISTIC_H
#define HEURISTIC_H

#include <stdlib.h> // abs
#include <math.h> // exp

#include "vars.h"
#include "consts.h"
#include "clause.h"

/*
 * This function tries to evaluate how important the var is
 * in the formula.
 */
static inline double
compute_value( atom_t *formula, atom_t *clauses_index, value_t var, int clause_n )
{
    // number of times the var appears positively or negatively in the formula
    int positive_occur_num = 0;
    int negative_occur_num = 0;

    for ( int i = 0; i < clause_n; ++i ){
        
        atom_t *clause = formula + (clauses_index[i]);
        atom_t *clause_end = formula + (clauses_index[i+1]);

        // for each atom in the clause, if it is an instance of [var]
        for ( atom_t *cur_atom = clause; cur_atom < clause_end; ++cur_atom ){
            if ( VARIABLE_NAME(*cur_atom) == var ){

                if ( IS_NEGATED(*cur_atom) )
                    negative_occur_num++;
                else
                    positive_occur_num++;
            }
        }

    }

    // total number of occurrences
    int total_occur_num = positive_occur_num + negative_occur_num;
    
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

    return exp( (double) (total_occur_num)/abs( negative_occur_num - positive_occur_num) );
}





/*
 * This function sorts its second argument, with
 * the first values more "interesting" than the last ones.
 * It is used to find on which vars to dispatch in a parallel way
 */
void
choose_immutable_vars( atom_t *formula, atom_t *clauses_index, value_t *vars, value_t *sorted_vars, int clause_n, int var_n );




#endif

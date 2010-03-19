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
 * This function sorts its second argument, with
 * the first values more "interesting" than the last ones.
 * It is used to find on which vars to dispatch in a parallel way
 */
void
sort_vars_by_value( atom_t *formula, atom_t *clauses_index, value_t *vars, int *sorted_vars, int clause_n, int var_n );

/*
* This function sets the immutable vars differently for each thread
*/
void
set_immutable_vars( value_t * all_vars, int *sorted_vars, int var_n, int thread_n);


#endif

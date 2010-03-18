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
choose_immutable_vars( atom_t *formula, atom_t *clauses_index, value_t *vars, value_t *sorted_vars, int clause_n, int var_n );

/*
* This function sets the immutable vars differently for each thread
*/
void
set_immutable_vars( value_t * all_vars, int var_n, int thread_n);


#endif

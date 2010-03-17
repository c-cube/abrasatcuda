/*
 * module for dpll calculations
 */

#ifndef DPLL_H
#define DPLL_H

#include <assert.h>

#include "consts.h"
#include "vars.h"
#include "clause.h"



// only function exported by a solver module
success_t solve_thread( atom_t* formula, atom_t* clauses_index, value_t* vars, int clause_n, int var_n );


#endif

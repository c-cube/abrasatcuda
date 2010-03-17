
#ifndef _BRUTE_FORCE_H
#define _BRUTE_FORCE_H


#include "clause.h"
#include "vars.h"
#include "consts.h"


success_t solve_thread( atom_t* formula, atom_t* clauses_index, value_t* vars, int clause_n, int var_n );


#endif

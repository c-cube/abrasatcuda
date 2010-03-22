#ifndef MULTI_THREAD_H
#define MULTI_THREAD_H

#include "vars.h"
#include "consts.h"
#include "clause.h"


int solve( atom_t *formula, atom_t* clauses_index, int clause_n, int var_n, int thread_n );


#endif

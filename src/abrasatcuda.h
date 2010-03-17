/*
 * main part.
 */


#ifndef ABRASATCUDA_H
#define ABRASATCUDA_H

#include "clause.h"
#include "consts.h"

int solve( atom_t *formula, atom_t* clauses_index, int clause_n, int var_n );

#endif


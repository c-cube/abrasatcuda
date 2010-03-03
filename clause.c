#include "clause.h"

/*
* enables iteration over clauses of the formula
* WARNING : this function DOES increment the index.
*
* preferred usage is : while ( NULL != ( current_clause = next_clause(...) ) )
*/
int
next_clause ( short * clause_array, short * * clause_pointer_array, int * satisfied_clauses_array , unsigned number_of_clauses, unsigned current_clause_index, clause * clause_struct )
{
  while ( satisfied_clauses_array[current_clause_index] )
    current_clause_index++; // we don't want to take care of satisfied clauses
  if ( current_clause_index >= number_of_clauses )
    return -1; // iteration is over
  clause_struct->start = clause_pointer_array[current_clause_index++];
  clause_struct->stop = clause_pointer_array[current_clause_index];
  return 0;
}



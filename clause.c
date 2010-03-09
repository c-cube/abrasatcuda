#include "clause.h"

/*
* enables iteration over clauses of the formula
* WARNING : this function DOES increment the index.
*
* preferred usage is : while ( -1 != next_clause(...)  )
*/
int
next_clause ( short int * clause_array, short int ** clause_pointer_array, int * satisfied_clauses_array , unsigned int number_of_clauses, unsigned int current_clause_index, clause * clause_struct ) {
  while ( satisfied_clauses_array[current_clause_index] )
    current_clause_index++; // we don't want to take care of satisfied clauses
  if ( current_clause_index >= number_of_clauses )
    return -1; // iteration is over
  clause_struct->start = clause_pointer_array[current_clause_index++];
  clause_struct->stop = clause_pointer_array[current_clause_index];
  return 0;
}

/*
* iteration over atoms of a clause
* usage is : while ( -1 != next_atom(...) )
*/
int
next_atom ( clause * clause_struct, short * current_atom )
{
  if ( current_atom == clause_struct->stop  || 
       current_atom < clause_struct->start ){
    return -1;
  }

  current_atom++;
  return 0;
}


#ifndef MY_CLAUSE_H
#define MY_CLAUSE_H

/* 
*  a is of type long
*  IS_USED extracts its first bit,
*  which is to be 1 if a is set
*/
#define IS_USED(a) ( a & 0x8000 )

/*
*  IS_NEGATED extracts 2nd bit,
*  which is 1 if the variable is negated
*/
#define IS_NEGATED(a) ( a & 0x4000 )

/*
* VARIABLE_NAME(a) gives the identifer of the variable
*/
#define VARIABLE_NAME(a) ( a & 0xCFFF )

/* 
*  NEGATE(a) returns not(a), it is to be used as such :
*  a = NEGATE(a)
*/
#define NEGATE(a) ( a ^ 0x4000 )

/*
*  UNUSE(a) sets a to be not used any longer
*  usage is : a = UNUSE(a)
*/
#define UNUSE(a) ( a & 0x7FFF )

typedef struct 
{
  short * clause_array;
  short * start;
  short * stop; /* is the start of the next clause, so should not dereferenced in an iteration over a clause */
} clause;

/*
* enables iteration over clauses of the formula
* WARNING : this function DOES increment the index.
*
* preferred usage is : while ( -1 != next_clause(...)  )
*/
int next_clause( short int * clause_array/* array of clauses*/,
                 short int ** clause_pointer_array /* array of pointers to begining of clauses */,
                 int * satisfied_clauses_array/* array of satisfied clauses */,
                 unsigned number_of_clauses/* number of clauses */,
                 unsigned current_clause_index/* the current clause index in the (short *) * array */,
                 clause * clause_struct/* pointer to the used clause struct */
               );

/*
* iteration over atoms of a clause
* usage is : while ( -1 != next_atom(...) )
*/
int
next_atom ( clause * clause_struct, /* the clause we're iterating over */
            short * current_atom /* pointer to the current atom, should belong to the current clause */
            );
#endif

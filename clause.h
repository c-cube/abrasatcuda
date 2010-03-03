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
extern clause * next_clause( short * /* array of clauses*/,
                      (short *) * /* array of pointers to begining of clauses */,
                      unsigned /* number of clauses */,
                      unsigned /* the current clause index in the (short *) * array */,
                      clause * /* pointer to the used clause struct */
                    )
#endif

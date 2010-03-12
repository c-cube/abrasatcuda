#ifndef MY_CLAUSE_H
#define MY_CLAUSE_H

#include <stdlib.h> // malloc
#include <string.h> // memcpy
#include <assert.h> // assert
#include <stdio.h> // printf

/*
 * logical atoms are represented as short
 */


typedef short atom_t;


/*
 * creates an atom from the relative number 
 * (as read in file)
 */
atom_t make_atom( int n );

/* 
*  a is of type long
*  IS_USED extracts its first bit,
*  which is to be 1 if a is set
*/
#define IS_USED(a) ( (a) & 0x8000 )
#define IS_USED_BINARY(a) ( IS_USED(a) > 0 ? 1 : 0 )


/*
*  IS_NEGATED extracts 2nd bit,
*  which is 1 if the variable is negated
*/
#define IS_NEGATED(a) ( (a) & 0x4000 )
#define IS_NEGATED_BINARY(a) ( IS_NEGATED(a) > 0 ? 1 : 0 )

int is_negated( atom_t atom );

/*
* VARIABLE_NAME(a) gives the identifer of the variable
*/
#define VARIABLE_NAME(a) ( (a) & 0x3FFF )

int variable_name( atom_t atom );

/* 
*  NEGATE(a) returns not(a), it is to be used as such :
*  NEGATE(a)
*/
#define NEGATE(a) ( (a) ^= 0x4000 )

/*
*  UNUSE(a) sets a to be not used any longer
*  usage is : UNUSE(a)
*/
#define UNUSE(a) ( (a) &= 0x7FFF )

//-----------------------------------------------------------------------------
/*
 * Clauses are array of atoms
 */

typedef atom_t* clause_t;


inline atom_t* clause_item( clause_t *clause, int n )
{
    return ((atom_t*)clause) + n;
}

inline void clause_put_there( clause_t* place, atom_t* atoms, int n )
{
    atom_t* start = (atom_t*) place;
    for (int i=0; i<n; ++i){
        start[i] = atoms[i];
    }
}


/*
* iteration over atoms of a clause. [clause] is the [n]_th clause in the [clauses_index] array.
* *[iterator] is affected to the address of the next atom of [clause].
* usage is : 
*
* atom_t *iterator = NULL;
* while ( -1 != next_atom(clause, clauses_index, n, &iterator){ process_atom(*iterator);} )
*/
inline int atom_iterate ( clause_t *clause, atom_t *clauses_index, int n, atom_t **iterator )
{
    if ( iterator == NULL )
        return -1;


    // initialization
    if ( *iterator == NULL ){
        *iterator = ((atom_t*) clause);
        return 0;
    } 
    
    if ( ++(*iterator) == (((atom_t*) clause) + clauses_index[n+1]) )
        return -1;

    return 0;
}




void clause_print( clause_t *clause, atom_t* clauses_index, int n );


//-----------------------------------------------------------------------------
/*
 * formulae are array of clause, but represented as short int array
 */

typedef atom_t* formula_t;

/*
 * returns the clause_t* associated with index n in the formula
 */
inline clause_t *formula_item( atom_t *formula, atom_t *clauses, int n)
{
    return (clause_t*) (formula + (clauses[n]));
}

/*
* given an index, finds the next clause in clause_array. *n is modified.
*
* preferred usage is : while ( -1 != clause_iterate(...)  )
*/

inline int clause_iterate( 
    atom_t *formula,
    atom_t *clauses_index_array,
    int length,
    int *n, 
    clause_t **iterator)
{
    if ( iterator == NULL )
        return -1;

    // if iterator is not initialized, ignore n
    if ( *iterator == NULL ){
        *n = 0;
        *iterator = (clause_t*) formula;
    } else if ( *n >= length ){
        return -1; // end of iteration
    } else {
        *iterator = (clause_t*) (formula+(clauses_index_array[++(*n)]));
    }
    return 0;
}

/*
 * builds a formula from an array of clauses.
 * Each clause must be correct (ie, in a row (as result of make_clause))
 * the function allocates memory and copies what is needed.
 */
atom_t formula_build( 
    atom_t **formula, 
    atom_t **clauses_index, 
    clause_t *clauses, 
    int *clauses_length, 
    int n );
   



inline void formula_print(
    atom_t *formula,
    atom_t *clauses_index,
    int n )
{
    clause_t *iterator = NULL;
    int cur=0;
    while ( clause_iterate( formula, clauses_index, n, &cur, &iterator ) != -1 ){
        clause_print( iterator, clauses_index, cur );
    }
} 


#endif

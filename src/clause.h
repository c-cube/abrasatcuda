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

// typedef atom_t* clause_t; // forget about it, it just obsfucates types.


inline atom_t* clause_item( atom_t* clause, int n )
{
    return ((atom_t*)clause) + n;
}



/*
* iteration over atoms of a clause. [clause] is the [n]_th clause in the [clauses_index] array.
* *[iterator] is affected to the address of the next atom of [clause].
* usage is : 
*
* atom_t *iterator = NULL;
* while ( -1 != atom_iterate(clause, clause_end, n, &iterator){ process_atom(*iterator);} )
*
* or :
* while ( -1 != atom_iterate(clauses_index[n], clauses_index[n+1], n, &iterator )){ ... }
*/
inline int atom_iterate ( atom_t *clause, atom_t *clause_end, atom_t **iterator )
{
    if ( iterator == NULL )
        return -1;


    // initialization
    if ( *iterator == NULL ){
        *iterator = clause;
        return 0;
    } 
    
    if ( ++(*iterator) == clause_end )
        return -1;

    return 0;
}




void clause_print( atom_t *clause, atom_t *clause_end );


//-----------------------------------------------------------------------------
/*
 * formulae are array of clause, but represented as short int array
 */

typedef atom_t* formula_t;

/*
 * returns the atom_t* associated with index n in the formula
 */
inline atom_t *formula_item( atom_t *formula, atom_t *clauses_index, int n)
{
    return formula + (clauses_index[n]);
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
    atom_t **iterator)
{
    if ( iterator == NULL )
        return -1;

    // if iterator is not initialized, ignore n
    if ( *iterator == NULL ){
        *n = 0;
        *iterator = formula;
    } else if ( *n >= length ){
        return -1; // end of iteration
    } else {
        *iterator = formula+(clauses_index_array[++(*n)]);
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
    atom_t *clauses, 
    int *clauses_length, 
    int n );
   



inline void formula_print(
    atom_t *formula,
    atom_t *clauses_index,
    int n )
{
    for (int i = 0; i<n; ++ i){
        clause_print( 
            formula_item( formula, clauses_index, i ), 
            formula_item( formula, clauses_index, i+1 ) );
        if (i<n-1)
            printf("\e[32m /\\ \e[m");
        else
            printf("\n");
    }
} 


#endif

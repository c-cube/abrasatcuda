/*
* utilities for handling clauses
*/


/*
 * logical atoms are represented as short
 */


#ifndef MY_CLAUSE_H
#define MY_CLAUSE_H

#include <stdlib.h> // malloc
#include <string.h> // memcpy
#include <assert.h> // assert
#include <stdio.h> // print

#include "consts.h"



typedef short atom_t;


/*
 * creates an atom from the relative number
 * (as read in file)
 */
static inline atom_t make_atom( int n )
{
    return ( 0x8000                               // used ?
             | (n<0 ? 0x4000 : 0x0)               // negated ?
             | (0x3FFF & (n<0 ? (0xFFFF ^ n)+1 : n) )
             // small part for the name, with binary complement if < 0
           );
}

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


static inline atom_t* clause_item( atom_t* clause, int n )
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
static inline int atom_iterate ( atom_t *clause, atom_t *clause_end, atom_t **iterator )
{
    if ( iterator == NULL )
        return FAILURE;


    // initialization
    if ( *iterator == NULL ){
        *iterator = clause;
        return SUCCESS;
    }

    if ( ++(*iterator) == clause_end )
        return FAILURE;

    return SUCCESS;
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
static inline atom_t *formula_item( atom_t *formula, atom_t *clauses_index, int n)
{
    return formula + (clauses_index[n]);
}

/*
* given an index, finds the next clause in clause_array. *n is modified.
*
* preferred usage is : while ( -1 != clause_iterate(...)  )
*/

static inline int clause_iterate(
    atom_t *formula,
    atom_t *clauses_index_array,
    int length,
    int *cur_index,
    atom_t **iterator)
{
    if ( iterator == NULL )
        return FAILURE;

    // if iterator is not initialized, ignore n
    if ( *iterator == NULL ){
        *cur_index = 0;
        *iterator = formula;
    } else {
        if ( *cur_index >= length ){
            return FAILURE; // end of iteration
        } else {
            *iterator = formula+(clauses_index_array[++(*cur_index)]);
        }
    }
    return SUCCESS;
}





static inline void formula_print(
    atom_t *formula,
    atom_t *clauses_index,
    int n )
{
    int i;
    for (i = 0; i<n; ++ i){
        clause_print(
            formula_item( formula, clauses_index, i ),
            formula_item( formula, clauses_index, i+1 ) );
        if (i<n-1)
            print("\033[36m /\\ \033[m\n");
        else
            print("\n");
    }
}


#endif

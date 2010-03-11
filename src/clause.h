#ifndef MY_CLAUSE_H
#define MY_CLAUSE_H

#include <stdlib.h> // malloc
#include <string.h> // memcpy
#include <assert.h>

/*
 * logical atoms are represented as short
 */


typedef short atom_t;


/*
 * creates an atom from the relative number 
 * (as read in file)
 */
inline atom_t make_atom( int n )
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

/*
* VARIABLE_NAME(a) gives the identifer of the variable
*/
#define VARIABLE_NAME(a) ( (a) & 0x3FFF )

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

typedef struct __clause_t
{
    atom_t *stop; /* is the start of the next clause, so should not get dereferenced 
  in an iteration over a clause (maybe last item)*/
    atom_t *clause_array; // starts right here !
} clause_t;

// returns the number of atoms+1, ie the size of the clause_array + 1
inline atom_t clause_length( clause_t *clause )
{
    atom_t answer = (clause->stop - clause->clause_array);
    return (atom_t) answer;
}


inline atom_t* clause_item( clause_t *clause, int n )
{
    atom_t* answer = clause->clause_array + n;
    assert( answer < clause->stop );
    return answer;
}

inline clause_t* make_clause( clause_t* place, int n )
{
    place->clause_array =  (place->clause_array);
    place->stop = ( (place->clause_array)) + (n+1);

    assert( clause_length( place ) == n+1 );

    return place;
}



/*
* iteration over atoms of a clause
* usage is : 
*
* atom_t *iterator = NULL;
* while ( -1 != next_atom(clause_ptr, &iterator){ process_atom(*iterator);} )
*/
inline int atom_iterate ( clause_t * clause, atom_t ** iterator )
{
    if ( iterator == NULL )
        return -1;


    // initialization
    if ( *iterator == NULL ){
        *iterator = (clause->clause_array);
        return 0;
    } 
    
    if ( ++(*iterator) == clause->stop )
        return -1;

    return 0;
}




inline void clause_print( clause_t *clause )
{
    atom_t *iterator = NULL;
    int is_first = 1;
    printf("\e[32m(\e[m");

    while ( atom_iterate( clause, &iterator) != -1 ){
        if (is_first)
            is_first = 0;
        else
            printf("\e[32m v \e[m");
        if (IS_NEGATED( *iterator ))
            printf("-");
        printf("%d", VARIABLE_NAME( *iterator ));

    } 
    printf("\e[32m)\e[m");
}


//-----------------------------------------------------------------------------
/*
 * formulae are array of clause, but represented as short int array
 */

typedef atom_t * formula_t;

/*
 * returns the clause_t* associated with index n in the formulae
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
        *iterator = (clause_t*) &(formula);
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
inline atom_t formula_build( atom_t **formula, atom_t **clauses_index, clause_t *clauses, int n )
{
    atom_t offset;

    assert( formula != NULL && *formula == NULL );
    assert( clauses_index != NULL && *clauses_index == NULL );

    int formula_size = 42 * n  ;
    *formula = malloc( formula_size * sizeof(atom_t) );
    *clauses_index = malloc(n * sizeof(atom_t));

    for (int i=0; i<n; ++i){
        while ( offset + clause_length( &clauses[i] ) >= formula_size ){
            // allocate more space if needed
            formula_size = (int) (formula_size * 1.5);
            *formula = realloc( *formula, formula_size );
        }
        
        // add clause to formula
        memcpy( *formula + offset, &clauses[i], clause_length(&clauses[i])+1 );

        atom_t old_offset = offset; // upgrade offset
        offset += clause_length(&clauses[i])+1;

        (*clauses_index)[i] = old_offset; // upgrade (i-th clause)->stop
        formula_item(*formula,*clauses_index,i)->stop = *formula + offset;
    }

    return offset;
}
   



inline void formula_print(
    atom_t *formula,
    atom_t *clauses_index,
    int length )
{
    clause_t *iterator = NULL;
    int n=0;
    while ( clause_iterate( formula, clauses_index, length, &n, &iterator ) != -1 ){
        clause_print( iterator );
    }
} 


#endif

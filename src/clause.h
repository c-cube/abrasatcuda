#ifndef MY_CLAUSE_H
#define MY_CLAUSE_H

typedef struct 
{
  short int * clause_array;
  short int * stop; /* is the start of the next clause, so should not get dereferenced 
  in an iteration over a clause */
} clause_t;


#define CLAUSE_LENGTH(c) ((int)((c).stop - (c).clause_array))

/*
 * creates an atom from the relative number 
 * (as read in file)
 */
inline short make_atom( int n )
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


/*
* given an index, finds the next clause in clause_array. *n is modified.
*
* preferred usage is : while ( -1 != clause_iterate(...)  )
*/
inline short int clause_iterate( 
    short int *formula,
    int *clauses_index_array,
    int length,
    int *n, 
    clause_t **iterator)
{
    if ( iterator == NULL )
        return -1;

    // if iterator is not initialized, ignore n
    if ( *iterator == NULL ){
        printf("coin\n");
        *n = 0;
        printf("coin\n");
        *iterator = (clause_t*) &formula[0];
        printf("coin\n");
        (*iterator)->stop =  length > 1 ? 
            &formula[clauses_index_array[1]] : NULL;
        printf("coin\n");
    } else if ( *n >= length ){
        return -1; // end of iteration
    } else {
        printf("pan\n");
        *iterator = (clause_t*) &formula[clauses_index_array[++(*n)]];
        (*iterator)->stop = (*n < length) ? 
            &formula[clauses_index_array[(*n)+1]] : NULL;
    }
    return 0;
}




/*
* iteration over atoms of a clause
* usage is : 
*
* short *iterator = NULL;
* while ( -1 != next_atom(clause_ptr, &iterator){ process_atom(*iterator);} )
*/
inline int atom_iterate ( clause_t * clause_struct, short ** iterator )
{
    if ( iterator == NULL )
        return -1;


    // initialization
    if ( * iterator == NULL ){
        *iterator = clause_struct->clause_array;
        return 0;
    } 
    
    if ( ++(*iterator) >= clause_struct->stop )
        return -1;

    return 0;
}




inline void clause_print( clause_t *clause )
{
    short int *iterator = NULL;
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


#endif

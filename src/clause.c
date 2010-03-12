#include "clause.h"


int is_negated( atom_t atom ){  return atom & 0x4000; }

int variable_name( atom_t atom ){   return atom & 0x3FFF; }



short make_atom( int n )
{
    return ( 0x8000                               // used ?
             | (n<0 ? 0x4000 : 0x0)               // negated ?
             | (0x3FFF & (n<0 ? (0xFFFF ^ n)+1 : n) ) 
             // small part for the name, with binary complement if < 0
           );
}


atom_t formula_build( 
    atom_t **formula, 
    atom_t **clauses_index, 
    clause_t *clauses, 
    int *clauses_length, 
    int n )
{
    atom_t offset = 0;

    assert( formula != NULL);
    assert( *formula == NULL);
    assert( clauses_index != NULL); 
    assert( *clauses_index == NULL);

    int formula_size = 42 * n  ;
    *formula = malloc( formula_size * sizeof(atom_t) );
    *clauses_index = malloc(n * sizeof(atom_t));

    for (int i=0; i<n; ++i){
        while ( offset + clauses_length[i] >= formula_size ){
            // allocate more space if needed
            formula_size = (int) (formula_size * 1.5);
            *formula = realloc( *formula, formula_size );
        }
        
        // add clause to formula
        memcpy( *formula + offset, &clauses[i], clauses_length[i] );

        atom_t old_offset = offset; // upgrade offset
        offset += clauses_length[i]+1;

        (*clauses_index)[i] = old_offset;
    }

    return offset;
}



void clause_print( clause_t *clause, atom_t* clauses_index, int n )
{
    atom_t *iterator = NULL;
    int is_first = 1;
    printf("\e[32m(\e[m");

    while ( atom_iterate( clause, clauses_index, n, &iterator) != -1 ){
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

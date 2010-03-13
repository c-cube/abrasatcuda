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





void clause_print( atom_t *clause, atom_t* clause_end )
{
    atom_t *iterator = NULL;
    int is_first = 1;
    printf("\033[32m(\033[m");

    while ( atom_iterate( clause, clause_end, &iterator) != -1 ){
        if (is_first)
            is_first = 0;
        else
            printf("\033[32m v \033[m");
        if (IS_NEGATED( *iterator ))
            printf("-");
        printf("%d", VARIABLE_NAME( *iterator ));
    } 
    printf("\033[32m)\033[m");
}

#include "clause.h"


int is_negated( atom_t atom ){  return atom & 0x4000; }

int variable_name( atom_t atom ){   return atom & 0x3FFF; }





void clause_print( atom_t *clause, atom_t* clause_end )
{
    atom_t *iterator = NULL;
    int is_first = 1;
    print("\033[32m(\033[m");

    while ( atom_iterate( clause, clause_end, &iterator) != -1 ){
        if (is_first)
            is_first = 0;
        else
            print("\033[32m v \033[m");
        if (IS_NEGATED( *iterator ))
            print("-");
        print("%d", VARIABLE_NAME( *iterator ));
    }
    print("\033[32m)\033[m");
}

#include <stdio.h>

#include "parser.h" // parse
#include "abrasatcuda.h"
#include "consts.h"

#include "dispatch.h" // solve






int main( int argc, char ** argv )
{
    
    if (argc < 2){
        printf("usage : abrasatcuda file\n");
        return 1;
    }

    char* file_path = argv[1];

    atom_t *formula = NULL;
    atom_t *clauses_index = NULL;
    int num_var, num_clause;

    HLINE
#ifdef DEBUG
    printf("parses file %s\n", file_path );
#endif
    parse( file_path, &formula, &clauses_index, &num_var, &num_clause );

    HLINE
#ifdef DEBUG
    printf("file parsed, formula of %d clauses and %d var built\n",num_clause, num_var );


    formula_print( formula, clauses_index, num_clause );

    HLINE
    printf("tries to solve\n");
#endif
    int answer = solve( formula, clauses_index, num_clause, num_var );

    printf("Answer : \033[31;4m%s\033[m \n", answer == SUCCESS ? "True" : "False" );


    // de-allocate memory
    free(formula);
    free(clauses_index);

    return 0;
}








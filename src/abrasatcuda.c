#include <stdio.h>

#include "parser.h"
#include "solve.h"
#include "abrasatcuda.h"






int main( int argc, char ** argv )
{
    
    if (argc < 2){
        printf("usage : abrasatcuda file");
        return 1;
    }

    char* file_path = argv[1];

    atom_t *formula = NULL;
    atom_t *clauses_index = NULL;
    int num_var, num_clause;

    printf("parses file %s", file_path );
    parse( file_path, &formula, &clauses_index, &num_var, &num_clause );

    printf("file parsed, formula of %d clauses and %d var built\n",
        num_clause, num_var );

    formula_print( formula, clauses_index, num_clause );

    int answer = solve( formula, clauses_index, num_clause );

    printf("\033[31;4mAnswer : %s\033[m \n", answer ? "Vrai" : "Faux" );


    // de-allocate memory
    free(formula);
    free(clauses_index);

    return 0;
}




/*
 * this function's purpose is to manage to solve the problem.
 * It relies on solve_thread (one or many instances) to do so.
 */

// forward declaration
int solve_thread( atom_t* formula, atom_t* clauses_index, char* vars, int n );

int solve( atom_t *formula, atom_t* clauses_index, int n )
{
    // allocates space for n vars
    char vars[n];

    return solve_thread( formula, clauses_index, vars, n );
}




int solve_thread( atom_t* formula, atom_t* clauses_index, char* vars, int n )
{
    

    return 0;
}

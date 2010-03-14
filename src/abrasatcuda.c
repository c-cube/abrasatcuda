#include <stdio.h>

#include "parser.h"
#include "solve.h"
#include "abrasatcuda.h"
#include "consts.h"






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

    HLINE
    printf("parses file %s", file_path );
    parse( file_path, &formula, &clauses_index, &num_var, &num_clause );

    HLINE
    printf("file parsed, formula of %d clauses and %d var built\n",
        num_clause, num_var );

    formula_print( formula, clauses_index, num_clause );

    HLINE
    printf("tries to solve\n");
    int answer = solve( formula, clauses_index, num_clause );

    printf("Answer : \033[31;4m%s\033[m \n", answer == SUCCESS ? "True" : "False" );


    // de-allocate memory
    free(formula);
    free(clauses_index);

    return 0;
}




/*
 * this function's purpose is to manage to solve the problem.
 * It relies on solve_thread (one or many instances) to do so.
 */


int solve( atom_t *formula, atom_t* clauses_index, int n )
{
    // allocates space for n vars
    value_t vars[n];

    // TODO : create CUDA threads, each with its own [vars] array,
    // and dispatch it in CUDA.
    // TODO : find the k most "interesting" vars, and create 2^k threads
    // with different var affectations.
    // TODO : think of a way to share information between threads (for example,
    // if a thread finds out that a var cannot be true (exhausted possibilities ?)
    // it may tell the other threads to set this var to 0)

    return solve_thread( formula, clauses_index, vars, n );
}




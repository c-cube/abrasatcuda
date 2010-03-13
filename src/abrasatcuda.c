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

    // TODO : create CUDA threads, each with its own [vars] array,
    // and dispatch it in CUDA.
    // TODO : find the k most "interesting" vars, and create 2^k threads
    // with different var affectations.
    // TODO : think of a way to share information between threads (for example,
    // if a thread finds out that a var cannot be true (exhausted possibilities ?)
    // it may tell the other threads to set this var to 0)

    return solve_thread( formula, clauses_index, vars, n );
}



/*
 * a single thread of execution. It is given an array of [vars] with some of those
 * immutable and already affected.
 * It must find out if clauses are satisfiables with this repartition, by
 * brute force over others vars.
 */
int solve_thread( atom_t* formula, atom_t* clauses_index, char* vars, int n )
{
    
    // initialize all free vars
    int current_var;
    initialize_truth_values( vars, &current_var, n );


    // compute which clauses are satisfied
    char satisfied_clauses[n];

    atom_t* clause;
    int cur_index;
    while ( clause_iterate( formula, clauses_index, n, &cur_index, &clause ) 
            != FAILURE )
    {
        int clause_satisfied = FAILURE;
        atom_t *clause_end = formula + clauses_index[cur_index+1];
        for ( atom_t *iterator = clause; iterator < clause_end; ++ iterator ){

            int name = VARIABLE_NAME(*iterator);
            int is_negative = IS_NEGATED(*iterator);

            // only immutable vars are interesting
            if ( ! IS_IMMUTABLE(vars[name]) )
                continue;

            if ( is_negative ){
                // clause satisfied by this atom !
                if ( ! TRUTH_VALUE(vars[name]) ){ 
                    clause_satisfied = 1;
                    break;
                }
            } else {
                // clause satisfied
                if ( TRUTH_VALUE(vars[name]) ){ 
                    clause_satisfied = 1;
                    break;
                }
            }
        }
        
        if (clause_satisfied)
            satisfied_clauses[cur_index] = 1;

    }

    // try all possibilities
    while ( next_combination( vars, &current_var, n ) != -1 ){
        int this_clause_ok = formula_is_satisfied( 
            formula, clauses_index, vars, satisfied_clauses, n );
        if ( this_clause_ok == SUCCESS )
            return SUCCESS; // success !
    }

    return FAILURE;

}

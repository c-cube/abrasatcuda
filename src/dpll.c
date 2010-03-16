
#include "dpll.h"
#include "solve.h" // value_print

// a simple "heuristic" (just picks up the first non-affected var it finds)
int heuristic( atom_t* formula, atom_t *clauses_index, value_t *vars, int clause_n, int var_n)
{
    // iterate on vars
    for (int i = 1; i <= var_n; ++i){
        if ( ! ( IS_AFFECTED(vars[i])
              || IS_IMMUTABLE(vars[i])))
        {
            return i;
        }
    }

    // no free var found !
    return -1; 
}


/*
 * main algorithm function
 */
success_t dpll(
    atom_t* formula,
    atom_t *clauses_index,
    value_t *vars,
    int clause_n,
    int var_n)
{
    
    printf("launches dpll with %d clauses and %d vars\n", clause_n, var_n );

    // manage false stack. We start from 1, because it allows us to detect exhaution easily on stack_depth == 0
    int stack_depth = 1;

    // initializes satisfied_clauses
    satisfied_t satisfied_clauses[clause_n];
    for (int i = 0; i < clause_n; ++i )
        satisfied_clauses[i] = 0;

    // int last_pushed_var = -1; // @DEPRECATED@
    int next_var = -1; 
    int last_pushed_var = -1;
    success_t propagate_sth = FAILURE;

    /*
     * Start. At this point, we have to update satisfied_clauses info, and see if 
     * the formula is still potentially satisfiable.
     * If it is, we jump on branch for further exploration.
     * If not, we have to change of branch; if we are on a positive choice branch,
     *      we just have to go on the negative one;
     *      if we are on a negative branch, we must backtrack because we exhausted 
     *      the branch.
     */
    start:
        printf("\033[31m->\033[m @start\n");
        // debug
        value_print( vars, var_n );
        satisfied_print( satisfied_clauses, clause_n );

        // exhausted all possibilities at root, loser !
        if ( stack_depth <= 0 )
            return FAILURE;


        // updates info on clauses
        if ( formula_is_satisfiable( formula, clauses_index, vars, satisfied_clauses,
                        stack_depth, clause_n, var_n ) == FALSE )
        {
            assert( stack_depth > 0 );
            goto failure;
        }

        // check if all clauses are satisfied
        if ( all_clauses_are_satisfied( satisfied_clauses, clause_n ) == TRUE ){
            printf("all clauses satisfied !\n");
            return SUCCESS; // win !
        }

        // try to propagate unit clauses.
        propagate_sth = unit_propagation( formula, clauses_index, 
            vars, satisfied_clauses, stack_depth, clause_n, var_n );


        if ( propagate_sth == SUCCESS ){
            printf("propagate successfull !\n");
            
            // if formula is no more satisfiable, we failed.
            if ( formula_is_satisfiable( formula, clauses_index, vars, satisfied_clauses,
                            stack_depth, clause_n, var_n ) == FALSE )
            {
                assert( stack_depth > 0 );
                goto failure;
            }  
        }             
        
        // this is not yet a failure nor a success, we have to dig deeper to find out.
        goto branch;

        
    /*
     * formula has still a chance to be satisfiable, so we 
     * choose a var, and test it with positive value.
     */
    branch:
        printf("\033[31m->\033[m @branch\n");
        next_var = heuristic( formula, clauses_index, vars, clause_n, var_n );

        // assert( next_var != -1 ); // all vars affected but formula not satisfiable ??
        if ( next_var == -1 ){
            if ( all_clauses_are_satisfied( satisfied_clauses, clause_n ) == TRUE ){
                return SUCCESS;
            } else {
                printf("all vars affected, but not all clauses satisfied ?!\n");

                stack_depth--;
                goto failure;
            }
        }

        // simulate a function call
        stack_depth++;

        printf("chooses var %d at stack depth %d\n", next_var, stack_depth );
        
        /*
         * first try. failure of the first branch will lead to 
         * the label "failure_positive".
         */

        // affect the var with positive value
        SET_AFFECTED(vars[next_var]);
        SET_STACK_DEPTH(vars[next_var], stack_depth);
        SET_TRUE(vars[next_var]);

        goto start;

    /*
     * We just met failure.
     * Now we have to recognize it to deal with it properly.
     */
    failure:
        printf("\033[31m->\033[m @failure [stack depth %d]\n", stack_depth);
        
        // exhausted all possibilities at root, loser !
        if ( stack_depth <= 0 )
            return FAILURE;

        // so we cancel changes done by the previous affectation
        unroll( vars, satisfied_clauses, stack_depth, clause_n, var_n);


        // what is the last pushed var ?
        // we have to recompute it because no information is stored about the previous choice.
        last_pushed_var = heuristic( formula, clauses_index, vars, clause_n, var_n );

        if ( last_pushed_var == -1 ){ // root of call stack
            printf("failure at stack depth %d, no last_pushed_var\n",stack_depth);
            return FAILURE; // at root + unsatisfiable ==> definitely unsatisfiable
        }

        if ( TRUTH_VALUE( vars[last_pushed_var] ) == 1 ){
            // the previous iteration was a test for positive affectation of this var
            // we now have to test the negative choice.

            // and go on with the new choice
            goto failure_positive;

        } else {
            // uh-oh, this var has been thoroughly tested without results, fail.

            goto failure_negative;
        }
        
    /*
     * the try with positive value has failed.
     * We remain at the same stack depth, but try with a negative value.
     */
    failure_positive:
        printf("\033[31m->\033[m @failure positive\n");
        printf("switching var %d to false\n", last_pushed_var);

        // there has been an unroll, remember that this var is still affected
        SET_AFFECTED(vars[last_pushed_var]);
        SET_STACK_DEPTH(vars[last_pushed_var], stack_depth);
        SET_FALSE(vars[last_pushed_var]);

        goto start;
        
    /* 
     * Uh-oh, the negative try is also a failure. So, we have to backtrack because
     * the previous choice was not the good one.
     */
    failure_negative:
        printf("\033[31m->\033[m @failure negative\n");

        // unroll every change made at this level and upper
        unroll( vars, satisfied_clauses, stack_depth, clause_n, var_n);

        // go back in the stack
        stack_depth--;

        // there has been a failure, now we have to deal with it on previous recursive call.
        goto failure;


    // end:
        printf("what are you doing here ???\n");
        assert(0);
        return FAILURE; // never reached
}




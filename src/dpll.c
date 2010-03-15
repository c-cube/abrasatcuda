
#include "dpll.h"
#include "solve.h" // value_print

// a simple "heuristic" (just picks up the first non-affected var it finds)
int heuristic(
    atom_t* formula,
    atom_t *clauses_index,
    value_t *vars,
    int clause_n,
    int var_n)
{
    // iterate on vars
    for (int i = 1; i <= var_n; ++i){
        if ( ! ( IS_AFFECTED(vars[i])
              || IS_IMMUTABLE(vars[i])))
        {
            return i;
        }
    }

    return -1; // no free var found !
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

    int last_pushed_var = -1;
    int next_var = -1;

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
        printf("@start\n");
        value_print( vars, var_n );

        // exhausted all possibilities at root, loser !
        if (stack_depth == 0 )
            return FAILURE;


        // updates info on clauses
        if ( formula_is_satisfiable( formula, clauses_index, vars, satisfied_clauses,
                        stack_depth, clause_n, var_n ) == FALSE ){

            assert( stack_depth > 0 );
            goto failure;
        }

        // check if all clauses are satisfied, and update information about it
        if ( all_clauses_are_satisfied( satisfied_clauses, clause_n ) == TRUE ){
            return SUCCESS; // win !
        }

        // try to propagate unit clauses.
        success_t propagate_sth = unit_propagation( formula, clauses_index, 
            vars, satisfied_clauses, stack_depth, clause_n, var_n );


        // if formula is no more satisfiable, we failed.
        if ( propagate_sth == SUCCESS ){
            if ( formula_is_satisfiable( formula, clauses_index, vars, satisfied_clauses,
                            stack_depth, clause_n, var_n ) == FALSE ){

                assert( stack_depth > 0 );
                goto failure;
            }         
        }             
        
        // this is not yet a failure nor a success, we have to dig deeper to find out.
        goto branch;

        
        

    /*
     * We just met failure.
     * Now we have to recognize it to deal with it properly.
     */
    failure:
        printf("@failure\n");
        
        // what is the last pushed var ?
        last_pushed_var = find_pushed( vars, stack_depth, var_n );

        if ( last_pushed_var == -1 ) // root of call stack
            return FAILURE; // at root + unsatisfiable ==> definitely unsatisfiable

        if ( TRUTH_VALUE( vars[last_pushed_var] ) == 1 ){
            // the previous iteration was a test for positive affectation of this var
            // we now have to test the negative choice.

            // so we cancel changes done by the previous affectation
            unroll( vars, satisfied_clauses, stack_depth, clause_n, var_n);

            // and go on with the new choice
            goto failure_positive;

        } else {
            // uh-oh, this var has been thoroughly tested without results, fail.

            goto failure_negative;
        }
        

    /*
     * formula has still a chance to be satisfiable, so we 
     * choose a var, and test it with positive value.
     */
    branch:
        printf("@branch\n");
        next_var = heuristic( formula, clauses_index, vars, clause_n, var_n );

        assert( next_var != -1 ); // all vars affected but formula not satisfiable ??
        printf("chooses var %d at stack depth %d\n", next_var, stack_depth );
        
        /*
         * first try. failure of the first branch will lead to 
         * the label "failure_positive".
         */

        // simulate a function call
        stack_depth++;

        // affect the var with positive value
        SET_AFFECTED(vars[next_var]);
        SET_STACK_DEPTH(vars[next_var], stack_depth);
        SET_TRUE(vars[next_var]);

        goto start;

    /*
     * the try with positive value has failed.
     * We remain at the same stack depth, but try with a negative value.
     */
    failure_positive:
        printf("@failure positive\n");
        SET_FALSE(vars[last_pushed_var]);

        goto start;
        
    /* 
     * Uh-oh, the negative try is also a failure. So, we have to backtrack because
     * the previous choice was not the good one.
     */
    failure_negative:
        printf("@failure negative\n");

        // go back in the stack
        stack_depth--;

        // unroll every change made in deeper levels
        unroll( vars, satisfied_clauses, stack_depth, clause_n, var_n);

        // there has been a failure, now we have to deal with it on previous recursive call.
        goto failure;


    // end:
        printf("what are you doing here ???\n");
        assert(0);
        return FAILURE; // never reached
}




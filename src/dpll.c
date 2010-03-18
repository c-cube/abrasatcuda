#include "dpll.h"
#include "interfaces/solve.h" // value_print
#include "clause.h"
#include "consts.h"
#include "vars.h"









/*
 * this function verifies if a formula has still a chance to be satisfiable 
 * with the current (partial) variable affectations. It also updates which clauses
 * are satisfied or not.
 * Arguments : 
 * [formula] : whole formula (raw array of atom_t)
 * [clauses_index] : array of size [clause_n]+1, with the offset of 
 *      each clause inside [formula]
 * [vars] : array of truth values
 * [satisfied_clauses] : array of boolean, to know which clauses are satisfied
 * [stack_depth] : current depth of the recursion stack
 * [clause_n] : number of clauses
 * [var_n] : number of var
 */
#ifdef PROF
truth_t 
#else
static inline truth_t 
#endif
formula_is_satisfiable(  
    atom_t* formula, 
    atom_t* clauses_index,  
    value_t* vars,
    satisfied_t* satisfied_clauses,
    unsigned int stack_depth,
    int clause_n,
    int var_n )
{
    // for each clause
    for (int i = 0; i < clause_n; ++i ){

        // this clause is already satisfied, next
        assert( STACK_DEPTH(satisfied_clauses[i]) <= stack_depth);

        if ( SATISFIED(satisfied_clauses[i]) )
            continue;

        atom_t *clause = formula + clauses_index[i];
        atom_t *clause_end = formula + clauses_index[i+1];
        
        atom_t *iterator;

        truth_t clause_satisfiable = FALSE;

        // for this clause, check if it is satisfied, or still has a chance
        for ( iterator = clause; iterator < clause_end; ++ iterator ){
            int name = VARIABLE_NAME(*iterator);
            // if this var is not affected, there may be still a chance
            if ( ! ( IS_AFFECTED(vars[name]) || IS_IMMUTABLE(vars[name]) ) ){
#ifdef DEBUG
                //printf("clause %d satisfiable thank to free var %d\n", i, name);
#endif
                clause_satisfiable = TRUE;
                // do NOT break, maybe another atom will show that this clause is satisfied.
            }

            /*
             * if we find a var which has a value, we can check if it makes the clause 
             * satisfied. In this case, we can memoize it and go to the next clause.
             */
            if ( IS_IMMUTABLE(vars[name]) || IS_AFFECTED(vars[name]) ){
                int is_negative = IS_NEGATED(*iterator);

                if ( is_negative ){
                    // clause satisfied
                    if ( TRUTH_VALUE(vars[name]) == FALSE ){ 
#ifdef DEBUG
                        //printf("clause %d satisfied at depth %d by atom %d\n",i,stack_depth,name);
#endif
                        SET_SATISFIED(satisfied_clauses[i]);
                        SET_STACK_DEPTH(satisfied_clauses[i], stack_depth);
                        clause_satisfiable = TRUE;
                        break;
                    }
                } else {
                    // clause satisfied
                    if ( TRUTH_VALUE(vars[name]) == TRUE ){ 
#ifdef DEBUG
                        //printf("clause %d satisfied at depth %d by atom %d\n",i,stack_depth,name);
#endif
                        SET_SATISFIED(satisfied_clauses[i]);
                        SET_STACK_DEPTH(satisfied_clauses[i], stack_depth);
                        clause_satisfiable = TRUE;
                        break;
                    }
                }
            }
        }

        // there is no free var or satisfying atom, the clause is obviously empty, fail !
        if ( clause_satisfiable == FALSE ){
#ifdef DEBUG
            value_print( vars, var_n );
            printf("clause %d not satisfiable ",i); clause_print( clause, clause_end ); printf("\n"); 
#endif
            return FALSE;
        }

    }
    
    return TRUE;
}


/*
 * this function returns TRUE if all clauses are satisfied
 */
#ifdef PROF
truth_t
#else
static inline truth_t 
#endif
all_clauses_are_satisfied( 
    satisfied_t *satisfied_clauses,
    int clause_n)
{

    for (int i = 0; i < clause_n; ++i ){
        if ( SATISFIED( satisfied_clauses[i] ) != TRUE )
            return FALSE;
    }
    return TRUE;
}







/*
 * This finds unit clauses and propagates them.
 * It returns whether or not the formula is still satisfiable after unit propagation
 */
#ifdef PROF
truth_t
#else
static inline truth_t 
#endif
unit_propagation( atom_t* formula, atom_t *clauses_index, value_t *vars, satisfied_t* satisfied_clauses, unsigned int stack_depth, int clause_n, int var_n )
{

    int index = 0;

    //for each clause
    while ( index < clause_n ){

        // if clause is already satisfied, just don't mind and go on
        if ( SATISFIED(satisfied_clauses[index]) == TRUE ){
            index++;
            continue;
        }
        
        // now searching if this clause is a unit clause

        atom_t *clause = formula + (clauses_index[index]);
        atom_t *clause_end = formula + (clauses_index[index+1]);

        int num_atom = 0; // number of non-affected atoms in this clause
        atom_t *unit_atom = NULL; // the unit atom (if existing)

        for ( atom_t *atom = clause; atom < clause_end; ++atom ){
            // we have an unaffected atom here
            if ( ! (  IS_AFFECTED(vars[VARIABLE_NAME(*atom)]) 
                   || IS_IMMUTABLE(vars[VARIABLE_NAME(*atom)]))){
                num_atom++;
                unit_atom = atom;
            }
        }

        /*
         * This is a unit clause. We therefore choose the truth value of the unit var,
         * update information about which clauses are satisfied, and 
         * __START FROM THE BEGINNING__ (some clauses may have become unit ones)
         */
        if ( num_atom == 1 ){
#ifdef DEBUG
            printf("unit clause %d, unit var %d", index, VARIABLE_NAME(*unit_atom));
            clause_print( clause, clause_end ); printf("\n"); 
#endif

            int name = VARIABLE_NAME(*unit_atom);

            SET_SATISFIED(satisfied_clauses[index]); // the clause is satisfied, by necessity
#ifdef DEBUG
            //printf("set clause %d to satisfied\n", index);
            //if ( SATISFIED(satisfied_clauses[index] ))
            //  printf("indeed satisfied\n");
#endif        
            SET_STACK_DEPTH(satisfied_clauses[index], stack_depth); // remember where we did that

            if ( IS_NEGATED(*unit_atom) )
                SET_FALSE(vars[name]);
            else
                SET_TRUE(vars[name]);
            // remember at what depth we change this var, and that it is affected
            SET_AFFECTED(vars[name]);
            SET_STACK_DEPTH(vars[name], stack_depth);
        

            // verify if the formula is still satisfiable (if not, we are in a bad branch)
            // also update which clauses are satisfied.
            truth_t is_still_satisfiable = formula_is_satisfiable( 
                formula, clauses_index, vars, satisfied_clauses, stack_depth, clause_n, var_n );

            // the propagation has shown that we are in a non-satisfiable branch
            if ( is_still_satisfiable == FALSE ){
                return FALSE;
            } else { // start from scratch
                index = 0;
                continue;
            }
        }

        index++; // next clause
    }

    // no fresh unit clause has been found, so return true (as far as we know,
    // all our changes have been checked, and the formula was still potentially
    // satisfiable before we began)
    return TRUE;
}


#ifdef PROF
void
#else
static inline void 
#endif
initialize_values( truth_t* vars, int var_n )
{
    for (int i=1; i <= var_n; ++ i){
        if ( ! IS_IMMUTABLE(vars[i]) ){
            SET_NON_IMMUTABLE(vars[i]);
            SET_NON_AFFECTED(vars[i]);
            SET_FALSE(vars[i]);
            SET_STACK_DEPTH(vars[i], 0);
        }
    }
}

#ifdef PROF
void
#else
static inline void 
#endif
initialize_satisfied ( satisfied_t * satisfied_clauses, int var_n)
{
    for (int i = 1; i <= var_n; ++i){ 
        SET_NON_IMMUTABLE( satisfied_clauses[i]);
        SET_NON_AFFECTED(satisfied_clauses[i]);
        SET_FALSE(satisfied_clauses[i]);
        SET_STACK_DEPTH(satisfied_clauses[i], 0);
    }       
}

/*
 * This function unrolls every change that happened after the 
 * false function call at depth [stack_depth].
 * It will search for every var affected and clause satisfied at a 
 * __higher or equal__ depth then the one given.
 */
#ifdef PROF
void
#else
static inline void 
#endif
unroll( value_t *vars, satisfied_t *satisfied_clauses, 
    unsigned int stack_depth, int clause_n, int var_n )
{
    for ( int i = 1; i <= var_n; ++i ){
        // all vars affected at this depth must be unaffected
        if ( STACK_DEPTH(vars[i]) >= stack_depth ){
            SET_NON_AFFECTED(vars[i]);
            SET_STACK_DEPTH(vars[i],0);
        }
    }

    for ( int i = 0; i < clause_n; ++i ){

        if ( STACK_DEPTH(satisfied_clauses[i] ) >= stack_depth ){
            SET_NOT_SATISFIED(satisfied_clauses[i]);
            SET_STACK_DEPTH(satisfied_clauses[i], 0);
        }
    }
}



/*
 * gives the number of the var chosen by an heuristic for
 * the next branch to explore
 */

// a simple "heuristic" (just picks up the first non-affected var it finds)
// TODO : find a better heuristic
#ifdef PROF
int
#else
static inline int 
#endif
heuristic( atom_t* formula, atom_t *clauses_index, value_t *vars, int clause_n, int var_n)
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
 * When a failure occur, we hav to find what was the last var we choosed
 * and pushed on the stack. This is a job for find_pushed_var.
 */
#ifdef PROF
atom_t
#else
static inline atom_t 
#endif
find_pushed_var( value_t *vars, unsigned int stack_depth, int var_n )
{
    atom_t answer = -1;
    for ( int i=1; i <= var_n; ++i ){
        if ( STACK_DEPTH(vars[i]) == stack_depth )
            answer = i;
    }

    assert( answer <= var_n );

    return answer;
}







#define INVARIANT_STACK {                               \
        assert( stack_depth_plus % 2 == 0);             \
        assert( stack_depth % 2 == 1);                  \
        assert( stack_depth_plus == stack_depth + 1); }

/*
 * tries to solve the problem with the DPLL algorithm
 *
 *
 * procedure DPLL (F)
 * Begin                                                        <=== label start
 * If F = ∅ then return ”satisﬁable”;
 * Else F ← UnitPropagation(F);
 * If nil ∈F then return ”unsatisﬁable”;                        <=== label check 
 * Else ( Branching Rule )                                      <=== label branch
 * Choose l a literal according to some heuristic H;
 * If DPLL(F ∪ {l})= satisﬁable then return ”satisﬁable” ;  
 * Else DPLL(F ∪ {¬ l})                                         <=== label failure_positive
 * End
 *
 */
success_t 
dpll(
    atom_t* formula,
    atom_t *clauses_index,
    value_t *vars,
    int clause_n,
    int var_n)
{
    
#ifdef DEBUG
    printf("launches dpll with %d clauses and %d vars\n", clause_n, var_n );
#endif

    /* 
     * manage false stack. We start from 1, because it allows us to detect exhaution easily on stack_depth == 0.
     * We add 2 to stack_depth at each branch. stack_depth holds the current *true* stack level (with only
     * one var affected at each branch), and stack_depth_plus holds the stack level in which unit clauses propagating
     * can store satisfied clauses and affected vars
     */
    unsigned int stack_depth = 1;
    unsigned int stack_depth_plus = stack_depth + 1; // always stack_depth_plus = stack_depth + 1

    // initializes satisfied_clauses
    satisfied_t satisfied_clauses[clause_n];
    for (int i = 0; i < clause_n; ++i )
        satisfied_clauses[i] = 0;

    int next_var = -1; 
    int last_pushed_var = -1;

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
#ifdef DEBUG
        printf("\033[31m->\033[m @start\n");
        value_print( vars, var_n );
        satisfied_print( satisfied_clauses, clause_n );
#endif

        INVARIANT_STACK

        // exhausted all possibilities at root, loser !
        if ( stack_depth <= 0 )
            return FAILURE;

        stack_depth_plus = stack_depth + 1; // stack_depth_plus is the stack depth of propagated variables

        INVARIANT_STACK

        // updates info on clauses. New affectations are put on stack_depth_plus
        if ( formula_is_satisfiable( formula, clauses_index, vars, satisfied_clauses,
                        stack_depth_plus, clause_n, var_n ) == FALSE )
        {
            assert( stack_depth > 0 );
            goto epic_fail;
        }

        // check if all clauses are satisfied
        if ( all_clauses_are_satisfied( satisfied_clauses, clause_n ) == TRUE ){
#ifdef DEBUG
            printf("all clauses satisfied !\n");
#endif
            return SUCCESS; // win !
        }

        // try to propagate unit clauses.
        truth_t is_still_satisfiable = unit_propagation( formula, clauses_index, 
            vars, satisfied_clauses, stack_depth_plus, clause_n, var_n );


        // if formula is no more satisfiable after propagation, we failed.
        if ( is_still_satisfiable == FALSE ){
            assert( stack_depth > 0 );
            goto epic_fail;
        }             
        
        // this is not yet a failure nor a success, we have to dig deeper to find out.
        goto branch;

        
    /*
     * formula has still a chance to be satisfiable, so we 
     * choose a var, and test it with positive value.
     */
branch:
#ifdef DEBUG
        printf("\033[31m->\033[m @branch\n");
#endif
        next_var = heuristic( formula, clauses_index, vars, clause_n, var_n );

        // assert( next_var != -1 ); // all vars affected but formula not satisfiable ??
        if ( next_var == -1 ){
            if ( all_clauses_are_satisfied( satisfied_clauses, clause_n ) == TRUE ){
                return SUCCESS;
            } else {
#ifdef DEBUG
                printf("all vars affected, but not all clauses satisfied ?!\n");
#endif

                // stack_depth -= 2; // TWO levels down FIXME : pertinent ?
                goto epic_fail;
            }
        }

        // simulate a function call
        stack_depth += 2;
        stack_depth_plus += 2;
        INVARIANT_STACK

#ifdef DEBUG
        printf("chooses var %d at stack depth %d\n", next_var, stack_depth );
#endif
        
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
epic_fail:
#ifdef DEBUG
        printf("\033[31m->\033[m @epic_fail [stack depth %d]\n", stack_depth);
#endif
        
        // exhausted all possibilities at root, loser !
        if ( stack_depth <= 0 )
            return FAILURE;


        // what is the last pushed var ?
        last_pushed_var = find_pushed_var( vars, stack_depth, var_n );

        // so we cancel changes done by the previous affectation
        unroll( vars, satisfied_clauses, stack_depth, clause_n, var_n);


        if ( last_pushed_var == -1 ){ // root of call stack
#ifdef DEBUG
            printf("epic_fail at stack depth %d, no last_pushed_var\n",stack_depth);
#endif
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
#ifdef DEBUG
        printf("\033[31m->\033[m @failure positive\n");
        printf("switching var %d to false\n", last_pushed_var);
#endif 

        INVARIANT_STACK

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
#ifdef DEBUG
        printf("\033[31m->\033[m @failure negative\n");
#endif

        // go back in the stack
        stack_depth -= 2;
        stack_depth_plus -= 2;

        INVARIANT_STACK

        // there has been a failure, now we have to deal with it on previous recursive call.
        goto epic_fail;


    // end:
        printf("what are you doing here ???\n");
        assert(0);
        return FAILURE; // never reached
}



/*
 * The function exported by the module
 * according to the solve.h interface
 */
success_t 
solve_thread( atom_t* formula, atom_t* clauses_index, value_t* vars, int clause_n, int var_n )
{
    initialize_values( vars, var_n );

    // current default implementation 
    truth_t answer = dpll( formula, clauses_index, vars, clause_n, var_n );


    value_print( vars, var_n );
    if( answer == SUCCESS )
        printf("yeah !\n");
    if ( answer == FAILURE )
        printf("oh noes !\n");

    return answer;

}

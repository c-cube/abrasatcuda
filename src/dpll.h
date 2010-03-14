/*
 * module for dpll calculations
 */

#ifndef _DPLL_H
#define _DPLL_H

#include "consts.h"
#include "vars.h"


/*
 * this function verifies if a formula has still a chance to be satisfiable 
 * with the current (partial) variable affectations. It also updates which clauses
 * are satisfied or not.
 * Arguments : 
 * [formula] : whole formula (raw array of atom_t)
 * [clauses_index] : array of size [n]+1, with the offset of each clause inside [formula]
 * [vars] : array of truth values
 * [satisfied_clauses] : array of boolean, to know which clauses are satisfied
 * [stack_depth] : current depth of the recursion stack
 * [n] : number of clauses
 */

inline truth_t formula_is_satisfiable( 
    atom_t* formula, 
    atom_t* clauses_index,  
    value_t* vars,
    satisfied_t* satisfied_clauses,
    int stack_depth,
    int n )
{
    // for each clause
    for (int i = 0; i<n; ++i ){

        // this clause is already satisfied, next
        if ( SATISFIED(satisfied_clauses[i]) )
            continue;

        atom_t *clause = formula + clauses_index[i];
        atom_t *clause_end = formula + clauses_index[i+1];
        
        atom_t *iterator;

        // for this clause, check if it is satisfied, or still has a chance
        int clause_satisfiable = FALSE;
        for ( iterator = clause; iterator < clause_end; ++ iterator ){
            int name = VARIABLE_NAME(*iterator);
            // if this var is not affected, there may be still a chance
            if ( ! IS_AFFECTED(vars[name]) ){
                clause_satisfiable = TRUE;
                break;
            }

            // at this point, the var is either immutable either affected.
            assert( IS_IMMUTABLE(vars[name]) || IS_AFFECTED(vars[name]) );
            int is_negative = IS_NEGATED(*iterator);

            if ( is_negative ){
                // clause satisfied
                if ( ! TRUTH_VALUE(vars[name]) ){ 
                    SET_SATISFIED(satisfied_clauses[i]);
                    SET_STACK_DEPTH(satisfied_clauses[i], stack_depth);
                    clause_satisfiable = TRUE;
                    break;
                }
            } else {
                // clause satisfied
                if ( TRUTH_VALUE(vars[name]) ){ 
                    SET_SATISFIED(satisfied_clauses[i]);
                    SET_STACK_DEPTH(satisfied_clauses[i], stack_depth);
                    clause_satisfiable = TRUE;
                    break;
                }
            }
        }

        // there is not free var or satisfying atom, the clause is therefore empty, fail !
        if ( clause_satisfiable == FALSE )
            return FALSE;

    }
    
    return TRUE;
}


/*
 * this function returns TRUE if all clauses are satisfied
 */
inline truth_t all_clauses_are_satisfied( 
    satisfied_t *satisfied_clauses,
    int n)
{

    for (int i = 0; i < n; ++i ){
        if ( ! SATISFIED( satisfied_clauses[i] ) )
            return FALSE;
    }
    return TRUE;
}


/*
 * This finds unit clauses and propagates them.
 */
inline success_t unit_propagation( atom_t* formula, atom_t *clauses_index, satisfied_t* satisfied_clauses, int stack_depth, int n )
{
    success_t did_something = FAILURE;

    //for each clause
    for ( int index = 0; index < n; ++index ){

        atom_t *clause = formula + (clauses_index[index]);
        atom_t *clause_end = formula + (clauses_index[index+1]);

        int num_atom = 0; // number of non-affected atoms in this clause
        atom_t *unit_atom = NULL; // the unit atom (if existing)

        for ( atom_t *atom = clause; atom < clause_end; ++atom ){
            // we have an unaffected atom here
            if ( ! (  IS_AFFECTED(vars[VARIABLE_NAME(*atom)]) 
                   || IS_IMMUTABLE(vars[VARIABLE_NAME(*atom)]))){
                num_atom++;
                unit_atom = atom - clause;
            }
        }

        // propagate the unit clause !
        if ( num_atom == 1 ){
            did_something = SUCCESS;
            
            int name = VARIABLE_NAME(*unit_atom);

            SET_SATISFIED(satisfied_clauses[index]); // the clause is satisfied, by choice 
            SET_STACK_DEPTH(satisfied_clauses[index], stack_depth); // remember where we did that

            if ( IS_NEGATED(*unit_atom) )
                SET_TRUE(vars[name]);
            else
                SET_FALSE(vars[name]);
            // remember at what depth we change this var
            SET_AFFECTED(vars[name]);
            SET_STACK_DEPTH(vars[name], stack_depth);
        }
    }
   
    return did_something;
}


/*
 * gives the number of the var chosen by an heuristic for
 * the next branch to explore
 */
inline int heuristic(
    atom_t* formula,
    atom_t *clauses_index,
    value_t *vars,
    int n)
{
    // TODO
    return 0;
}




/*
 * tries to solve the problem with the DPLL algorithm
 */
inline success_t dpll(
    atom_t* formula,
    atom_t *clauses_index,
    int n)
{
    // manage false stack. We start from 1, because it allows us to detect exhaution easily on stack_depth == 0
    int stack_depth = 1;

    // initializes satisfied_clauses
    satisfied_t satisfied_clauses[n];
    for (int i = 0; i<n; ++i )
        satisfied_clauses[i] = 0;

    // do we have to return back ?
    int main_branch = 1;

    while ( 1 ){

        // exhausted all possibilities, loser !
        if (stack_depth == 0 )
            return FAILURE;

        // check if all clauses are satisfied, and update information about it
        if ( all_clauses_are_satisfied( satisfied_clauses, n ) == TRUE ){
            // win !
            return SUCCESS;
        }

        // try to propagate unit clauses.
        success_t propagate_sth = unit_propagation( formula, clauses_index, 
            satisfied_clauses, stack_depth, n );

        // something has changed.
        if ( propagate_sth == SUCCESS ){

            // the empty clause is present. This branch just failed.
            if ( formula_is_satisfiable( formula, clauses_index, vars, satisfied_clauses,
                                                            stack_depth, n ) == FALSE ){

                assert( stack_depth > 0 );

                // what is the last pushed var ?
                int last_pushed_var = find_last_pushed( vars, n );

                if ( TRUTH_VALUE( vars[last_pushed_var] ) == 1 ){
                    // the previous iteration was a test for positive affectation of this var
                    // we now have to test the negative choice.

                    // so we cancel changes done by the previous affectation
                    unroll( vars, satisfied_clauses, stack_depth, n);

                    // and go on with the new choice
                    SET_FALSE( vars[last_pushed_var] );
                    continue;
                } else {
                    // uh-oh, this var has been thoroughly tested without results, fail.
                    unroll( vars, satisfied_clauses, stack_depth, n);
                    SET_NON_AFFECTED( vars[last_pushed_var]);
                    SET_STACK_DEPTH( vars[last_pushed_var], 0);
                    continue;
                }
            }
        }

        // here, we branch !

        int next_var = heuristic( formula, clauses_index, vars, n );
        
        /*
         * first try. The second try (negative choice) is made
         * in the previous "if" statement, where we fail.
         */

        // simulate a function call
        stack_depth++;

        // affect the var with positive value
        SET_AFFECTED(vars[next_var])
        SET_STACK_DEPTH(vars[next_var], stack_depth);
        SET_TRUE(vars[next_var]);


    }

    // TODO
    return 0;
}




#endif

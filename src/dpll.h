/*
 * module for dpll calculations
 */

#ifndef _DPLL_H
#define _DPLL_H

#include <assert.h>

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

inline truth_t formula_is_satisfiable(  
    atom_t* formula, 
    atom_t* clauses_index,  
    value_t* vars,
    satisfied_t* satisfied_clauses,
    int stack_depth,
    int clause_n,
    int var_n )
{
    // for each clause
    for (int i = 0; i<clause_n; ++i ){

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
    int clause_n)
{

    for (int i = 0; i < clause_n; ++i ){
        if ( ! SATISFIED( satisfied_clauses[i] ) )
            return FALSE;
    }
    return TRUE;
}


/*
 * This finds unit clauses and propagates them.
 */
inline success_t unit_propagation( atom_t* formula, atom_t *clauses_index, value_t *vars, satisfied_t* satisfied_clauses, int stack_depth, int clause_n, int var_n )
{
    success_t did_something = FAILURE;

    //for each clause
    for ( int index = 0; index < clause_n; ++index ){

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
int heuristic(
    atom_t* formula,
    atom_t *clauses_index,
    value_t *vars,
    int clause_n,
    int var_n);

/*
 * this function finds which var of [vars] has been pushed at this [stack_depth].
 */
inline int find_pushed( value_t *vars, int stack_depth, int var_n )
{
    int win_index = -1;
    for (int i = 0; i<var_n; ++i ){
        if ( IS_AFFECTED(vars[i]) && STACK_DEPTH(vars[i] == stack_depth) ){
            assert( win_index == -1 ); // only one var by push
            win_index = i;
        }   
    }
    return win_index;
}


/*
 * This function unrolls every change that happened after the 
 * false function call at depth [stack_depth].
 * It will search for every var affected and clause satisfied at a higher depth
 * then the one given.
 * It does not affect changes occured at the level [stack_depth].
 */
inline void unroll( value_t *vars, satisfied_t *satisfied_clauses, 
    int stack_depth, int clause_n, int var_n )
{
    for ( int i = 0; i < var_n; ++i ){
        // all vars affected at this depth must be unaffected
        if ( STACK_DEPTH(vars[i]) > stack_depth ){
            SET_NON_AFFECTED(vars[i]);
            SET_STACK_DEPTH(vars[i],0);
        }
    }

    for ( int i = 0; i < clause_n; ++i ){

        if ( STACK_DEPTH(satisfied_clauses[i] ) > stack_depth ){
            SET_NOT_SATISFIED(satisfied_clauses[i]);
            SET_STACK_DEPTH(satisfied_clauses[i], 0);
        }
    }
}

            
            


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
success_t dpll(
    atom_t* formula,
    atom_t *clauses_index,
    value_t *vars,
    int clause_n,
    int var_n);



#endif

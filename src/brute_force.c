#include "brute_force.h"

#include "consts.h"
#include "vars.h"
#include "solve.h"


/*
 * this function verifies if a formula has still a chance to be satisfiable 
 * with the current (partial) variable affectations. It also updates which clauses
 * are satisfied or not.
 * Arguments : 
 * [formula] : whole formula (raw array of atom_t)
 * [clauses_index] : array of size [clause_n]+1, with the offset of 
 *      each clause inside [formula]
 * [vars] : array of truth values
 * [clause_n] : number of clauses
 * [var_n] : number of var
 */
inline truth_t is_satisfiable(  
    atom_t* formula, 
    atom_t* clauses_index,  
    value_t* vars,
    int clause_n,
    int var_n )
{
    // for each clause
    for (int i = 0; i < clause_n; ++i){

        atom_t *clause = formula + clauses_index[i];
        atom_t *clause_end = formula + clauses_index[i+1];
        
        atom_t *iterator;

        // for this clause, check if it is satisfied, or still has a chance
        truth_t clause_satisfiable = FALSE;
        for ( iterator = clause; iterator < clause_end; ++ iterator ){

            int name = VARIABLE_NAME(*iterator);

            // the var is either immutable either affected.
            assert( IS_IMMUTABLE(vars[name]) || IS_AFFECTED(vars[name]) );
            int is_negative = IS_NEGATED(*iterator);

            if ( is_negative ){
                // clause satisfied
                if ( TRUTH_VALUE(vars[name]) == FALSE ){ 
#ifdef DEBUG
                    printf("clause %d satisfied by atom %d\n", i, name);
#endif
                    clause_satisfiable = TRUE;
                    break;
                }
            } else {
                // clause satisfied
                if ( TRUTH_VALUE(vars[name]) == TRUE ){ 
#ifdef DEBUG
                    printf("clause %d satisfied by atom %d\n", i, name);
#endif
                    clause_satisfiable = TRUE;
                    break;
                }
            }
        }

        // there is not free var or satisfying atom, the clause is obviously empty, fail !
        if ( clause_satisfiable == FALSE ){
#ifdef DEBUG
            printf("clause %d not satisfiable\n",i);
#endif
            return FALSE;
        }

    }
    
    return TRUE;
}


/*
 * this function finds the next combination of binary values
 * for items of the array. It returns -1 if all possibilities have been enumerated.
 * It ignores immutable vars.
 * [vars] is the array of value_t (each representing the binary value of a var), mutated each iteration
 * [cur] is a reference to the current pending var, mutated each iteration.
 * [var_n] is the number of vars; The length of [vars] is [var_n]+1 (var 0 does not exist)
 */

inline success_t next_combination( value_t*vars, int *cur, int var_n )
{

    assert( *cur >= 1);
    assert( *cur <= var_n);
    
    truth_t advanced = FALSE;
    while (1){

        // check for termination. The last var is [var_n], not [var_n]-1
        if (*cur == var_n && (TRUTH_VALUE(vars[*cur]) || IS_IMMUTABLE(vars[*cur]))){
            return FAILURE;
#ifdef DEBUG
            printf("next_combination failed on cur = %d with ", *cur); value_print( vars, var_n); 
#endif
        }


        // do not consider immutable values
        if (IS_IMMUTABLE(vars[*cur])){ 
            ++(*cur);
            continue; 
        }

        // omg this var is not affected yet !
        assert( IS_AFFECTED(vars[*cur]) );

        if (TRUTH_VALUE(vars[*cur])){
            SET_FALSE(vars[*cur]);
            ++(*cur);
            advanced = TRUE; // remember to go back after
            continue;
        }

        // this var is affected to 0, switch it to 1.
        assert(TRUTH_VALUE(vars[*cur]) == FALSE);
        SET_TRUE(vars[*cur]);
        break;
    }

    // if advanced, remember to start from the beginning next time
    if ( advanced == TRUE )
        *cur = 1;

    return SUCCESS;
}


/*
 * this function initializes an array of truth value before
 * we can iterate on combinations on it.
 * It mainly SET_AFFECTED all the truth values and set them to 0
 */
inline void initialize_truth_values( value_t* vars, int var_n )
{
    for ( int i = 1; i <= var_n; ++i ){
        if ( ! IS_IMMUTABLE(vars[i]) ){
            SET_AFFECTED(vars[i]);
            SET_FALSE(vars[i]);
        }
    }
}

/*
 * TODO : a correct implementation, for benchmarking
 * a brute force solver, iterating over all possibilities until it exhausts them
 * or finds a satisfying affectation of vars
 */
success_t brute_force(atom_t* formula, atom_t* clauses_index, 
    value_t* vars, int clause_n, int var_n)
{
    // initialize all free vars
    initialize_truth_values( vars, var_n );

    int cur = 1;

#ifdef DEBUG
    value_print(vars,var_n);
    printf("tries every possibility !\n");
#endif
    
    /*
     * try every possibilities until exhaustion or until we find that all clauses are satisfied
     */
    while ( next_combination( vars, &cur, var_n ) != FAILURE ){
#ifdef DEBUG
        value_print( vars, var_n );
#endif

        // compute satisfiability at this point
        truth_t all_ok = is_satisfiable( formula, clauses_index, 
                vars, clause_n, var_n );

        if ( all_ok == FALSE ){
            continue; // not ok
#ifdef DEBUG
            printf("fail\n");
#endif
        } else {
#ifdef DEBUG
            printf("all clauses ok\n");
#endif
            return SUCCESS;
        }


    }

    return FAILURE;
}






// this is the entry point of a thread
success_t solve_thread( atom_t* formula, atom_t* clauses_index, value_t* vars, int clause_n, int var_n )
{


    // current implementation 
    truth_t answer = brute_force( formula, clauses_index, vars, clause_n, var_n );


    if( answer == SUCCESS )
        value_print( vars, var_n );

    return answer;

}

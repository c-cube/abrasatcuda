#include "brute_force.h"


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
    

    int advanced = 0;
    while (1){

        // check for termination. The last var is [var_n], not [var_n]-1
        if (*cur == var_n && (TRUTH_VALUE(vars[*cur]) == 1 || IS_IMMUTABLE(vars[*cur]))){
            return FAILURE;
            printf("next_combination failed on cur = %d with ", *cur); value_print( vars, var_n); 
        }


        // do not consider immutable values
        if (IS_IMMUTABLE(vars[*cur])){ 
            ++(*cur);
            continue; 
        }

        // omg this var is not affected yet !
        //printf( "cur = %d, var[cur] = %d\n", *cur,vars[*cur]);
        assert( IS_AFFECTED(vars[*cur]) );

        if (TRUTH_VALUE(vars[*cur])){
            SET_FALSE(vars[*cur]);
            ++(*cur);
            advanced = 1; // remember to go back after
            continue;
        }

        // this var is affected to 0, switch it to 1.
        assert(TRUTH_VALUE(vars[*cur]) == 0);
        SET_TRUE(vars[*cur]);
        break;
    }

    if ( advanced )
        *cur = 1;

    return SUCCESS;
}


/*
 * this function initializes an array of truth value before
 * we can iterate on combinations on it.
 * It mainly SET_AFFECTED all the truth values and set them to 0
 */
inline void initialize_truth_values( value_t* vars, int *cur, int var_n )
{
    int has_found_mutable = 0;

    *cur = 1;
    for (int i = 1; i <= var_n; ++i ){
        if ( ! IS_IMMUTABLE(vars[i]) ){
            SET_AFFECTED(vars[i]);
            SET_FALSE(vars[i]);

            // set *cur to the first interesting var
            if ( ! has_found_mutable ){
                has_found_mutable = 1;
                *cur = i;
            }
        }
    }
}

/*
 * TODO : a correct implementation, for benchmarking
 * a brute force solver, iterating over all possibilities until it exhausts them
 * or finds a satisfying affectation of vars
 */
inline success_t brute_force(atom_t* formula, atom_t* clauses_index, 
    value_t* vars, int clause_n, int var_n)
{
    // initialize all free vars
    int current_var;
    initialize_truth_values( vars, &current_var, var_n );

    printf("current array of vars : "); value_print(vars,var_n);

    // compute which clauses are satisfied
    satisfied_t satisfied_clauses[var_n];

    int cur_index;
    for (cur_index = 0; cur_index < var_n; ++cur_index ){
        
        int clause_satisfied = 0;
        atom_t* clause = formula + clauses_index[cur_index];
        atom_t *clause_end = formula + clauses_index[cur_index+1];
        printf("currently clause %d between %d and %d\n", cur_index, clauses_index[cur_index], clauses_index[cur_index+1]);
        // try to see if any of the atoms is positive
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
        
        if (clause_satisfied){
            satisfied_clauses[cur_index] = 1;
            printf("clause satisfied : "); clause_print( clause, clause_end ); printf("\n");
        }
    }

    printf("tries every possibility !\n");
    
    // try all possibilities
    while ( next_combination( vars, &current_var, var_n ) != FAILURE ){
        printf("combination "); value_print( vars, var_n );
        int this_clause_ok = formula_is_satisfiable( formula, clauses_index, 
                vars, satisfied_clauses, 0, clause_n, var_n );
        if ( this_clause_ok == SUCCESS )
            return SUCCESS; // success !
        printf("fail\n");
    }

    return FAILURE;
}






// this is the entry point of a thread
success_t solve_thread( atom_t* formula, atom_t* clauses_index, value_t* vars, int clause_n, int var_n )
{
    initialize_values( vars, var_n );

    // current default implementation 
    truth_t answer = brute_force( formula, clauses_index, vars, clause_n, var_n );


    if( answer == SUCCESS )
        value_print( vars, var_n );

    return answer;

}

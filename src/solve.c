

#include "solve.h"

int formula_is_satisfied( 
    atom_t* formula, 
    atom_t* clauses_index,  
    char* vars,
    char* satisfied_clauses,
    int n )
{
    for (int i = 0; i<n; ++i ){

        // this clause is already satisfied, next
        if ( satisfied_clauses[i] )
            continue;

        atom_t *clause = formula + clauses_index[i];
        atom_t *clause_end = formula + clauses_index[i+1];
        
        atom_t *iterator;

        // for this clause, check if it is satisfied, or still has a chance
        int clause_satisfiable = SUCCESS;
        for ( iterator = clause; iterator < clause_end; ++ iterator ){
            int name = VARIABLE_NAME(*iterator);
            int is_negative = IS_NEGATED(*iterator);

            if ( is_negative ){
                if ( ! vars[name] ){ // clause satisfied by this atom !
                    clause_satisfiable = 1;
                    break;
                }
            } else {
                if ( vars[name] ){ // clause satisfied
                    clause_satisfiable = 1;
                    break;
                }
            }
        }

        if ( ! clause_satisfiable )
            return FAILURE;

    }
    
    return SUCCESS;
}



/*
 * utilities for solver.
 */

/* LICENSE :
DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
                   Version 2, December 2004 

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net> 

Everyone is permitted to copy and distribute verbatim or modified 
copies of this license document, and changing it is allowed as long 
as the name is changed. 

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION 

    0. You just DO WHAT THE FUCK YOU WANT TO.
*/

#ifndef _SOLVE_H
#define _SOLVE_H


#include <stdio.h>
#include <assert.h>
#include "clause.h"


/*
 * type representing a truth value for a single variable.
 */
typedef char value_t;

/*
 * operations on truth values.
 * MSB bit defines mutability of var.
 * 6th bit (next one) defines if the var is affected
 * the 0th bit (LSB) defines the truth value of the var
 */

#define IS_IMMUTABLE(x) ((x) & 0x80)

#define IS_AFFECTED(x) ((x) & 0x40)

#define SET_AFFECTED(x) ((x) |= 0x40)
#define SET_NON_AFFECTED(x) ((x) &= 0xBF)

#define SET_IMMUTABLE(x) ((x) |= 0x80)

#define TRUTH_VALUE(x) ((x) & 0x01)
#define SET_TRUE(x) ((x) |= 0x01)
#define SET_FALSE(x) ((x) &= 0xFE)

// prints truth values in a nice way
inline void value_print( value_t* values, int n )
{
    for (int i=1; i<=n; ++i){
        int escape_sequence = 0;
        if ( IS_IMMUTABLE(values[i] ))
            escape_sequence = 31; // red
        else
        {
            if (IS_AFFECTED(values[i]))
                escape_sequence = 32; // green
            else
                escape_sequence = 34; // blue
        } 
        printf("%d=\033[%dm%d\033[m, ", i, escape_sequence, TRUTH_VALUE(values[i]));
    }
    printf("\n");
}




/*
 * this function finds the next combination of binary values
 * for items of the array. It returns -1 if all possibilities have been enumerated.
 * It ignores immutable vars.
 * [vars] is the array of char (each representing the binary value of a var), mutated each iteration
 * [cur] is a reference to the current pending var, mutated each iteration.
 * [n] is the number of vars; The length of [vars] is [n]+1 (var 0 does not exist)
 */

inline int next_combination( char *vars, int *cur, int n )
{
    

    int advanced = 0;
    while (1){

        // check for termination. The last var is [n], not [n]-1
        if (*cur == n && (TRUTH_VALUE(vars[*cur]) == 1 || IS_IMMUTABLE(vars[*cur]))){
            return FAILURE;
            printf("next_combination failed on cur = %d with ", *cur); value_print( vars, n); 
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
inline void initialize_truth_values( char* vars, int *cur, int n )
{
    int has_found_mutable = 0;

    *cur = 1;
    for (int i = 1; i <= n; ++i ){
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
 * this function verifies if a formula has still a chance to be satisfiable 
 * with the current variable affectations.
 * Arguments : 
 * [formula] : whole formula (raw array of atom_t)
 * [clauses_index] : array of size [n]+1, with the offset of each clause inside [formula]
 * [vars] : array of truth values
 * [satisfied_clauses] : array of boolean, to know which clauses are satisfied
 * [n] : number of clauses
 */

int formula_is_satisfied( 
    atom_t* formula, 
    atom_t* clauses_index,  
    char* vars,
    char* satisfied_clauses,
    int n );


#endif



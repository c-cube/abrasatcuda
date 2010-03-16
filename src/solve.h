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
#include "vars.h"
#include "dpll.h"

// prints truth values in a nice way
inline void value_print( value_t* values, int var_n )
{
    printf("    values : ");
    for (int i=1; i<=var_n; ++i){
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
        printf("%d=\033[%dm%d\033[m", i, escape_sequence, TRUTH_VALUE(values[i]));

        if (IS_AFFECTED(values[i]) || IS_IMMUTABLE(values[i]))
            printf("[%d], ", STACK_DEPTH(values[i]));
        else
            printf(", ");
    }
    printf("\n");
}


// prints satisfied clauses in a nice way
inline void satisfied_print( satisfied_t *satisfied_clauses, int clause_n )
{
    printf("    clauses : ");
    int escape_sequence = 34;
    char signal_char = '_';
    for (int i = 0; i < clause_n; ++ i){
        if ( SATISFIED(satisfied_clauses[i]) ){
            escape_sequence = 32;
            signal_char = '|';
        } else {
            signal_char = '_';
            escape_sequence = 34;
        }
        printf( "%d=\033[%dm%c\033[m, ", i, escape_sequence, signal_char );
    }
    printf("\n");
}












/*
 * a single thread of execution. It is given an array of [vars] with some of those
 * immutable and already affected.
 * It must find out if clauses are satisfiables with this repartition, by
 * brute force over others vars.
 */
success_t solve_thread( atom_t* formula, atom_t* clauses_index, value_t* vars, 
    int clause_n, int var_n );




#endif



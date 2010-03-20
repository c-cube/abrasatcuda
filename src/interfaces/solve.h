/*
 * utilities for solver.
 */

#ifndef SOLVE_H
#define SOLVE_H


#include <stdio.h>
#include <assert.h>

#include "../clause.h"
#include "../vars.h"
#include "../consts.h"

// prints truth values in a nice way
static inline void value_print( value_t* values, int var_n )
{
    printf("    \033[39;4mvalues :\033[m ");
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

        //if (IS_AFFECTED(values[i]) || IS_IMMUTABLE(values[i]))
        if ( STACK_DEPTH(values[i]) > 0 )
            printf("[%d], ", STACK_DEPTH(values[i]));
        else
            printf(", ");
    }
    printf("\n");
}


// prints satisfied clauses in a nice way
static inline void satisfied_print( satisfied_t *satisfied_clauses, int clause_n )
{
    printf("    \033[39;4mclauses :\033[m ");
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
        printf( "%d=\033[%dm%c\033[m", i, escape_sequence, signal_char );

        // print stack depth, if any
        //if ( STACK_DEPTH(satisfied_clauses[i]) > 0 )
        //    printf("[%d], ", STACK_DEPTH(satisfied_clauses[i]));
        //else
        //    printf(", ");

        printf(", ");
    }
    printf("\n");
}












/*
 * a single thread of execution. It is given an array of [vars] with some of those
 * immutable and already affected.
 * It must find out if clauses are satisfiables with this repartition, by
 * brute force over others vars.
 */
#ifdef CUDA
__device__
#endif
success_t solve_thread( atom_t* formula, atom_t* clauses_index, value_t* vars, 
    int clause_n, int var_n );




#endif



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
#define SET_NON_AFFECTED(x) ((x) ^= 0x40)

#define SET_IMMUTABLE(x) ((x) |= 0x80)

#define TRUTH_VALUE(x) ((x) & 0x01)
#define SET_TRUTH_VALUE(x,truth) ((x) ^= (0x01 & truth))

// prints truth values in a nice way
inline void value_print( value_t* values, int n )
{
    for (int i=0; i<n; ++i){
        int escape_sequence = 0;
        if ( IS_IMMUTABLE(values[i] ))
            escape_sequence = 31; // red
        else if (IS_AFFECTED(values[i]))
            escape_sequence = 32; // green
        else
            escape_sequence = 34; // blue
          
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
    // check for termination. The last var is [n], not [n]-1
    if (*cur == n && TRUTH_VALUE(vars[*cur]) == 1)
        return -1;

    while (1){

        // do not consider immutable values
        if (IS_IMMUTABLE(vars[*cur])){ 
            (*cur)++;
            continue; 
        }

        // omg this var is not affected yet !
        assert( IS_AFFECTED(vars[*cur]) );

        if (TRUTH_VALUE(vars[*cur])){
            SET_TRUTH_VALUE(vars[*cur], 0);
            ++(*cur);
            continue;
        }

        // this var is affected to 0, switch it to 1.
        assert(TRUTH_VALUE(vars[*cur]) == 0);
        SET_TRUTH_VALUE(vars[*cur], 1);
        return 0;
    }

}







#endif



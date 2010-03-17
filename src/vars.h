/*
 * module for variables array manipulation
 * and satisfied clauses array manipulation
 */

#ifndef VARS_H
#define VARS_H

// TODO : write tests for stack manipulation !!!


#include "consts.h"


/*
 * type representing a truth value for a single variable.
 */
typedef short value_t;

/*
 * operations on truth values.
 * MSB bit (15th) defines mutability of var.
 * 14th bit (next one) defines if the var is affected
 * bits 13 to 1 define stack depth 
 * the 0th bit (LSB) defines the truth value of the var
 */

#define IS_IMMUTABLE(x) ((x) & 0x8000)

#define IS_AFFECTED(x) ((x) & 0x4000)

#define SET_AFFECTED(x) ((x) |= 0x4000)
#define SET_NON_AFFECTED(x) ((x) &= 0xBFFF)

#define SET_IMMUTABLE(x) ((x) |= 0x8000)
#define SET_NON_IMMUTABLE(x) ((x) &= 0x7FFF)

#define TRUTH_VALUE(x) ((x) & 0x0001)

#define SET_TRUE(x) ((x) |= 0x0001)
#define SET_FALSE(x) ((x) &= 0xFFFE)


/*
 * type representing whether a clause is satisfied or not. 
 * LSB bit (0) defines whether the clause is satisfied or not
 * As for truth value, bits 1 to 13 are dedicated to stack information
 */
typedef short satisfied_t;

#define SATISFIED(x) ((x) & 0x0001)

#define SET_SATISFIED(x) ((x) |= 0x0001)
#define SET_NOT_SATISFIED(x) ((x) &= 0xFFFE)




/* 
 * operations on stack depth information
 * bits 13 to 1 define stack depth of either a truth value, either a clause satisfied info
 * 
 * max depth = 2^13 = 8192
 */


#define STACK_DEPTH(x) (((x) & 0x3FFE) >> 1)
#define SET_STACK_DEPTH(x,depth) ((x) = ((x) & 0xC001) | (depth) << 1)

#endif

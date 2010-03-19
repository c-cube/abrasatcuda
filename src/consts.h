/*
 * file with some useful consts
 * intented to be included everywhere
 */

#ifndef CONSTS_H
#define CONSTS_H


#include <stdio.h> // print, flockfile, funlockfile

// for backtracking

typedef int success_t;

#define SUCCESS (0)
#define FAILURE (-1)


// for predicates
typedef short truth_t;

#define TRUE (1)
#define FALSE (0)


#define HLINE print("-------------------------------------------------------\n");

/*
 * a thread safe version of print : it
 * prevents several threads to access stdout
 * at the same time.
 */
#if PARALLEL == pthread
#define print(args...)         do {                             \
    flockfile(stdout);                                          \
    printf( args );                                             \
    funlockfile(stdout);                                        \
} while(0)
#else
#define print(args...)  printf(args)
#endif


#endif

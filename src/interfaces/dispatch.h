/*
 * This header defines an interface that the main() can use to solve 
 * a problem. It provides only a function, solve(), which purpose
 * is to dispatch solving work on one or many threads (or even on
 * network...) and then gather the results to give them back to 
 * the caller.
 * It uses a solve_thread (module implementing solve.h) as a basic 
 * computing unit.
 */



#ifndef DISPATCH_H
#define DISPATCH_H

#include "../consts.h"
#include "../clause.h"
#include "../vars.h"



/*
 * this function's purpose is to manage to solve the problem.
 * It relies on solve_thread (one or many instances) to do so.
 */


    // TODO : create CUDA threads, each with its own [vars] array,
    // and dispatch it in CUDA.
    // TODO : find the k most "interesting" vars, and create 2^k threads
    // with different var affectations.
    // TODO : think of a way to share information between threads (for example,
    // if a thread finds out that a var cannot be true (exhausted possibilities ?)
    // it may tell the other threads to set this var to 0)



#endif


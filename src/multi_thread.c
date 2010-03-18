#include "multi_thread.h"
#include "interfaces/solve.h" // solve_thread
#include "heuristic.h" // choose_immutable_vars


#include <pthread.h> // pthread stuff


// gets the number of threads at compile time
#ifndef THREAD_NUM
#define THREAD_NUM 2
#warning should define THREAD_NUM !
#endif

// struct used to pass several args through pthread_create
struct solve_args
{
    atom_t *formula;
    atom_t *clauses_index;
    value_t *vars;
    int clause_n;
    int var_n;
};


/* 
 * This function is the one executed on a fresh thread. It mainly
 * calls solve_thread() with the right arguments.
 *
 * It must :
 *      de-encapsulate args, 
 *      run solve() with them,
 *      give the results back to caller
 */
void *
thread_func( void *void_args )
{
    // creates independant args from the struct
    struct solve_args *args = (struct solve_args*) void_args;
    atom_t *formula = args->formula;
    atom_t *clauses_index = args->clauses_index;
    value_t *vars = args->vars;
    int clause_n = args->clause_n;
    int var_n = args->var_n;

#ifdef DEBUG
    printf("thread %d has vars ", pthread_self()); 
    value_print(vars, var_n);
#endif

    // runs the solve_thread() function
    success_t result = solve_thread( formula, clauses_index, vars, clause_n, var_n );

#ifdef DEBUG
    printf("thread %d has found %s\n", unsigned long int) pthread_self(), result == SUCCESS ? "true" : "false" );
#endif

    // now notify the main thread
    // TODO
}



/*
 * just encapsulates its args and launches thread
 */
static inline void 
launch_thread( atom_t* formula, atom_t *clauses_index, value_t *vars, int clause_n, int var_n, pthread_t *thread )
{
    // create a struct to encapsulate args
    struct solve_args *args = malloc(sizeof(struct solve_args));
    args->formula = formula;
    args->clauses_index = clauses_index;
    args->vars = vars;
    args->clause_n = clause_n;
    args->var_n = var_n;

    // launches thread
    pthread_create( thread, NULL, &thread_func, (void*) args );

}



/*
 * this solves the problem on several thread
 */
int 
solve( atom_t *formula, atom_t* clauses_index, int clause_n, int var_n )
{
#ifdef DEBUG
    printf("uses %d threads\n", THREAD_NUM);
#endif
    // create structure to hold pthread_t
    pthread_t threads[THREAD_NUM];

    // allocate space to hold private thread vars data
    // everything is initialized at 0
    value_t *all_vars = calloc(THREAD_NUM * (var_n+1), sizeof(value_t));

    // determine how to choose immutable vars
    value_t sorted_vars[var_n+1];
    choose_immutable_vars( formula, clauses_index, all_vars, sorted_vars, clause_n, var_n );

    // sets immutable vars (differently for each thread...)
    set_immutable_vars( all_vars, var_n, THREAD_NUM);

    // starts THREAD_NUM threads
    for (int i = 0; i < THREAD_NUM; ++i ){
        value_t *cur_vars = all_vars + (i * (var_n+1));
#ifdef DEBUG
        printf("launches thread %d with vars ", i); value_print( cur_vars, var_n); 
#endif
        
        // really launches this thread
        launch_thread( formula, clauses_index, cur_vars, clause_n, var_n, threads + i ); 
    }

}


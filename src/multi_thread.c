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
    pthread_mutex_t *mutex_answer;
    pthread_cond_t *cond_answer;
    int *thread_terminated;
    success_t *success_answer;
    value_t **vars_answer;
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


    // runs the solve_thread() function
    success_t result = solve_thread( formula, clauses_index, vars, clause_n, var_n );

#ifdef DEBUG
    print("thread %lu has found %s\n", (unsigned long int) pthread_self(), result == SUCCESS ? "true" : "false" );
#endif

    // now, notify the main thread
    pthread_mutex_lock( args->mutex_answer );
        ++ *(args->thread_terminated);
#ifdef DEBUG
        print("currently, %d threads have finished\n", *(args->thread_terminated) );
#endif
        // give the good combination back
        if ( result == SUCCESS ){
            *(args->vars_answer) = vars;
        }
        *(args->success_answer) = result;
        // signal the main thread that this has changed
        pthread_cond_signal( args->cond_answer );
    pthread_mutex_unlock( args->mutex_answer );

    free( void_args );
    return NULL;
}



/*
 * just encapsulates its args and launches thread
 */
static inline void 
launch_thread( atom_t* formula, atom_t *clauses_index, value_t *vars, int clause_n, int var_n, pthread_t *thread,
    pthread_mutex_t *mutex_answer, pthread_cond_t *cond_answer, 
    int *thread_terminated, success_t *success_answer, value_t **vars_answer)
{
    // create a struct to encapsulate args
    struct solve_args *args = (struct solve_args*) malloc(sizeof(struct solve_args));
    args->formula = formula;
    args->clauses_index = clauses_index;
    args->vars = vars;
    args->clause_n = clause_n;
    args->var_n = var_n;
    args->mutex_answer = mutex_answer;
    args->cond_answer = cond_answer;
    args->thread_terminated = thread_terminated;
    args->success_answer = success_answer;
    args->vars_answer = vars_answer;

    // launches thread
    pthread_create( thread, NULL, &thread_func, (void*) args );

}



/*
 * this solves the problem on several thread
 */
success_t 
solve( atom_t *formula, atom_t* clauses_index, int clause_n, int var_n )
{
#ifdef DEBUG
    print("uses %d threads\n", THREAD_NUM);
#endif
    // create structure to hold pthread_t
    pthread_t threads[THREAD_NUM];
    
    // create a mutex and a cond to synchronize threads with main thread
    pthread_mutex_t mutex_answer = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond_answer = PTHREAD_COND_INITIALIZER;
    // shared values used to answer
    success_t success_answer = FAILURE;
    int thread_terminated = 0;
    value_t *vars_answer = NULL;

    // allocate space to hold private thread vars data
    // everything is initialized at 0
    value_t *all_vars = calloc(THREAD_NUM * (var_n+1), sizeof(value_t));

    // determine how to choose immutable vars
#ifdef DEBUG
    print("sorts vars by value\n");
#endif
    int sorted_vars[var_n+1];
    sort_vars_by_value( formula, clauses_index, all_vars, sorted_vars, clause_n, var_n );

    // sets immutable vars (differently for each thread...)
#ifdef DEBUG
    print("chooses immutable vars and sets them\n");
#endif
    set_immutable_vars( all_vars, sorted_vars, var_n, THREAD_NUM );

    // starts THREAD_NUM threads
    for (int i = 0; i < THREAD_NUM; ++i ){
        value_t *cur_vars = all_vars + (i * (var_n+1));
#ifdef DEBUG
        print("launches thread %d with vars ", i); value_print( cur_vars, var_n); 
#endif
        
        // really launches this thread
        launch_thread( formula, clauses_index, cur_vars, clause_n, var_n, 
            threads + ((pthread_t) i), &mutex_answer, &cond_answer, 
            &thread_terminated, &success_answer, &vars_answer ); 
    }

    // wait until all threads notify they have finished, or one said 'success'
    while (1){
        // lock the mutex
        pthread_mutex_lock( &mutex_answer );
        // release the mutex and wait for a notification
        if ( success_answer == FAILURE ){
            pthread_cond_wait( &cond_answer, &mutex_answer );
        }
        // ok, one of the threads returned true !
        if ( success_answer == SUCCESS ){
#ifdef DEBUG
            print("global success ! ");
#endif
            value_print( vars_answer, var_n );
            return SUCCESS;
        } else {
#ifdef DEBUG
            print("%d threads have died without success\n", thread_terminated );
#endif
            // all threads have died without success
            if ( thread_terminated >= THREAD_NUM )
                return FAILURE;
        }

        pthread_mutex_unlock( &mutex_answer );
    }

}


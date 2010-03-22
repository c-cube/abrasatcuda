#include <stdio.h>
#include <dlfcn.h>

#include "parser.h" // parse
#ifndef CUDA
#include "abrasatcuda.h"
#endif
#include "consts.h"
#ifdef CUDA
#include "solve.h"
#endif

#include "interfaces/dispatch.h" 
#ifdef CUDA
#include "solve.h"
#endif


// gets the number of threads at compile time
#ifndef THREAD_NUM
#define THREAD_NUM 2
#warning should define THREAD_NUM ! (set to 2)
#endif





int main( int argc, char ** argv )
{
    
    // if no arg is supplied, error
    if (argc < 3){
        print("usage : abrasatcuda lib file [thread number]\n");
        return 1;
    }

    // TODO use getopt ?
    char* lib_path = argv[1];
    char* file_path = argv[2];

    int thread_n;
    if ( argc > 3 )
        thread_n = atoi(argv[3]);
    else
        thread_n = THREAD_NUM;

    atom_t *formula = NULL;
    atom_t *clauses_index = NULL;
    int num_var, num_clause;

    /*
     * Now we try to open the dynamic lib containing
     * the functions to solve the problem.
     */
    success_t (*solve)( atom_t *formula, atom_t* clauses_index, 
        int clause_n, int var_n, int thread_n );
    void *lib_handle = dlopen( lib_path, RTLD_LAZY );
    if ( lib_handle == NULL ){
        print("unable to load file %s\n", lib_path);
        exit(1); // TODO : find the good error
    }

    solve = dlsym( lib_handle, "solve" );
    if ( solve == NULL ){
        print("unable to find symbol solve in dynamic lib\n");
        exit(1);
    }
#ifdef DEBUG
    print("dynamic lib opened\n");
#endif


    // ok, we now have the function !


    // parses data file
    HLINE
#ifdef DEBUG
    print("parses file %s\n", file_path );
#endif
    parse( file_path, &formula, &clauses_index, &num_var, &num_clause );

    HLINE
#ifdef DEBUG
    print("file parsed, formula of %d clauses and %d var built\n",num_clause, num_var );

    formula_print( formula, clauses_index, num_clause );

    HLINE
    print("tries to solve\n");
#endif


    int answer = (*solve)( formula, clauses_index, num_clause, num_var, thread_n );

    print("Answer : \033[31;4m%s\033[m \n", answer == SUCCESS ? "True" : "False" );


    // de-allocate memory
    free(formula);
    free(clauses_index);


    // unload library *completely*
    if ( dlclose( lib_handle ) != 0 ){
        print("error on dlclose() : %s\n", dlerror());

    }

    return 0;
}








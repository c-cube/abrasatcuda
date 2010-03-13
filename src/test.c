/*
 * miscalleanous tests file
 */

#include <assert.h>
#include <stdio.h>


#include <unistd.h>
//#define PAUSE sleep(1);
#define PAUSE 

#define HLINE printf("-------------------------------------------------------\n");


/*
 * modules to test
 */

#include "list.h"
#include "parser.h"
#include "clause.h"
#include "solve.h"

// feel the power !
#include "utils.h"


/*
 * tests the list.h module
 */

void test_list()
{

    printf( "testing list.h... \n" );
    PAUSE

    list_t l;
    LIST_NODE_T a,b,c;

    list_init( &l );

    assert( l.is_empty );
    assert( l.node == NULL );

    list_push( &l, &a );

    assert( a.next == &a );
    assert( a.previous == &a );
    assert( l.node == &a );
    assert( ! l.is_empty );
    LIST_NODE_T *iterator1 = NULL;
    assert( list_iterate( &l, &iterator1 ) != -1 );
    assert( iterator1 == &a );
    assert( list_iterate( &l, &iterator1 ) == -1 );

    assert( list_member( &l, &a ) );
    assert( list_member( &l, &a ) );
    assert( ! list_member( &l, &b ) );
    

    list_append( &l, &b );

    assert( a.next == &b );
    assert( a.previous == &b );
    assert( b.previous == &a );
    assert( b.next == &a );
    assert( list_member( &l, &a ) );
    assert( list_member( &l, &b ) );
    assert( list_member( &l, &b ) );


    list_push( &l, &c );

    assert( b.next == &c );
    assert( c.next == &a );
    assert( b.previous == &a );
    assert( c.previous == &b );
    assert( list_length( &l ) == 3 );
    assert( l.node == &c );

    assert( list_pop( &l ) == &c );

    assert( a.next == &b );
    assert( b.previous == &a );
    assert( b.next == &a );
    assert( a.previous == &b );
    assert( list_length( &l ) == 2 );
    assert( list_length( &l ) == 2 );
    assert( l.node == &a );

    LIST_NODE_T *iterator = NULL;;
    list_iterate( &l, &iterator );
    assert( iterator == &a );
    list_iterate( &l, &iterator );
    assert( iterator == &b );
    assert( list_iterate( &l, &iterator ) == -1 );

    printf( "\033[44mOK\033[m !\n" );
    HLINE
}




/*
 * tests parser.c
 */


void test_parser()
{

    printf( "testing parser.c... \n" );
    PAUSE

    FILE* input = fopen( "./tests/example.cnf", "r" );

    assert( input != NULL );

    list_t *lines = read_lines( input );

    fclose( input );
    
    assert( lines != NULL );
    assert( ! lines->is_empty );
    printf( "nbr de lignes : %d\n", list_length( lines ));
    printf( "nbr de lignes : %d\n", list_length( lines ));
    assert( list_length( lines ) == 8 );

    LIST_NODE_T *iterator = NULL;
    while ( list_iterate( lines, &iterator ) != -1 ) {
        printf( "line : "); printf( "%s", line_of_list_node(iterator)->content );
    } 


    atom_t *formula = NULL;
    atom_t *clauses_index = NULL;
    int num_clause, num_var;

     
    printf("begins parsing clauses\n");
    parse_lines( lines, &formula, &clauses_index, &num_var, &num_clause ); 

    printf("clauses parsed, now checking content\n");

    assert(formula != NULL);
    assert(clauses_index != NULL);

    int i;
    for (i=0; i<num_clause;++i)
        printf("atom : %d\n", VARIABLE_NAME(clauses_index[i]));

    int n=0;
    for (int i=0; i<n; ++i){
        clause_print( 
            formula_item( formula, clauses_index, i), 
            formula_item( formula, clauses_index, i+1) );
        printf("\n");
    }
        
    formula_print( formula, clauses_index, num_clause ); 

    printf( "\033[44mOK\033[m !\n" );
    HLINE
}


/*
 * tests clause functions
 */
void test_clause()
{
    printf( "testing clause.h... \n" );
    atom_t *a = malloc(6 * sizeof(atom_t));
    *clause_item(a, 0) = make_atom(4); printf("atome : %hu\n", VARIABLE_NAME(*clause_item(a, 0)));
    *clause_item(a, 1) = make_atom(-3); printf("atome : %hu\n", VARIABLE_NAME(*clause_item(a, 1)));
    *clause_item(a, 2) = make_atom(2); printf("atome : %hu\n", VARIABLE_NAME(*clause_item(a, 2)));
    *clause_item(a, 3) = make_atom(-627); printf("atome : %hu\n", VARIABLE_NAME(*clause_item(a, 3)));


    //assert( clause_length(a) == 5 );

    /*
    atom_t *iterator = NULL;
    while ( atom_iterate( a, &iterator ) != -1 ){
        printf( "atom with identity %u. is it negative : %u\n", 
            VARIABLE_NAME( *iterator ), IS_NEGATED_BINARY( *iterator ) );

        assert( IS_USED( *iterator ) );
    }
    */


    NEGATE( *clause_item(a, 0)); 
    assert( IS_NEGATED_BINARY( *clause_item(a, 0)) );
    NEGATE( *clause_item(a, 0)); 
    NEGATE( *clause_item(a, 1)); 
    assert( ! IS_NEGATED_BINARY( *clause_item(a, 1)) );
    NEGATE( *clause_item(a, 1)); 

    UNUSE( *clause_item(a, 2));
    assert( ! IS_USED( *clause_item(a, 2) ));
    assert( ! IS_USED( *clause_item( a,2) ));


    // printf("clause a = "); clause_print( a ); printf( "\n" );
    

    atom_t *b = malloc(3*sizeof(atom_t));
    atom_t *c = malloc(3*sizeof(atom_t));

    *clause_item(b, 0) = make_atom(3); 
    *clause_item(b, 1) = make_atom(4);
    *clause_item(c, 0) = make_atom(1); 
    *clause_item(c, 1) = make_atom(2);

    printf("builds clause\n");

    // atom_t* formula = malloc( (3+1+2+1+2+1)*sizeof(atom_t));
    // atom_t *clauses_index = NULL;
    
    // proceed !
    //atom_t offset = formula_build( &formula, &clauses_index, truc, 3 ); 
    //printf("clause built : %d atom_t items long\n", offset);
    //assert( offset == 8 );

    //for (int i=0; i<3; ++i){
    //    printf("clause : "); clause_print( (atom_t*) (formula+(clauses_index[i])) ); printf("\n");
    //}

    //printf("prints clause\n");
    //formula_print( formula, clauses_index, 3 );

    printf( "\033[44mOK\033[m !\n" );
    HLINE
}

void test_solve()
{
    printf( "testing solve.h... \n" );


    char a = 0;
    printf("a = 0 is immutable : %d, affected : %d, has truth value : %d\n",
        IS_IMMUTABLE(a), IS_AFFECTED(a), TRUTH_VALUE(a));
    SET_TRUTH_VALUE(a,1);
    SET_IMMUTABLE(a);
    SET_AFFECTED(a);

    assert( IS_IMMUTABLE(a));
    assert( IS_AFFECTED(a));
    assert( TRUTH_VALUE(a));

    SET_NON_AFFECTED(a);
    assert( TRUTH_VALUE(a));
    assert( ! IS_AFFECTED(a));

    printf("a = 0 is immutable : %d, affected : %d, has truth value : %d\n",
        IS_IMMUTABLE(a), IS_AFFECTED(a), TRUTH_VALUE(a));


    // iterations
    printf("builds a 5 var-length truth value table\n");

    char tab[5];
    
    SET_IMMUTABLE(tab[0]);
    SET_AFFECTED(tab[0]);
    SET_AFFECTED(tab[1]);
    SET_AFFECTED(tab[2]);
    SET_AFFECTED(tab[3]);
    SET_AFFECTED(tab[4]);
    SET_TRUTH_VALUE(tab[0],1);
    SET_TRUTH_VALUE(tab[2],1);

    value_print( tab, 5 );

    printf("showing all iterations\n");

    int cur = 0;
    while ( next_combination( tab, &cur, 5) != 1){
        value_print(tab,5);
    }

    printf( "\033[44mOK\033[m !\n" );
    HLINE
}



/*
 * run all test successively
 */
int main(int argc, char** argv){

    HLINE

    test_list(); 
    test_clause();
    test_parser();
    test_solve();

    return 0;

}




/*
 * miscalleanous tests file
 */

#include <assert.h>
#include <stdio.h>


#include <unistd.h>
//#define PAUSE sleep(1);
#define PAUSE 

#define HLINE printf("----------------------------\n");


/*
 * modules to test
 */

#include "list.h"
#include "parser.h"
#include "clause.h"

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

    printf( "\e[44mOK\e[m !\n" );
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
    
    assert( lines != NULL );
    assert( ! lines->is_empty );
    printf( "nbr de lignes : %d\n", list_length( lines ));
    printf( "nbr de lignes : %d\n", list_length( lines ));
    assert( list_length( lines ) == 8 );

    LIST_NODE_T *iterator = NULL;
    while ( list_iterate( lines, &iterator ) != -1 ) {
        printf( "line : "); printf( "%s", container_of(iterator,line_t,list_node)->content );
    } 


    short int *formula = NULL;
    short int *clauses_array = NULL;
    int num_clause, num_var;

     
    printf("begins parsing clauses\n");
    parse_lines( lines, &formula, &clauses_array, &num_var, &num_clause ); 

    printf("clauses parsed, now checking content\n");
    LIST_NODE_T *iterator = NULL;
    while( clause_iterate( formula, iterator ) != -1 ){

        

    printf( "\e[44mOK\e[m !\n" );
    HLINE
}


/*
 * tests clause functions
 */
void test_clause()
{
    printf( "testing clause.h... \n" );
    clause_t a;
    a.clause_array = malloc(4*sizeof(short));
    a.clause_array[0] = make_atom(4); printf("atome : %hu\n", VARIABLE_NAME(make_atom(4)));
    a.clause_array[1] = make_atom(-3); printf("atome : %hu\n", VARIABLE_NAME(make_atom(-3)));
    a.clause_array[2] = make_atom(2); printf("atome : %hu\n", VARIABLE_NAME(make_atom(2)));
    a.clause_array[3] = make_atom(-627); printf("atome : %hu\n", VARIABLE_NAME(make_atom(-627)));
    a.stop = ((short*) a.clause_array)+4;

    short *iterator = NULL;
    while ( atom_iterate( &a, &iterator ) != -1 ){
        printf( "atom with identity %u. is it negative : %u\n", 
            VARIABLE_NAME( *iterator ), IS_NEGATED_BINARY( *iterator ) );

        assert( IS_USED( *iterator ) );
    }


    NEGATE(a.clause_array[0]); 
    assert( IS_NEGATED_BINARY(a.clause_array[0]) );
    NEGATE(a.clause_array[1]); 
    assert( ! IS_NEGATED_BINARY(a.clause_array[1]) );

    UNUSE(a.clause_array[2]);
    assert( ! IS_USED( a.clause_array[2] ));
    
    printf( "\e[44mOK\e[m !\n" );
    HLINE
}



/*
 * run all test successively
 */
int main(){

    HLINE

    test_list(); 
    test_parser();
    test_clause();

    return 0;

}

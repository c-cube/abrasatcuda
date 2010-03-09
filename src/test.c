/*
 * miscalleanous tests file
 */

#include <assert.h>
#include <stdio.h>


#include <unistd.h>
#define PAUSE sleep(1);


/*
 * modules to test
 */

#include "list.h"
#include "parser.h"

// feel the power !
#include "utils.h"


/*
 * tests the list.h module
 */

void test_list(){

    printf( "testing list.h... " );
    PAUSE

    list_t l;
    LIST_NODE_T a,b,c;

    list_init( &l );

    assert( l.is_empty );
    assert( l.node == NULL );

    list_push( &l, &a );

    assert( a.next == &a );
    assert( a.previous == &a );
    assert( list_member( &l, &a ) );
    assert( ! list_member( &l, &b ) );
    

    list_append( &l, &b );

    assert( a.next == &b );
    assert( a.previous == &b );
    assert( b.previous == &a );
    assert( b.next == &a );
    assert( list_member( &l, &a ) );
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
    assert( l.node == &a );

    LIST_NODE_T **iterator;
    iterate( &l, iterator );
    assert( *iterator == &a );
    iterate( &l, iterator );
    assert( *iterator == &b );
    assert( iterate( &l, iterator ) == 0 );

    printf( "OK !\n" );
}




/*
 * tests parser.c
 */


void test_parser()
{
    printf( "testing parser.c... " );
    PAUSE

    char mode[] = {'r', '\0'};
    FILE* input = fopen( "./tests/example.cnf", mode );

    assert( input != NULL );


    line_t *lines = read_lines( input );
    assert( lines != NULL );
    printf( "nbr de lignes : %d\n", list_length( &lines->list_node ));
    assert( list_length( &lines->list_node ) == 9 );

    LIST_NODE_T *iterator = &lines->list_node;
    do {
        printf( "line : "); printf( "%s", container_of(iterator,line_t,list_node)->content );
    } while ( iterate( &lines->list_node, &iterator ) != 0 );
        

    printf("OK !\n" );


    return;
}


/*
 * run all test successively
 */
int main(){

    test_list(); 
    test_parser();

    return 0;

}

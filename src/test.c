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



/*
 * tests the list.h module
 */

void test_list(){

    printf( "testing list.h... " );
    PAUSE

    LIST_NODE_T a,b,c;

    list_init( &a );

    assert( a.next == &a );
    assert( a.previous == &a );
    assert( a.alone == 0 );
    assert( list_member( &a, &a ) );
    assert( ! list_member( &a, &b ) );
    

    list_add( &a, &b );

    assert( a.next == &b );
    assert( a.previous == &b );
    assert( b.previous == &a );
    assert( b.next == &a );
    assert( b.alone == 0 );
    assert( list_member( &a, &a ) );
    assert( list_member( &a, &b ) );


    list_add( &b, &c );

    assert( b.next == &c );
    assert( c.next == &a );
    assert( b.previous == &a );
    assert( c.previous == &b );


    list_remove( &b );

    assert( a.next == &c );
    assert( c.previous == &a );
    assert( c.next == &a );
    assert( a.previous == &c );
    assert( b.next == &b );
    assert( b.previous == &b );
    assert( b.alone == 1 );

    LIST_NODE_T *iterator = &a;
    iterate( &a, &iterator );
    assert( iterator == &c );
    assert( iterate( &a, &iterator ) == 0 );

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

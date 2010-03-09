/*
 * fichier de tests divers
 */

#include <assert.h>
#include <stdio.h>

#include "list.h"


void test_list(){

    printf( "teste list.h... " );

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

    printf( "OK !\n" );
}


/*
 * run all test successively
 */
int main(){

    test_list(); 



    return 0;

}

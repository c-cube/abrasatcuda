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
        printf( "line : "); printf( "%s", line_of_list_node(iterator)->content );
    } 


    atom_t *formula = NULL;
    int *clauses_index = NULL;
    int num_clause, num_var;

     
    printf("begins parsing clauses\n");
    parse_lines( lines, &formula, &clauses_index, &num_var, &num_clause ); 

    printf("clauses parsed, now checking content\n");

    assert(formula != NULL);
    assert(clauses_index != NULL);

    int i;
    for (i=0; i<num_clause;++i)
        printf("atom : %d\n", VARIABLE_NAME(clauses_index[i]));

    clause_t *clause_iterator = NULL;
    int n=0;
    while( clause_iterate( formula, clauses_index, num_clause, &n, &clause_iterator ) != -1 ){
        clause_print( clause_iterator );
        printf("\n");
    }
        
        

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
    a.clause_array = malloc(4*sizeof(atom_t));
    a.clause_array[0] = make_atom(4); printf("atome : %hu\n", VARIABLE_NAME(make_atom(4)));
    a.clause_array[1] = make_atom(-3); printf("atome : %hu\n", VARIABLE_NAME(make_atom(-3)));
    a.clause_array[2] = make_atom(2); printf("atome : %hu\n", VARIABLE_NAME(make_atom(2)));
    a.clause_array[3] = make_atom(-627); printf("atome : %hu\n", VARIABLE_NAME(make_atom(-627)));
    a.stop = a.clause_array+4;

    assert( CLAUSE_LENGTH(a) == 4 );

    atom_t *iterator = NULL;
    while ( atom_iterate( &a, &iterator ) != -1 ){
        printf( "atom with identity %u. is it negative : %u\n", 
            VARIABLE_NAME( *iterator ), IS_NEGATED_BINARY( *iterator ) );

        assert( IS_USED( *iterator ) );
    }


    NEGATE(a.clause_array[0]); 
    assert( IS_NEGATED_BINARY(a.clause_array[0]) );
    NEGATE(a.clause_array[0]); 
    NEGATE(a.clause_array[1]); 
    assert( ! IS_NEGATED_BINARY(a.clause_array[1]) );
    NEGATE(a.clause_array[1]); 

    UNUSE(a.clause_array[2]);
    assert( ! IS_USED( a.clause_array[2] ));
    assert( ! IS_USED( CLAUSE_ITEM(a,2) ));


    printf("clause a = "); clause_print( &a ); printf( "\n" );
    

    clause_t b; b.clause_array = malloc(2*sizeof(atom_t));
    clause_t c; c.clause_array = malloc(2*sizeof(atom_t));

    b.clause_array[0] = make_atom(3); 
    b.clause_array[1] = make_atom(4);
    c.clause_array[0] = make_atom(1); 
    c.clause_array[1] = make_atom(2);
    b.stop = b.clause_array + 2;
    c.stop = c.clause_array + 2;

    printf("builds clause\n");
    clause_t truc[] = {a,b,c};
    atom_t *d;
    atom_t *clauses_index;
    atom_t offset = clause_build( &d, &clauses_index, truc, 3 ); 
    printf("clause built : %d atom_t items long\n", offset);
    assert( offset == 8 );

    printf("prints clause\n");
    formula_print( d, clauses_index, 3 );

    printf( "\e[44mOK\e[m !\n" );
    HLINE
}



/*
 * run all test successively
 */
int main(){

    HLINE

    test_list(); 
    test_clause();
    test_parser();

    return 0;

}

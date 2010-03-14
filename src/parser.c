#include <string.h>
#include <assert.h>



#include "parser.h"



// drop a line of comments
#define DROP_COMMENT(iterator) if ( (iterator)->content[0] == 'c' ) continue; 

#define DEBUG printf

#define CLAUSE_N(formula,clauses_index,n) ((clause_t*) formula[(*clauses_index)[n]])

int parse( const char* file_path, atom_t **formula, atom_t **clauses_index, int *num_var, int *num_clause )
{

    // open file
    FILE* input = fopen( file_path, "r" );

    // read lines from file
    list_t *lines = read_lines( input );
    fclose( input );

    // parse each line into the formula
    return parse_lines( lines, formula, clauses_index, num_var, num_clause );
}    
   
/*
 * this function reads the file and puts lines
 * in a list
 */
list_t *read_lines( FILE* input )
{
    // list of lines
    list_t *lines = malloc(sizeof(line_t));
    list_init( lines );

    // read lines
    char* linePtr;
    line_t *new_line;
    size_t n = 0;
    while ( getline( &linePtr, &n, input ) != FAILURE ) {
        new_line = malloc(sizeof( line_t ));        
        new_line->content = linePtr;
        list_append( lines, &(new_line->list_node) );

        linePtr = NULL; // asks getline to allocate a new buffer for next line
    }

    return lines;

}
        

/*
 * this function is intended to read each line and build the formula
 */
int parse_lines( list_t* lines, atom_t ** formula, atom_t **clauses_index, int *num_var, int *num_clause ){

    int formula_length = 0;
    int offset_in_formula = 0;
    int clause_index = 0;
    int atom_num = 0;

    list_t *clauses = malloc(sizeof(list_t));
    list_init( clauses );
    
    // assertions
    assert( *formula == NULL );
    assert( *clauses_index == NULL );

    line_t *iterator = NULL;
    LIST_NODE_T *list_iterator = NULL;
    int is_looking_for_pb = 1;
    int current_token = 0; 
    char *offset_in_line, *offset_in_line_bis;
    // for each line (list_pop returns a list node)
    //DEBUG("starts loop of parse_lines\n");
    while ( (list_iterator = list_pop( lines )) != NULL ){
        iterator = line_of_list_node(list_iterator);
        //DEBUG( "parser loop with line %s", iterator->content );
        
        // before beginning of clause list, initialize everything
        if (is_looking_for_pb){
            DROP_COMMENT(iterator)
            //DEBUG("hello problem line\n");
            assert( iterator->content[0] == 'p' ); // either comment or problem at first
            assert( strncmp(iterator->content, "p cnf", 5) == 0 ); // match pb with cnf

            // drop the "p cnf" at the beginning
            sscanf( iterator->content + 5, "%d %d", num_var, num_clause);  
            is_looking_for_pb = 0;

            DEBUG("problem of size #var = %d, #clause = %d\n", *num_var, *num_clause);
            // allocate formula
            formula_length = *num_clause * *num_var * 5;
            *formula = malloc( formula_length * sizeof(short int)) ; // what a beautiful heuristic !
            // allocate clause_array. It has a bonus slot for the last pointer
            *clauses_index = malloc( (*num_clause+1) * sizeof(atom_t*) );
            clause_index = 0;
            offset_in_formula = 0;
            
            // create first clause structure
            (*clauses_index)[0] = offset_in_formula; // clause_t here

            continue;
        }

        DROP_COMMENT(iterator) // if line is a comment, drop it

        //DEBUG("searching for clauses\n");

        // ok, now we are sure we have a clause line.
        
        offset_in_line = iterator->content;
        offset_in_line_bis = offset_in_line;
        while ( 1 ){
            current_token = (int) strtol( offset_in_line, &offset_in_line_bis, 10 );

            // If formula is full, extend it.
            if ( formula_length - offset_in_formula <= 2 ){
                formula_length = (int) (1.5 * formula_length); // add some place
                *formula = realloc( *formula, formula_length * sizeof(short int) );
            }

            // special cases
            if ( offset_in_line_bis == offset_in_line )
                break;
            if ( current_token == 0 ){
                // remember where new clause begins
                (*clauses_index)[ ++ clause_index ] = offset_in_formula; // clause_t here
                //DEBUG("clauses_index[%d] = %d\n", clause_index, offset_in_formula );
                offset_in_line = 0; // begin next line
                assert( (*clauses_index)[clause_index] == offset_in_formula );
                break;
            } else {
                offset_in_line = offset_in_line_bis; 
                atom_t current_atom = make_atom( current_token );
                
                atom_num++;

                (*formula)[ offset_in_formula++ ] = current_atom;
            }
        }

        continue;   // next line

    }
    DEBUG("number of clauses = %d, clause_index = %d\n", *num_clause, clause_index );
    assert( clause_index == *num_clause );
    assert( offset_in_formula == atom_num );
    printf("offset_in_formula = %d, num_clause = %d,  clause_index[n] = %d\n", offset_in_formula, 
        *num_clause, (*clauses_index)[*num_clause]);
    assert( offset_in_formula == (*clauses_index)[*num_clause] );

    return SUCCESS; 
}
        





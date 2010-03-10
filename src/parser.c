#include <string.h>
#include <assert.h>



#include "parser.h"



// drop a line of comments
#define DROP_COMMENT(iterator) if ( (iterator)->content[0] == 'c' ) continue; 

#define DEBUG printf

#define CLAUSE_N(formula,clauses_index,n) ((clause_t*) formula[(*clauses_index)[n]])

int parse( const char* file_path, short int **formula, int **clauses_index, int *num_var, int *num_clause )
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
    while ( getline( &linePtr, &n, input ) != -1 ) {
        new_line = malloc(sizeof( line_t ));        
        new_line->content = linePtr;
        list_append( lines, &(new_line->list_node) );

        //DEBUG("%s", linePtr );

        linePtr = NULL; // asks getline to allocate a new buffer for next line
    }

    return lines;

}
        

/*
 * this function is intended to read each line and build the formula
 */
int parse_lines( list_t* lines, short int ** formula, int **clauses_index, int *num_var, int *num_clause ){

    int formula_length = 0;
    int offset_in_formula = 0;
    int clause_index = 0;

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
            *clauses_index = malloc( (*num_clause+1) * sizeof(short int *) );
            clause_index = 0;
            offset_in_formula = 0;
            
            // create first clause structure
            (*clauses_index)[0] = offset_in_formula; // clause_t here
            offset_in_formula = offset_in_formula + (sizeof(clause_t))/sizeof(short int); 
            CLAUSE_N(formula, clauses_index, 0)->clause_array = 
                    (*formula) + offset_in_formula; // atoms just further


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

            //DEBUG( "token read : %d\n", current_token);
            // special cases
            if ( offset_in_line_bis == offset_in_line )
                break;
            if ( current_token == 0 ){
                // remember where new clause begins
                DEBUG("puts index %d in clauses_index[%d]\n", offset_in_formula, clause_index+1);
                (*clauses_index)[ ++ clause_index ] = offset_in_formula; // clause_t here
                int old_offset_in_formula = offset_in_formula;
                offset_in_formula = offset_in_formula + (sizeof(clause_t))/sizeof(short int); 
                CLAUSE_N(formula, clauses_index, clause_index)->clause_array = 
                        (*formula) + offset_in_formula; // atoms just further
                if (clause_index >= 1){
                    CLAUSE_N(formula, clauses_index, clause_index-1)->stop = (*formula) + old_offset_in_formula;
                    break;
                }
            }

            offset_in_line = offset_in_line_bis;
            short int current_atom = make_atom( current_token );
            
            //DEBUG("atom read : %d at offset_in_formula %d\n",VARIABLE_NAME(current_atom), offset_in_formula);
            
            (*formula)[ offset_in_formula++ ] = current_atom;

        }

        continue;   // next line

    }

    return 0; 
}
        





#include "parser.h"

// getline !
#define _GNU_SOURCE
#include <stdio.h>


int parse( const char* file_path, short int ** formula )
{

    // open file
    const char mode[] = {'r', '\0'};
    FILE* input = fopen( file_path, mode );

    // read lines from file
    line_t *lines = read_lines( input );
    fclose( input );

    // parse each line into the formula
    return parse_lines( lines, formula );
}    
   

line_t *read_lines( FILE* input )
{
    line_t *first_line = malloc(sizeof( line_t ));
    line_t *current_line = first_line;

    first_line->content = "premiere ligne";

    // read lines
    char* linePtr;
    while ( getline( &linePtr, NULL, input ) ){
        line_t *new_line = malloc(sizeof( line_t ));
        new_line->content = linePtr;
        list_add( &(current_line->list_node), &(new_line->list_node) ); // adds line to the list
        current_line = new_line;

        linePtr = NULL; // asks getline to allocate a new buffer for next line
    }

    return first_line;

}
        

int parse_lines( line_t* lines, short int ** formula ){

    return 0; // TODO : real code
}
        





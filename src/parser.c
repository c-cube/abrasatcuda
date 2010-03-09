#include "parser.h"



int parse( const char* file_path, short int ** formula )
{

    // open file
    FILE* input = fopen( file_path, "r" );

    // read lines from file
    list_t *lines = read_lines( input );
    fclose( input );

    // parse each line into the formula
    return parse_lines( lines, formula );
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

        //printf("%s", linePtr );

        linePtr = NULL; // asks getline to allocate a new buffer for next line
    }

    return lines;

}
        

/*
 * this function is intended to read each line and build the formula
 */
int parse_lines( list_t* lines, short int ** formula ){




    return 0; // TODO : real code
}
        





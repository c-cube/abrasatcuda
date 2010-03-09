#ifndef _PARSER_H


#include <stdlib.h>
#include <stdio.h>

#include "clause.h"
#include "list.h"


// handles a line of input
typedef struct {
    char *content;
    LIST_NODE_T list_node;
} line_t;


/*
 * utilities functions
 */

line_t *read_lines( FILE* input );

int parse_lines( line_t* lines, short int ** formula );


/*
 * parses the file into the formula 
 * returns 0 on failure, 1 on success
 */

int parse( const char* file_path, short int ** formula ); 




#define _PARSER_H 1
#endif

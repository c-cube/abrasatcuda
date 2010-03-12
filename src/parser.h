/*
 * utilities for parsing CNF files.
 */

/* LICENSE :
DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
                   Version 2, December 2004 

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net> 

Everyone is permitted to copy and distribute verbatim or modified 
copies of this license document, and changing it is allowed as long 
as the name is changed. 

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION 

    0. You just DO WHAT THE FUCK YOU WANT TO.
*/



#ifndef _PARSER_H
#define _PARSER_H 1

// getline !
#define _GNU_SOURCE
#include <stdio.h>

#include "clause.h"
#include "list.h"
#include "utils.h"


// handles a line of input
typedef struct {
    char        *content;
    LIST_NODE_T list_node;
} line_t;


// get the line from its line_t->list_node
#define line_of_list_node(line) container_of(line,line_t,list_node)

/*
 * utilities functions
 */

list_t *read_lines( FILE* input );

int parse_lines( list_t* lines, atom_t ** formula, atom_t **clauses_index, int *num_var, int *num_clause );


/*
 * parses the file into the formula 
 * returns 0 on failure, 1 on success
 */

int parse( const char* file_path, atom_t ** formula, atom_t **clauses_index, int *num_var, int *num_clause );



#endif



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


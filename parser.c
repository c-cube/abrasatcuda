#include "parser.h"
#include <stdio.h>

int parse( const char* file_path, short int ** formula )
{


    char *mode = {'r', '\0'};
    FILE* input = fopen( file_path, mode );

    char buf[256];

    while ( fgets( buf, 256, input ) > 0 )
    {
        
        // skips comment lines
        if ( buf[0] == 'c' )
            continue;
        

        





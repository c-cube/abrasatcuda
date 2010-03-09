#ifndef _LIST_H


#include <stdlib.h>

/*
 * module de listes chainées inspirées de celles du noyau linux
 */



typedef struct list_node_t      
{                                       
   struct list_node_t* next;                   
   struct list_node_t* previous;               
   unsigned short int alone;            
} LIST_NODE_T ;

/*
 * initializes node as the first (and only)
 * member of a list.
 */
#define list_init( node )   do {            \
    (node)->previous = node;                \
    (node)->next = node;                    \
    (node)->alone = 0;                      \
    } while (0)

/*
 * adds obj to the list, just after node.
 */
#define list_add( node, obj )   do {        \
    LIST_NODE_T *my_node, *my_obj;          \
    my_node = (node);                       \
    my_obj = (obj);                         \
    my_node->next->previous = my_obj;       \
    my_obj->next = my_node->next;           \
    my_node->next = my_obj;                 \
    my_obj->previous = my_node;             \
    my_obj->alone = 0;                      \
    my_node->alone = 0;                     \
    } while (0)     


/*
 * removes node from the list it belongs to.
 * unspecified behavior if member of no list.
 */
#define list_remove( node )  do {                 \
    LIST_NODE_T *my_node;                         \
    my_node = (node);                             \
    my_node->next->previous = my_node->previous;  \
    my_node->previous->next = my_node->next;      \
    my_node->previous = my_node;                  \
    my_node->next = my_node;                      \
    my_node->alone = 1;                           \
    } while (0)


/*
 * sets iterator to the address of the next element of list.
 * usage : 
 *
 * iterator = &start;
 * do { process_item( *iterator); } while ( iterate( start, iterator ));
 */
inline unsigned short int iterate( LIST_NODE_T* start, LIST_NODE_T** iterator )
{
    if ( *iterator == NULL || (*iterator)->next == start )
        return 0;
    *iterator = (*iterator)->next; 
    return 1;
}


/*
 * tests for membership of obj to the list
 * node belongs to. Returns 1 on success, 0 on failure.
 */
inline unsigned short int list_member( LIST_NODE_T* node, LIST_NODE_T* obj )
{
    LIST_NODE_T* iterator;
    iterator = node;
    do {
        if ( iterator == obj )
            return 1;
    } while ( iterate( node, &iterator ) != 0 );

    return 0;
}
   
#define _LIST_H 1
#endif

#ifndef LIST_H
#define LIST_H 1


#include <stdlib.h>

/*
 * module de listes chainées inspirées de celles du noyau linux
 */



typedef struct list_node_t      
{                                       
   struct list_node_t* next;                   
   struct list_node_t* previous;               
} LIST_NODE_T ;


typedef struct coincoin {
    LIST_NODE_T* node;
    unsigned short int is_empty;
} list_t;

/*
 * initializes the list handle
 */
#define list_init( list )  do {             \
    (list)->node = NULL;                    \
    (list)->is_empty = 1;                   \
    } while(0)


/*
 * initializes node as the first (and only)
 * member of a list.
 */
#define list_item_init( node )   do {       \
    (node)->previous = node;                \
    (node)->next = node;                    \
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
    } while (0)


/*
 * sets iterator to the address of the next element of list.
 * usage : 
 *
 * LIST_NODE_T* iterator = NULL;
 * list_t start;
 * while ( list_iterate( start, &iterator )) { process_item( iterator ); } 
 */
static inline short int list_iterate( list_t *list, LIST_NODE_T** iterator )
{
    if ( iterator == NULL )
        return -1;

    // initialization
    if ( *iterator == NULL ){
        *iterator = list->node;
        return 0;
    }
    // step
    if ( (*iterator)->next == list->node )
        return -1;
    *iterator = (*iterator)->next; 
    return 0;
}


/*
 * tests for membership of obj to the list
 * node belongs to. Returns 1 on success, 0 on failure.
 */
static inline unsigned short int list_member( list_t *list, LIST_NODE_T* obj )
{
    if ( list == NULL || list->is_empty )
        return 0;

    LIST_NODE_T* iterator = NULL;
    while ( list_iterate( list, &iterator ) != -1 ) {
        if ( iterator == obj )
            return 1;
    } 

    return 0;
}


/*
 * length of a list
 */
static inline int list_length( list_t* list )
{
    if ( list == NULL || list->is_empty )
        return 0;

    LIST_NODE_T* iterator = NULL;
    int answer = 0;
    while ( list_iterate( list, &iterator ) != -1 ) {
        answer++;
    } 
    return answer;
}

static inline void list_push( list_t* list, LIST_NODE_T* obj )
{
    if ( list == NULL )
        return;

    if ( list->is_empty ){
        list->node = obj;
        list_item_init( obj );
        list->is_empty = 0;
    } else {
        // insert before head
        list_add( list->node->previous, obj );
        list->node = obj;
    }
}

static inline void list_append( list_t *list, LIST_NODE_T* obj )
{
    if ( list == NULL )
        return;

    if ( list->is_empty ){
        list->node = obj;
        list_item_init( obj );
        list->is_empty = 0;
    } else {
        list_add( list->node->previous, obj );
    }
}



static inline LIST_NODE_T *list_pop( list_t* list )
{
    if ( list == NULL )
        return NULL;

    LIST_NODE_T *answer = list->node;

    if (answer == NULL)
        return answer;

    // case of a single-item list
    if ( list->node->next == list->node ){
        list->node = NULL;
        list->is_empty = 1;
    } else {
        list->node = list->node->next;
        list_remove( answer );
    }
    return answer;
}

   
#endif

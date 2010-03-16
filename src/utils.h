/*
 * thanks to the linux kernel :)
 */


/*
 * finds the offset of [MEMBER] in [TYPE] (which is a struct), by 
 * casting 0 in (TYPE*) and taking the address of the member.
 */

#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)


/*
 * finds a pointer to the struct of type [type], which member [member] has address 
 * [ptr]. Its basically allows to find a structure from one of its member, 
 * assuming you know its type.
 */
#define container_of(ptr, type, member) ({                                  \
                const typeof( ((type *)0)->member ) *__mptr = (ptr);        \
                (type *)( (char *)__mptr - offsetof(type,member) );})

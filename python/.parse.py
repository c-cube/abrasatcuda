#!/usr/bin/env python
"""parser for CNF problem, which then passes the formula to solve
to the dynamic library.
"""

# parser for the file
from sys import argv
from os import environ

import ansi # colored output

import ctypes
from ctypes import pointer, c_short, byref, POINTER, cast, addressof, sizeof, c_int


def ptr_array_access( array, Type, n ):
    "accesses the n-th item of array by address"
    return cast(addressof(array) + n*sizeof(Type), POINTER(Type))

#--------------------------------------------------------------------------------
# import some things from binary
assert environ["LD_LIBRARY_PATH"] == "."
abrasatcuda = ctypes.CDLL( "libabrasatcuda.so" )

# make an atom from an int
_make_atom = abrasatcuda.make_atom
_make_atom.restype = c_int
_make_atom.argtypes = [c_int]
def make_atom( i ):
    answer = _make_atom( c_int(i) )
    return answer


# make a formula from clauses
formula_build = abrasatcuda.formula_build
formula_build.restype = c_int

# solve the problem
solve = abrasatcuda.solve

# manipulate atoms
variable_name = abrasatcuda.variable_name
is_negated = abrasatcuda.is_negated


def make_clause( atoms ):
    "concatenates atoms into a clause"
    clause = (c_int * len(atoms))()
    for i, atom in enumerate(atoms):
        assert( type(atom) == c_int )
        clause[i] = atom
    return clause


def print_clause( clause_ptr, length ):
    "prints the clause a pointer to which is given"
    answer = ansi.inGreen( "(" )
    l = []
    for i in range(length):
        atom = ptr_array_access( clause_ptr, c_int, i ).contents
        temp = ""
        if is_negated(atom):
            temp = ansi.inBlue( "~" )
        temp += str( variable_name(atom) )
        l.append( temp )
    answer += ansi.inGreen( " v " ).join(l)
    return answer + ansi.inGreen( ")" )

            

#------------------------------------------------------------------------------
def parse_lines( lines ):
    "parses lines into a list of clauses"
    
    clauses_list = []
    clauses_array = None
    clauses_length = None
        
    clause_num, var_num = 0, 0
    cur_clause = []
    for line in ( x for x in lines if len(x) > 0 and x[0] != 'c' ):
        tokens = line.split()
        print tokens
        if tokens[0] == 'p':
            assert tokens[1] == "cnf"
            var_num, clause_num = int(tokens[2]), int(tokens[3])
            print "problem has %d clauses and %d var" % (clause_num, var_num )
            # creates an array of [clause_num] pointers
            clauses_length = ((c_int) * clause_num)()
            continue
        else:
            for t in tokens:
                atom = int(t)
                # if end of clause
                if atom == 0:
                    clauses_length[len(clauses_list)] = len(cur_clause)
                    clauses_list.append( make_clause( cur_clause ) )
                    cur_clause = []
                else:
                    atom = make_atom(atom)
                    cur_clause.append( c_int(atom) )
    
    print "all lines read"
    assert len(clauses_list) == clause_num, "number of clause must match problem spec"
    
    clauses_array = ((POINTER(c_int)) * clause_num)()
    for i, clause in enumerate( clauses_list ):
        clauses_array[i] = ptr_array_access( clause, c_int, i )

    return (clauses_array, clauses_length)
        







#------------------------------------------------------------------------------
# main stuff

def print_help():
    "prints help at CLI"
    print """usage : parse file [options]
    This is intented to create structures for SAT-solving, and give it
    to the library to get solved.
    """

def main():
    # needs at least one arg
    if len(argv) == 1:
        print_help()
        return

    print "opens file", argv[1]
    fd_input = open( argv[1], 'r')

    print "parses file"
    lines = fd_input.xreadlines()

    (clauses, clauses_length) = parse_lines( lines )
    n = len(clauses)

    fd_input.close()

    print "has found  %d clauses" % n
    for index in range(n):
        print "length of %d-th clause : %d" % (index, clauses_length[index] )
        print print_clause( clauses[index], clauses_length[index] )

    print "clauses printed !"

    formula = (POINTER(c_int))()
    clauses_index = (POINTER(c_int))()
    
    formula_build( byref(formula), byref(clauses_index), clauses, clauses_length, n )
    
    print "formula built !"
    for i in range(n):
        abrasatcuda.clause_print( formula[i], clauses_index, n)

    print "tries to solve"
    solve( formula, clauses, n )


if __name__ == "__main__":
    main()


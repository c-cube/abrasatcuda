#!/usr/bin/env python
"""parser for CNF problem, which then passes the formula to solve
to the dynamic library.
"""

# parser for the file
from sys import argv
from os import environ

import ansi # colored output

import ctypes
from ctypes import pointer, c_short, byref, POINTER

#--------------------------------------------------------------------------------
# import some things from binary
assert environ["LD_LIBRARY_PATH"] == "."
abrasatcuda = ctypes.CDLL( "libabrasatcuda.so" )

# make an atom from an int
make_atom = abrasatcuda.make_atom
make_atom.restype = c_short


# make a formula from clauses
formula_build = abrasatcuda.formula_build
formula_build.restype = c_short

# solve the problem
solve = abrasatcuda.solve




def make_clause( atoms ):
    clause = (c_short * len(atoms))()
    for i, atom in enumerate(atoms):
        clause[i] = atom
    return clause


def print_clause( clause ):
    answer = ansi.inGreen( "(" )
    l = []
    for i in clause:
        temp = ""
        if (i & 0x4000):
            temp = ansi.inBlue( "~" )
        temp += (i & 0x3FFF )
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
            clauses_array = ((POINTER(c_short)) * clause_num)()
            clauses_length = ((c_short) * clause_num)()
            continue
        else:
            for t in tokens:
                atom = int(t)
                # if end of clause
                if atom == 0:
                    clauses_list.append( make_clause( cur_clause ) )
                    clauses_length[len(clauses_list)-1] = len(cur_clause)
                    cur_clause = []
                else:
                    cur_clause.append( make_atom(atom) )
    
    print "all lines read"
    assert len(clauses_list) == clause_num, "number of clause must match problem spec"
    for i in range(len(clauses_array)):
        clauses_array[i] = pointer(clauses_list[i])

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

    print "has found  %d clauses" % n
    for index in range(n):
        print_clause( clauses[index].content )

    formula = pointer(c_short(0))
    formula_build( byref(formula), clauses, clauses, clauses_length, n )

    print "tries to solve"
    solve( formula, clauses, n )


if __name__ == "__main__":
    main()


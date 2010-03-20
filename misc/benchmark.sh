#!/bin/sh

# script used to try all instances of the program on a bench of 
# test files, and give the results back.

# It does *not* check if the answers given by programs are good or not.

if [ $# -le 1 ]; then
    echo "usage : ./misc/benchmark.sh program_1 program_2 ... program_n
    To be executed in the root of the project, with tests files in ./tests/"
    exit -1
fi


# file to hold results
OUTPUT=benchmark

# loops on programs
for testfile in ./tests/aim-50-1_6-yes1-4.cnf ./tests/dubois2*.cnf ./tests/par8-1-c.cnf ./tests/quinn.cnf ; do
    for i in $@; do
        echo "tests $i with file $testfile" ;
        { time "$i" "$testfile"; } >> ${OUTPUT} 2>&1 ;
    done ;
done








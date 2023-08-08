#!/bin/bash
name="sdllib test"
files="test.c"
libs="-l:matrix.a -l:sdllib.a -lm"
bash compile.sh "./"
gcc "$files" -o "$name" -L./libs $libs
./"$name"
rm ./"$name"
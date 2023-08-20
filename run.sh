#!/bin/bash
set -e
name="sdllib"
files="test_sdllib.c"
libs="-l:matrix.a -l:sdllib.a -lm"
bash compile.sh "./"
gcc "$files" -o "$name" -L./libs $libs
./"$name"
rm ./"$name"
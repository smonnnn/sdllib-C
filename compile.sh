#!/bin/bash
set -e
name="sdllib"
output_location=$(realpath "$1")
flags=""
files="sdllib.c"

declare -A dependencies
dependencies["matrix"]="https://github.com/smonnnn/Matrix.git"

if ! [ -d "$output_location"/libs ]; then
	mkdir "$output_location"/libs
fi

for d in ${!dependencies[@]}; do
	if [ -d "$output_location"/libs/"$d"/.git ]; then
		cd ./libs/"$d"
		git pull origin main
	else
		git clone ${dependencies["$d"]} "$output_location"/libs/"$d"
		cd ./libs/"$d"
	fi
	bash "$output_location"/libs/"$d"/compile.sh "$output_location"
	cp -r "$output_location"/libs/**/*.h "$output_location"/libs/
	cd ../../
done

gcc $flags -c "$files" -o "$output_location"/libs/"$name".o
ar rcs "$output_location"/libs/"$name".a "$output_location"/libs/"$name".o
rm "$output_location"/libs/*.o
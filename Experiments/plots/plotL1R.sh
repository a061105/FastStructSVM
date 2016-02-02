#!/bin/bash

argList="'$1'"
fileName="$1"

shift

for arg in $*
do
	argList="$argList,'$arg'"
	fileName=${fileName}_$(./pathToFile $arg)
done

echo $argList
echo $fileName

matlab  -r "plotL1R($argList)" > log
mv $fileName.eps ~/public_html/figures/

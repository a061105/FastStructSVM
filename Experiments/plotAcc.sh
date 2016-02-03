#!/bin/bash

argList="'$1'"
fileName="$1"

shift

prog=$1

shift

for arg in $*
do
	argList="$argList,'$arg'"
	fileName=${fileName}_$(./pathToFile $arg)
done

echo $argList
echo $fileName

matlab -nodisplay -r "${prog}($argList)" > log
chmod 777 $fileName.pdf
mv $fileName.pdf /u/ianyen/public_html/figures/
#mv $fileName.eps ~/public_html/figures/

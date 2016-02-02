
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

matlab -nodisplay -r "plotObj($argList)" > log
mv $fileName.pdf /u/ianyen/public_html/figures/
#mv $fileName.eps ~/public_html/figures/

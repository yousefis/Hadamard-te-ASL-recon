#!/usr/bin/env bash

#!/bin/sh
SourceDIR='/media/sf_shared/Sophie_data/3-brain_nii/'
DestiDIR='/media/sf_shared/Sophie_data/3-brain_nii/'
for FILE in "$SourceDIR"*.nii
do
#    echo $FILE
    new_file=$(echo $FILE | cut -d '/' -f 6-)
    new_file1=$(echo $new_file | cut -d '.' -f 1)

    echo "${new_file1}"


mkdir "$SourceDIR""${new_file1}"
/usr/local/fsl/bin/fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o "$SourceDIR""${new_file1}" "$SourceDIR""${new_file1}"/"${new_file1}"

done


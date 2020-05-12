#!/usr/bin/env bash

#!/bin/sh
SourceDIR='/media/sf_shared/Sophie_data/2-resampled_nii/'
DestiDIR='/media/sf_shared/Sophie_data/3-brain_nii/'
for FILE in  "$SourceDIR"*.nii
do
#    echo $FILE
    new_file=$(echo $FILE | cut -d '/' -f 6-)
    new_file1=$(echo $new_file | cut -d '.' -f 1)

    echo "${new_file1}_brain"



/usr/local/fsl/bin/bet "${SourceDIR}""${new_file1}" "${DestiDIR}""${new_file1}_brain"  -f 0.5 -g 0
done


/usr/local/fsl/bin/fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o /media/sf_shared/Sophie_data/test/1_Acc_pp120111_T1W3D__PACS_SENSE_4_1_brain /media/sf_shared/Sophie_data/test/1_Acc_pp120111_T1W3D__PACS_SENSE_4_1_brain
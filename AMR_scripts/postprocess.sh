#!/bin/bash

sed -r 's/(@@ )|(@@ ?$)//g' $1 > $2

python postprocess_AMRs.py -f $2 -s ../s2s_amr_parser/data/test.txt $3

python reformat_single_amrs.py -f $2.restore.final -e .form

cd ~/AMR-gs-master/tools/fast_smatch
sh compute_smatch.sh ~/s2s_amr_parsing/s2s_amr_parser/output/test.amr.restore.final.form ../../data/AMR/amr_2.0/test.txt
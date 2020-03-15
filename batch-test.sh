#!/bin/bash
var=$(sbatch N1_comet-48.slurm | (awk '{print $4}'))
a=$(date +%s%N)
echo $var
state="PENDING"
while [ "$state" != "RUNNING" ]
do
sleep 1
state=$(sacct -j $var | awk 'FNR == 3 {print $6}')
done
b=$(date +%s%N)
diff=$((b-a))
printf "Queuing Time: %s.%s\n" "${diff:0: -9}" "${diff: -9:3}"

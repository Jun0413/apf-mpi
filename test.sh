a=$(date +%s%N)
#sbatch N1_comet-48.slurm
mpirun -np 24 ./apf -n 400 -i 2000 -x 2 -y 12 -k
b=$(date +%s%N)
diff=$((b-a))
printf "Total time passed(Execution Time + Queuing Time): %s.%s\n" "${diff:0: -9}" "${diff: -9:3}"

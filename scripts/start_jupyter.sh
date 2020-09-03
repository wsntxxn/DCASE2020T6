#!/bin/bash
#SBATCH --mem-per-cpu 8G
#SBATCH --time 1-0:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output logs/jupyter-notebook-%J.log

# get tunneling info
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
#cluster=$(hostname -f | awk -F"." '{print $2}')
cluster="cu01"

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# e.g. farnam:
# module load Python/2.7.13-foss-2016b 

# DON'T USE ADDRESS BELOW. 
# DO USE TOKEN BELOW
jupyter-notebook --no-browser --port=${port} --ip=${node} --NotebookApp.token='' --NotebookApp.password=''

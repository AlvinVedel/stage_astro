#!/bin/bash
#SBATCH --job-name=compute_metrics_v2         # nom du job
#SBATCH -C v100                # decommenter pour reserver uniquement des GPU V100 32 Go
#SBATCH -A dnz@v100
#####SBATCH --partition=gpu_p2l          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)   ==> V100 
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=15           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU V100)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=5:00:00              # temps maximum d'execution demande (HH:MM:SS)   # 20H
#SBATCH --output=../logs/compute_metrics_v2_%j.out      # nom du fichier de sortie
#SBATCH --error=../logs/compute_metrics_v2_%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --qos=qos_gpu-t3 

module purge

module load tensorflow-gpu/py3/2.4.1
 
set -x
 
python -u ../scripts/compute_metrics_v2_5.py

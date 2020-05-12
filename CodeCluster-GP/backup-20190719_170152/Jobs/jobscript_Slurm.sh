#!/bin/bash 
#SBATCH --job-name=GP_20190719_170152
#SBATCH --output=/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/synthesize_asl_loss/CodeCluster-GP/backup-20190719_170152/Jobs/output.txt
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6500
#SBATCH --partition=LKEBgpu
#SBATCH --gres=gpu:1 
#SBATCH --time=0 
#SBATCH --nodelist=res-hpc-lkeb03 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/lkeb-hpc/hsokootioskooyi/Program/cudnn7.4-for-cuda9.0/cuda/lib64/
source /exports/lkeb-hpc/syousefi/TF1LO/bin/activate
echo "on Hostname = $(hostname)"
echo "on GPU      = $CUDA_VISIBLE_DEVICES"
echo
echo "@ $(date)"
echo
python /exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/synthesize_asl_loss/CodeCluster-GP/backup-20190719_170152/run.py --where_to_run Cluster 

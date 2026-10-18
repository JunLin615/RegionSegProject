#!/bin/bash
#SBATCH --time=20:00:00               # 运行时间
#SBATCH --gres=gpu:a100:1             # 请求1块A100 GPU
#SBATCH --cpus-per-task=8             # 每个任务使用8个CPU
#SBATCH --ntasks=1                    # 任务数
#SBATCH --output=output.txt           # 输出文件
#SBATCH --error=error.txt             # 错误文件
#SBATCH --mem=64G                     # 请求内存

# 定义数据文件夹和模型存储文件夹路径（根据实际情况修改）
DATA_DIR="./dataset/yellow_260118"  # 数据根文件夹（包含images子文件夹）
MODEL_DIR="./dataset/yellow_260118/model_save"  # 模型保存路径

module load cuda/12.2
conda activate torch24
cd /ibex/user/$USER/data/RegionSegProject
# 打印分配的 GPU 列表
echo "Allocated GPUs: $SLURM_JOB_GPUS"
# 设置 CUDA_VISIBLE_DEVICES 环境变量，以便只使用分配的 A100 GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,gpu_name --format=csv,noheader | grep 'A100' | awk -F ',' '{print $1}' | paste -sd ",")
# 运行程序时传入路径参数
python DeepLabV3.py --data_dir "$DATA_DIR" --model_dir "$MODEL_DIR"
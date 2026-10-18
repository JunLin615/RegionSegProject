本系列脚本用于识别显色SERS的样品区域。


## boundary2mask.py
python boundary2mask.py --json "JSON路径" --img_dir "图片文件夹路径" --out_dir "输出路径"
由于使用cv2的文件读取，所以路径不要有中文


python boundary2mask.py --json "E:\stage\KAUST Gan\paper work\SERS Mu\RegionIdentification\yellow_260118\261118_yellow.json" --img_dir "E:\stage\KAUST Gan\paper work\SERS Mu\RegionIdentification\yellow_260118\images" --out_dir "E:\stage\KAUST Gan\paper work\SERS Mu\RegionIdentification\yellow_260118"
## DeepLabV3.py

对应run_trainC251202.sh
数据集结构：

运行文件夹
-DeepLabV3C251231.py
-run_trainC251231.sh
-dataset
--数据文件夹
---images
---masks
--model_save

sbatch run_trainC251202.sh

## 和服务器交互数据



cmd
cd E:\working\linux_send\RegionSegProject\temporary
scp -r 260119\* lij0y@glogin.ibex.kaust.edu.sa:/ibex/user/lij0y/data/RegionSegProject/
scp -r "yellow_260118" lij0y@glogin.ibex.kaust.edu.sa:/ibex/user/lij0y/data/RegionSegProject/dataset
训练：
服务器上
cd /ibex/user/lij0y/data/RegionSegProject/
conda activate torch24
sbatch run_trainC251231.sh
检查作业状态
squeue -u $USER

查看进度
服务器上
hostname查看具体机器
tensorboard --logdir=. --port=7007 --bind_all
本地
ssh -L 7009:login510-22:7009 lij0y@glogin.ibex.kaust.edu.sa

下载模型：

cmd
cd E:\stage\KAUST Gan\paper work\SERS Mu\RegionIdentification\R6G251231
scp -r lij0y@glogin.ibex.kaust.edu.sa:/ibex/user/lij0y/data/RegionSegProject/dataset/R6G251231/model_save/ model_save
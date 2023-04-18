#ls /usr/local | grep cuda # cuda cuda-11.0
#nvidia A100 (ampere architecture)
conda create -n mmtrack python=3.7 -y
conda activate mmtrack

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
pip install -U openmim
mim install mmengine
mim install "mmcv-full<1.7.0" # 1.6.2
mim install "mmdet<3.0.0"
python -c "import mmdet, mmcv; print(mmdet.__version__, mmcv.__version__)" # 2.28.2 1.6.2
git clone https://github.com/eplatero97/DeepFASORT
cd DeepFASORT
pip install -r requirements/build.txt
pip install -v -e .
pip install git+https://github.com/JonathonLuiten/TrackEval.git
python -c "import mmdet, mmcv, mmcls; print(mmdet.__version__, mmcv.__version__, mmcls.__version__)" # 2.28.2 1.6.2 0.25.0
pip install -U numpy # 1.21.6
python demo/demo_mot_vis.py configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py --input demo/demo.mp4 --output mot.mp4

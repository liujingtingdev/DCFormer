## Environment
The project is developed under the following environment:
- Python 3.8.10
- PyTorch 1.11.0
- CUDA 11.3
For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 

## Dataset
### Human3.6M
#### Preprocessing
1. We follow the previous state-of-the-art method [CA-PF](https://github.com/CHUNYUWANG/H36M-Toolbox/blob/master/README.md) to set up RGB images from the Human3.6M dataset. All RGB images should be put under `code_root/H36M-Toolbox/images/`. 
2. Download the [CA-PF](https://github.com/QitaoZhao/ContextAware-PoseFormer/blob/main/README.md)'s preprocessed Human3.6M labels [here](https://drive.google.com/drive/folders/1OYKWnu_5GPLRfceD3Psf4-JZkloBodKx) and unzip it to `DCFormer/data/`.
3. Download (COCO) pre-trained weights for HRNet-32/HRNet-48 from https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA and place it under `DCFormer/data/pretrained/`
4. Please access [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) to download the model and code. Run `generate_h36m_depth.py` to generate depth maps for Human3.6M.



### MPI-INF-3DHP
#### Preprocessing
1. Please download the original dataset from https://vcai.mpi-inf.mpg.de/3dhp-dataset/. We refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for generating `.npz` files (`data_train_3dhp.npz` and `data_test_3dhp.npz`). To simplify the process, we have provided the scripts `pre_load_3dhp.py` and `pre_load_3dhp_test.py`. 
- Run `pre_load_3dhp.py` to generate `data_train_3dhp.npz` and the pre-processed RGB images for the training set.
- Run `pre_load_3dhp_test.py` to generate `data_test_3dhp.npz` and the pre-processed RGB images for the test set.
After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `DCFormer_mpi/dataset/` directory.
2. Download (COCO) pre-trained weights for HRNet-32/HRNet-48 from (https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA)and place it under `DCFormer_mpi/dataset/pretrained/`
3. Please access [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) to download the model and code. Run `generate_3dhp_depth.py` to generate depth maps for MPI-INF-3DHP.
Note: We did not use Deformable Context Extraction for this dataset as our input is ground truth 2D keypoint.


## Training
After dataset preparation, you can train the model as follows:
### Human3.6M
You can train Human3.6M with the following command:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config <PATH-TO-CONFIG> --logdir ./logs
```
where config files are located at `experiments/human36m/`. 
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config experiments/human36m/human36m_single.yaml --logdir ./logs
```
### MPI-INF-3DHP
You can train MPI-INF-3DHP with the following command:
```
python run_3dhp.py -f 1 -b 160 --train 1 --lr 0.0007 -lrd 0.97
```


## Evaluation
| Dataset  | Depth model | Checkpoint|
|----------|---------------|-----------|
|Human3.6M|Depth-Anythingv2-l|[download](https://drive.google.com/file/d/1o6n5RDo2OYpjh5pascBcR5cR6PjL1Y0y/view?usp=drive_link)|
|Human3.6M|Depth-Anythingv2-b|[download](https://drive.google.com/file/d/1qG7QbigtKTCHxvd-mRaEM4anpPW2vp3m/view?usp=drive_link)|
|Human3.6M|Depth-Anythingv2-s|[download](https://drive.google.com/file/d/1vnRgZxkcPQ7qPwV3gJXpZ2CWXyUW1wpW/view?usp=drive_link)|
|MPI-INF-3DHP|Depth-Anythingv2-b|[download](https://drive.google.com/file/d/1Pkqc1j5xxSRsJYCqC1JE1GIMevuQQp_o/view?usp=drive_link)|




After downloading the weight from table above, you need place the pre-trained model weights (`best_epoch_{DEPTH_MODEL_NAME}.bin`) under `code_root/DCFormer/checkpoint/`, and run: evaluate Human3.6M models by:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config <PATH-TO-CONFIG> --logdir ./logs --eval
```

For MPI-INF-3DHP dataset, you can download the checkpoint with T = 81 and put in `code_root/DCFormer_mpi/checkpoint/` directory, then you can run:
```
python run_3dhp.py -f 1 -b 160 --train 0 --reload 1
```
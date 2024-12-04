# 3D-Model-Reconstruction from 2D images
# A Novel Hybrid Ensemble Approach For 3D Object <br /> Reconstruction from Multi-View Monocular RGB images for Robotic Simulations 
Ajith Balakrishnan, Dr. Sreeja S


implementation of paper -  Refine3DNet: Scaling Precision in 3D Object Reconstruction from Multi-View RGB Images using Attention (https://arxiv.org/abs/2412.00731)

## (1) Architecture
![Overview Image](https://github.com/Ajithbalakrishnan/3D-Model-Reconstruction/blob/master/imgs/arch_overview.png)
![Arch Image](https://github.com/Ajithbalakrishnan/3D-Model-Reconstruction/blob/master/imgs/solution_arch.png)
## (2) STSO-JTSO Algorithm
![Teaser Image](https://github.com/Ajithbalakrishnan/3D-Model-Reconstruction/blob/master/imgs/stsojtso.png)
## (3) Sample Results
![Teaser Image](https://github.com/Ajithbalakrishnan/3D-Object-Reconstruction-from-Multi-View-Monocular-RGB-images/blob/master/imgs/qualitativeoutpus.png)

## (4) Data
#### 3D-R2N2 Dataset
[https://github.com/chrischoy/3D-R2N2](https://github.com/chrischoy/3D-R2N2)
#### LSM Dataset
[https://github.com/akar43/lsm](https://github.com/akar43/lsm)

## (5) Released Model
#### Trained on 3D-R2N2 dataset
will release soon

## (6) Requirements
python  3.5

tensorflow 1.11.0 

numpy 1.13.3

scipy 0.19.0

matplotlib 2.0.2

skimage 0.13.0

## (7) Run
#### Training
python main_AttSets.py

#### Test Demo (Download released model first)
python demo_AttSets.py

## (8) Citation
If you use the paper or code for your research, please cite :
```
@article{Refine3DNet,
  title={Refine3DNet: Scaling Precision in 3D Object Reconstruction from Multi-View RGB Images using Attention},
  author={Ajith Balakrishnan, Dr Sreeja S, Dr Linu Shine},
  journal={ICVGIP},
  year={2024},
  doi={https://doi.org/10.1145/3702250.3702292}
}
```

## (9) Special Thanks To
https://github.com/Yang7879/AttSets
//
https://github.com/chrischoy/3D-R2N2
//
https://arxiv.org/pdf/1901.11153.pdf 

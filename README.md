# 3D-Model-Reconstruction - TF Implimentation of Attaets
# Attentional Aggregation of Deep Feature Sets <br /> for Multi-view 3D Reconstruction
Bo Yang, Sen Wang, Andrew Markham, Niki Trigoni. arXiv preprint arXiv:1808.00758, 2018.

## (1) Architecture
![Arch Image](https://github.com/Yang7879/AttSets/blob/master/attsets_arch.png)
## (2) Optimization
![Teaser Image](https://github.com/Yang7879/AttSets/blob/master/attsets_optim.png)
## (3) Sample Results
![Teaser Image](https://github.com/Yang7879/AttSets/blob/master/attsets_sample.png)

## (4) Data
#### 3D-R2N2 Dataset
[https://github.com/chrischoy/3D-R2N2](https://github.com/chrischoy/3D-R2N2)
#### LSM Dataset
[https://github.com/akar43/lsm](https://github.com/akar43/lsm)

## (5) Released Model
#### Trained on 3D-R2N2 dataset, 70M
[https://drive.google.com/open?id=1A1ihqMDfZLrjQeCFWEjgp-WYb810_om-](https://drive.google.com/open?id=1A1ihqMDfZLrjQeCFWEjgp-WYb810_om-)
## (6) Requirements
python 2.7 or 3.5

tensorflow 1.2 +

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
If you use the paper or code for your research, please cite:
```
@inProceedings{Yang18b,
  title={Attentional Aggregation of Deep Feature Sets for Multi-view 3D Reconstruction},
  author = {Bo Yang
  and Sen Wang
  and Andrew Markham
  and Niki Trigoni,
  booktitle={arXiv preprint arXiv:1808.00758},
  year={2018}
}
```

# Few-shot Class Incremental Learning with Subspace from Learned Weights (KCC 2022)
Project repository for KHU Capstone Design 1

---  

## Experimental results
|         Method         | Reuse Novel |     1     |     2     |     8     |  Average  |
|:----------------------:|:-----------:|:---------:|:---------:|:---------:|:---------:|
|      Subspace Reg.     |      X      | **62.27** |   47.30   |   18.98   |   34.14   |
|      Subspace Reg.     |      O      | **62.27** |  _47.45_  |   19.62   |  _34.25_  |
| Semantic Subspace Reg. |      X      |   57.33   |   44.00   |   19.14   |   32.69   |
| Semantic Subspace Reg. |      O      |   57.33   |   44.50   | **21.56** |   33.58   |
|     Linear Mapping     |      X      |  _59.87_  | **47.60** |   18.62   |   33.26   |
|     Linear Mapping     |      O      |  _59.87_  |   46.80   |  _21.12_  | **34.33** |

---

## Environment
- Python 3.6.9  
- Pytorch 1.9.0  
- CUDA 11.1  
- RTX 3090
---

## Dataset preparation  
Download `all.pickle` and `class_labels.txt` from [google drive](https://drive.google.com/drive/folders/1muZBsSYkZgoXxvMJDkp8vMDqpOen86J9?usp=sharing).  
Put these files on `data/miniImageNet/`.

---

## Train a model on the base classes  
To train a model on the base class, run `scripts/continual/slurm_run_backbone.sh`.
``` bash
bash scripts/continual/slurm_run_backbone.sh
```
The trained model will be saved on `dumped/backbones/continual/resnet18/${seed}/base20/`.

---

## Train a model on the novel classes
This repository provides three subspace regularization techinques including reuse of the learned weights.  
Please check `--reuse_novel` option to reuse the novel classifier weights.

---

### Subspace regularization
```bash
bash scripts/continual/slurm_subspace_reg.sh
```

---

### Semantic subspace regularization
```bash
bash scripts/continual/slurm_semantic_subspace_reg.sh
```

---

### Linear mapping
```bash
bash scripts/continual/slurm_linear_mapping.sh
```

---

## Acknowledgement
This repository is based on https://github.com/feyzaakyurek/subspace-reg.

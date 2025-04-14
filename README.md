# Towards Safer and Understandable Driver Intention Prediciton 🚗🛡

<p align="center">
  <img src="./figures/Teaser_Diagram_Ver2.0_page-0001.jpg" alt="VCBM Application">
</p>   

---

## Installation :wrench:

For running SlowFAST models;

Our python environement is identical to [SlowFAST](https://github.com/facebookresearch/SlowFast.git), we recommend following their installation instructions.

For running I3D and DINO models;

```shell
git clone https://github.com/Mukil07/DIP-Explainability.git
cd models
conda env create -f environment.yml

export PYTHONPATH="./:$PYTHONPATH"
```

Additionally have to install transformers library and Pytorch Video, 
```shell

git clone https://github.com/facebookresearch/pytorchvideo.git
pip install -e .
```
---

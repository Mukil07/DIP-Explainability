# TOWARDS SAFER AND UNDERSTANDABLE DRIVER INTENTION PREDICITON 

<p align="center">
  <img src="./figures/Teaser_Diagram_Ver2.0_page-0001.jpg" alt="VCBM Application">
</p>   

---

## Installation :wrench:
Our python environement is identical to [SlowFAST](https://github.com/facebookresearch/SlowFast.git), we recommend following their installation instructions:

```shell
conda create --name=vcbm python=3.10
conda activate vcbm

git clone https://github.com/Mukil07/DIP-Explainability.git
cd DIP-Explainability
pip install -r requirements.txt

export PYTHONPATH="./:$PYTHONPATH"
```

Additionally have to install transformers library and Pytorch Video, 
```shell

git clone https://github.com/facebookresearch/pytorchvideo.git
pip install -e .
```
---

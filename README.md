# Multi Task Multi-Scale Contrastive Knowledge Distillation for Efficient Medical Image Segmentation

**Thesis Title -** **"Advancing Medical Image Segmentation Through Multi-Task and Multi-Scale Contrastive Knowledge Distillation"**.

This is my master’s thesis, where I investigate the feasibility of knowledge transfer between neural networks for medical image segmentation tasks, specifically focusing on the transfer from a larger multi-task “Teacher” network to a smaller “Student” network. 

<p align="center">
  <img alt="Light" src="/assets/T1-S1.png" width="46%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Light" src="/assets/T1-S2.png" width="45%">
</p>

<p align="center">
  <img alt="Light" src="/assets/Results.png" width="74%">
&nbsp; &nbsp; &nbsp; &nbsp;
</p>

## MTMS-Contrastive Knowledge Distillation
The overall architecture of our multi-task multi-scale contrastive knowledge distillation framework for segmentation. 

![alt text](/assets/MTMS-KD.png?raw=true)

## Contrastive Learning
Representation of Contrastive Pairs. A beginner’s guide to Contrastive Learning can be found [here.](https://www.v7labs.com/blog/contrastive-learning-guide)

![alt text](/assets/CL.png?raw=true)

## Knowledge Distillation
Teacher-Student Framework for Knowledge Distillation. A beginner’s guide to Knowledge Distillation can be found [here.](https://www.v7labs.com/blog/knowledge-distillation-guide)

![alt text](/assets/KD.png?raw=true)

## Multi-Task Teacher Network
We trained two teacher models T1 and T2, one a multi-task pre-trained U-Net and a multi-task TransUNet, respectively.

![alt text](/assets/MT-Teacher.png?raw=true)

## Student Network
The student model, a simplified version of the teacher model, is significantly smaller in scale and is trained on only 50% of the data compared to the teacher model.

![alt text](/assets/Student.png?raw=true)

## Datasets Used
The CT spleen segmentation dataset from the medical image decathlon is used for all the experiments. Below are the links to the processed 2D images from the CT spleen dataset -
* [**Medical Image Decathlon**](https://medicaldecathlon.com/) 
* [**Processed Spleen Segmentation**](https://drive.google.com/drive/folders/1dOwuEVNXVCYg5LuZruJskyzcyi18XyIf?usp=sharing)

## Steps to Use the Framework
### Step 1 - Clone the repository to your desired location:

```bash
git clone https://github.com/RisabBiswas/MTMS-Med-Seg-KD
cd MTMS-Med-Seg-KD
```
### Step 2 - Process Data
There are two options - Either download the .NIFTI file and convert them to 2D slices using the [conversion script](https://github.com/RisabBiswas/MTMS-Med-Seg-KD/blob/main/Data%20Processing/convert_nifti_to_2D_resized.py) or, you can use the processed spleen dataset, which can be downloaded from the above link. 

The data is already split into training and testing datasets. 

**Input CT Volume of Spleen Dataset -**
<p align="center">
  <img src="/assets/GIF_1_IM_6_60_6_82.gif" width="230" /> 
<!--   <img src="/assets/GIF_1_IM_3_01_3_26.gif" width="220" /> -->
  <img src="/assets/GIF_1_IM_8_01_8_24_.gif" width="230" />
  <img src="/assets/GIF_1_IM_22_40_22_70.gif" width="230" />
</p>

<p align="center">
  <img src="/assets/GIF_1_M_6_60_6_82.gif" width="230" /> 
<!--   <img src="/assets/GIF_1_M_3_01_3_26.gif" width="220" /> -->
  <img src="/assets/GIF_1_M_8_01_8_24_.gif" width="230" />
  <img src="/assets/GIF_1_M_22_40_22_70.gif" width="230" />
</p>

**Processed 2D Slices -** 
<p align="center">
  <img src="/assets/DP_1.png" width="230" /> 
<!--   <img src="/assets/GIF_1_IM_3_01_3_26.gif" width="220" /> -->
  <img src="/assets/DP_2.png" width="228" />
</p>

Additionally, other binary segmentation datasets that can be explored are - 
* [**DRIVE (Digital Retinal Images for Vessel Extraction)**](https://paperswithcode.com/dataset/drive)
* [**RITE (Retinal Images vessel Tree Extraction)**](https://paperswithcode.com/dataset/rite)
* [**ISIC Dataset**](https://challenge.isic-archive.com/data/)
* [**Brain Tumor Dataset**](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
* [**2D Brain Tumor Segmentation Dataset**](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation/code)
* Colorectal Polyp Segmentation Dataset -
  * [**KVASIR-SEG**](https://datasets.simula.no/downloads/kvasir-seg.zip)
  * [**CVC-ClinicDB**](https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=0)
  * [**BKAI-IGH NeoPolyp**](https://www.kaggle.com/competitions/bkai-igh-neopolyp/data)
  * [**CVC-300**](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579)






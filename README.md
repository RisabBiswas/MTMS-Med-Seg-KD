# Multi Task Multi-Scale Contrastive Knowledge Distillation for Efficient Medical Image Segmentation

**Thesis Title -** **[Advancing Medical Image Segmentation Through Multi-Task and Multi-Scale Contrastive Knowledge Distillation](https://arxiv.org/abs/2406.03173)**

This is my master’s thesis, where I investigate the feasibility of knowledge transfer between neural networks for medical image segmentation tasks, specifically focusing on the transfer from a larger multi-task “Teacher” network to a smaller “Student” network using a multi-scale contrastive learning approach. 

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

## Results 
Below are a few quantitative and qualitative results. KD(T1, S1) and KD(T1, S2) are the results obtained from our proposed method. More detailed results and ablation studies can be found in the thesis. 

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
<p align="center">
  <img alt="Light" src="/assets/MTMS-KD.png" width="74%">
</p>

## Contrastive Learning
Representation of Contrastive Pairs. A beginner’s guide to Contrastive Learning can be found [here.](https://www.v7labs.com/blog/contrastive-learning-guide)
<p align="center">
  <img alt="Light" src="/assets/CL.png" width="65%">
</p>
<!-- ![alt text](/assets/CL.png?raw=true) -->

## Knowledge Distillation
Teacher-Student Framework for Knowledge Distillation. A beginner’s guide to Knowledge Distillation can be found [here.](https://www.v7labs.com/blog/knowledge-distillation-guide)
<p align="center">
  <img alt="Light" src="/assets/KD.png" width="65%">
</p>

## Multi-Task Teacher Network
We trained two teacher models T1 and T2, one a multi-task pre-trained U-Net and a multi-task TransUNet, respectively.
<p align="center">
  <img alt="Light" src="/assets/MT-Teacher.png" width="65%">
</p>

## Student Network
The student model, a simplified version of the teacher model, is significantly smaller in scale and is trained on only 50% of the data compared to the teacher model.
<p align="center">
  <img alt="Light" src="/assets/Student.png" width="65%">
</p>

## Datasets Used
The CT spleen segmentation dataset from the medical image decathlon is used for all the experiments. Below are the links to the processed 2D images from the CT spleen dataset -
* [**Medical Image Decathlon**](https://medicaldecathlon.com/) 
* [**Processed Spleen Segmentation**](https://drive.google.com/drive/folders/1dOwuEVNXVCYg5LuZruJskyzcyi18XyIf?usp=sharing)

## Additional Datasets
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
 
Other multi-class segmentation datasets that can be explored are - 
* [**Synapse Multi-Organ CT Dataset**](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
* [**ACDC Dataset**](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
* [**AMOS Multi-Modality Abdominal Multi-Organ Segmentation Challenge**](https://amos22.grand-challenge.org/)
* [**BraTS 2022**](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

## Steps to Use the Framework
### Step 1 - Clone the repository to your desired location:

```bash
git clone https://github.com/RisabBiswas/MTMS-Med-Seg-KD
cd MTMS-Med-Seg-KD
```
### Step 2 - Process Data
There are two options - Either download the .NIFTI file and convert them to 2D slices using the [conversion script](https://github.com/RisabBiswas/MTMS-Med-Seg-KD/blob/main/Data%20Processing/convert_nifti_to_2D_resized.py) or, you can use the processed spleen dataset, which can be downloaded from the above link. 

The data is already split into training and testing datasets. 

**> Input CT Volume of Spleen Dataset -**
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

**> Processed 2D Slices -** 
<p align="center">
  <img src="/assets/Processed Slices.png"  width="69%" /> 
</p>

### Step 3 - Train the Teacher Network
Training the multi-task teacher network (T1 or T2) is straightforward. Now that you have already created data folders, to train the T1 model, follow the below commands. 
```bash
cd Multi-Task Teacher Network (T1)
```
or,
```bash
cd Multi-Task Teacher Network (T2)
```
Run the training script -
```bash
python train.py
```
You can experiment with different weight values for the reconstruction loss. Additionally, for all the experiments I have used DiceBCE loss as the choice of loss function. You can try other loss functions as well such as Dice Loss. 

The pre-trained weights can also be downloaded from below -
* T1 - Will be uploaded soon!
* T2 - Will be uploaded soon!

### Step 4 - Inference on the Teacher Network
Once the teacher network is trained, to run inference, follow the below command -
```bash
python inference.py
```
also, you can look at the metrics by running the following - 
```bash
python metrics.py
```

### Step 4 - Train the Student Network (S1 or S2) W/o Knowledge Distillation
Before performing knowledge distillation and analysing its effect on the student model, we would like to train the student model and see its performance w/o any knowledge transfer from the teacher network. 
```bash
cd Student Network (S1)
```
Run the training script -
```bash
python train.py
```
Run the inference script -
```bash
python inference.py
```
Also, you can look at the metrics by running the following - 
```bash
python metrics.py
```
The pre-trained weights can also be downloaded from below -
* S1 - Will be uploaded soon!
* S2 - Will be uploaded soon!

### Step 5 - Train the Student Network (S1 or S2) With Knowledge Distillation
The steps to train the student model with contrastive knowledge distillation are similar and straightforward - 
```bash
cd KD_Student Network (T1-S1)
```
Run the training script -
```bash
python train_Student.py
```
Run the inference script -
```bash
python inference.py
```
Also, you can look at the metrics by running the following - 
```bash
python metrics.py
```
The knowledge distillation is performed at various scales, which can be customised in the training code. 

## Further Exploration
Currently, the architecture has only been tested on binary segmentation tasks and there is still room for further exploration such as - 
* Experiment on multi-class segmentation task.
* Try other contrastive loss.

## Acknowledgement
I extend my heartfelt gratitude to my guru 🙏🏻 [Dr. Chaitanya Kaul](https://www.linkedin.com/in/chaitanya-kaul-859330b6/) for his visionary guidance and unwavering support throughout my project. His mentorship has significantly shaped me as a researcher and a better individual. I am profoundly grateful for his invaluable contributions to my professional and personal growth. 

## Authors
- [Risab Biswas](https://www.linkedin.com/in/risab-biswas/)
- [Dr. Chaitanya Kaul](https://www.linkedin.com/in/chaitanya-kaul-859330b6/) 


## Read the Thesis 
You can find it [here](https://arxiv.org/abs/2406.03173) if you want to read the thesis. And if you like the project, we would appreciate a citation to the original work:
```
@misc{biswas2024multitask,
      title={Multi-Task Multi-Scale Contrastive Knowledge Distillation for Efficient Medical Image Segmentation}, 
      author={Risab Biswas},
      year={2024},
      eprint={2406.03173},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Contact 
If you have any questions, please feel free to reach out to <a href="mailto:risabbiswas19@gmail.com" target="_blank">Risab Biswas</a>.

## Conclusion
I appreciate your interest in my research. The code should not have any bugs, but if there are any, I am are sorry about that. Do let us know in the issues section, and we will fix it ASAP! Cheers! 




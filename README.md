# Multi Task Multi-Scale Contrastive Knowledge Distillation for Medical Image Segmentation

Advancing Medical Image Segmentation Through Multi-Task and Multi-Scale Contrastive Knowledge Distillation.

<p align="center">
  <img alt="Light" src="/assets/T1-S1.png" width="42%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="/assets/T1-S2.png" width="41%">
</p>

## Description
This master’s thesis investigates the feasibility of knowledge transfer between neural networks for medical image segmentation tasks, specifically focusing on the transfer from a larger multi-task “Teacher” network to a smaller “Student” network. 

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



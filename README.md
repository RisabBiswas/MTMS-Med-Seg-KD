# Multi Task Multi-Scale Contrastive Knowledge Distillation for Medical Image Segmentation

Advancing Medical Image Segmentation Through Multi-Task and Multi-Scale Contrastive Knowledge Distillation.

## Description
This master’s thesis investigates the feasibility of knowledge transfer between neural networks for medical image segmentation tasks, specifically focusing on the transfer from a larger multi-task “Teacher” network to a smaller “Student” network. 

## Knowledge Distillation
Teacher-Student Framework for Knowledge Distillation
![alt text](/assets/KD.png?raw=true)

## Multi-Task Teacher Network
We trained two teacher models T1 and T2, one a multi-task pre-trained U-Net and a multi-task TransUNet, respectively.
![alt text](/assets/MT-Teacher.png?raw=true)

## Student Network
The student model, a simplified version of the teacher model, is significantly smaller in scale and is trained on only 50% of the data compared to the teacher model.
![alt text](/assets/Student.png?raw=true)

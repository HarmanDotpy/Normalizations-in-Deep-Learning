# Normalizations in Deep Learning

We do experimental work to report the impact of different normalization schemes like batch norm, instance norm, group norm, batch-instance norm, layer norm in deep CNN models (using CIFAR-10).

The detailed report can be found [here](https://github.com/sm354/COL870-Assignment-1/blob/main/Report.pdf). 

## ResNet and various Normalization Schemes

We use ResNet model [[He et al., 2016](https://arxiv.org/abs/1512.03385)] to solve image classification task. Normalization techniques (applied just before activation function) - Batch Norm, Instance Norm, Batch-Instance Norm, Layer Norm, and Group Norm are compared. All are coded from scratch in pytorch ([scripts](https://github.com/sm354/COL870-Assignment-1/tree/main/ResNet%20and%20Normalizations)). In addition to self-written modules for normalization layers, batch norm of pytorch (torch_bn) and no normalization (nn) variants of ResNet are also compared. 



![img](https://lh3.googleusercontent.com/ojhv6r8p3G8tnAMOtiII-4heIE2UL57OIfJLVYyw6Q5LVNmuugUrJmY1MoNCVmAJVRWMKkGe2dkUNdKuldXqJginRdrdPg0pHVRKd_dI8Y1ebYr_6_dmOC6wV1MK5q80IskOG6PN)

![img](https://lh3.googleusercontent.com/QpXYRudfXJX4cSszgMIgJZv9pxGIiEP1dYW6K9d6Lc9gHHHeozYlA-Q570jV3yXlhXFlC28xQfuP-gu1zwVMKJCX9oY7KtOAfjV-E8-7wgc7evpqPp8Az7XNxfYW8Ho13Uf_rH1B)

We also compare the feature evolution throughout learning:-

No Normalization            |  Batch-Instance Normalization normalization
:-------------------------:|:-------------------------:
![img](https://lh5.googleusercontent.com/VSiC9ONIpHZCW1E91nf92QbwdA4XV4PHDLNCWy4U74JEGjlg6CbQYA3GT-ZZiOzFS8aIuU91NjO288aGqG9Ca4VoU2ibsgjTqCvIL6K3z9TRHGHsRkssa752mp-hUuH7zwboN27u) |  ![img](https://lh6.googleusercontent.com/sIpydqM5F_1owAH7c_d_vC_W5BmlR0mzW8IBw93PC9bNwCELnojq6fJyy5XovA1fHvITqqVCQ1tVbjFiTCcJH5uhPF0wzD6V1Y8ekVUWxUVrleSi6zoM56yJcBreYRMz5Osm3JO0)
















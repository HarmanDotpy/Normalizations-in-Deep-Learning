# Normalizations in Deep Learning

We do experimental work to report the impact of different normalization schemes like batch norm, instance norm, group norm, batch-instance norm, layer norm in deep CNN models (using CIFAR-10).

The detailed report can be found [here](https://github.com/HarmanDotpy/Normalizations-in-Deep-Learning/blob/main/Normalizations_report.pdf). 

## For reproducing the experiments
### Training
```bash
python train_cifar.py --normalization <norm_type> --data_dir ./dataset/cifar-10-batches-py --output_file ./saving_results/1.1.pth --n 2 --num_epochs 2
```
here ***norm_type*** is one of torch_bn, bn, in, bin, ln, gn, nn which represent different normalization schemes. torch_bn means inbuilt batch norm of pytorch, bn, in, bin, ln, gn, nn mean batch norm, instance norm, batch-instance norm, layer norm, group norm, and no normalization respectively.

- ***n*** sets the number of layers. Total layers of the model will be 6n+2.
- ***num_epochs*** sets the number of epochs for training.
- ***output_file*** is the path to save the trained model.
- ***data_dir*** is for giving the directory where data will be downloaded

### Testing
```bash
python test_cifar.py --model_file ./pretrained_models/part_1.1.pth --normalization <norm_type> --n 2 --test_data_file ./sample_test_data/cifar_test.csv --output_file ./saving_results/1.1_test_out.csv
```
- ***model_file*** is the path to path to the saved model which we get from training.
- ***test_data_file*** is for giving the test images.
- ***output_file*** is the file (csv) in which the predictions are written to.
rest all as explained in training



## ResNet and various Normalization Schemes

We use ResNet model [[He et al., 2016](https://arxiv.org/abs/1512.03385)] to solve image classification task. Normalization techniques (applied just before activation function) - Batch Norm, Instance Norm, Batch-Instance Norm, Layer Norm, and Group Norm are compared. 
All are coded from scratch in pytorch. In addition to self-written modules for normalization layers, batch norm of pytorch (torch_bn) and no normalization (nn) variants of ResNet are also compared. 



![img](https://lh3.googleusercontent.com/ojhv6r8p3G8tnAMOtiII-4heIE2UL57OIfJLVYyw6Q5LVNmuugUrJmY1MoNCVmAJVRWMKkGe2dkUNdKuldXqJginRdrdPg0pHVRKd_dI8Y1ebYr_6_dmOC6wV1MK5q80IskOG6PN)

![img](https://lh3.googleusercontent.com/QpXYRudfXJX4cSszgMIgJZv9pxGIiEP1dYW6K9d6Lc9gHHHeozYlA-Q570jV3yXlhXFlC28xQfuP-gu1zwVMKJCX9oY7KtOAfjV-E8-7wgc7evpqPp8Az7XNxfYW8Ho13Uf_rH1B)

We also compare the feature evolution throughout learning:-

No Normalization            |  Batch-Instance Normalization normalization
:-------------------------:|:-------------------------:
![img](https://lh5.googleusercontent.com/VSiC9ONIpHZCW1E91nf92QbwdA4XV4PHDLNCWy4U74JEGjlg6CbQYA3GT-ZZiOzFS8aIuU91NjO288aGqG9Ca4VoU2ibsgjTqCvIL6K3z9TRHGHsRkssa752mp-hUuH7zwboN27u) |  ![img](https://lh6.googleusercontent.com/sIpydqM5F_1owAH7c_d_vC_W5BmlR0mzW8IBw93PC9bNwCELnojq6fJyy5XovA1fHvITqqVCQ1tVbjFiTCcJH5uhPF0wzD6V1Y8ekVUWxUVrleSi6zoM56yJcBreYRMz5Osm3JO0)

## Some Results

<img width="800" alt="Screenshot 2021-05-27 at 11 28 30 PM" src="https://user-images.githubusercontent.com/50492433/119874604-6ac5e400-bf43-11eb-925e-d5eca42cd2d6.png">


### References
[1]: Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer Normalization.
2016. URL http://arxiv.org/abs/1607.06450.

[2]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and
Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016,
pages 770???778. IEEE Computer Society, 2016. doi: 10.1109/CVPR.2016.90.
URL https://doi.org/10.1109/CVPR.2016.90.

[3]: Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep
network training by reducing internal covariate shift. 32nd International Conference on Machine Learning, ICML 2015, 1:448???456, 2015.
Hyeonseob Nam and Hyo-Eun Kim. Batch-instance normalization for adaptively style-invariant neural networks. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicol`o Cesa-Bianchi, and Roman Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr??eal, Canada, pages 2563???
2572, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/
018b59ce1fd616d874afad0f44ba338d-Abstract.html.

[4]: Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for
large-scale image recognition. In Yoshua Bengio and Yann LeCun, editors,
3rd International Conference on Learning Representations, ICLR 2015, San
Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL
http://arxiv.org/abs/1409.1556.

[5]: Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance Normalization: The Missing Ingredient for Fast Stylization. (2016), 2016. URL
http://arxiv.org/abs/1607.08022.

[6]: Yuxin Wu and Kaiming He. Group Normalization. International Journal of
Computer Vision, 128(3):742???755, 2020.


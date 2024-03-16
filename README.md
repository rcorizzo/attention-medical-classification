# Attention-Based Medical Knowledge Injection in Deep Image Classification Models
This repository contains the implementation of experiments about Attention-Based Medical Knowledge Injection in Deep Image Classification. The appendix of this paper can be accessed through [View PDF](docs/Appendix_Medical_Attention.pdf)

## Data preparation
Download and extract the NIH chest x-ray data from https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345. All compressed packages of images can be downloaded in batch through the .py script contained in the "images" folder. Structure after downloading and extracting all files of images for NIH data:
```
/data/NIH/
  images_01/
    00000001_000.png
    00000001_001.png
  images_02/
    00001336_000.png
    00001337_000.png
  ...
```
Data structure required for model training and evaluation (after running data_prepare.py in data folder):
```
/data/class1_class2/
  train/
    class1/
      img1.png
    class2/
      img2.png
  val/
    class1/
      img3.png
    class/2
      img4.png
  test/
    class1/
      img5.png
    class2/
      img6.png
```
Masks are predicted by pretrained segmentation U-Net model with Ottawa images (undisclosed)
```
/data/Ottawa_masks_512/
  img1.png
  img2.png
  img3.png
```

## Training and evaluation
To run the classification task with cross entropy loss:
```
python model_loss_ce.py 
--path: Path of data, ../data/class-name/ 
--nclass: Number of classes, 2 or 3 
--gpu: Specify which gpu to use, not required, default is 0
```
To run the classification task with the proposed loss:
```
python model_loss_attent.py 
--path: Path of data, ../data/class-name/  
--nclass Number of classes, 2 or 3  
--task: Specify what classes (e: effusion, p: pneumothorax, c: cardiomegaly, nf: no finding) are included, ne/np/nc/nep 
--lamda: lambda, default is 0.5 
--thresh: threshold, default is 0.3 
--isAdaptive whether utilize the adaptive lambda, not required
--gpu: Specify which gpu to use, not required, default is 0
```
This should give a results.txt with performances on validation and testing sets respectively.

__Examples__

To train a PVT model with cross entropy loss, 3 classes:
```
python pvt_loss_ce.py --path ../data/nofind_effusion_pneumothorax/ --nclass 3
```
To train a VGG16 model with proposed loss, binary classes (no finding vs effusion), lambda = 0.25, threshold is 0.7:
```
python vgg_loss_attent.py --path ../data/nofind_effusion/ --nclass 2 --task ne lamda 0.25 --thresh 0.7
```
To train a ResNet50 model with proposed loss, binary classes (no finding vs pneumothorax), lambda is adaptive, threshold is 0.9:
```
python resnet_loss_attent.py --path ../data/nofind_pneumothorax/ --nclass 2 --task np  --thresh 0.9 --isAdaptive
```

### Note

In practical, the bounding box for Effusion in the paper is obtained based on the masks corresponding to the images in the training data. As Ottawa is a private dataset, information related to it is replaced by fixed values in this repository.

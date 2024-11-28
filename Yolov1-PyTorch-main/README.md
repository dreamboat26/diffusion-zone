YOLOv1 Implementation in Pytorch
========

This repository implements Yolo, specifically [Yolov1](https://arxiv.org/pdf/1506.02640) with training, inference and mAP evaluation in PyTorch.
The repo has code to train Yolov1 on voc dataset. Specifically I trained on trainval images of VOC 2007+2012 dataset.
For testing, I use VOC2007 test set.

## YOLOv1 Explanation and Implementation Video

<a href="https://youtu.be/TPD9AfY7AHo">
   <img alt="YOLOv1 Explanation and Implementation" src="https://github.com/user-attachments/assets/ccc0935b-b314-4841-9dc2-c594462b9062" width="300">
</a>

## Sample Output by training YOLOv1 on VOC 2007+2012 dataset 
Prediction(Top) | Class Grid Map(Bottom)
</br>
<img src="https://github.com/user-attachments/assets/b5201466-3025-426e-bf08-6213c0bc69ea" width="200">
<img src="https://github.com/user-attachments/assets/3f41051a-dcbc-462f-99e1-79feffedf759" width="200">
<img src="https://github.com/user-attachments/assets/6a27eb4f-635e-4402-a3b3-ae936729cb75" width="200">
<img src="https://github.com/user-attachments/assets/f772275f-73f7-4d1a-a0d6-d60374d4baba" width="200">
</br>

<img src="https://github.com/user-attachments/assets/049aef7d-501b-4474-b5e3-ae355d4e3829" width="200">
<img src="https://github.com/user-attachments/assets/3181f2a4-f2b5-4ea8-a666-1b5c6c600c23" width="200">
<img src="https://github.com/user-attachments/assets/2b2f4ba1-c0ee-4097-9e05-75b9533a9b46" width="200">
<img src="https://github.com/user-attachments/assets/f47affa5-9cc0-4184-902a-c977f720078c" width="200">
</br>

## Data preparation
For setting up the VOC 2007+2012 dataset:
* Create a data directory inside Yolov1-Pytorch
* Download VOC 2007 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the `VOC2007` directory inside `data` directory
* Download VOC 2007 test data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the  `VOC2007` directory and name it as `VOC2007-test` directory inside `data`
  * Download VOC 2012 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the  `VOC2012` directory inside `data`
    * Ensure to place all the directories inside the data folder of repo according to below structure
        ```
        Yolov1-Pytorch
            -> data
                -> VOC2007
                    -> JPEGImages
                    -> Annotations
                    -> ImageSets
                -> VOC2007-test
                    -> JPEGImages
                    -> Annotations
                -> VOC2012
                    -> JPEGImages
                    -> Annotations
                    -> ImageSets
            -> tools
                -> train.py
                -> infer.py
            -> config
                -> voc.yaml
            -> model
                -> yolov1.py
            -> loss
                -> yolov1_loss.py
            -> dataset
                -> voc.py
        ```

## For training on your own dataset

* Update the path for `train_im_sets`, `test_im_sets` in config
* Modify dataset file `dataset/voc.py` to load images and annotations accordingly specifically `load_images_and_anns` method
* Update the class list of your dataset in the dataset file.
* Dataset class should return the following:
    ```
  im_tensor(C x H x W) , 
  target{
        'yolo_targets' : S x S x (5B+C) (this is the target used by yolo loss)
        'bboxes': Number of Gts x 4 (this is in x1y1x2y2 format normalized from 0-1 and usedonly during evaluation)
        'labels': Number of Gts,
        }
  file_path(just used for debugging)
  ```g

## Differences from Yolov1 paper
Below are the differences from the paper
* Resnet-34 backbone used instead of Darknet
* Batchnorm layers in yolo specific 4 convolutional layers added
* Learning rate of 1E-2 ended up being too high in my experiments so I changed it to 1E-3(without warmup) and then decaying by factor of 0.5 after 50,75,100, 125 epochs. 
* Other hyper-parameters have directly been picked from paper and have not been tuned.
* With linear prediciton layers, I was only getting mAP of ~52% . With following changes that increased to ~58%
  * Sigmoid for box predictions. `use_sigmoid` parameter in config
  * 1x1 conv layers for yolo prediction layers instead of fc layers. `use_conv` parameter in config
  * To get the same prediction layers as paper, set `use_conv` and `use_sigmoid` as False in config.

## For modifications 
* In case you have GPU which does not support 64 batch size, you can use a smaller batch size like 16 and then have `acc_steps` in config set as 4.
* For uing a different backbone you would have to change the following:
  * Modify `features` in `yolo.py` to whatever is the backbone you desire.
  * In config change `backbone_channels` to whatever is the number of channels in feature map returned by new backbone.
  * Also change `conv_spatial_size` if required, to whatever is the final size of feature map just before prediction layers(so the fc layers or 1x1 conv layers). That means spatial size after backbone layers and 4 detection conv layers.

# Quickstart
* Create a new conda environment with python 3.10 then run below commands
* ```git clone https://github.com/explainingai-code/Yolov1-PyTorch.git```
* ```cd Yolov1-PyTorch```
* ```pip install -r requirements.txt```
* For training/inference use the below commands passing the desired configuration file as the config argument in case you want to play with it. 
* ```python -m tools.train``` for training Yolov1 on VOC dataset
* ```python -m tools.infer --evaluate False --infer_samples True``` for generating inference predictions
* ```python -m tools.infer --evaluate True --infer_samples False``` for evaluating on test dataset

## Configuration
* ```config/voc.yaml``` - Allows you to play with different components of Yolov1 on voc dataset  


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of `task_name` key in config will be created

During training of Yolov1 the following output will be saved 
* Latest Model checkpoint in ```task_name``` directory

During inference the following output will be saved
* Sample prediction outputs for images in ```task_name/samples/preds/*.jpeg``` 
* Sample grid class outputs for images in ```task_name/samples/grid_cls/*.jpeg``` 

## Citations
```
@article{DBLP:journals/corr/RedmonDGF15,
  author       = {Joseph Redmon and
                  Santosh Kumar Divvala and
                  Ross B. Girshick and
                  Ali Farhadi},
  title        = {You Only Look Once: Unified, Real-Time Object Detection},
  journal      = {CoRR},
  volume       = {abs/1506.02640},
  year         = {2015},
  url          = {http://arxiv.org/abs/1506.02640},
  eprinttype    = {arXiv},
  eprint       = {1506.02640},
  timestamp    = {Mon, 13 Aug 2018 16:48:08 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/RedmonDGF15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

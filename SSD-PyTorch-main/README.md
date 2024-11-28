SSD Implementation in Pytorch
========

This repository implements SSD, with training, inference and mAP evaluation in PyTorch.
Most of the code is just parts of pytorch ssd implementation and all I have done is gotten rid of abstractions and commented the code.

The repo provides code to train on voc dataset. Specifically I trained on trainval images of VOC 2007 dataset and for testing, I use VOC2007 test set.

## SSD Explanation and Implementation Video
<a href="https://youtu.be/c_nEue9itwg">
   <img alt="SSD Explanation and Implementation" src="https://github.com/user-attachments/assets/663754cf-93a7-4b7a-9a0f-ff094f73e90a" width="400">
</a>


## Result by training SSD on VOC 2007 dataset 
One should be able to get **71-72% mAP** by training on VOC 2007 trainval images(**68% reported in paper**).

Adding 2012 trainval we should be able to get **>77% mAP**

<img src="https://github.com/user-attachments/assets/e21e3344-a0b7-4c91-b06d-6b83f62df0b0" width="250">
<img src="https://github.com/user-attachments/assets/0d128c3e-d4ab-4335-a18f-77b7553f9634" width="250">
<img src="https://github.com/user-attachments/assets/1c588ab8-975e-4ece-bb2e-679d6b9fb18d" width="250">
</br>

Here's an evaluation result that I got after training 100 epochs.
```
Class Wise Average Precisions
AP for class aeroplane = 0.7552
AP for class bicycle = 0.8384
AP for class bird = 0.7025
AP for class boat = 0.6543
AP for class bottle = 0.3411
AP for class bus = 0.8355
AP for class car = 0.8611
AP for class cat = 0.8682
AP for class chair = 0.4798
AP for class cow = 0.7453
AP for class diningtable = 0.7092
AP for class dog = 0.8582
AP for class horse = 0.8506
AP for class motorbike = 0.8259
AP for class person = 0.7721
AP for class pottedplant = 0.3939
AP for class sheep = 0.7300
AP for class sofa = 0.7626
AP for class train = 0.8615
AP for class tvmonitor = 0.7260
Mean Average Precision : 0.7286
```


## Data preparation
For setting up the VOC 2007 dataset:
* Create a data directory inside SSD-Pytorch
* Download VOC 2007 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the `VOC2007` directory inside `data` directory
* Download VOC 2007 test data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the  `VOC2007` directory and name it as `VOC2007-test` directory inside `data`
* If you want to use 2012 trainval images as well, then download VOC 2012 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and copy the  `VOC2012` directory inside `data`
  * Ensure to place all the directories inside the data folder of repo according to below structure
      ```
      SSD-Pytorch
          -> data
              -> VOC2007
                  -> JPEGImages
                  -> Annotations
                  -> ImageSets
              -> VOC2007-test
                  -> JPEGImages
                  -> Annotations
              -> VOC2012 (if needed)
                  -> JPEGImages
                  -> Annotations
                  -> ImageSets
          -> tools
              -> train.py
              -> infer.py
          -> config
              -> voc.yaml
          -> model
              -> ssd.py 
          -> dataset
              -> voc.py
      ```

## For training on your own dataset

* Update the path for `train_im_sets`, `test_im_sets` in config
* If you want to train on 2007+2012 trainval then have `train_im_sets` as `['data/VOC2007', 'data/VOC2012'] `
* Modify dataset file `dataset/voc.py` to load images and annotations accordingly specifically `load_images_and_anns` method
* Update the class list of your dataset in the dataset file.
* Dataset class should return the following:
    ```
  im_tensor(C x H x W) , 
  target{
        'bboxes': Number of Gts x 4 (this is in x1y1x2y2 format normalized from 0-1)
        'labels': Number of Gts,
        'difficult': Number of Gts,
        }
  file_path
  ```


## For modifications 
* In case you have GPU which does not support large batch size, you can use a smaller batch size like 2 and then have `acc_steps` in config set as 4(to mimic 8 batch size training).
* For using a different backbone you would have to change the following:
  * Change the backbone, extra conv layers and creation of feature maps in initialization of SSD model
  * Ensure the `out_channels` is correctly set as the channels in all feature maps to be used for prediction [here](https://github.com/explainingai-code/SSD-PyTorch/blob/main/model/ssd.py#L316)
  * In the forward method call the backbone and extra conv layers and ensure `outputs` is correctly set as list of feature maps [here](https://github.com/explainingai-code/SSD-PyTorch/blob/main/model/ssd.py#L472)

# Quickstart
* Create a new conda environment with python 3.10 then run below commands
* ```git clone https://github.com/explainingai-code/SSD-PyTorch.git```
* ```cd SSD-PyTorch```
* ```pip install -r requirements.txt```
* For training/inference use the below commands passing the desired configuration file as the config argument in case you want to play with it. 
* ```python -m tools.train``` for training SSD on VOC dataset
* ```python -m tools.infer --evaluate False --infer_samples True``` for generating inference predictions
* ```python -m tools.infer --evaluate True --infer_samples False``` for evaluating on test dataset

## Configuration
* ```config/voc.yaml``` - Allows you to play with different components of SSD on voc dataset  


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of `task_name` key in config will be created

During training of SSD the following output will be saved 
* Latest Model checkpoint in ```task_name``` directory

During inference the following output will be saved
* Sample prediction outputs for images in ```task_name/samples```

## Citations
```
@article{DBLP:journals/corr/LiuAESR15,
  author       = {Wei Liu and
                  Dragomir Anguelov and
                  Dumitru Erhan and
                  Christian Szegedy and
                  Scott E. Reed and
                  Cheng{-}Yang Fu and
                  Alexander C. Berg},
  title        = {{SSD:} Single Shot MultiBox Detector},
  journal      = {CoRR},
  volume       = {abs/1512.02325},
  year         = {2015},
  url          = {http://arxiv.org/abs/1512.02325},
  eprinttype    = {arXiv},
  eprint       = {1512.02325},
  timestamp    = {Wed, 12 Feb 2020 08:32:49 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/LiuAESR15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

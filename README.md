This branch is the implementation of the paper "A Comprehensive Study for Center Loss".

* [Files](#files)
* [Trained_Model](#trained_model)
* [Contact](#contact)
### Implement_Details
The overall pipeline is the same as center loss (https://github.com/ydwen/caffe-face). 

In this paper, we use [CAISA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html), [VGG-Face2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
For the details of removing overlapping ID, please refer to https://github.com/happynear/FaceDatasets

Preprocessing
We use MTCNN (https://github.com/kpzhang93/MTCNN_face_detection_alignment) to detect five facial keypoints and use them to align the faces. For the alignment details, please see https://github.com/ydwen/caffe-face/blob/caffe-face/face_example/extractDeepFeature.m

Training
The training details can be found on according prototxt.
Note that the batch size we used is 512.

### Loss layer
Center Loss & Generalized Center Loss

        layer {
             name: "generalized_center_loss"
             type: "CenterLoss"
             bottom: "fc5"
             bottom: "label"
             top: "center_loss"
             top: "count"
             param {
                 lr_mult: 1
                 decay_mult: 1
             }
             center_loss_param {
                 num_output: 7994
                 margin:5 ##radius##
                 center_filler {
                   type: "xavier"
                 }
             }
             loss_weight: 0.01
             loss_weight: 0.0
         }
Advanced Center Loss & Generalized Center Loss

        layer {
             name: "advanced_center_loss"
             type: "SharedCenterLoss"
             bottom: "fc5"
             bottom: "label"
             top: "shared_center_loss"
             top: "count"
             param {
               name:"center"
               lr_mult: 1
               decay_mult: 1
             }
             param {
               lr_mult: 1
               decay_mult: 0
             }
             shared_center_loss_param {
               num_output: 7994
               margin:5 ##radius##
               gamma_shared: True ##share weights##
               center_filler {
                 type: "xavier"
               }
               gamma_filler{
                 type:"constant"
                 value:1
               }
             }
             loss_weight: 0.01
             loss_weight: 0.0
           }

### Files
- caffe
  * caffe.proto
  * center_loss_layer.hpp
  * center_loss_layer.cpp
  * center_loss_layer.cu
  * shared_center_loss_layer.hpp
  * shared_center_loss_layer.cpp
  * shared_center_loss_layer.cu  
- deploy_prototxt
  * resnet4.prototxt
  * resnet10.prototxt
  * resnet20.prototxt
  * resnet36.prototxt
  * resnet64.prototxt
- exp4_2
  * Parameter sharing
  * Loss Weight
  * Radius
  * Training_set
  * Depth
- exp4_3
  * softmax
  * softmax + contrastive
  * normface
  * coco
  * SphereFace
  * softmax + CL
  * softmax + ACL
  * softmax + ACL-γ
  * coco + ACL-γ
  * sphere + ACL-γ
- exp4_4 & exp4_5 & exp4_6
  * softmax
  * softmax + CL
  * softmax+ ACL-γ
  * softmax + CL (ρ=5)
  * softmax+ ACL-γ (ρ=5)
  * sphere+ ACL-γ (ρ=5)
- training_list
  * [link](https://drive.google.com/open?id=1RGchdWY-Yjz4kqB2kqj15jseK90NB3Rn)

### Trained_Model
- exp4_2
  * [link](https://drive.google.com/open?id=1w-Tx-N8jDEXsOi_akPPTN-jZcRC-21FP)
- exp4_3
  * [link](https://drive.google.com/open?id=1WMPbY_dwqs1jeyVu6wy1OqdykfTgkCWw)
- exp4_4 & exp4_5 & exp4_6
  * [link](https://drive.google.com/open?id=1yZdA-CGVgb07brunz8reJVE2jRaKzhIL)

### Contact 
- [Yandong Wen](http://ydwen.github.io/)
- [Kaipeng Zhang](http://kpzhang93.github.io/)


### License
Copyright (c) Yandong Wen, Kaipeng Zhang
All rights reserved.
MIT License

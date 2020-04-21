# Image recognition model collection
Collection of classic image recognition models, e.g.ResNet, Alexnet, VGG19, inception_V4 in Tensorflow. 


## The format of dataset
Firstly, Use unzip data.zip to unzipped it, and then images belonging to different categories are placed in different folders.


## Training networks
| models      | paper                                    | commands                     |  introduction       |
| ----------  | :-----------:                            | :-----------:                |  :-----------:      |
| Alexnet     | [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)| `python main.py --type_of_model Alexnet` | [Alexnet](https://blog.csdn.net/qq_41776781/article/details/94437671) 
|             |                                          |                              |                              
| VGG19       | [VGG19](https://arxiv.org/abs/1801.01401)| `python main.py --type_of_model VGG19` |  [VGG19](https://blog.csdn.net/qq_41776781/article/details/94452085) 
|             |                                          |                              |         
|Inception_V4 | [Inception_V4](https://arxiv.org/abs/1606.03498)  | `python main.py --type_of_model inception_V4`  |  [Inception_V4](https://blog.csdn.net/qq_41776781/article/details/94476538) 
|             |                                          |                              |                    
| ResNet      | [ResNet](https://arxiv.org/abs/1606.03498) | `python main.py --type_of_model ResNet --resnet_type 50`  |  [ResNet](https://blog.csdn.net/qq_41776781/article/details/94459299) 

## An explanation
It's easy to use the above recognition models with kears, but by reading papers and analyzing source code, which could help in improving programming ability. If you want to know basic something about these models, you can visit my [CSDN](https://blog.csdn.net/qq_41776781/category_9291732.html), 


# Detail about Parameters settings for fine-tuning Stable Diffusion:
 This SD fine-tuned model consists of four distinct datasets: MNIST, SVHN, CIFAR10, Imagenet for class pizza, and Imagenet for class teddy.

 ## 1. Custom Dataset Settings:
 To prepare a custom dataset for training Lora-stable diffusion, selecting and regularizing an appropriate set of images for training is mandatory.
 - Regulation images are optional. If you want to maintain flexibility inside the model, do not let it go outside the authentic class images and allow it to pick 
   random content from that prompt beyond the class. Here, we are using a real dataset, so regularization of images is required.
- parameters for selecting reg img and img for training data are selected according to the dataset requirements: Mnist, SVHN, and CIFAR 10 small and blurred images. Thus choosing maximum images for training. While imagenet images are good quality and have large pixels, few are needed. Still, it depends on data project requirements (optional); some can select between 50 -100 train images without regularization or less than from it and use a regularization set of images.
- After selecting img for training, regularization images are selected by using the following formula:

 Regularization-image = train images x repeat             
                     =  40 x 30
                      = 1200   ( For MNIST, SVHN, CIFAR10)
                  ** For imagenet dataset **
                      = 37  x 25
                      = 925
## 2. Captioning:  
    continue..........


  
 | **Dataset**       | **Image-type/T.classes** | \# train images per class | \# repeat | **class-label** | **triggerword (Instance Prompt)** | **Class (Class prompt)** | **Prompt**                      |
|-------------------|--------------------------|---------------------------|-----------|-----------------|----------------------------------|--------------------------|---------------------------------|
| **MNIST**         | Greyscale/10             | 40                        | 30        | 0               | Z0ero                            | Number0                  | A photo of Z0ero Number0        |
|                   |                          |                           |           | 1               | one1                             | Number1                  | A photo of one1 Number1         |
|                   |                          |                           |           | 2               | two2                             | Number2                  | A photo of two2 Number2         |
|                   |                          |                           |           | ...             | ...                              | ...                      | ...                             |
| **SVHN**          | Color/10                 | 40                        | 30        | 0               | HouseNo0                         | Hnumber0                 | A photo of HouseNo0 Hnumber0    |
|                   |                          |                           |           | 1               | HouseNo1                         | Hnumber1                 | A photo of HouseNo1 Hnumber1    |
|                   |                          |                           |           | 2               | HouseNo2                         | Hnumber2                 | A photo of HouseNo2 Hnumber2    |
|                   |                          |                           |           | ...             | ...                              | ...                      | ...                             |
| **CIFAR10**       | Color/10                 | 40                        | 30        | 0: Airplane     | A1plane0                         | cifar10_0                | A photo of A1plane0 cifar10_0   |
|                   |                          |                           |           | 1: Automobile   | car1                             | cifar10_1                | A photo of car1 cifar10_1       |
|                   |                          |                           |           | 2: bird         | bird2                            | cifar10_2                | A photo of bird2 cifar10_2      |
|                   |                          |                           |           | ...             | ...                              | ...                      | ...                             |
| **IN-1k (pizza)** | Color/1                  | 37                        | 25        | pizza           | 1pizza                           | pizza_slice              | A photo of 1pizza pizza_slice   |
| **IN-1k (teddy)** | Color/1                  | 37                        | 25        | teddy-bear      | 1toy_bear                        | teddy_bear               | A photo of 1toy_bear teddy_bear |

 

## 2. training parameters:


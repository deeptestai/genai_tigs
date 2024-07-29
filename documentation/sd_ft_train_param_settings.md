# Detail about Parameters settings for fine-tuning Stable Diffusion:
 This SD fine-tuned model consists of four distinct datasets: MNIST, SVHN, CIFAR10, Imagenet for class pizza, and Imagenet for class teddy.

 ## 1. Custom Dataset Settings:
 To prepare a custom dataset for training Lora-stable diffusion, selecting and regularizing an appropriate set of images for training is mandatory.
 - Regularization images are optional. To maintain flexibility within the model, it is advisable not to let the model generate content outside the authentic class images and to allow it to 
   explore random content within the scope of the prompt. However, since we are using a real dataset, incorporating regularization images is necessary.
- We selected a training set of 40 images for MNIST, SVHN, and CIFAR-10, supplemented with 1,200 regularization images. These images, being from real datasets, are relatively small and often 
  blurred, necessitating the choice of a larger image set. For ImageNet, where the images are already bright and of high quality, we selected 37 images along with 925 regularization images. 
  Without regularization, it is possible to select only 50-100 training images and still effectively train the model.
- After selecting the images for training, the regularization images are chosen using the following formula:

  Regularization-image = train images x repeat   
 
                      =  40 x 30
                      
                      = 1200   ( For MNIST, SVHN, CIFAR10)
                      
            ** For imagenet dataset **
            
                      = 37  x 25
                      
                      = 925
## 2. Captioning:  
     Fine-tuning the SD model is essential, as precise captioning is required to generate appropriate images based on prompts. We utilized BLIP captioning.

### Guidelines for Captioning Images:

- Select a single keyword, incorporating any digit to ensure uniqueness.
- Treat a word without a space as a one word word.
- Use the superclass to represent regularization images, ensuring that the class word represents the entire class with a unique name. In our case, one1, HouseNo0,A1plane0 and so on.
-We trained the models for MNIST, SVHN, and CIFAR10 with 10 classes each, except for Imagenet, which was trained for two classes.

#### Prompts for all classes are mentioned below:
**MNIST prompt for 10 classes:**  ["A photo of Z0ero Number0","A photo of one1 Number1","A photo of two2 Number2 ","A photo of three3 Number3","A photo of Four4 Number4","A photo of Five5 Number5","A photo of Six6 Number6","A photo of Seven7 Number7 ","A photo of Eight8 Number8","A photo of Nine9 Number9"]
**SVHN prompt for 10 classes:**[ "A photo of HouseNo0 Hnumber0","A photo of HouseNo1 Hnumber1","A photo of HouseNo2 Hnumber2 ","A photo of HouseNo3 Hnumber3","A photo of HouseNo4 Hnumber4","A photo of HouseNo5 Hnumber5","A photo of HouseNo6 Hnumber6","A photo of HouseNo7 Hnumber7 ","A photo of HouseNo8 Hnumber8","A photo of HouseNo9 Hnumber9"]
**CIFAR10 prompt for 10 classes:**[ "A photo of A1plane0 cifar10_0","A photo of car1 cifar10_1","A photo of bird2 cifar10_2","A photo of cat3 cifar10_3","A photo of deer4 cifar10_4","A photo of dog5 cifar10_5","A photo of frog6 cifar10_6","A photo of horse7 cifar10_7","A photo of ship8 cifar10_8","A photo of truck9 cifar10_9"]
**Imagenet prompt for 10 classes:**[  "A photo of 1pizza pizza_slice",  "A photo of 1teddy_toy teddy_bear"]




  
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


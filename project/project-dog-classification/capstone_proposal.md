# Machine Learning Engineer Nanodegree

## Capstone Proposal
Kurie Jumi
June 20, 2019

## Proposal

### Domain Background

Applications of image recognition and classification have started to become integrated into our lives from customs at the airport to surveillance for no-checkout convenience stores. Further research and development is taking place in all fields from entertainment, automobile, healthcare, security, retail, construction, beauty, agriculture and many more. We can expect to see more and more applications in the future disrupting the current industries and leaveraging technology to support human workers. 

In this project, a web application is created to classify dog breeds based on input image. When supplied with an image of a human, the application will identify the resembling dog breed. [Related academic research](https://www.pnas.org/content/pnas/115/25/E5716.full.pdf) applies more advanced CNN and deep learning models to automatically identify, count and describe wild animals. This research is an example of how classification of animals helps us understand the natural ecosystems without disrupting the habitat. Understanding CNN (and similar models) is the first step towards image classification and application of these models can help solve wide range of problems that can be easily integrated into other existing systems and services.


### Problem Statement

The goal of this project is to create an application that does the following according to input image
- if a dog is detected in the image, return the predicted breed.
- if a human is detected in the image, return the resembling dog breed.
- if neither is detected in the image, provide output that indicates an error.

To accomplish this, we will do the following
- Import Datasets & Preprocess images as necessary
- Detect Humans (from pre-trained models)
- Detect Dogs (from pre-trained models)
- Create a CNN to Classify Dog Breeds (from scratch with accuracy > 10%) 
- Create a CNN to Classify Dog Breeds (using transfer learning with accuracy > 60%)
- Write & Test the Algorithm
- Evaluate & Improve
- Deploy the model & create api to access model through web app


### Datasets and Inputs (approx. 2-3 paragraphs)

The data set selected by Udacity for this project include a Udacity unique [dog breed](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) dataset and [LFW](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) dataset created and maintained by researchers in University of Massachusetts, Amherst. 

The dog breed dataset consist of test, train and valid file each with the same set of 133 different dog breed files with images inside. The dog breed dataset will be used to test the pre-trained model to dect dogs as well as testing the CNN to classify dog breed.

```
.
└── dogImages
    ├── test
    |   ├── 001.Affenpinscher
    |   ├── 002.Afghan_hound
    |   ├── ...
    |   └── 133.Yorkshire_terrier   
    ├── train
    |   ├── 001.Affenpinscher
    |   ├── 002.Afghan_hound
    |   ├── ...
    |   └── 133.Yorkshire_terrier     
    └── valid
        ├── 001.Affenpinscher
        ├── 002.Afghan_hound
        ├── ...
        └── 133.Yorkshire_terrier  
```

The LFW human face dataset consist of files named firstname_lastname and an image file (or files) in them. The LFW dataset will be used for human face detection for the pretrained model. 

```
.
└── lfw
    ├── Aaron_Eckhart
    |   └── Aaron_Eckhart_0001.jpg
    ├── Aaron_Guiel
    |   └── Aaron_Guiel_0001.jpg
    ├── ...
    └── Zydrunas_Ilgauskas
        └── Zydrunas_Ilgauskas_0001.jpg
```

As the goal of the project is to predict dog breed, the data set we use must be specific for dogs and must contain dog breed names. The dog image file has test and train files for dog breed training and testing as well as valid file for dog detection. The LFW human face dataset has many files with names and images of that person. We don't need to classify people, so the number of image data is enough for human detection. Overall, we can say that the provided data is enough to do all of the above task stated in the problem statement. 


### Solution Statement

In the problem statement, we have already specified the steps we will take to complete this task. For each of the task, the algorithms/libraries will be introduced.

First, we must define a pre-trained model for human detection and dog detection. The project will use OpenCV's implementation of Haar feature-based cascade classifiers for detecting human faces and pre-trained detector [haarcascade_frontalface_alt.xml](https://github.com/opencv/opencv/tree/master/data/haarcascades). For the dog detection, the [VGG-16](https://pytorch.org/docs/master/torchvision/models.html) from torchvision trained on ImageNet(http://www.image-net.org/) will be used. As these models are pre-trained and dog and human are not so similar visually, we assume the accuracy to be at least 90% and we will confirm using the datasets we have shared above.

Next, we will need to come up with a base model to use to classify dog images as well as evaluation metric to know when to stop training further. For this project, we will attempt to create CNN model from scratch to classify dog breeds after a dog has been detected. The minimal test accuracy to pass this portion is set to 10%. The reason for the low accuracy setting is due to the fact that there are set of classes that are very similar visually (even human cannot distinguish). There are more than 100 classes, which makes the random guess accuracy to 1%, so an accuracy of more than 10% is very unlikely to be due to chance. 


### Benchmark Model (approximately 1-2 paragraphs)

The model that will be used as base for dog breed classification (transwer learning) is VGG-16. Some models that can be used as benchmark includes AlexNet, Inception-v3, ResNet-50 to name a few. There are more models that can be compared, and they are provided through pytorch webpage explaining [torchvision models](https://pytorch.org/docs/master/torchvision/models.html). 

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.


### Evaluation Metrics (approx. 1-2 paragraphs) 

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).


### Project Design (approx. 1 page)

1. Import Datasets & Preprocess images as necessary
2. Detect Humans (from pre-trained models)
3. Detect Dogs (from pre-trained models)
4. Create a CNN to Classify Dog Breeds (from scratch with accuracy > 10%) 
5. Create a CNN to Classify Dog Breeds (using transfer learning with accuracy > 60%)
6. Write & Test the Algorithm
7. Evaluate & Improve
8. Deploy the model & create api to access model through web app

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.


-----------

### Reference

- [Udacity project](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification)
- [Udacity dog images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
- [Udacity provided human lfw images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)
- [LFW Human Face Images](http://vis-www.cs.umass.edu/lfw/)
Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller.
Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.

- [Snapshot Serengeti Project](https://medium.com/coinmonks/automated-animal-identification-using-deep-learning-techniques-41039f2a994d)  
- [CNN pytorch 1](https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/) for object detection
- [CNN pytorch 2](https://www.kaggle.com/puneetgrover/training-your-own-cnn-using-pytorch) for OCR / text recognition 
- [CNN benchmark 1](https://github.com/alyato/CNN-models-comparison)
- [CNN benchmark 2](https://github.com/jcjohnson/cnn-benchmarks#alexnet)
- [pytorch models](https://pytorch.org/docs/master/torchvision/models.html)
“Torchvision.models.” Torchvision.models - PyTorch Master Documentation, pytorch.org/docs/master/torchvision/models.html.
- [training/validation plot](https://qiita.com/hiroyuki827/items/213146d551a6e2227810)
- [Dog breed methods](https://towardsdatascience.com/dog-breed-classification-hands-on-approach-b5e4f88c333e)
- [Kaggle dog breed](https://www.kaggle.com/c/dog-breed-identification/data)
- [CNN transfer learning in pytorch](https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce)
- [VGG-16 model]
Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman arXiv:1409.1556
- [top-1 score](https://stats.stackexchange.com/questions/156471/imagenet-what-is-top-1-and-top-5-error-rate)
First, you make a prediction using the CNN and obtain the predicted class multinomial distribution (∑𝑝𝑐𝑙𝑎𝑠𝑠=1).
Now, in the case of top-1 score, you check if the top class (the one having the highest probability) is the same as the target label.
In the case of top-5 score, you check if the target label is one of your top 5 predictions (the 5 ones with the highest probabilities).
In both cases, the top score is computed as the times a predicted label matched the target label, divided by the number of data-points evaluated.
Finally, when 5-CNNs are used, you first average their predictions and follow the same procedure for calculating the top-1 and top-5 scores.
- [Animal detection example] Norouzzadeh, Mohammad Sadegh, et al. “Automatically Identifying, Counting, and Describing Wild Animals in Camera-Trap Images with Deep Learning.” Proceedings of the National Academy of Sciences, vol. 115, no. 25, 2018, doi:10.1073/pnas.1719367115.
-[OpenCV](https://opencv.org/)
“OpenCV.” OpenCV, opencv.org/.
- [harcascade feature]
"haarcascade frontalface" OpenCV, https://github.com/opencv/opencv/tree/master/data/haarcascades.
Viola, P., and M. Jones. “Rapid Object Detection Using a Boosted Cascade of Simple Features.” Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001, 15 Apr. 2003, doi:10.1109/cvpr.2001.990517.
- [ImageNet](www.image-net.org/) 
“ImageNet.” ImageNet, www.image-net.org/.
- [Summary of techniques](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)
Krizhevsky, Alex, et al. “ImageNet Classification with Deep Convolutional Neural Networks - Semantic Scholar.” Undefined, 1 Jan. 1970, www.semanticscholar.org/paper/ImageNet-Classification-with-Deep-Convolutional-Krizhevsky-Sutskever/2315fc6c2c0c4abd2443e26a26e7bb86df8e24cc.
- [Inception-v3](https://arxiv.org/abs/1512.00567) 
Szegedy, Christian, et al. “Rethinking the Inception Architecture for Computer Vision.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, doi:10.1109/cvpr.2016.308.
- [ResNet]
He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, doi:10.1109/cvpr.2016.90.
- [Inception-v3 survey](https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c)
- [Udacity dog_app.ipynb](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/dog_app.ipynb)
Udacity. “Udacity/Deep-Learning-v2-Pytorch.” GitHub, 28 May 2019, github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification.
- [Azure Machine Learning](https://github.com/Azure/MachineLearningNotebooks)
- [MLA format online](https://owl.purdue.edu/owl/research_and_citation/mla_style/mla_formatting_and_style_guide/mla_general_format.html) Create citation for book, magazine, newspaper, website, journal, film, other for free!

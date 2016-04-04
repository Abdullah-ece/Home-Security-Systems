
## Features of a Smart Home Security System

- Facial Recognition
- Object type detection
- Sound Analysis
- Speech recognition

The first step to before being able to do Image or audio analyses would be to extract relevant frames from the video streams in real time. This is crucial to a smart interactive device and requires extensive down sizing of the data to run the models on the features identified most relevant. One also needs the device to identify and react to certain events (owner coming home, break-in etc) through a frame by frame comparative analysis.

Let us start with the event that there is a disturbance and the image frames and audio data is fed into the trained model to classify the event into pre-defined classes (simplest cast being intrusion vs non intrusion).

## Facial Recognition

Given a frame, let us start with the features that we would extract from it to first look for faces within the scenario and then if we find one, to match it with the available "registered" face repository. Given pictures of the home-owner/family, we will have an extensively pre trained model. We would also like that the features are constantly updated given that especially for children, the facial key-points can vary as they grow into adults. This could be accomplished by access to owners' Facebook profile or a manually updated picture repository.

Facial detection algorithms would work in three steps once the input become available:

- Human or not Human
- Registered person or not (binary)
- Which user is it?

### Human or not a Human

This can be accomplished easily through pre-labelled data from ImageNet or other similar online repositories. Discussed in section on Object Classification

### Registered person or not (binary)

**Features**

Extracting the eigenfaces: One of the major issues with image recognition is the sheer number of features that we need to deal with. A simple way to get around this can be Principal Component Analysis (PCA) which reformulates new features from the old to maximise the explained variance in the data.  These new compressed features derived from human faces are now referred to as "eigenfaces".

***Potential vulnerabilities***

One issue that plagues facial recognition algorithms and their accuracy are lighting conditions which can botch our pixel intensity maps, different angles and the natural facial contortions we present while displaying emotions. Google's Android phone OS, which has a face unlock mode, gets around this by getting the user to first train the phone by taking images of their face in different lighting conditions and angles. 

**Libraries**

Caffe though frequently used, pales when compared to Theano or simple libaries like FANN. Deep convoluted neural networks rule the game when it comes to face classification models. While face recognition continues to be highly inaccurate, a method called 'Deep Dense Face Detector' by Yahoo researchers hopes to improve that.

### Which user is it?

It would be computation expensive to train a model on K classes, hence one way is to do succesive binary classification comaring the derived human image to the repositary of the users with access to the house. Once can use the Siamese Network Training provided by Caffe to perform this classfication.

## Object type detection

### Training the Models

Given the large amount of data needed to train CNNs, the online pre-labelled datasets such as Pascal V0C or ImageNet are state of the art in training object classifiers. CloudCV (<http://cloudcv.org/objdetect/>) goes a step further and provides a combined, downsized dataset with relevant features extracted.

### Caffe: 
Performs image classification using convoluted neural nets and includes OpenCV, Blas libraries allowing efficient computation in C++. It also includes libraries for Python (pycaffe) and Matlab (matcaffe) as well as supports a variety of input formats for images. Another big plus are the pre trained models offered by Caffe's [Model Zoo][7].

## Sound Analysis

The field of 'Computational Auditory Scene Analysis' aims at accurately classifying acoustic sequences to detect sound events related to specific activities or 'audio based environment modelling'.

Acoustic Analysis can give us a large amount of information about the surroundings that video/image analysis alone cannot contextualize. Examples of this include screaming, gunshots, glass breaking, footsteps from the vicinity, etc. Other features of the person like gender, emotion can be improved in accuracy with additional audio input which can classify the pitch, loudness and other vocal pointers to classify the situation/person.

A Conventional Acoustic Sounds Recognition Classifier is the Hidden Markov Model with Gaussian Mixture Models, while ANN, Decision Trees and SVM can be used for further Discriminative classification. K-means clustering or trained classification can be employed to identify deviations from the background noise.

**Sounds Analysis steps**

- Human/Non Human Sound
  - If Human, Speech recognition
  - If not Human, Emergency/Non Emergency Sound

### GMM-HMM vs Deep Neural Networks

Most current systems use HMM combined with GMM models for temporal anlaysis however one can easily outperform these methods with Deep neural networks with a few layers. Party since HMM-GMM models rely on discarding large amounts of information to extract certain varibales to train upon, which can cause serious performance issues when the data is mostly tiny modulations of a small numer of parameters, like Speech.

On the other hand, GMM's while less accurate, are faster to compute, less expensive and more reliable. They are sometimes used in combination with deep neural networks to offer a good initial approximation.

Hence for non speech classification auditary classification, it makes sense to use GMM-HMM models trained on the background (specific to the environment), human, non-human sound inputs. Special features based on specific functionality can be trained using binary/multiclass classification techniques (screaming, gunshots, explosions, etc). 

## Speech recognition

Since speech recognition is a pattern recognition problem, it makes sense to use deep neural network models. As we discussed in the previous point, Deep neural networks (DNN) using HMM for specific frames have been found to have a superior performance to Gaussian Mixture Models combined with Hidden Markov Model (HMM) classifiers.

Depending on whether we will use a speaker dependent or a speaker independent system, the training of the model can be done either through large repositaries or using openly avaliable pre-trained models for the latter. While in case of speaker dependent system, the model can be easily trained by the user with text based voice training.

### Speaker dependent vs Speaker Independent models

A speaker dependent system would be superior to the speaker independent system in context based training, where a dependent system would accurately identify speech commands over a larger vocabulary space. It has also been shown to provide superior accuracy in general commands capturing the nuances of a speakers voice, tone and accent.

For the purposes of a home based security system, to be used by 1-3 people, a speaker dependent system would be vastly more useful in terms of accuracy even when the commands are altered or distorted, as well as the flexibility of commands that can be used. This system would also be superior when exposed to background disturbance in isolating the speaker's distinct voice atributes.

### Static vs Dynamic approach

Since we are looking at keyword detection rather than text transcription, we can get a good performance through static (using all of the speech at once as input) rather than dynamic approaches (small windows of speech used to make local classification decisions which are then compiled to make a global decision).  

### Performance Issues

Real-time Image, Audio analysis requires a prohibitive amount of computational processing power and this could easily become a major issue. While the training of the model takes place in its 'rest' mode by the user and is usually the most computationally expensive step, the prediction step would require a non null amount of time to process frames, audio time series and update the status every few seconds. Relatively recent Convolutional Network Cascades hold the potential to enable the coexistence high accuracy as well as high performance for visual detection systems. These methods focus on dropping false positive suggestions like the background in the early stages and focus on more advanced features only for inputs that successfully clear the earlier low resolution  stages as [Li, et.al. demonstrate in this paper][6].

## References:

[1]: [Multi-view Face Detection Using Deep Convolutional Neural Networks, S. Sudhakal et.al., Yahoo](http://www.cvrobot.net/wp-content/uploads/2015/06/ICMR-2015-Multi-view-Face-Detection-Using-Deep-Convolutional-Neural-Networks.pdf)

[2]: [Acoustic Detection of Human Activities in Natural Environments", Stavros Ntalampiras et. al.](http://ec.europa.eu/environment/life/project/Projects/index.cfm?fuseaction=home.showFile&rep=file&fil=AMIBIO_periodical_article_EN_2012.pdf)

[3]: [Technologies for Smart Sensors and Sensor Fusion, by Kevin Yallup, Krzysztof Iniewski, CRC Press](https://books.google.ch/books?id=y27OBQAAQBAJ&pg=PA392&dq=human+activity+sound+detection+classifier&hl=en&sa=X&ved=0ahUKEwiOr6i7ofXLAhXIoQ4KHaDpCaIQ6AEIHTAA#v=onepage&q=human%20activity%20sound%20detection%20classifier&f=false)

[4]: [An Abnormal Sound Detection and Classification System for Surveillance Applications, Cheung-Fat Chan and Eric W.M. Yu, 18th European Signal Processing Conference](http://www.ee.cityu.edu.hk/~cfchan/sound_detection.pdf)

[5]: [Deep Neural Networks for Acoustic Modeling in Speech Recognition, Geoffrey Hinton et.al.](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/38131.pdf)

[6]: [A Convolutional Neural Network Cascade for Face Detection](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)

[7]: [Model Zoo, Caffe](https://github.com/BVLC/caffe/wiki/Model-Zoo)


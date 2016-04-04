
## Main features of a home security system

- Facial Recognition
- Object type detection
- Smoke detection
- Sound Analysis
- Speech recognition

## Facial Recognition

Given a picture of a particular scene, let us start with the features that we would extract from it to first identify Faces within the scenario and then if we find one, to match it with the available "cleared" face repository. Given pictures from the home-owner, we will have a training dataset to train our models upon and extract relevant features. We would also like that the features are constantly updated given that especially for children, the facial key-points can vary as they grow into adults.

Facial detection algorithms would work in three steps once the input become available:

- Human or not a Human
- Registered person or not (binary)
- Which one is it(Dad, Mom, Robin the son)

### Human or not a Human

Refer to Object Classification

### Registered person or not (binary)

**Features**

Extracting the eigenfaces: One of the major issues with image recognition is the sheer number of features that we need to deal with. A simple way to get around this can be Principal Component Analysis (PCA) which reformulates new features from the old to maximise the explained variance in the data. We can call it a compression of the features, keeping only the most descriptive. These new compressed features are referred to as "eigenfaces".

***Potential vulnerabilities***

One issue that plagues facial recognition algorithms and their accuracy are lighting conditions which can botch our pixel intensity maps, different angles and the natural facial contortions we present while displaying emotions. Google's Android phone OS, which has a face unlock mode, gets around this by getting the user to first train the phone by taking images of their face in different lighting conditions and angles. 

**Libraries**

Caffe though frequently used, pales when comapred to Theano or simple libaries like FANN. Deep convoluted neural networks rule the roost when it comes to face classification models. While face recognition continues to be highly inaccurate, a method called 'Deep Dense Face Detector' by Yahoo researchers hopes to improve that. [1]

### Which one is it(Dad, Mom, Robin the son)

It would be computation expensive to train a model on K classes, hence one way is to do succesive binry classification comaring the derived human image to the repositary of the users with access to the house. Once can use the Siamese Network Training provided by Caffe (Model Zoo) to perform this classfication.

## Object type detection

### Training the Models

Given the large amount of data needed to train CNNs, the online pre-labelled datasets such as ImageNet can be incredibly useful in training one's classifiers

### Caffe: 
This deep learning framework perform image classification using convoluted neural nets and includes OpenCV, Blas libraries allowing efficient computation in C++. It also includes libraries for Python (pycaffe) and Matlab (matcaffe) allowing interfacing with these enviroments as well as supports a variety of input formats for images. Another big plus are the pre trained models offered by Caffe.

## Sound Analysis

The field of 'Computational Auditory Scene Analysis' aims at accurately classifying acoustic sequences to detect sound events related to specific activities or 'audio based environment modelling'.

Acoustic Analysis can give us a large amount of information about the surroundings that video/image analysis alone cannot contextualize. Examples of this include screaming, gunshots, glass breaking, footsteps from the vicinity, etc. Other features of the person like gender, emotion can be improved in accuracy with additional audio input which can classify the pitch, loudness and otehr vocal pointers to classify the situation/person.

A Conventional Acoustic Sounds Recognition Classifier is the Hidden Markov Model with Gaussian Mixture Models, while ANN, Decision Trees and SVM can be used for further Discriminative classification. K-means clustering or trained classification can be employed to identify deviations from the background noise.

### GMM-HMM vs Deep Neural Networks

Most current systems use HMM combined with GMM models for temporal anlaysis however one can easily outperform these methods with Deep neural networks with several layers. HMM-GMM models rely on discarding large amounts of information to extract certain varibales to train upon, which can cause serious performance issues when the data amounts to a tiny modulation of a small numer of parameters, Speech, case in point.

GMM's while less accurate, are faster to compute, less expensive and more reliable. They are sometimes used in combinationw ith deep neural networks to offer a good initial approximation.

For non speech classification auditary classification, one can use models trained on the background (specific to the environment), human, non-human sound inputs. Special features based on specific functionality can be trained using binary/multiclass classification techniques (screaming, gunshots, explosions, etc). Another way would be to first classify based on human vs non-human sounds and go further on multiclass or speech recognition based on the output.

## Speech recognition

Since speech recognition is a pattern recognition problem, it makes sense to use deep neural network models. As we discussed in the previous point, Deep neural networks (DNN) using HMM for specific frames have been found to have a superior performance to Gaussian Mixture Models combined with Hidden Markov Model (HMM) classifiers.

Depending on whether we will use a speaker dependent or a speaker independent system, the training of the model can be done either through large repositaries or using openly avaliable pre-trained models for the latter. While in case of speaker dependent system, the model can be trained by the user with text based voice training.

### Speaker dependent vs Speaker Independent models

A speaker dependent system would be superior to the speaker independent system in context based training, where a dependent system can accuratly identify speech commands over a larger vocabulary. It has also been shown to provide sueprior accuracy in general commands capturing the nuances of a speakers voice and accent.

For the purposes of a home based security system, to be used by 1-3 people, a speaker dependent system would be vastly more useful in terms of accuracy even when the commands are altered or distorted, as well as the flexibility of commands that can be used. This system would also be superior when exposed to background disturbance in isolating the speaker's distinct voice atributes.

### Static vs Dynamic approach

Since we are looking at keyword detection rather than text transcription, we can get a good performance through static (using all of the speech at once as input) as well as dynamic approaches (small windows of speech used to make local classification decisions which are then compiled to make a global decision).  


## Smoke Detection and Home layout analysis

## References:

[1]: [http://www.cvrobot.net/wp-content/uploads/2015/06/ICMR-2015-Multi-view-Face-Detection-Using-Deep-Convolutional-Neural-Networks.pdf](Multi-view Face Detection Using Deep Convolutional Neural Networks, S. Sudhakal et.al., Yahoo)

[2]: [http://ec.europa.eu/environment/life/project/Projects/index.cfm?fuseaction=home.showFile&rep=file&fil=AMIBIO_periodical_article_EN_2012.pdf](Acoustic Detection of Human Activities in Natural Environments", STAVROS NTALAMPIRAS et. al.)

[3]: Technologies for Smart Sensors and Sensor Fusion, by Kevin Yallup, Krzysztof Iniewski, CRC Press

[4]: [http://www.ee.cityu.edu.hk/~cfchan/sound_detection.pdf](An Abnormal Sound Detection and Classification System for Surveillance Applications, Cheung-Fat Chan and Eric W.M. Yu, 18th European Signal Processing Conference)

[5]: [http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/38131.pdf](Deep Neural Networks for Acoustic Modeling in Speech Recognition, Geoffrey Hinton et.al.)

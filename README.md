# Speech-Emotion-Recognization
SPEECH AND EMOTION RECOGNIZATION
Hari Pranesh M
Bennett University Greater Noida, Uttar Pradesh, India
1haripranesh.m@gmail.com
Abstract. Speech is thought of as the broadest and generally normal mechan-ism of correspondence. Discourse can pass on a plenty of data in regards to one's psychological, social, passionate qualities. Additionally, discourse feeling acknowledgment related work can support deflecting digital wrongdoings. Re-search on discourse feeling acknowledgment taking advantage of simultaneous AI methods has been on the top for quite a while. Various strategies like Recurrent Neural Network (RNN), Deep Neural Network (DNN), otherworldly component extraction and a lot more have been applied on various datasets. This paper presents a special Convolution Neural Network (CNN) based discourse feeling acknowledgment framework. A model is created and taken care of with crude discourse from explicit dataset for preparing, arrangement and testing purposes with the assistance of top of the line GPU. At long last, it emerges with a persuading exactness regarding 60.00% which is better contrasted with some other comparable errand on this dataset by a huge degree. This work will be powerful in creating conversational and social robots and distributing every one of the subtleties of their opinions.
Keywords:Speech, feelings, Deep Neural Network (DNN), Convolution Neural Network (CNN)
1	Introduction
In today's world, we are slowly entering into an era where textual inputs are taken over by speech inputs that is medium of communication with machine with voice which obviously a better one which we use nowadays with ALEX and Google Assis-tants. Speech Recognition is a fast growing technology that involve speech to text and vice versa; Human to Machine interaction and vice versa; Automatic Translation system that is input in any language converting it into English.As this field grows faster, the need of inference of emotion from the speech also increases as it can solve many problems in a better way and also by understanding the user’s mindset/ feelings and also to respond accordingly. There are different methods to get the emotion that are computer vision, Signal processing Machine Learning and Neural Network. It can be mainly used in Call Centers and during voice call conversations as of to detect the emotion of the customer. The Method that we have chosen in our project is Convolution Neural Networks and this method has given the wide success to many of the projects. Some of the project done by using CNN Model are Face Detection, Handwritten digits recognition, etc. Here the convolution refers to mathematical activity in our project we mainly used some functions like SoftMax unit, Relu and Dropout.
2	Related Work
In the Earlier Studies, they searched for a relation between speech acoustics and the emotion. To determine the correlation with speaker’s emotion systematic analysis of parameters or group of parameters were done. They mainly used the classifiers to analyses the emotion which was not that much effective. Over the other commonly used models supervised deep learning neural network models where far better in analyzing the emotion. In real time speech processing involves continuous signal inputs, fast processing and generating within a short span of time to meet the real-world needs. So, our model has to be faster as much as possible in generating the great output. In all the first studies, it is assumed that the data input is noise free, with no interference in any means, uncompressed and clear in quality. It is unknown that till which extend the models can handle the real time errors.

3	Block Diagram
 
Figure:-1

 
Figure:-2
Firstly, we will get the input of the audio signal. Then we will try to extract the fea-ture and remove the unwanted noise from the input which we have received. Then, we will try to enhance the feature which we have extracted from the input. Then we will Train the model using classifier. After that we will detect the emotion of the speech. Then we will display the emotion predicted by our model.

4	Proposed Methodology
In this project we have executed using conventional neural network. The architecture of the system is as follows:
 

                                                                Figure-3
The audio files are taken as inputs and intensity normalization is being applied over them.The model is trained over conventional neural network with respect to weights and output is generated as a numerical value corresponding to four emotions (i.e. happy, sad, angry and neutral).

5	Proposed Algorithm
 

The algorithm of our system is as above. Firstly, we will give the inputs files to the system. Then we will extract the features using the LIBROSA, a Python library.(In this project the number of features extracted is=180). Then, we will divide the data in to train and test and there after we will construct a CNN model with layers to train the data. Then testing will be done and results will be displayed (i.e.Happy,Sad, Angry and Neutral).
Parameter	Model 1	Model 2	Model 3	Model 4
Convolution filter size	3x3	3x3	3x3	3x3
Activation function	Re-lu(2)andSoftMax(1)	Relu(3) andSoft-Max(1)	Relu(3) andSoft-Max(1)	Relu(4) andSoft-Max(1)
Dropout fac-tor	0.1	0.25	0.1	0.1
Optimizer Stochastic	Gra-dient Des-cent	Gra-dient Des-cent	Gradient Descent	Gradient Descent
Learning rate	0.00005	0.00005	0.00005	0.00005
Cross Validation 	sparse_categori-cal_crossentropy	sparse_categori-cal_crossentropy	sparse_categori-cal_crossentropy	sparse_categori-cal_crossentropy
Table-1


6	Experimental Setup
For execution of this project we have used a system with core i7, 10th Gen, 3.1 GHz processer, witha Ram of 16GB. We have used the RAVDESS dataset.We have extracted 180 features.  We have used a CNN model with two hidden layers of dropout values 0.1 and “Relu" activation and for the final activation layer we have used "Softmax" function (since our output is non binary).
We have used Sparse Categorical cross entropy and we have plotted the performance in a heat map. We have tried three more CNN models with increased number of hid-den layers but accuracy decreases.
Layer (Type)	Output Shape	Param #
conv1d (Conv1D)	(None, 180, 128)	768
activation (Activation)	(None, 180, 128)	0
dropout (Dropout)	(None, 180, 128)	0
max_pooling1d (MaxPooling1D)	(None, 22, 128)	0
Conv1d_1 (Conv1D)	(None, 22, 128)	82048
Activation_1 (Activation)	(None, 22, 128)	0
Dropout_1 (Dropout)	(None, 22, 128)	0
Flatten (Flatten)	(None, 2816)	0
Dense (Dense)	(None, 8)	22536
Activation_2 (Activation)	(None, 8)	0
Total Params:105,352
Trainable params:105,352
Non-trainable params:0
Table-2(Model-1)
Layer (Type)	Output Shape	Param #
conv1d_2 (Conv1D)	(None, 180, 128)	768
activation_3 (Activation)	(None, 180, 128)	0
dropout_2 (Dropout)	(None, 180, 128)	0
max_pooling1d_1 (MaxPooling1D)	(None, 22, 128)	0
Conv1d_3 (Conv1D)	(None, 22, 128)	82048
Activation_4 (Activation)	(None, 22, 128)	0
Max_pooling1d_2 (Maxpooling1D)	(None, 2, 128)	0
Dropout_3 (Dropout)	(None, 2, 128)	0
Conv1d_4 (Conv1D)	(None, 2, 128)	82048
Activation_5 (Activation)	(None, 2, 128)	0
Dropout_4 (Dropout)	(None, 2, 128)	0
Flatten_1 (Flatten)	(None, 256)	0
Dense_1 (Dense)	(None, 8)	2056
Activation_6 (Activation)	(None, 8)	0

	Table 3(model-2)


Layer (Type)	Output Shape	Param #
conv1d_5 (Conv1D)	(None, 180, 128)	768
activation_7 (Activation)	(None, 180, 128)	0
dropout_5 (Dropout)	(None, 180, 128)	0
max_pooling1d_3 (MaxPooling1D)	(None, 22, 128)	0
Conv1d_6 (Conv1D)	(None, 22, 128)	82048
Activation_8 (Activation)	(None, 22, 128)	0
Max_pooling1d_4 (Maxpooling1D)	(None, 2, 128)	0
Dropout_6 (Dropout)	(None, 2, 128)	0
Conv1d_7 (Conv1D)	(None, 2, 128)	82048
Activation_9 (Activation)	(None, 2, 128)	0
Dropout_7 (Dropout)	(None, 2, 128)	0
Flatten_2 (Flatten)	(None, 256)	0
Dense_2 (Dense)	(None, 8)	2056
Activation_10 (Activation)	(None, 8)	0





Total Params:248,968
Trainable params:248.968
Non-trainable params:0

7	Dataset Description
For this project we have used the Ryerson Audio-Visual Database of Emo-tional Speech and Song (RAVDESS) dataset. It contains 1440 records: 60 prelimi-naries for every actor x 24 actors = 1440. The RAVDESS contains 24 expert actors (12 masculine, 12 feminine), expressing two lexically-matched articulations in an unbiased North American intonation.  Speech feelings incorporate disgust, surprise, fearful, angry, sad and happy articulations. 
The dataset contains two types of sentences in different emotions that is "Kids are talking by the door” and "Dogs are sitting by the door”. 

 
                                  Figure-5
 
                                    Figure-6
8	Preprocessing
In conventional narrow-band information transmission frameworks, the bandwidth capacity of signal was restricted to lessen the transmission bit rates. In communication, for instance, the recurrence scope of speech used to be restricted to the range from 300 Hz to 3.4 kHz. It was to the point of guaranteeing a fundamental degree of speech coherence yet at the expense of high voice quality. Almost certainly, such serious data transfer capacity decrease brought about a significant decrease in the emotional data conveyed by speakers.

9	Future Extraction
The extraction of the right and relevant collection of characteristics is the most crucial element of Speech Emotion Recognition. The emotion is identified based on the pitch and intensity of the feature in the dataset given. Let’s give you an example in which a high -intensity pitch is associated with both happiness and anger. The pitch with low intensity is correlated with the boredom and sadness. The selection of which and how many characteristics are required for automated speech detection is a critical problem in this project. The first level entails increasing efficiency and consistency. Our project considers parameters such as pitch, intensity, formant, and speech rate.

 
                                                        Figure-7
10	Future Scope
The result of this project is an CNN model of detecting the emotion of the speech signal irrespective of the gender and the language of the speaker. Despite the success, we have to still possibilities of enhancement - We can to add more emotion that can be predicted, and have to try produce output more accurate and flexible to handle interferences in any means from the surrounding/environment for this we can collect the data in adverse listening conditions, and data with noise. And can train the model with more training data, so as to increase the accuracy of the model appreciably. Can try to reduce the processing time and produce the output as fast as possible, so that we can meet the real-world needs. And also have to reduce the space used by the model, so that it can be attached to the applications more easily.

11	Result
We have built a model with an accuracy range from 65-75% at different time for predicting the emotions. We have achieved the highest accuracy with the neural network of two hidden layers and a activation layer of Softmax function. So, as if we want to achieve the higheraccuracy we have to increase the training data.



Confusion Matrix	Model 1	Model 2	Model 3	Model 4
accuracy 	0.6309523809523809 	0.5297619047619048 	0.5178571428571429 	0.5119047619047619 
F1 score 	0.6350817401138928	0.49887563588126377	0.5301055098129632	0.5049139543798995
Recall 	0.6309523809523809	0.5297619047619048	0.5178571428571429	0.5119047619047619
Precision 	0.6457470206127564	0.47852581380883275	0.5874169838495279	0.5333013031847885
True
Positive	28	28	26	11
False
Positive	27	37	40	17
True
Negative	32	27	23	22
False
Negative	17	23	19	17
                                                            Table-6





 
                            Figure-8

Heatmaps of our model:-
 
                            Figure-9(Model-1)

 
Figure-10(Model-2)

 
                                                           Figure-11(Model-3)

 
                                                        Figure-12(Model-4)
12	Conclusion
In the Project we used Convolution Neural Networks to train and test our dataset. The SoftMax function is used to calculate the probability distribution of output layer. The main observation here is that the training of dataset works well in CNN and testing datasets are varying across the datasets. Despite on behalf of our success of the project we want to improve the performance of our dataset to another level.

13	Reference
[1] El Ayadi M, Kamel MS, Karray F. Survey on speech emotion recognition: Features, classification schemes, and databases. Pattern recognition. 2011 Mar 1;44(3):572-87.
[2] Ingale AB, Chaudhari DS. Speech emotion recognition. International Journal of Soft Computing and Engineering (IJSCE). 2012 Mar;2(1):235-8.
[3] Schuller B, Rigoll G, Lang M. Hidden Markov model-based speech emotion recognition. In2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceed-ings.(ICASSP'03). 2003 Apr 6 (Vol. 2, pp. II-1). Ieee.
[4] Nwe TL, Foo SW, De Silva LC. Speech emotion recognition using hidden Markov models. Speech communication. 2003 Nov 1;41(4):603-23.
[5] Fayek HM, Lech M, Cavedon L. Evaluating deep learning architectures for Speech Emotion Recog-nition. Neural Networks. 2017 Aug 1;92:60-8.
[6] Swain M, Routray A, Kabisatpathy P. Databases, features and classifiers for speech emotion recog-nition: a review. International Journal of Speech Technology. 2018 Mar;21(1):93-120.
[7] Huang Z, Dong M, Mao Q, Zhan Y. Speech emotion recognition using CNN. InProceedings of the 22nd ACM international conference on Multimedia 2014 Nov 3 (pp. 801-804).
[8] Schuller BW. Speech emotion recognition: Two decades in a nutshell, benchmarks, and ongoing trends. Communications of the ACM. 2018 Apr 24;61(5):90-9.
[9] Han K, Yu D, Tashev I. Speech emotion recognition using deep neural network and extreme learning machine. InInterspeech 2014 2014 Sep 1.
[10] Wu S, Falk TH, Chan WY. Automatic speech emotion recognition using modulation spectral fea-tures. Speech communication. 2011 May 1;53(5):768-85.
[11] Nogueiras A, Moreno A, Bonafonte A, Mariño JB. Speech emotion recognition using hidden Mar-kov models. InSeventh European conference on speech communication and technology 2001.
[12] Mirsamadi S, Barsoum E, Zhang C. Automatic speech emotion recognition using recurrent neural networks with local attention. In2017 IEEE International conference on acoustics, speech and signal processing (ICASSP) 2017 Mar 5 (pp. 2227-2231). IEEE.
[13] Chen L, Mao X, Xue Y, Cheng LL. Speech emotion recognition: Features and classification mod-els. Digital signal processing. 2012 Dec 1;22(6):1154-60.
[14] Jain M, Narayan S, Balaji P, Bhowmick A, Muthu RK. Speech emotion recognition using support vector machine. arXiv preprint arXiv:2002.07590. 2020 Feb 3.
[15] Liu ZT, Xie Q, Wu M, Cao WH, Mei Y, Mao JW. Speech emotion recognition based on an im-proved brain emotion learning model. Neurocomputing. 2018 Oct 2;309:145-56.

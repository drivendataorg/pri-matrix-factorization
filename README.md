[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' style="max-width:85%;>](https://www.drivendata.org/)
<br><br>

<img src='https://s3.amazonaws.com/drivendata-public-assets/chimp.jpg' height=400>

# Pri-matrix Factorization
## Goal of the Competition
 Camera traps are triggered by motion or heat, and passively record the behavior of species in the area without significantly disturbing their natural tendencies. But camera traps can't yet automatically label the species they observe: it takes the valuable time of expert researchers, or thousands of citizen scientists, to [label](https://www.chimpandsee.org/#/) this data. Can you help automate the species-tagging process, freeing up time and resources to focus on higher-level research and conservation efforts?

## What's in this Repository
This repository contains code volunteered from leading competitors in the [Pri-matrix Factorization](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/) DrivenData challenge.

#### Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).


## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | --- | --- | ---
1 | dmytro | 0.012288 | 0.012371 |Fine-tuned multiple ImageNet pre-trained convolutional neural networks on randomly selected frames from video clips using stratified 4 folds split. Predicted classes for 32 frames per clip and combined predictions in a way time specific information is dropped and only combined prediction statistics used. Trained the second level models (xgboost, neural network, lightgbm) on out of fold CNN model predictions.
2 | ZFTurbo | 0.013594 | 0.013717 | The solution is divided into the following stages: preprocessing, metadata extraction, extract image hashes, extract features with pre-trained neural nets, train neural net models based on 3D convolutions, train neural net models for audio data, train neural net models for extracted, features from pre-trained nets, train recurrent neural net models for image hashes, validation for all neural nets, process test data for all neural nets, generate neighbor features, run models of second level, final weighted ensemble.
3 | AVeysov SKolbachev | 0.015307 |   0.015416 | Extract skip features from best modern CNNs (resnet, inception v4, inception-resnet 2, nasnet among others); use meta-data and attention layer (in essence trainable pooling) for classification on each set of features; use all the features together for an ensemble models; use several folds + weak semi supervised approach + blend the resulting models.

#### Winners and Zamba Blog Post:

#### Benchmark Blog Post: ["Long-term Recurrent Convolutional Network"](http://drivendata.co/blog/pri-matrix-factorization-benchmark/)

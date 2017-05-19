# Deep Learning with Python Tutorial

Welcome! Our goal in this repository is to install some of the necessary python packages for running deep neural networks, and to run some of these networks on some sample data from the Oxford 17 Flowers dataset (linked here: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/). We'll be particularly interested in convolutional neural networks (CNNs), and how one can use data augmentation and pre-trained networks to increase the effectiveness of these CNNs on small datasets.

Our goals will be as follows:

1. (Optional/Homework) Download the Oxford 17 Flowers dataset, and organize it into a format that Keras can work with. This has already been done via the Scripts/Preprocess_Data.py program.

2. Augment a sample image of a flower, and inspect the results, via the Scripts/Augment_Images.py program. You can see the results of this step at Results/augmentation_test

3. Train a deep neural network using Convolutional Neural Networks (CNNs) on the augmented Oxford 17 Flowers dataset. Classify it on a test set of flowers not trained in the original network. Output weights from this network can be found at Results/Basic_Network_Weights.h5

4. Use bottlenecking at the widely-used and successful VGGNet to greatly increase our accuracy from the previous step in much less time. Output weights from this network can be found at Results/Pretrained_Network_Weights.

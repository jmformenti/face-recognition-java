# Face Recognition pipeline in Java

A face recognition pipeline implementation to do face detection and face identification in Java.

Integrated pretrained third party models for face detection and embedding calculation using [DJL](https://djl.ai/).

Face identification implemented using cosine distance between calculated embeddings and target face.

# Build

Commands to build the project:

    git clone https://github.com/jmformenti/face-recognition-java.git
    cd face-recognition-java
    mvn package

A fatjar `face-recognition-java-${VERSION}.jar` will be placed in target dir.

# Quickstart

1. Prepare data, one root directory with one subdirectory with images of each person (for example, see `src/test/resources/images/train`).

2. Generate embeddings file.

    java -jar target/face-recognition-java-${VERSION}.jar embed -p /path/to/root/images -e embeddings.dat

3. Recognize faces in one image.

    java -jar target/face-recognition-java-${VERSION}.jar predict -e embeddings.dat -p /path/to/image

# Pretrained models

Two models are used:

|Model|Type|References|
|--|--|--|
|PaddlePaddle (DJL,flavor=server)|Face detection|https://docs.djl.ai/jupyter/paddlepaddle/face_mask_detection_paddlepaddle.html<br>https://paddledetection.readthedocs.io/featured_model/FACE_DETECTION_en.html|
|[20180402-114759](https://drive.google.com/uc?export=download&id=1TDZVEBudGaEd5POR5X4ZsMvdsh1h68T1) [converted to TorchScript](https://djl.ai/docs/pytorch/how_to_convert_your_model_to_torchscript.html)|Face embeddings|https://github.com/timesler/facenet-pytorch|

# Further work

Tasks to improve the performance:

 1. Apply face alignment before embeddings calculation.
 2. Use some classifier to pass from an embedding to the person label.

# References

 1. [Deep Java Library](https://djl.ai/)
 2. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. [VGGFace2: A dataset for recognising face across pose and age](https://arxiv.org/pdf/1710.08092.pdf), International Conference on Automatic Face and Gesture Recognition, 2018.
 3. _Jason Brownlee_ PhD, Machine Learning Mastery, [How to Perform Face Recognition With VGGFace2 in Keras](https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/)
 4. [5 Celebrity Faces Dataset](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset) used for testing.



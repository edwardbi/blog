#Tensorflow with TFlearn Implementation of RCNN model

Around 2013 to 2014, Girshick et al proposed a new state-of-art deep convolutional neural networks system on object detection called RCNN. This system is shown to outperform the previous state-of-art result, namely, the deep convolutional neural network system on image classification tasks. Aside from proposing a new framework, the author also put a lot of thought when writting the paper "Region-based Convolutional Networks for Accurate Object Detection and Semantic Segmentation". Furthermore, the system is open-sourced from the author's group that people may try to learn what is going on with the system. However, one misfortunate thing is that there is no Tensorflow implementation of such system. As a result, I decide to make the system myself according to both the instruction provided in the paper and the many online tutorials and system discussion blogs. Before we discuss what is happening to the code, let's first take a look what is RCNN system. 
<br><br>

##Brief Introduction of RCNN
The RCNN system is based off Alexnet, a variant of the Convolutional Neural Network structure which is proposed by Alex Krizhevsky from University of Toronto. The structure of Alexnet is shown in graph below:
![Alexnet](http://upload.semidata.info/new.eefocus.com/article/image/2015/09/28/5608f4354fed7.jpg)<br>
In our implementation, we use 5 layers of convolutional layers plus two layers of fully-connected layers followed by a softmax classfication structure. In the first two convolutional layers, the image feature maps are going through convolution followed by max-pooling and local normalization to form the output which is going to be the input to the next layer. Starting from the third convolutional layers until the fifth one, convolution with the kernel is the only operation. After the five layers of convolutional processes, resulted feature maps are feed to a two layers fully-connected layers for feature generation. The final result is predicted with a softmax output layer to classify the object. Alexnet is typically trained with Imagenet, in our case, as my personal computer is limited in processing power and storage space, the 17 flowers dataset is used to train both the Alexnet and the RCNN. <br>
As Alexnet may already classify objects, why should we even use RCNN to start with? For images with only one object what is takig the dominate space of the picture, such as selfie, we may safely assume Alexnet is capable of performing object classification already. However, in real-life situation, the dominate object is not taking a majority space in the picture or many objects may exist in the same picture. For example, a system trained with human images and horse images may have difficulty classify an image with human riding on top of the horse. Thus, the Alexnet method is not idea in such commonplace situation. RCNN, on the other hand, works better with under such case. <br>
The idea behind the RCNN algorithm is that it uses traditional edge-detection or image processing techniques to try to make many image segment proposals that the traditional algorithm "thinks" as objects. At training time, as it is infeasible of asking human annotators to mark every objects in the image, an evaluation metric called IOU is used to calculate the percentage intersection between the proposal image segments with the object segment provided by the human annotator. Segment proposals which passed the threthold value is thought of as positive examples while the others are categorize as background: the negative examples. These examples are feed to a pre-trained Alexnet with only the N-way softmax classification layer (N corresponds to the number of pre-trained classes) replaced by a randomly initialized (L+1)-way softmax classification layer (L corresponds to the number of fine-tunning categories) for fine tunning. The resulted fine-tuned Alexnet is used to extract 4096 feature vectors of any image segment from its last fully-connected layer, the layer right before the softmax classification layer. Final classification is made by feeding the feature vector through a number of pre-trained bi-class SVMs which every single SVM corresponding to a single object. Now having to know the structure of this system, let's take a look at how to realize it with Tensorflow. 
<br><br>

##The RCNN Model
To implement the model, there are a couple of existing projects and/or codes that we may directly use to accelerate our work. First of all, defining Alexnet structures by Tensorflow can be tedious and time-consuming. To allow an easy implementation, we adopt the tflearn wrapper of Tensorflow to make the code easy to achieve. The tflearn project exist at [here](https://github.com/tflearn/tflearn)<br>
According to their online document, taking convolutional layers as an example, what you used to do with 
<pre><code>
with tf.name_scope('conv1'):
    W = tf.Variable(tf.random_normal([5, 5, 1, 32]), dtype=tf.float32, name='Weights')
    b = tf.Variable(tf.random_normal([32]), dtype=tf.float32, name='biases')
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.add_bias(W, b)
    x = tf.nn.relu(x)
</code></pre>
All you need to do now is
<pre><code>
tflearn.conv_2d(x, 32, 5, activation='relu', name='conv1')
</code></pre>
Which is obviously a lot easier to realize in this situation. With this handy tool, we use the Alexnet implementation provided by Github user [ck196](https://github.com/ck196/tensorflow-alexnet)'s work, with the following structure as the network structure:
<pre><code>
def create_alexnet(num_classes):
    # Building 'AlexNet'
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')
    network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.001)
    return network
</code></pre>
When fine-tunning the network, tflearn allows an easy way by including the variable restore=False at the last fully_connected layer of the system, this makes the create_alexnet function appear as follow:
<pre><code>
def create_alexnet(num_classes, restore=True):
    # Building 'AlexNet'
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax', restore=restore)
    network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.001)
    return network
</code></pre>
The variable restore is default to True, which means the system will restore all variables of the saved model to fit to the model. However, when a pre-trained model is loaded for fine-tunning, we set the restore to False, thus, only the last layer, where we have the restore=restore code, is not taking the model variable. This way, only the last fully connected layer is required to learn its new parameters from randomly init variables. Finally, how do we take the feature vectors from the system? There are many ways. One way is to get the last fully connected layer before the softmax's name to get its unique weights and biases. An much easier way will be using the same model but delete the softmax layer as well as the dropout layer before it. </br>
Knowing the tricks with the backbone model of the RCNN, another question is how can we gennerate image segment proposals? Luckly, we may use the selectivesearch library of python to achieve this task. With pip install selectivesearch, we can directly import the selectivesearch function to achieve the process. <br>
After the image proposal is made, another question of important is how do we define the IOU to decide which image segment is taken as the foreground and which is considered as background. The idea is relatively simple: if we know there is intersection between two regions, after randing the rectanglur vertices we can always find the area of intersection. And the formula for calculating the intersetion rate is defined to be the area of intersection devide by the area of union. To achieve this logic, we first decide on the situations where intersection regions may happen. The entire logic of IOU is given by the code segments below:
<pre><code>
# IOU Part 1, intersection detection plus intersection area calculation
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False
    if if_intersect == True:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1] 
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter
# IOU Part 2, get the IOU score
def IOU(ver1, vertice2):
    # vertices in four points
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3] 
        area_2 = vertice2[4] * vertice2[5] 
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False
</code></pre>
Finally, we may use the scikit-learn's LinearSVC module for svm training. According to the paper, when fine-tuning the Alexnet, the CNN is baised towards abundancy of data while when training the SVM, the SVM classifier is baised towards small amount of data. Thus, when decideing the IOU threthold for CNN, we define it to be 0.5 to allow more data to be marked as positive. When training the SVM, as we don't need that much positive cases, we restrict the threthold to be 0.3 to reduce the learning amount.<br>
At classification time, we train a cascade of SVM classifiers, which one SVM representing one class of object. The code can be obtained at [here](https://github.com/edwardbi/DeepLearningModels/tree/master/RCNN).



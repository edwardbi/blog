#Tensorflow Implementation of CBOW model

Mikolov's Word2Vec provides an amazing way for unsupervised learning of textual documents. The result, according to Mikolov,
is able to automaticlly formulate words as vector embeddings that contain the word's sematic as well as syntactic meaning. An 
famous example is the vector "France" minus the vector "Paris" plus the vector "Rome" is close to the vector "Greece". As the 
model is so powerful, various implementations are provided in various programming langugages in different frameworks. For example, 
Gensim library in Python has the exact implementation of methods in orginal C code. The result is shown to be effective and the 
way to use the library in python is easy. Meanwhile, Tensorflow, the Google-open-sourced machine learning/deep learning framework, 
also has its own implementation of the Word2Vec models. Unfortunately, their implementation has only the Skip-Gram model available, 
thus, it is a good programming exercise to construct the CBOW model based on Tensorflow's Skip-Gram implementation. In this blog, 
we will exam how to modify the model in the word2vec_basic.py file to make it running in the CBOW structure instead of the Skip-Gram 
structure. 
<br><br>
##Brief Introduction of Word2Vec

Word2Vec contains two models proposed by Mikolov in his paper Efficient Estimation of Word Representations in Vector Space. Two models 
are presented from this paper: the CBOW model and the Skip-Gram model. For those of you who has certain knowledge of Neural Networks, 
the model structures are presented in the graph below: ![Word2Vec](http://s8.sinaimg.cn/large/001Sy0Jmty6Lzce4wJN17&690)<br>
As the model requires a big training set for better performance, training speed may be a concern. Mikolov further presented two 
alternative training methods for effecitively training the model, they are Hierarchical Softmax and Negative Sampling. The former 
method is working on the idea of constructing a binary huffman tree with words at leave level at the output level. The tree is 
constructed prior to the training of the model and accroding to word frequence. During training, the probability of the word given its 
context (assumed CBOW model) is obtained by traversing the word along its huffman tree pathway from root to leaf. Binary classification 
is used at each level of the tree at traversing time and each non-leaf node's probability is multipied together to count as the probability 
of the final output. Negative Sampling, on the other hand, implements on the idea of noise-contrastive estimation method, which means 
to randomly assume negative examples with the correct target as positive examples to train the model. In Tensorflow, the provided tutorial 
implements the Skip-Gram structure together with the Negative Sampling training methods. The CBOW model is left as a practise in this case.
<br><br>

##The CBOW Model
There are couple things the demo version of Word2Vec from Tensorflow is used that is not shared from the original C version. The first 
thing is the limitation imposed on the dictionary size. For faster training and better accuracy, Mikolov dropped words below certain probability. 
However, the Tensorflow version seems to put a solid restriction on the size of the dictionary instead. The code snipped is shown below:
<pre><code>
# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary
</code></pre>
The size is limited to 50k and this is achieved by the line "count.extend(collections.Counter(words).most_common(vocabulary_size - 1))". 
To remove this limitation, we can direction use the function "Counter().most_common()" without parameter, which will formulate the count list with 
all unique word-count paris from the input data. The modeified code is shown below:
<pre><code>
# Step 2: Build the dictionary and replace rare words with UNK token.
def build_dataset(words, min_cut_freq):
  count_org = [['UNK', -1]]
  count_org.extend(collections.Counter(words).most_common())
  count = [['UNK', -1]]
  for word, c in count_org:
    word_tuple = [word, c]
    if word == 'UNK': 
        count[0][1] = c
        continue
    if c > min_cut_freq:
        count.append(word_tuple)
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary
</code></pre>
where after we have all words' word-count pairs, we re-store them in a new list with count smaller than the min_cut_freq dropped. <br>
Afterwards, we see that the input to the graph is given by the "generate_batch()" function, which generates two lists, with one input 
correspoinding to one output. For example, if the inputs' ids in dictionary is [20, 14, 31, 33, 0, 321, 4231, 2], a Skip-Gram input is 
with window wize 3 is going to be [14, 14, 31, 31, 33, 33, 0, 0, 321, 321, 4231, 4231 ], 
and the labels is going to be [[20], [31], [14], [33], [31], [0], [33], [321], [0], [4231], [321], [2]]. For CBOW, what we want is the have 
the following format of input and labels: [[20, 31], [14, 33], [31, 0], [33, 321], [0, 4231], [321, 2]], [[14], [31], [33], [0], [321], [4231]]
Thus, we can modify the "generate_batch()" function as follow:
<pre><code>
data_index = 0

def generate_cbow_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    batch_temp = np.ndarray(shape=(num_skips), dtype=np.int32)
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch_temp[j] = buffer[target]
    batch[i] = batch_temp
    labels[i,0] = buffer[skip_window]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels
</code></pre>
where we define a temp list to store the context of predicted words, push it to the batch after the collection is done. This code sigment 
will produce the desired output as explained above.
<br>
Finally, as the Skip-Gram model is working on the idea of using one word to represent the enviroment while the CBOW model is summing up the 
enviroment to formuate the output, we need to modify the graph. First of all, we need to modify the tensor placeholder for the input 
as a tensor of size batch_size by context_size. The code to achieve this is:
<pre><code>
train_inputs = tf.placeholder(tf.int32,shape=[batch_size, skip_window * 2])
</code></pre>
Then, we need to have the window_size embeddings to add together and take the average to form the output. The code is
<pre><code>
reduced_embed = tf.div(tf.reduce_sum(embed, 1), skip_window*2)
</code></pre>
The overall graph part is thus
<pre><code>
with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32,shape=[batch_size, skip_window * 2])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    reduced_embed = tf.div(tf.reduce_sum(embed, 1), skip_window*2)
    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, reduced_embed, train_labels,
                     num_sampled, vocabulary_size))
</code></pre>
Finally, at the session section, replace the generate_batch function with the generate_CBOW_batch function defined above, the CBOW model 
is achieved. For detail of the model, please refer to my opensourced project CBOW at [here](https://github.com/edwardbi/CBOW)

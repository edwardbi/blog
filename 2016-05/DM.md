#Gensim and Tensorflow Implementation of DM model

Paragraph vectors is the other word-embedding topics proposed by Le and Mikolov in their work "Distributed Representations of Words
and Documents". As an extension of word vectors, the models can be seen as an extension to the origianal CBOW and Skip-Gram models 
of Word2Vec with a special input dimension called paragraph id, a same-to word-embedding size vector that lives in a different 
dimension from the word vectors. The purpose of such model are two fold: 
- Include longer textual information into the model so word embedding will hopefully be more accurate (The Distriubted Memory Model)
- Create a sentence or document level word-embedding (Both the DM model and the DBOW model) <br>

## Brief Introduction of Paragraph Models
Similar to Word2Vec, there are two variants of the model of paragraph models, they are the PV-DM and the PV-DBOW model. The PV-DM model
is based off the CBOW model while the PV-DBOW model is assemble the Skip-Gram model. Both structures are shown in graph below:
</br>![PV-DM](http://img1.tuicool.com/vMZvuy.png!web)![PV-DBOW](http://img2.tuicool.com/FvQJfq.png!web)<br>

The models can be trained with the same methods used in Word2Vec and recently, it is used for classifying emotional responses of the 
IMDb movie rating dataset. 

In this blog, I am going to focus on the DM model. Gensim will be used firstly to treat the DM model as a black box, then, the Tensorflow 
version of the DM model is going to be implemented basing on my understanding of the model. If you find any mistake, please contact me 
at edward920210@126.com to point out the problem in my code. I will be happy to have a discussion with you, maybe you can help me to see 
the things I wasn't notice of! I'm here to thank you in advance!

## The DM Model
Firstly, we may use the Gensim deep learning library to have a taste of this model. There is a function already exist in Gensim for 
doing so. The Doc2Vec library is responsible for generating the paragraph vectors. However, different from the word vectors, the blog 
for training this model is not clearly documented to the level of the Word2Vec model's blog. I am going to present my code for doing so, 
let's get started!
<pre><code>
from gensim.models import doc2vec
from collections import namedtuple

import csv
import re
import string

# We use the wikipedia dataset to train the paragraph vector
reader = csv.reader(open("wikipedia.csv"))
count = 0
data = ''
for row in reader:
    count = count + 1
    if count > 301:
       break
    else:
        data += row[1]

# Setup a regex to split paragraph to sentences. 
# We assume sentences ending with . ? or !. There are sentences that
# have . in the middle such as Mr.Wang, as the majoirty sentences are
# okay and this program is mostly serve as a demo, I decide to ignore 
# these cases. 
sentenceEnders = re.compile('[.?!]')
data_list = sentenceEnders.split(data)

# I created a namedtuple with words=['I', 'love', 'NLP'] and tags=['SEN_1']
# to represent an input sentence
LabelDoc = namedtuple('LabelDoc', 'words tags')
exclude = set(string.punctuation)
all_docs = []
count = 0
for sen in data_list:
    word_list = sen.split()
    # For every sentences, if the length is less than 3, we may want to discard it
    # as it seems too short. 
    if len(word_list) < 3:
        continue
    tag = ['SEN_' + str(count)]
    count += 1
    sen = ''.join(ch for ch in sen if ch not in exclude)
    all_docs.append(LabelDoc(sen.split(), tag))

# Print out a sample for one to view what the structure is looking like    
print all_docs[0:10]

# Mikolov pointed out that to reach a better result, you may either want to shuffle the 
# input sentences or to decrease the learning rate alpha. We use the latter one as pointed
# out from the blog provided by http://rare-technologies.com/doc2vec-tutorial/
model = doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
model.build_vocab(all_docs)
for epoch in range(10):
    model.train(all_docs)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay    

# Finally, we save the model
model.save('my_model.doc2vec')
</code></pre>
Afterwards, we need a way to test the model, here is the code for it:
<pre><code>
import random
import numpy as np
import string
# Randomly choose a sentence id
doc_id = np.random.randint(model.docvecs.count) 
print doc_id
# Calculate the simialr sentences by using the docvecs.most_similar function
sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)
# We first print out the target sentence
print('TARGET' , all_docs[doc_id].words)
# Then, from the sims result, we obtain the top-8 results to print them out
count = 0
for i in sims:
    if count > 8:
        break
    pid = int(string.replace(i[0], "SEN_", ""))
    print(i[0],": ", all_docs[pid].words)
    count += 1
</code></pre>
The results of a random example is shown below:
</p>
>8136
>('TARGET', ['Maldonado', 'holds', 'two', 'notable', 'knockout', 'victories', 'over', 'Maiquel', 'Falc\xc3\xa3o'])
>('SEN_8152', ': ', ['Maldonado', 'was', 'expected', 'to', 'face', 'Aaron', 'Rosa', 'at'])
>('SEN_8147', ': ', ['Notable', 'victories', 'from', 'this', 'period', 'include', 'Jessie', 'Gibbs', 'Vitor', 'Miranda', 'and', 'two', 'TKO', 'victories', 'over', 'Maiquel', 'Falc\xc3\xa3o'])
>('SEN_3945', ': ', ['Wright', 'Zoological', 'Museum', 'Emily', 'Graslie'])
>('SEN_10040', ': ', ['Green', 'Professor', 'John', 'B'])
>('SEN_6144', ': ', ['British', 'blues', 'boom'])
>('SEN_6045', ': ', ['Ohio', 'State', 'went', '6\xe2\x80\x933', 'during', 'this', 'period'])
>('SEN_10981', ': ', ['Morrison', 'was', 'knocked', 'out', 'in', 'the', 'sixth', 'round'])
>('SEN_6295', ': ', ['Australia', 'and', 'New', 'Zealand'])
>('SEN_7465', ': ', ['1434', 'CE', 'called', 'QutbulMadar', 'and', 'is', 'centered', 'around', 'his', 'shrine', 'dargah', 'at', 'Makanpur', 'Kanpur', 'district', 'Uttar', 'Pradesh'])
</p>
</br>
It is interesting to see that the target sentence mentioned Maldonado, while the most closed sentence is also about the same person. 
Moreover, the second-most close one is about notable victories. Consequently, we know that the paragraph vector is working to some 
extent. </br>
Now, let's try to look into the actual structure of the DM model and use Tensorflow to realize this model. Below is my way of doing so: 
First of all, I use the wikipedia training corpus to train the model, thus, the same dataset pre-processing and formatting codes are 
used directly from the Gensim model presented above. Afterwards, the build_data function shall be modified because this time, the training 
corpus is not a text file full of words. There file are currently arranged in the format of namedtuple. The code for doing so is shown below: 
<pre><code>
# Function taking in the input data and a minimum word frequence
def build_dataset(input_data, min_cut_freq):
  # Firstly, we collect the words into a list for the counter function
  # to use to generate the count. 
  words = []
  for i in input_data:
        for j in i.words:
            words.append(j)
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
  data = []
  unk_count = 0
  for tup in input_data:
    word_data = []
    for word in tup.words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0
        unk_count += 1
      # Here, we collect the words' index to a temp variable
      # called word_data to get ready to push into the data
      word_data.append(index)
    # push the words with the sentence tag to the data
    data.append(LabelDoc(word_data, tup.tags))    
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary
</code></pre>
We also need to modify the generate_batch function. We would like to have the data generated from the function above as input, the output 
should be exactly the same as CBOW model except with one more label: the sentence the window belongs to. Below is the code for doing so: 
<pre><code>
word_index = 0
sentence_index = 0

def generate_DM_batch(batch_size, num_skips, skip_window):
    global word_index
    global sentence_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # Define an extra variable called para_labels for the paragraph labels
    para_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) 
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[sentence_index].words[word_index])
        sen_len = len(data[sentence_index].words)
        if sen_len - 1 == word_index: # reaching the end of a sentence
            # Reset the word index to 0 as a new sentences is incoming. 
            word_index = 0
            # Update the sentence_index
            sentence_index = (sentence_index + 1) % len(data)
        else: # increase the word_index by 1
            word_index += 1 
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
        para_labels[i, 0] = sentence_index
        buffer.append(data[sentence_index].words[word_index])
        sen_len = len(data[sentence_index].words)
        if sen_len - 1 == word_index: # reaching the end of a sentence
            word_index = 0
            sentence_index = (sentence_index + 1) % len(data)
        else: # increase the word_index by 1
            word_index += 1 
    return batch, labels, para_labels
</code></pre>
The structure of the model is given in the code snippet below:
<pre><code>
#paragraph vector place holder, same dimension as the train_labels placeholder
train_para_labels = tf.placeholder(tf.int32,shape=[batch_size, 1])

# Look up embeddings for inputs.
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed_word = tf.nn.embedding_lookup(embeddings, train_inputs)

# Look up embeddings for paragraph inputs 
para_embeddings = tf.Variable(tf.random_uniform([paragraph_size, embedding_size], -1.0, 1.0))
embed_para = tf.nn.embedding_lookup(para_embeddings, train_para_labels)
# Concat the word embeddings with the paragraph embeddings to form the same input as in the model structure
embed = tf.concat(1, [embed_word, embed_para])
# Average the embeddings
reduced_embed = tf.div(tf.reduce_sum(embed, 1), skip_window*2 + 1)

# The loss is calculated with the reduced_embed
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, reduced_embed, train_labels,
                     num_sampled, vocabulary_size))

# In a session, you should call the generate_DM_batch function
batch_inputs, batch_labels, batch_para_labels = generate_DM_batch(batch_size, num_skips, skip_window)
feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels, train_para_labels: batch_para_labels}
</code></pre>
Above is the code segment that is different from the CBOW model discussed in [previous blog](https://github.com/edwardbi/blog/blob/master/2016-05/CBOW.md) 
I strongly encourage you to test it out if you have Tensorflow installed to your programming enviorment, and the result is currently really bad, I am working 
on to incorporate decreasing learning rate or shuffle of sentences to see if a better result can be obtained. 

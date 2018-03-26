import tensorflow as tf
import numpy as np 
from filereader import Filereader as F

tf.set_random_seed(0)
tf.logging.set_verbosity(tf.logging.INFO)

f = F(path="./data/")
X_train, Y_train, N, M = f.getData(sample=60000)
X_test, Y_test, N_test, M_test = f.getData(dataset="testing",sample=10000)

print(X_train.shape, Y_train.shape)

learning_rate = 0.001
training_epochs = 100 

number_of_inputs = X_train.shape[1]
number_of_outputs = Y_train.shape[1]

layer_1_nodes = 200
layer_2_nodes = 200
layer_3_nodes = 10

elu_alpha = 1.2
epochs = 100
batch_size = 250
l2_reg = 1e-4

def accuracy(predictions, actuals):
    return np.mean(np.argmax(actuals,axis=1)==np.argmax(predictions,axis=1))

with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
    Y = tf.placeholder(tf.float32, shape=(None, 10))

with tf.variable_scope('layers', reuse=tf.AUTO_REUSE, initializer=tf.keras.initializers.he_normal(seed=1),
                       regularizer=tf.keras.regularizers.l2(l=l2_reg)):
    layer_1_output = tf.keras.activations.elu(tf.layers.dense(inputs=X, units=layer_1_nodes, use_bias=True, name="layer1"),alpha=elu_alpha)
    layer_2_output = tf.keras.activations.elu(tf.layers.dense(inputs=layer_1_output, units=layer_2_nodes, use_bias=True, name="layer2"), alpha=elu_alpha)
    layer_3_output = tf.layers.dense(inputs=layer_2_output, units=layer_3_nodes, use_bias=True, name="layer3")

with tf.name_scope('predictions'):
    predictions = tf.nn.softmax(layer_3_output)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=layer_3_output))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

with tf.name_scope('metrics'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, axis=-1), tf.argmax(predictions, axis=-1)),tf.float32))

with tf.name_scope('dataset'):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.shuffle(buffer_size=batch_size,seed=1,reshuffle_each_iteration=True)
    dataset = dataset.repeat()

    dataset_init = dataset.make_initializable_iterator()
    dataset_iter = dataset_init.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    sess.run(dataset_init.initializer, feed_dict={X: X_train, Y: Y_train})

    for epoch in range(epochs) :
        for batch in range(X_train.shape[0]//batch_size):
            X_batch, Y_batch = sess.run(dataset_iter)
            sess.run(train, feed_dict={X: X_batch, Y: Y_batch})

        train_acc = sess.run(accuracy, feed_dict={X:X_train, Y:Y_train})
        print("epoch {}: Training accuracy = {}".format(epoch+1, train_acc))

    test_acc = sess.run(accuracy, feed_dict={X:X_test, Y:Y_test})
    print("Final Testing accuracy: {}".format(test_acc))


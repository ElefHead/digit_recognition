import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from filereader import Filereader

tf.enable_eager_execution()
tf.executing_eagerly()
f = Filereader('./data/')

x_train, y_train, r,c = f.getData(dataset="training", sample=60000)
x_test, y_test, r,c = f.getData(dataset="testing", sample=10000)

learning_rate = 0.001
training_epochs = 100

elu_alpha = 1.2
epochs = 100
batch_size = 250
l2_reg = 1e-4


layer_1_nodes = 200
layer_2_nodes = 200
output_layer_nodes = y_train.shape[1]


# Eager Batching
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).\
    shuffle(buffer_size=batch_size).\
    batch(batch_size=batch_size).\
    repeat()


# model
class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.dense1 = tf.layers.Dense(units=layer_1_nodes, kernel_initializer=tf.keras.initializers.he_normal(seed=1), activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2())
        self.dense2 = tf.layers.Dense(units=layer_2_nodes, kernel_initializer=tf.keras.initializers.he_normal(seed=1), activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2())
        self.dense3 = tf.layers.Dense(units=output_layer_nodes, kernel_initializer=tf.keras.initializers.he_normal(seed=1), kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, input):
        result = self.dense1(input)
        result = self.dense2(result)
        result = self.dense3(result)
        return result


model = MNISTModel()


def predict(model, inputs):
    return tf.nn.softmax(model(inputs))


def loss(model, inputs, targets):
    prediction = model(inputs)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=targets))


def accuracy(target, prediction):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(
                    target,
                    axis=1
                ),
                tf.argmax(
                    prediction,
                    axis=1
                )
            ),
            dtype=tf.float32
        )
    )


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
num_batches_per_epoch = x_train.shape[0] // batch_size

for epoch in range(epochs):
    for batch, (x_batch, y_batch) in enumerate(tfe.Iterator(dataset)):

        optimizer.minimize(lambda : loss(model, x_batch, y_batch),
                           global_step=tf.train.get_or_create_global_step())

        if batch >= num_batches_per_epoch:
            break

    print("Training accuracy after step {:03d}: {:.5f}".format(epoch+1, accuracy(y_train, predict(model, x_train))))

print("Testing accuracy {:.5f}".format(accuracy(y_test, predict(model, x_test))))


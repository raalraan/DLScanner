import numpy as np
import random
from ..utilities.try_imports import try_tensorflow
tf = try_tensorflow()
keras = try_tensorflow('keras')
#################################
def MLP_Classifier(function_dim,num_FC_layers,neurons):
    ''' Function to create MLP classfier.
    Input args:
    function_dim: dimensions of the input
    num_FC_layers: Number of the fully connected layers
    neurons: number of neurons in each FC layer
    output args:
             MLP classifier network
    '''
    inp =keras.layers.Input((function_dim, ))
    x = keras.layers.Dense(neurons,activation=None)(inp)
    for _ in range(num_FC_layers):
      x = keras.layers.Dense(neurons,activation='relu')(x)
    output = keras.layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model(inp,output)
    return model
#################################
def MLP_Regressor(function_dim,num_FC_layers,neurons):
    ''' Function to create MLP classfier.
    Input args:
    function_dim: dimensions of the input
    num_FC_layers: Number of the fully connected layers
    neurons: number of neurons in each FC layer
    output args:
             MLP classifier network
    '''
    inp =keras.layers.Input((function_dim, ))
    x = keras.layers.Dense(neurons,activation=None)(inp)
    for _ in range(num_FC_layers):
      x = keras.layers.Dense(neurons,activation='relu')(x)
    output = keras.layers.Dense(1,activation='linear')(x)
    model = keras.Model(inp,output)
    return model 
    
###########################################
### The following is related to the siilarity learning classifier#    
###########################################
def make_pairs(x, y):
    '''Function to create positive and negative paris of input data.
   Positive pairs has label =1 and negative pairs have labels =0.
   Input Args:
        x: input data of dimension (n, function dimension)
        y: vector label of each input with dimension (n)
   Output Args:
        x: pairs of the input with dimension (n,2,function dimension)
        y: labels of the positive and negative pairs.
                  
   Exampel to run:   pairs_train, labels_train = make_pairs(Xf, obs1f)
    '''
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(int(num_classes))]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[int(label1)])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), abs(np.array(labels).astype("float32")-1)
########################
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
###############
def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be c
                lassified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )
    return contrastive_loss
#############
def similariy_classifier(x_train, y_train,function_dim,latent_dim,neurons,num_layers,epochs, learning_rate,batch_size,verbose=0):
  ''' Function to create similarity classfier-First training part of the network.
    Input args:
    function_dim: dimensions of the input
    latent_dim: Dimension of the latent space
    output args:
             similarity classifier network
  '''
  xtrain,ytrain = make_pairs(x_train, y_train)
  input_ = keras.layers.Input((function_dim, ))   
  x = keras.layers.Dense(128, activation="tanh")(input_)
  x = keras.layers.Dense(64, activation="tanh")(x)
  x = keras.layers.Dense(32, activation="tanh")(x)
  x = keras.layers.Dense(latent_dim, activation="linear")(x)
  embedding_network = keras.Model(input_, x)
  input_1 = keras.layers.Input((function_dim, ))
  input_2 = keras.layers.Input((function_dim, ))
  tower_1 = embedding_network(input_1)
  tower_2 = embedding_network(input_2)
  merge_layer = keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
  output_layer = keras.layers.Dense(1, activation="linear")(merge_layer)
  model_S = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
  model_S.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss(margin=1))
  model_S.fit([xtrain[:,0],xtrain[:,1]], ytrain,epochs=epochs, batch_size=batch_size,verbose=0)
  ####### Second training step ######
  inp = keras.layers.Input((function_dim, ))
  x = (embedding_network(inp))
  x = keras.layers.Dense(neurons,activation=None)(inp)
  x = keras.layers.Dense(neurons,activation='relu')(x)
  output = keras.layers.Dense(1,activation="sigmoid")(x)
  model = keras.Model(inp,output)
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.BinaryCrossentropy())
  model.fit(x_train,y_train,epochs=epochs, batch_size=batch_size,verbose=0)
  return model      

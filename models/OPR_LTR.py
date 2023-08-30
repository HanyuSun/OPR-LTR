import tensorflow as tf
from tensorflow.python.keras.initializers import Identity, glorot_uniform, Zeros
from tensorflow.python.keras.layers import Dropout, Input, Layer, Embedding, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

def delta(x):

  return x + 0.01

class GraphConvolution(Layer):  # ReLU(AXW)

    def __init__(self, units, frequents,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0,
                 seed=1024, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.freqs = frequents
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.seed = seed

    def build(self, input_shapes):
        # print("ok")
        self.num_node = input_shapes[0][-2]
        # print(num_node)
        # print("ok")
        # feature_dim = input_shapes[0][-1]
        # print(feature_dim)
        self.kernel_loc = self.add_weight(shape=(self.num_node,
                                self.units),
                        initializer=glorot_uniform(
                            seed=self.seed),
                        regularizer=l2(self.l2_reg),
                        name='kernel_loc', )
        
        self.kernal_freq = self.add_weight(shape=(self.num_node,
                                self.freqs),
                        initializer=glorot_uniform(
                            seed=self.seed),
                        regularizer=l2(self.l2_reg),
                        name='kernal_freq', )
        
        self.kernal_combine = self.add_weight(shape=(self.num_node,
                         self.units + self.freqs),
                         initializer=glorot_uniform(
                            seed=self.seed),
                        regularizer=l2(self.l2_reg),
                        name='kernel_comb',)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_node, self.units),
                                        initializer=Zeros(),
                                        name='bias', )

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)

        self.built = True

    def call(self, inputs, training=None, **kwargs):
        # feature_node, feature_edge, adj = inputs
        feature, adj = inputs
        adj_c = adj[:,:, :self.num_node]
        adj_f = adj[:,:, self.num_node:]
        # print("real_time_state_shape: ", feature.shape)
        # print("adj: ", adj.shape)
        # print("spatial_adj: ", adj_c.shape)
        # print("turn_over_event_adj: ", adj_f.shape)
        feature_vertex = tf.matmul(adj_c, feature)
        # print("feature_vertex: ", feature_vertex.shape)
        output_vertex = tf.multiply(feature_vertex, self.kernel_loc)
        # print("output_vertex: ", output_vertex.shape)
        output_edge = tf.multiply(adj_f, self.kernal_freq)
        # print("output_edge: ", output_edge.shape)
        output = tf.reduce_sum(tf.multiply(tf.concat([output_vertex, output_edge],-1), self.kernal_combine), -1, keepdims=True)
        # print("output: ", output.shape) # output_shape = (None, 48, 1)
        if self.use_bias:
            output += self.bias
        # print("output_shape: ",output.shape)
        act = self.activation(output)  
        # print(" call ok")
        # act._uses_learning_phase = features._uses_learning_phase
        return act

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation,
                  'dropout_rate': self.dropout_rate,
                  'l2_reg': self.l2_reg,
                  'use_bias': self.use_bias,
                  'feature_less': self.feature_less,
                  'seed': self.seed
                  }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ReadOut(Layer):
  ## this layer combines all list into one column
    def __init__(self, units,
              activation=tf.nn.sigmoid, dropout_rate=0.5,
              use_bias=True, l2_reg=0,
              seed=1024, **kwargs):
      super(ReadOut, self).__init__(**kwargs)
      self.units = units
      self.use_bias = use_bias
      self.l2_reg = l2_reg
      self.dropout_rate = dropout_rate
      self.activation = activation
      self.seed = seed

    def build(self, input_shapes):
        # print("ok")
        self.num_node = input_shapes[-2]
        # print(num_node)
        # print("ok")
        # feature_dim = input_shapes[0][-1]
        # print(feature_dim)
        self.kernel = self.add_weight(shape=(self.units,
                                self.num_node),
                        initializer=glorot_uniform(
                            seed=self.seed),
                        regularizer=l2(self.l2_reg),
                        name='kernel_loc', )  

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units, 1),
                                        initializer=Zeros(),
                                        name='bias', )

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.built = True

    def call(self, inputs, training=None, **kwargs):

        feature = inputs
        output = tf.matmul(self.kernel, feature)
        if self.use_bias:
            output += self.bias

        act = self.activation(output) 
        return act   

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation,
                  'dropout_rate': self.dropout_rate,
                  'l2_reg': self.l2_reg,
                  'use_bias': self.use_bias,
                  'feature_less': self.feature_less,
                  'seed': self.seed
                  }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  

class ReadOut2(Layer):

  ## this layer gives each list a line
    def __init__(self,
              activation=tf.nn.leaky_relu, dropout_rate=0.5,
              use_bias=True, l2_reg=0,
              seed=1024, **kwargs):
      super(ReadOut2, self).__init__(**kwargs)
      self.use_bias = use_bias
      self.l2_reg = l2_reg
      self.dropout_rate = dropout_rate
      self.activation = activation
      self.seed = seed

    def build(self, input_shapes):
        # print("ok")
        self.num_node = input_shapes[0][-2]
        # print(num_node)
        # print("ok")
        # feature_dim = input_shapes[0][-1]
        # print(feature_dim)
        # self.kernel = self.add_weight(shape=(self.num_node * self.num_node,
        #                         self.num_node),
        #                 initializer=glorot_uniform(
        #                     seed=self.seed),
        #                 regularizer=l2(self.l2_reg),
        #                 name='kernel_loc', )
        self.kernel1 = self.add_weight(shape=(self.num_node,
                                self.num_node),
                        initializer=glorot_uniform(
                            seed=self.seed),
                        regularizer=l2(self.l2_reg),
                        name='kernel_loc', )
        
        self.kernel2 = self.add_weight(shape=(self.num_node,
                                self.num_node),
                        initializer=glorot_uniform(
                            seed=self.seed),
                        regularizer=l2(self.l2_reg),
                        name='kernel_loc', )

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_node, self.num_node),
                                        initializer=Zeros(),
                                        name='bias', )

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)

        self.built = True

    def call(self, inputs, training=None, **kwargs):

        feature, adj = inputs
        adj_c = adj[:,:, :self.num_node]
        # output = tf.matmul(self.kernel, feature)
        output = tf.multiply(feature, self.kernel1)
        output = tf.matmul(self.kernel2, output)
        output = tf.reshape(output,[-1, self.num_node, self.num_node])
        print("ReadOut2_output_shape: ",output.shape)

        if self.use_bias:
            output += self.bias

        act = self.activation(output) 
        result = tf.multiply(act, adj_c)
        return result   

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation,
                  'dropout_rate': self.dropout_rate,
                  'l2_reg': self.l2_reg,
                  'use_bias': self.use_bias,
                  'feature_less': self.feature_less,
                  'seed': self.seed
                  }
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  

class ReadOut3(Layer):

  ## this layer gives each list a line
    def __init__(self,
              activation=tf.nn.leaky_relu, dropout_rate=0.5,
              use_bias=True, l2_reg=0,
              seed=1024, **kwargs):
      super(ReadOut3, self).__init__(**kwargs)
      self.use_bias = use_bias
      self.l2_reg = l2_reg
      self.dropout_rate = dropout_rate
      self.activation = activation
      self.seed = seed

    def build(self, input_shapes):
        # print("ok")
        self.num_node = input_shapes[0][-2]
        # print(num_node)
        # print("ok")
        self.feature_dim = input_shapes[0][-1]
        self.feature0_dim = input_shapes[2][-1]
        self.adjmax = input_shapes[1][-1]
        # print(feature_dim)
        self.kernel = self.add_weight(shape=(self.num_node,
                                (self.feature0_dim+self.adjmax-self.num_node)),
                        initializer=glorot_uniform(
                            seed=self.seed),
                        regularizer=l2(self.l2_reg),
                        name='kernel', )

        self.kernel1 = self.add_weight(shape=(self.num_node,
                                self.num_node),
                        initializer=glorot_uniform(
                            seed=self.seed),
                        regularizer=l2(self.l2_reg),
                        name='kernel1', )
        
        self.kernel2 = self.add_weight(shape=(self.num_node,
                                self.num_node),
                        initializer=glorot_uniform(
                            seed=self.seed),
                        regularizer=l2(self.l2_reg),
                        name='kernel2', )        
        
        # self.kernel3 = self.add_weight(shape=(self.num_node,
        #                         self.num_node),
        #                 initializer=glorot_uniform(
        #                     seed=self.seed),
        #                 regularizer=l2(self.l2_reg),
        #                 name='kernel2', )
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_node, self.num_node),
                                        initializer=Zeros(),
                                        name='bias', )

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)

        self.built = True

    def call(self, inputs, training=None, **kwargs):
        feature, adj, feature0 = inputs
        adj_c = adj[:,:, :self.num_node]
        adj_f = adj[:,:, self.num_node:]
        # output = tf.matmul(self.kernel, feature)
        ## individual feature from Q and Z
        Q = tf.concat([feature0, adj_f],-1) # [None, 48, 1+5]
        print("Q:", Q.shape)
        outputQ = tf.reduce_sum(tf.multiply(Q, self.kernel), -1, keepdims=True) # [None, 48, 1]
        print("outputQ:", outputQ.shape)
        outputZ = tf.reshape(feature,[-1,1,self.num_node])   # [None, 48, 1]
        print("outputZ:", outputZ.shape)
        # inter Q_Z
        outputM = tf.multiply(tf.multiply(outputQ, self.kernel1) + tf.multiply(outputZ, self.kernel2), adj_c)
        # feature_m = tf.reshape(tf.tile(feature_T,[1,1,self.num_node]),[1,self.num_node,self.num_node]) # [None, 48 ,48]
        # feature_m = tf.concat([feature_m, tf.reshape(outputQ,[-1,1,self.num_node])],-1) # [None, 48, ,49]
        print("output: ", outputM.shape)
        if self.use_bias:
            outputM += self.bias      
        # output = tf.matmul(outputM, self.kernel3) # [None, 48, ,48]
        # print("ReadOut2_output_shape: ",output.shape)
        act = self.activation(outputM) 
        result = tf.multiply(act, adj_c)
        return result   

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation,
                  'dropout_rate': self.dropout_rate,
                  'l2_reg': self.l2_reg,
                  'use_bias': self.use_bias,
                  'feature_less': self.feature_less,
                  'seed': self.seed
                  }
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 




def GCN(adj_dim=48, feature_dim=5, node_dim=48, num_class=1, num_layers=3, activation=tf.nn.relu, dropout_rate=0.5, l2_reg=0, feature_less=False):
    Adj = Input(shape=(adj_dim,adj_dim+feature_dim))  # Input(shape=(None,), sparse=True)

    print("Adj shape: ", Adj)
    X_in = Input(shape=(node_dim, num_class), )
    print("X_in shape: ", X_in)
    h = X_in
    for i in range(num_layers):
        # if i == num_layers - 1:
        #     activation = tf.nn.softmax
        #     n_hidden = num_class
        h = GraphConvolution(1, 5, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg)([h,Adj])
    h2 = ReadOut3()([h,Adj,X_in])    
    # h2 = ReadOut2()([h,Adj])
    # h2 = ReadOut(530)(h)
    print("ok")
    output = h2
    print("output shape: ", output)
    model = Model(inputs=[X_in, Adj], outputs=output)

    return model


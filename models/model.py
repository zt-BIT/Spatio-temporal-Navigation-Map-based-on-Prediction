import tensorflow as tf
from tensorflow.python.util import nest
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper
#LIBRARY OF TENSORFLOW
import sys
sys.path.append('..')
from utils.util_MDN import *
#SELF DEFINED OBJECT
OBJ_NUM = 7
COMPONENTS = 6
def get_a_cell(lstm_size, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop
tf.reset_default_graph()
class Model():
    def __init__(self, args):
        #bool type params
        self.use_attention = args.use_attention
        self.attn_input_feeding = args.attn_input_feeding

        self.use_MDN = args.use_mdn  # if use MDN mode
        self.use_beam_search = args.use_beam_search
        self.attn_input_feeding = args.attn_input_feeding
        self.use_regularization = args.use_regularization
        #int params
        self.encoder_len = args.encoder_len
        self.decoder_len = args.decoder_len
        self.crd_num = args.input_size  # including the (x,y) of the seven cars
        self.hidden_size = args.hidden_size
        self.hidden_layers = args.hidden_layers
        self.mixtures = args.components  # num of mixture denesity netowrks
        self.obj_num = args.obj_num
        self.batch_size = args.batch_size
        self.beam_width = 2
        self.y_out_dim = 14  # the(x,y) of ego+6nbrs
        #others
        self.attention_type = args.attention_type
        self.learning_rate = args.learning_rate
        self.regularizer = tf.contrib.layers.l2_regularizer(0.01)
        self.decoder_outputs = []
        #placeholder
        self.X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.encoder_len, self.crd_num], name="input_data")
        self.y = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,self.decoder_len,self.y_out_dim], name="ground_truth") 
        self.input_data = tf.unstack(self.X, axis=1)
        self.y_labels = tf.unstack(self.y, axis=1)
        self.drop_out = tf.placeholder(dtype=tf.float32, name="drop_out")
        #variables
        self.W_out = tf.Variable(tf.random_normal([self.hidden_size, self.y_out_dim], stddev=0.01), name="W_out")
        self.b_out = tf.Variable(tf.constant(0.1, shape=[self.y_out_dim]), name="b_out")

    #LSTM MODEL
    def LSTM_model(self): 
        def lstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True, forget_bias=1.0)
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.drop_out)
            return lstm
        with tf.name_scope("LSTM") as scope:
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.hidden_layers)])   
            encoder_outputs, state = tf.contrib.rnn.static_rnn(cell, self.input_data, dtype=tf.float32)
            self.encoder_outputs = encoder_outputs
        with tf.variable_scope("rnn_decoder"):
            decoder_cell, self.initialstate = self.build_decoder_cell(state)
            W = tf.Variable(tf.truncated_normal([self.hidden_size, self.y_out_dim]), name='lstm_W')
            b = tf.Variable(tf.truncated_normal([self.y_out_dim]), name='lstm_b')
            self.regularization = self.regularizer(W)
            for i in range(len(self.y_labels)):
                if i == 0:
                    prev = self.input_data[-1]
                    prev1 =  tf.slice(prev,[0,3], [self.batch_size,14])
                    decoder_output, state = decoder_cell(prev1, self.initialstate)
                    prev = tf.matmul(decoder_output, W) + b
                    self.decoder_outputs.append(prev)      
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    decoder_output, state = decoder_cell(prev, state)
                    prev = tf.matmul(decoder_output, W) + b
                    self.decoder_outputs.append(prev)
        return self.decoder_outputs
    #BLSTM MODEL
    def bidir_LSTM_model(self):
        def bilstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.drop_out)
            return lstm
        with tf.name_scope("bidir_LSTM") as scope:
            #assert self.hidden_size % 2 == 0, "hidden_size must be even number for bidir-LSTM"
            fw_units=[self.hidden_size/2, self.hidden_size/2]
            bw_units=[self.hidden_size/2, self.hidden_size/2]
            fw_cells=[tf.contrib.rnn.BasicLSTMCell(unit) for unit in fw_units] 
            bw_cells=[tf.contrib.rnn.BasicLSTMCell(unit) for unit in bw_units] 
            encoder_outputs, state_fw,state_bw=tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, self.input_data, dtype=tf.float32)
            encoder_final_state_c0 = tf.concat((state_fw[0].c, state_bw[0].c), 1)
            encoder_final_state_h0 = tf.concat((state_fw[0].h, state_bw[0].h), 1)
            encoder_final_state_c1 = tf.concat((state_fw[1].c, state_bw[1].c), 1)
            encoder_final_state_h1 = tf.concat((state_fw[1].h, state_bw[1].h), 1)
            self.initialstate0 = tf.contrib.rnn.LSTMStateTuple(c = encoder_final_state_c0, h = encoder_final_state_h0)
            self.initialstate1 = tf.contrib.rnn.LSTMStateTuple(c = encoder_final_state_c1, h = encoder_final_state_h1)
            selfinitialstate = tuple([self.initialstate0,self.initialstate1])
            self.encoder_outputs=encoder_outputs
        with tf.variable_scope("bilstm_decoder"):
            decoder_cell, self.initialstate = self.build_decoder_cell(selfinitialstate)
            W = tf.Variable(tf.truncated_normal([self.hidden_size, self.y_out_dim]), name='bilstm_W')
            b = tf.Variable(tf.truncated_normal([self.y_out_dim]), name='bilstm_b')
            self.regularization = self.regularizer(W)
            for i, inp in enumerate(self.y_labels):
                if i == 0:       
                    prev = self.input_data[-1]
                    prev1 = tf.slice(prev,[0,3], [self.batch_size,14])           
                    decoder_output, state = decoder_cell(prev1, self.initialstate)                  
                    prev = tf.matmul(decoder_output, W) + b
                    self.decoder_outputs.append(prev)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    decoder_output, state = decoder_cell(prev1, state)    
                    prev = tf.matmul(decoder_output, W) + b     
                    self.decoder_outputs.append(prev)       
            return self.decoder_outputs

    #CONSTRUCT DECODER CELL WITH ATTENTION MECHANISM
    def build_decoder_cell(self, state):
        """
        state: [tuple] (cell_state, hidden_state)
        """
        encoder_outputs = tf.transpose(self.encoder_outputs, [1,0,2])
        #a list of 50 in length,and each element of the list is of size[batch_size, hidden_size]
        encoder_last_state = state
        encoder_inputs_length = self.encoder_len
        self.decoder_cell_list = [get_a_cell(self.hidden_size, self.drop_out) for i in range(self.hidden_layers)]
        decoder_initial_state = state
        
        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length 
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beam_search:
            print ("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(self.encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(lambda s: seq2seq.tile_batch(s, self.beam_width), self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(self.encoder_len, multiplier=self.beam_width)
        if self.use_attention:
            self.attention_mechanism = attention_wrapper.BahdanauAttention(num_units=self.hidden_size, memory=encoder_outputs,) 
            if self.attention_type.lower() == 'luong':
                self.attention_mechanism = attention_wrapper.LuongAttention(num_units=self.hidden_size, memory=encoder_outputs,)
            def attn_decoder_input_fn(inputs, attention):
                if not self.attn_input_feeding:
                    return inputs
                # Essential when use_residual=True
                _input_layer = Dense(self.hidden_size, dtype=self.dtype, name='attn_input_feeding')
                return _input_layer(array_ops.concat([inputs, attention], -1))           
        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
            self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_size,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=decoder_initial_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state
        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beam_search else self.batch_size * self.beam_width
        initial_state=[state for state in decoder_initial_state]
        initial_state[-1]=self.decoder_cell_list[-1].zero_state(batch_size=self.batch_size, dtype=tf.float32)
        decoder_initial_state = tuple(initial_state)
        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

    def mdn_process(self, xyvalueList, paramsList):
        """
        xyvalueList: [List] of [X_value, Y_value]
        paramList  : [List] of [mux, muy, sigmax, sigmay, rho, pi]
        """
        x_value, y_value = xyvalueList[0], xyvalueList[1]
        mu1, mu2, s1, s2 = paramsList[0], paramsList[1], paramsList[2], paramsList[3]
        rho, pi = paramsList[4], paramsList[5]
        max_pi = tf.reduce_max(pi, 1, keepdims = True)
        pi = tf.subtract(pi, max_pi)
        pi = tf.exp(pi)
        normalize_pi = tf.reciprocal(tf.reduce_sum(pi, 1, keepdims = True))
        pi = tf.multiply(normalize_pi,pi)
        s1 = tf.exp(s1)
        s2 = tf.exp(s2)
        rho = tf.tanh(rho)
        px1x2 = tf_2d_normal(x_value, y_value, mu1, mu2, s1, s2, rho)
        px1x2_mixed = tf.reduce_sum(tf.multiply(px1x2, pi), 1)
        loss_seq = -tf.log(tf.maximum(px1x2_mixed, 1e-10))#avoid zero divided
        return loss_seq

    def MDN_model(self, LSTM_type='LSTM'):
        """ 
        LSTM_type: [str] LSTM or BLSTM 
        """
        self.use_MDN = True
        if LSTM_type == 'LSTM':      
            outputs = self.LSTM_model()       
        elif LSTM_type == 'BLSTM':
            print ("using blstm model")
            outputs = self.bidir_LSTM_model()
        else:
            raise "You should specify the right model before running MDN"
        with tf.name_scope("Output_MDN") as scope:    
            params = 6 * self.obj_num # [mu1 mu2 sigma1 sigma2 rho pi] * obj_num
            output_units = self.mixtures * params  #6*7*6         
            outputs_tensor = tf.concat(axis=0, values=outputs[:-1])
            #LINEAR LAYER 
            W_o = tf.Variable(tf.random_normal([outputs_tensor.get_shape().as_list()[1], output_units], stddev=0.01), name='mdn_W')
            b_o = tf.Variable(tf.constant(0.5, shape=[output_units]), name='mdn_b')
            params_tensor = tf.matmul(outputs_tensor, W_o)+b_o

        with tf.name_scope('MDN_over_next_vector') as scope:
            params_tensor = tf.reshape(params_tensor, (self.decoder_len - 1, self.batch_size, output_units))
            params_tensor = tf.transpose(params_tensor, [1, 2, 0])
            MDN_X = tf.transpose(self.y, [0, 2, 1])
            x_next = tf.subtract(MDN_X[:,:,1:], MDN_X[:,:,:-1])
            self.xyvalue_split = tf.split(axis=1, num_or_size_splits=2*self.obj_num, value=x_next)
            self.params_split = tf.split(axis=1, num_or_size_splits=params, value=params_tensor)
            loss_sum = 0.0
            for obj_num in range(self.obj_num):
                loss_obj = self.mdn_process(self.xyvalue_split[2*obj_num:2*(obj_num+1)], self.params_split[6*obj_num:6*(obj_num+1)])#List
                loss_sum += loss_obj
            loss_ave = tf.multiply(loss_sum,1/self.obj_num)
            self.cost_seq = abs(tf.reduce_mean(loss_ave))
            

    def Evaluating(self):
        with tf.name_scope("evaluating") as scope:
            self.cost = 0.0 
            for i in range(len(self.y_labels)):           
                self.cost += tf.sqrt(tf.reduce_sum(tf.square(self.y_labels[i] - self.decoder_outputs[i])))
            if self.use_MDN:  
                self.cost += self.cost_seq
            if self.use_regularization:
                self.cost+= self.regularization
            tvars = tf.trainable_variables()           
            grads,_ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 1)          
            self.global_step = tf.Variable(0, trainable=False)         
            lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.97, staircase=True)
            tf.summary.scalar("cost", self.cost) 
            tf.summary.scalar("learningRate", self.learning_rate)       
            # for var in tf.trainable_variables(): 
            #     tf.summary.histogram(var.name, var) 
            # Merge all summaries into a single op 
            self.merged_summary_op = tf.summary.merge_all() 
            self.saver = tf.train.Saver(max_to_keep = 1)
            optimizer = tf.train.AdamOptimizer(lr)      
            #self.train_op = optimizer.minimize(self.cost, global_step=self.global_step)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            # calculate training parameters number
            self.numel = tf.reduce_sum([tf.size(var) for var in tvars])


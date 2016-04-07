import os, sys, time, warnings
import cPickle as pkl
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

import data_engine, metrics
from common import *

def weight_variable(shape, name):
    raise NotImplementedError('do not use this')
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)

def bias_variable(shape, name):
    raise NotImplementedError('do not use this')
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name)

def get_param(signature, params):
    # params is a dictionary
    # the naming is in the format of 'some_name:int'
    for key in params.keys():
        name = key.split(':')[0]
        if name == signature:
            return params[key]
    raise NotImplementedError('%s does not exist'%signature)

def tensor_mat_multiply(tensor, matrix, tensor_ndim=3):
    # to simulate theano tensor mat mul (a,b,c)*(c,d) -> (a,b,d)
    assert tensor_ndim == 3 # only support ndim=3 for now
    a = tf.shape(tensor)[0]
    b = tf.shape(tensor)[1]
    c = tf.shape(tensor)[2]
    d = tf.shape(matrix)[1]
    tensor_ = tf.reshape(tensor, tf.pack([a*b, c]))
    result = tf.matmul(tensor_, matrix)
    result = tf.reshape(result, tf.pack([a, b, d]))
    #result.set_shape([a, b, c]) # a,b,c cannot be tensor
    return result

class Attention_LSTM_decoder(object):
    def __init__(self, options):
        self.options = options
        
    def init_params(self):
        print 'init params of attention lstm decoder'
        ctx_dim = self.options['ctx_dim']
        word_dim = self.options['dim_word']
        dim = self.options['dim']
        # lstm params
        W = numpy.concatenate([norm_weight(word_dim, dim),
                               norm_weight(word_dim, dim),
                               norm_weight(word_dim, dim),
                               norm_weight(word_dim, dim)], axis=1)
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        lstm_W = tf.Variable(W, name='lstm_W')
        lstm_U = tf.Variable(U, name='lstm_U')
        lstm_b = tf.Variable(numpy.zeros((4*dim,), dtype='float32'), name='lstm_b')
        lstm_Wc = tf.Variable(norm_weight(ctx_dim, dim*4), name='lstm_Wc')
        # attention params
        lstm_Wc_att = tf.Variable(norm_weight(ctx_dim, ortho=False), name='lstm_Wc_att')
        lstm_Wd_att = tf.Variable(norm_weight(dim, ctx_dim), name='lstm_Wd_att')
        lstm_b_att = tf.Variable(numpy.zeros((ctx_dim,), dtype='float32'), name='lstm_b_att')
        lstm_U_att = tf.Variable(norm_weight(ctx_dim, 1), name='lstm_U_att')
        lstm_c_att = tf.Variable(numpy.zeros((1,), dtype='float32'), name='lstm_c_att')
        
        self.params = [lstm_W, lstm_U, lstm_b, lstm_Wc, lstm_Wc_att, lstm_Wd_att,
                       lstm_b_att, lstm_U_att, lstm_c_att]
        names = [param.name for param in self.params]
        self.params = OrderedDict(zip(names, self.params))
        
    def fprop(self, emb, mask, context, init_state, init_memory):
        # emb (t,m,word_dim), mask (t,m), ctx (m, f, ctx_dim),
        # init_state, init_memory (m, dim)
        print 'fprop of attention lstm decoder'
        W = get_param('lstm_W', self.params) # (word_dim, 4*dim)
        U = get_param('lstm_U', self.params) # (dim, 4*dim)
        Wc = get_param('lstm_Wc', self.params) # (ctx_dim, 4*dim)
        b = get_param('lstm_b', self.params) # (4*dim,)
        Wc_att = get_param('lstm_Wc_att', self.params) # (ctx_dim, ctx_dim)
        Wd_att = get_param('lstm_Wd_att', self.params) # (dim, ctx_dim)
        b_att = get_param('lstm_b_att', self.params) # (ctx_dim,)
        U_att = get_param('lstm_U_att', self.params) # (ctx_dim, 1)
        c_att = get_param('lstm_c_att', self.params) # (1,)

        nsteps = tf.shape(emb)[0]
        n_samples = tf.shape(emb)[1]
        pctx = tensor_mat_multiply(context, Wc_att) + b_att # (m,f,ctx_dim)
        state_below = tensor_mat_multiply(emb, W) + b # (t,m,4*dim)
        state_below_ta = tensor_array_ops.TensorArray(dtype=state_below.dtype, size=nsteps)
        state_below_ta = state_below_ta.unpack(state_below)
        mask_ta = tensor_array_ops.TensorArray(dtype=mask.dtype, size=nsteps)
        mask_ta = mask_ta.unpack(mask)
        H = tensor_array_ops.TensorArray(dtype=emb.dtype, size=nsteps)
        counter = tf.constant(0, dtype=dtypes.int32, name="time")
        
        def step(t, h, c, H_ta_t):
            # weighted sum with attention on context
            pstate_ = tf.matmul(h, Wd_att) # (m,ctx_dim)
            pctx_ = pctx + tf.expand_dims(pstate_, 1) # (m,f,ctx_dim)
            pctx_ = tf.tanh(pctx_)
            alpha = tf.squeeze(tensor_mat_multiply(pctx_, U_att)) + c_att # (m,f)
            ctx_ = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1) # (m, ctx_dim)
            # standard LSTM
            preact = tf.matmul(h, U) # (m,4*dim)
            x_ =  state_below_ta.read(t) # (m,4*dim)
            m = mask_ta.read(t) # (m,)
            ctx_ = tf.matmul(ctx_, Wc) # (m, 4*dim)
            preact = preact + x_ + ctx_
            i, o, f, inp = tf.split(1, 4, preact)
            i = tf.sigmoid(i) # (m, dim)
            o = tf.sigmoid(o)
            f = tf.sigmoid(f)
            inp = tf.tanh(inp)
            c_new = f * c + i * inp
            c_new = tf.expand_dims(m, 1) * c_new + tf.expand_dims(1. - m, 1) * c_new
            h_new = o * tf.tanh(c_new)
            h_new = tf.expand_dims(m, 1) * h_new + tf.expand_dims(1. - m, 1) * h_new
            
            H_ta_t = H_ta_t.write(t, h_new)
            return t + 1, h_new, c_new, H_ta_t
        unused_final_time, final_h, final_c, final_H = control_flow_ops.While(
            cond=lambda t, _1, _2, _3: t < nsteps,
            body=step,
            loop_vars=(counter, init_state, init_memory, H),
            parallel_iterations=10,
            back_prop=True,
            swap_memory=False,
            name='attention_conditional_lstm'
            )
        final_H = final_H.pack()
        return final_H
    
    def test(self):
        n_timesteps = 5
        n_samples = 8
        n_frames = 10
        self.options['dim_word'] = 64
        self.options['ctx_dim'] = 128
        self.options['dim'] = 32
        dim_word = self.options['dim_word']
        dim = self.options['dim']
        ctx_dim = self.options['ctx_dim']
        seed = 1234
        
        ctx = tf.random_uniform(shape=[n_samples, n_frames, ctx_dim], seed=seed)
        emb = tf.random_uniform(shape=[n_timesteps, n_samples, dim_word], seed=seed)
        mask = tf.ones([n_timesteps, n_samples])
        init_state = tf.zeros(shape=[n_samples, dim])
        init_memory = tf.zeros(shape=[n_samples, dim])
        self.init_params()
        states = self.fprop(emb, mask, ctx, init_state, init_memory) # (t,m,dim)
        cost = tf.reduce_sum(states)
        # training
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        with tf.Session() as session:
            tf.initialize_all_variables().run()
            for i in range(10000):
                t0 = time.time()
                train_cost, _ = session.run([cost, train_op])
                print 'cost %.4f, minibatch time %.3f'%(train_cost, time.time()-t0)

class Model(object):
    def __init__(self, options, channel=None):
        self.options = options
        self.channel = channel
        
    def init_params(self):
        print 'model init params'
        ctx_dim = self.options['ctx_dim']
        word_dim = self.options['dim_word']
        dim = self.options['dim']
        n_words = self.options['n_words']
        # word embedding
        Wemb = tf.Variable(norm_weight(n_words, word_dim), name='Wemb')
        # mlp to initialize h and c of lstm
        ff_init_state_W = tf.Variable(norm_weight(ctx_dim, dim),
                                      name='ff_init_state_W')
        ff_init_state_b = tf.Variable(numpy.zeros((dim,), dtype='float32'),
                                      name='ff_init_state_b')
        ff_init_memory_W = tf.Variable(norm_weight(ctx_dim, dim),
                                       name='ff_init_memory_W')
        ff_init_memory_b = tf.Variable(numpy.zeros((dim,), dtype='float32'),
                                      name='ff_init_memory_b')
        
        # word prediction
        ff_logistic_W = tf.Variable(norm_weight(dim, n_words),
                                    name='ff_logistic_W')
        ff_logistic_b = tf.Variable(numpy.zeros((n_words,), dtype='float32'),
                                    name='ff_logistic_b')

        params = [
            Wemb,
            ff_init_state_W, ff_init_state_b,
            ff_init_memory_W, ff_init_memory_b,
            ff_logistic_W, ff_logistic_b
            ]
        names = [param.name for param in params]
        self.params = OrderedDict(zip(names, params))
        self.params.update(self.decoder.params)
        
    def fprop(self):
        # (n_words, n_samples)
        print 'model fprop'
        x = tf.placeholder(tf.int64, name='x') # (t, m)
        mask = tf.placeholder(tf.float32, name='mask') # (t, m)
        ctx = tf.placeholder(tf.float32, name='ctx') # (m,f,1024)
        mask_ctx = tf.placeholder(tf.float32, name='mask_ctx') # (m,f)
        self.x = x
        self.mask = mask
        self.ctx = ctx
        self.mask_ctx = mask_ctx
        n_timesteps = tf.shape(x)[0]
        n_samples = tf.shape(x)[1]

        # build graph
        # --------------------------------------
        # word embeddings
        emb_ = get_param('Wemb', self.params)
        emb = tf.reshape(
            tf.gather(emb_, tf.reshape(x, [-1])),
            tf.pack([n_timesteps, n_samples, self.options['dim_word']]))
        emb_zeros = tf.zeros_like(emb)
        header = tf.expand_dims(tf.gather(emb_zeros, 0), 0)
        rest = tf.gather(emb, tf.range(0, n_timesteps-1))
        emb_shifted = tf.concat(0, [header, rest])
        # mean ctx
        counts = tf.reduce_sum(mask_ctx, 1, keep_dims=True)
        ctx_mean = tf.reduce_sum(ctx, 1) / counts # (m, 1024)
        # init h and c
        init_state = tf.matmul(ctx_mean, get_param('ff_init_state_W', self.params)) + \
          get_param('ff_init_state_b', self.params)
        init_memory = tf.matmul(ctx_mean, get_param('ff_init_memory_W', self.params)) + \
          get_param('ff_init_memory_b', self.params)

        # lstm
        proj_h = self.decoder.fprop(emb_shifted, mask, ctx, init_state, init_memory) # (t,m,h)
        # word prediction
        logit = tensor_mat_multiply(proj_h, get_param('ff_logistic_W', self.params)) + \
          get_param('ff_logistic_b', self.params) # (t,m,n_words)
        # loss function
        a = tf.shape(logit)[0]
        b = tf.shape(logit)[1]
        c = tf.shape(logit)[2]
        logits = tf.reshape(logit, tf.pack([a*b, c]))
        labels = tf.reshape(x, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) # (t*m)
        loss_reshaped = tf.reshape(loss, tf.pack([a, b])) # (t, m)
        loss_to_optimize = tf.reduce_sum(loss_reshaped * mask, 0) # (m,)
        loss_to_optimize = tf.reduce_mean(loss_to_optimize)
        
        return loss_to_optimize
    
    def train(self):
        # init data engine
        self.engine = data_engine.Movie2Caption(
            'attention',
            self.options.dataset,
            self.options.video_feature,
            self.options.batch_size,
            self.options.valid_batch_size,
            self.options.maxlen,
            self.options.n_words,
            self.options.K,
            self.options.OutOf)
        self.options['ctx_dim'] = self.engine.ctx_dim

        self.decoder = Attention_LSTM_decoder(self.options)
        self.decoder.init_params()
        # init params
        self.init_params()

        # build model 
        cost = self.fprop()
        # optimizer
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        optimizer = tf.train.GradientDescentOptimizer(self.options.lrate)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        # start training
        with tf.Session() as session:
            tf.initialize_all_variables().run()
            uidx = 0
            for eidx in xrange(self.options['max_epochs']):
                for idx in self.engine.kf_train:
                    tags = [self.engine.train[index] for index in idx]
                    uidx += 1
                    x, mask, ctx, ctx_mask = data_engine.prepare_data(
                        self.engine, tags)
                    t0 = time.time()
                    train_cost, _ = session.run(
                        [cost, train_op],
                        feed_dict={
                            self.x: x,
                            self.mask: mask,
                            self.ctx: ctx,
                            self.mask_ctx: ctx_mask
                            })
                    print 'cost %.4f, minibatch time %.3f'%(train_cost, time.time()-t0)

def train_from_scratch(state, channel):
    t0 = time.time()
    print 'training an attention model'
    model = Attention(options, channel)
    model.train()
    print 'training time in total %.4f sec'%(time.time()-t0)

def test_att_cond_lstm():
    from config import config
    model = Attention_LSTM_decoder(config.attention)
    model.test()

def test_model():
    from config import config
    config.attention.dim_word = 64
    config.attention.dim = 32
    model = Model(config.attention)
    model.train()
    
if __name__ == '__main__':
    #test_att_cond_lstm()
    test_model()

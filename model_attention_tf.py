import os, sys, time, warnings, copy
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
        
    def fprop(self, emb, mask, context, init_state, init_memory, one_step=False):
        # in training time
        # emb (t,m,word_dim), mask (t,m), ctx (m, f, ctx_dim),
        # init_state, init_memory (m, dim)

        # in the sampling time
        # emb (m, word_dim), mask (1, 1), ctx (f, ctx_dim)
        # init_state, init_memory (1, dim)
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

        if one_step:
            # this is for the case of building the sampling function
            emb = tf.expand_dims(emb, 0) # (1,m,word_dim)
            context = tf.expand_dims(context, 0) # (1,f,ctx_dim)
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
            tt = tensor_mat_multiply(pctx_, U_att)
            tt = tf.reshape(tt, tf.pack([tf.shape(tt)[0], tf.shape(tt)[1]]))
            alpha = tt + c_att # (m,f)
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
        return final_H, final_h, final_c
    
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
        states, _, _ = self.fprop(emb, mask, ctx, init_state, init_memory) # (t,m,dim)
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
        
    def build_model(self):
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
        proj_h, _, _ = self.decoder.fprop(
            emb_shifted, mask, ctx, init_state, init_memory) # (t,m,dim)
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
    
    def build_sampler(self):
        print 'build sampler'
        ctx_s = tf.placeholder(tf.float32, name='ctx') # (f, d)
        ctx_mask_s = tf.placeholder(tf.float32, name='ctx_mask') # (f)
        counts = tf.reduce_sum(ctx_mask_s)
        ctx_mean = tf.reduce_sum(ctx_s, 0, keep_dims=True) / counts # (1, d)
        init_state = tf.matmul(ctx_mean, get_param('ff_init_state_W', self.params)) + \
          get_param('ff_init_state_b', self.params)
        init_memory = tf.matmul(ctx_mean, get_param('ff_init_memory_W', self.params)) + \
          get_param('ff_init_memory_b', self.params)
        # next word
        x_s = tf.placeholder(tf.int32, name='x') # a vector
        init_state_s = tf.placeholder(tf.float32, name='init_state')
        init_memory_s = tf.placeholder(tf.float32, name='init_memory')
        emb_ = get_param('Wemb', self.params)
        emb = control_flow_ops.cond(
            tf.equal(tf.shape(x_s)[0], 1),
            lambda: tf.zeros((1, self.options['dim_word'])),
            lambda: tf.gather(emb_, x_s))
        
        mask = tf.ones([1, 1])
        H, next_state, next_memory = self.decoder.fprop(
            emb, mask, ctx_s, init_state_s, init_memory_s, one_step=True)
        # H: (1,1,dim)
        # word prediction
        logit = tensor_mat_multiply(H, get_param('ff_logistic_W', self.params)) + \
          get_param('ff_logistic_b', self.params) # (t,m,n_words)
        # loss function
        logit = tf.squeeze(logit, [0])
        next_probs = tf.nn.softmax(logit)
        next_sample = tf.argmax(next_probs, 1)
        return [ctx_s, ctx_mask_s, x_s, init_state_s, init_memory_s,
                init_state, init_memory,
                next_probs, next_sample, next_state, next_memory] 
        
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
        cost = self.build_model()
        [ctx_s_ph, ctx_mask_s_ph, x_s_ph, init_state_s_ph, init_memory_s_ph,
         init_state_tv, init_memory_tv,
         next_probs_tv, next_sample_tv,
        next_state_tv, next_memory_tv]  = self.build_sampler()
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
                    def sample_execute(from_which):
                        print '------------- sampling from %s ----------'%from_which
                        if from_which == 'train':
                            x_s = x
                            mask_s = mask
                            ctx_s = ctx
                            ctx_mask_s = ctx_mask

                        elif from_which == 'valid':
                            idx = self.engine.kf_valid[numpy.random.randint(
                                1, len(self.engine.kf_valid) - 1)]
                            tags = [self.engine.valid[index] for index in idx]
                            x_s, mask_s, ctx_s, ctx_mask_s = data_engine.prepare_data(
                                self.engine, tags)
                        for jj in xrange(numpy.minimum(10, x_s.shape[1])):
                            sample, score = self.gen_sample(
                                ctx_s_ph, ctx_mask_s_ph, x_s_ph,
                                init_state_s_ph, init_memory_s_ph,
                                init_state_tv, init_memory_tv, next_probs_tv, next_sample_tv,
                                next_state_tv, next_memory_tv,
                                session, ctx_s[jj], ctx_mask_s[jj], k=5, maxlen=30)
                            best_one = numpy.argmin(score)
                            sample = sample[0]
                            print 'Truth ',jj,': ',
                            for vv in x_s[:,jj]:
                                if vv == 0:
                                    break
                                if vv in self.engine.word_idict:
                                    print self.engine.word_idict[vv],
                                else:
                                    print 'UNK',
                            print
                            for kk, ss in enumerate([sample]):
                                print 'Sample (', kk,') ', jj, ': ',
                                for vv in ss:
                                    if vv == 0:
                                        break
                                    if vv in self.engine.word_idict:
                                        print self.engine.word_idict[vv],
                                    else:
                                        print 'UNK',
                            print
                    sample_execute(from_which='train')
                    #sample_execute(from_which='valid')

    def gen_sample(self, ctx_s_ph, ctx_mask_s_ph, x_s_ph,
                   init_state_s_ph, init_memory_s_ph,
                   init_state_tv, init_memory_tv,
                   next_probs_tv, next_sample_tv, next_state_tv, next_memory_tv,
                   session, ctx0, ctx_mask, k, maxlen, stochastic=False):
        sample = []
        sample_score = []

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = []
        hyp_memories = []

        # [(26,1024),(512,),(512,)]
        rval  = session.run([init_state_tv, init_memory_tv],
                           feed_dict={ctx_s_ph: ctx0,
                                      ctx_mask_s_ph: ctx_mask})
        next_state = []
        next_memory = []
        n_layers_lstm = 1
        
        for lidx in xrange(n_layers_lstm):
            next_state.append(rval[lidx])
            #next_state[-1] = next_state[-1].reshape([live_k, next_state[-1].shape[0]])
        for lidx in xrange(n_layers_lstm):
            next_memory.append(rval[n_layers_lstm+lidx])
            #next_memory[-1] = next_memory[-1].reshape([live_k, next_memory[-1].shape[0]])
        next_w = -1 * numpy.ones((1,)).astype('int32')
        # next_state: [(1,512)]
        # next_memory: [(1,512)]
        for ii in xrange(maxlen):
            # return [(1, 50000), (1,), (1, 512), (1, 512)]
            # next_w: vector
            # ctx: matrix
            # ctx_mask: vector
            # next_state: [matrix]
            # next_memory: [matrix]
            rval = session.run(
                [next_probs_tv, next_sample_tv, next_state_tv, next_memory_tv],
                feed_dict={ctx_s_ph: ctx0,
                            ctx_mask_s_ph: ctx_mask,
                            x_s_ph: next_w,
                            init_state_s_ph: next_state[0],
                            init_memory_s_ph: next_memory[0]})
            next_p = rval[0]
            next_w = rval[1] # already argmax sorted
            next_state = []
            for lidx in xrange(n_layers_lstm):
                next_state.append(rval[2+lidx])
            next_memory = []
            for lidx in xrange(n_layers_lstm):
                next_memory.append(rval[2+n_layers_lstm+lidx])
            if stochastic:
                sample.append(next_w[0]) # take the most likely one
                sample_score += next_p[0,next_w[0]]
                if next_w[0] == 0:
                    break
            else:
                # the first run is (1,50000)
                cand_scores = hyp_scores[:,None] - numpy.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k-dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size # index of row
                word_indices = ranks_flat % voc_size # index of col
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
                new_hyp_states = []
                for lidx in xrange(n_layers_lstm):
                    new_hyp_states.append([])
                new_hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    new_hyp_memories.append([])

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_hyp_scores[idx] = copy.copy(costs[ti])
                    for lidx in xrange(n_layers_lstm):
                        new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                    for lidx in xrange(n_layers_lstm):
                        new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                for lidx in xrange(n_layers_lstm):
                    hyp_states.append([])
                hyp_memories = []
                for lidx in xrange(n_layers_lstm):
                    hyp_memories.append([])

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        for lidx in xrange(n_layers_lstm):
                            hyp_states[lidx].append(new_hyp_states[lidx][idx])
                        for lidx in xrange(n_layers_lstm):
                            hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
                hyp_scores = numpy.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_w = numpy.array([w[-1] for w in hyp_samples])
                next_state = []
                for lidx in xrange(n_layers_lstm):
                    next_state.append(numpy.array(hyp_states[lidx]))
                next_memory = []
                for lidx in xrange(n_layers_lstm):
                    next_memory.append(numpy.array(hyp_memories[lidx]))

        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])
        return sample, sample_score
        
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

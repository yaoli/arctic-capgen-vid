'''
Build a soft-attention-based video caption generator
'''
import theano
import theano.tensor as tensor
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy
import os, sys, socket, shutil
import time
import warnings
from collections import OrderedDict

from sklearn.cross_validation import KFold
from scipy import optimize, stats

import data_engine
import metrics
import common


from common import *

base_path = None
hostname = socket.gethostname()
lscratch_dir = None

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

def validate_options(options):
    if options['ctx2out']:
        warnings.warn('Feeding context to output directly seems to hurt.')
    if options['dim_word'] > options['dim']:
        warnings.warn('dim_word should only be as large as dim.')
    return options

class Attention(object):
    def __init__(self, channel=None):
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'ff': ('self.param_init_fflayer', 'self.fflayer'),
            'lstm': ('self.param_init_lstm', 'self.lstm_layer'),
            'lstm_cond': ('self.param_init_lstm_cond', 'self.lstm_cond_layer'),
            }
        self.channel = channel

    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return (eval(fns[0]), eval(fns[1]))

    def load_params(self, path, params):
        # load params from disk
        pp = numpy.load(path)
        for kk, vv in params.iteritems():
            if kk not in pp:
                raise Warning('%s is not in the archive'%kk)
            params[kk] = pp[kk]

        return params

    def init_tparams(self, params, force_cpu=False):
        # initialize Theano shared variables according to the initial parameters
        tparams = OrderedDict()
        for kk, pp in params.iteritems():
            if force_cpu:
                tparams[kk] = theano.tensor._shared(params[kk], name=kk)
            else:
                tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams

    def param_init_fflayer(self, options, params, prefix='ff', nin=None, nout=None):
        if nin == None:
            nin = options['dim_proj']
        if nout == None:
            nout = options['dim_proj']
        params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01)
        params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')
        return params

    def fflayer(self, tparams, state_below, options,
                prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
        return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[
            _p(prefix,'b')])

    # LSTM layer
    def param_init_lstm(self, options, params, prefix=None, nin=None, dim=None):
        assert prefix is not None
        if nin == None:
            nin = options['dim_proj']
        if dim == None:
            dim = options['dim_proj']
        # Stack the weight matricies for faster dot prods
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix,'U')] = U
        params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

        return params

    # This function implements the lstm fprop
    def lstm_layer(self, tparams, state_below, options, prefix='lstm', mask=None,
                   forget=False, use_noise=None, trng=None, **kwargs):
        nsteps = state_below.shape[0]
        dim = tparams[_p(prefix,'U')].shape[0]

        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
            init_state = tensor.alloc(0., n_samples, dim)
            init_memory = tensor.alloc(0., n_samples, dim)
        else:
            n_samples = 1
            init_state = tensor.alloc(0., dim)
            init_memory = tensor.alloc(0., dim)

        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            elif _x.ndim == 2:
                return _x[:, n*dim:(n+1)*dim]
            return _x[n*dim:(n+1)*dim]

        def _step(m_, x_, h_, c_, U, b):
            preact = tensor.dot(h_, U)
            preact += x_
            preact += b

            i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
            o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
            c = tensor.tanh(_slice(preact, 3, dim))

            if forget:
                f = T.zeros_like(f)
            c = f * c_ + i * c
            h = o * tensor.tanh(c)
            if m_.ndim == 0:
                # when using this for minibatchsize=1
                h = m_ * h + (1. - m_) * h_
                c = m_ * c + (1. - m_) * c_
            else:
                h = m_[:,None] * h + (1. - m_)[:,None] * h_
                c = m_[:,None] * c + (1. - m_)[:,None] * c_
            return h, c, i, f, o, preact

        state_below = tensor.dot(
            state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
        U = tparams[_p(prefix, 'U')]
        b = tparams[_p(prefix, 'b')]
        rval, updates = theano.scan(
            _step,
            sequences=[mask, state_below],
            non_sequences=[U,b],
            outputs_info = [init_state, init_memory, None, None, None, None],
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True,
            profile=False)
        return rval

    # Conditional LSTM layer with Attention
    def param_init_lstm_cond(self, options, params,
                             prefix='lstm_cond', nin=None, dim=None, dimctx=None):
        if nin == None:
            nin = options['dim']
        if dim == None:
            dim = options['dim']
        if dimctx == None:
            dimctx = options['dim']
        # input to LSTM
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W

        # LSTM to LSTM
        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix,'U')] = U

        # bias to LSTM
        params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

        # context to LSTM
        Wc = norm_weight(dimctx,dim*4)
        params[_p(prefix,'Wc')] = Wc

        # attention: context -> hidden
        Wc_att = norm_weight(dimctx, ortho=False)
        params[_p(prefix,'Wc_att')] = Wc_att

        # attention: LSTM -> hidden
        Wd_att = norm_weight(dim,dimctx)
        params[_p(prefix,'Wd_att')] = Wd_att

        # attention: hidden bias
        b_att = numpy.zeros((dimctx,)).astype('float32')
        params[_p(prefix,'b_att')] = b_att

        # attention:
        U_att = norm_weight(dimctx,1)
        params[_p(prefix,'U_att')] = U_att
        c_att = numpy.zeros((1,)).astype('float32')
        params[_p(prefix, 'c_tt')] = c_att

        if options['selector']:
            # attention: selector
            W_sel = norm_weight(dim, 1)
            params[_p(prefix, 'W_sel')] = W_sel
            b_sel = numpy.float32(0.)
            params[_p(prefix, 'b_sel')] = b_sel

        return params

    def lstm_cond_layer(self, tparams, state_below, options, prefix='lstm',
                        mask=None, context=None, one_step=False,
                        init_memory=None, init_state=None,
                        trng=None, use_noise=None,mode=None,
                        **kwargs):
        # state_below (t, m, dim_word), or (m, dim_word) in sampling
        # mask (t, m)
        # context (m, f, dim_ctx), or (f, dim_word) in sampling
        # init_memory, init_state (m, dim)
        assert context, 'Context must be provided'

        if one_step:
            assert init_memory, 'previous memory must be provided'
            assert init_state, 'previous state must be provided'

        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        # mask
        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        dim = tparams[_p(prefix, 'U')].shape[0]

        # initial/previous state
        if init_state == None:
            init_state = tensor.alloc(0., n_samples, dim)
        # initial/previous memory
        if init_memory == None:
            init_memory = tensor.alloc(0., n_samples, dim)

        # projected context
        pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[
                _p(prefix, 'b_att')]
        if one_step:
            # tensor.dot will remove broadcasting dim
            pctx_ = T.addbroadcast(pctx_,0)
        # projected x
        state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[
            _p(prefix, 'b')]

        Wd_att = tparams[_p(prefix,'Wd_att')]
        U_att = tparams[_p(prefix,'U_att')]
        c_att = tparams[_p(prefix, 'c_tt')]
        if options['selector']:
            W_sel = tparams[_p(prefix, 'W_sel')]
            b_sel = tparams[_p(prefix,'b_sel')]
        else:
            W_sel = T.alloc(0., 1)
            b_sel = T.alloc(0., 1)
        U = tparams[_p(prefix, 'U')]
        Wc = tparams[_p(prefix, 'Wc')]
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        def _step(m_, x_, # sequences
                  h_, c_, a_, ct_, # outputs_info
                  pctx_, ctx_, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc, # non_sequences
                  dp_=None, dp_att_=None):
            # attention
            pstate_ = tensor.dot(h_, Wd_att)
            pctx_ = pctx_ + pstate_[:,None,:]
            pctx_list = []
            pctx_list.append(pctx_)
            pctx_ = tanh(pctx_)
            
            alpha = tensor.dot(pctx_, U_att)+c_att
            alpha_pre = alpha
            alpha_shp = alpha.shape
            alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
            ctx_ = (context * alpha[:,:,None]).sum(1) # (m,ctx_dim)
            if options['selector']:
                sel_ = tensor.nnet.sigmoid(tensor.dot(h_, W_sel) + b_sel)
                sel_ = sel_.reshape([sel_.shape[0]])
                ctx_ = sel_[:,None] * ctx_
            preact = tensor.dot(h_, U)
            preact += x_
            preact += tensor.dot(ctx_, Wc)

            i = _slice(preact, 0, dim)
            f = _slice(preact, 1, dim)
            o = _slice(preact, 2, dim)
            if options['use_dropout']:
                i = i * _slice(dp_, 0, dim)
                f = f * _slice(dp_, 1, dim)
                o = o * _slice(dp_, 2, dim)
            i = tensor.nnet.sigmoid(i)
            f = tensor.nnet.sigmoid(f)
            o = tensor.nnet.sigmoid(o)
            c = tensor.tanh(_slice(preact, 3, dim))

            c = f * c_ + i * c
            c = m_[:,None] * c + (1. - m_)[:,None] * c_

            h = o * tensor.tanh(c)
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
            rval = [h, c, alpha, ctx_, pstate_, pctx_, i, f, o, preact, alpha_pre]+pctx_list
            return rval
        if options['use_dropout']:
            _step0 = lambda m_, x_, dp_, h_, c_, \
                a_, ct_, \
                pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc: _step(
                m_, x_, h_, c_,
                a_, ct_,
                pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc,  dp_)
            dp_shape = state_below.shape
            if one_step:
                dp_mask = tensor.switch(use_noise,
                                        trng.binomial((dp_shape[0], 3*dim),
                                                      p=0.5, n=1, dtype=state_below.dtype),
                                        tensor.alloc(0.5, dp_shape[0], 3 * dim))
            else:
                dp_mask = tensor.switch(use_noise,
                                        trng.binomial((dp_shape[0], dp_shape[1], 3*dim),
                                                      p=0.5, n=1, dtype=state_below.dtype),
                                        tensor.alloc(0.5, dp_shape[0], dp_shape[1], 3*dim))
        else:
            _step0 = lambda m_, x_, h_, c_, \
                a_, ct_, pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc: _step(
                m_, x_, h_, c_, a_, ct_, pctx_, context,
                    Wd_att, U_att, c_att, W_sel, b_sel, U, Wc)

        if one_step:
            if options['use_dropout']:
                rval = _step0(
                    mask, state_below, dp_mask, init_state, init_memory, None, None,
                    pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc)
            else:
                rval = _step0(mask, state_below, init_state, init_memory, None, None,
                              pctx_, context, Wd_att, U_att, c_att, W_sel, b_sel, U, Wc)
        else:
            seqs = [mask, state_below]
            if options['use_dropout']:
                seqs += [dp_mask]
            rval, updates = theano.scan(
                _step0,
                sequences=seqs,
                outputs_info = [init_state,
                                init_memory,
                                tensor.alloc(0., n_samples, pctx_.shape[1]),
                                tensor.alloc(0., n_samples, context.shape[2]),
                                None, None, None, None, None, None, None, None],
                                non_sequences=[pctx_, context,
                                               Wd_att, U_att, c_att, W_sel, b_sel, U, Wc],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False, mode=mode, strict=True)

        return rval
    """---------------------------------------------------------------------------------"""
    """---------------------------------------------------------------------------------"""
    """---------------------------------------------------------------------------------"""
    def init_params(self, options):
        # all parameters
        params = OrderedDict()
        # embedding
        params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

        if options['encoder'] == 'lstm_bi':
            print 'bi-directional lstm encoder on ctx'
            params = self.get_layer('lstm')[0](options, params, prefix='encoder',
                                          nin=options['ctx_dim'], dim=options['encoder_dim'])
            params = self.get_layer('lstm')[0](options, params, prefix='encoder_rev',
                                          nin=options['ctx_dim'], dim=options['encoder_dim'])
            ctx_dim = options['encoder_dim'] * 2 + options['ctx_dim']

        elif options['encoder'] == 'lstm_uni':
            print 'uni-directional lstm encoder on ctx'
            params = self.get_layer('lstm')[0](options, params, prefix='encoder',
                                          nin=options['ctx_dim'], dim=options['dim'])
            ctx_dim = options['dim']

        else:
            print 'no lstm on ctx'
            ctx_dim = options['ctx_dim']

        # init_state, init_cell
        for lidx in xrange(options['n_layers_init']):
            params = self.get_layer('ff')[0](
                options, params, prefix='ff_init_%d'%lidx, nin=ctx_dim, nout=ctx_dim)
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_state', nin=ctx_dim, nout=options['dim'])
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_memory', nin=ctx_dim, nout=options['dim'])
        # decoder: LSTM
        params = self.get_layer('lstm_cond')[0](options, params, prefix='decoder',
                                           nin=options['dim_word'], dim=options['dim'],
                                           dimctx=ctx_dim)
        
        # readout
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_logit_lstm',
            nin=options['dim'], nout=options['dim_word'])
        if options['ctx2out']:
            params = self.get_layer('ff')[0](
                options, params, prefix='ff_logit_ctx',
                nin=ctx_dim, nout=options['dim_word'])
        if options['n_layers_out'] > 1:
            for lidx in xrange(1, options['n_layers_out']):
                params = self.get_layer('ff')[0](
                    options, params, prefix='ff_logit_h%d'%lidx,
                    nin=options['dim_word'], nout=options['dim_word'])
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_logit',
            nin=options['dim_word'], nout=options['n_words'])
        return params

    def build_model(self, tparams, options):
        trng = RandomStreams(1234)
        use_noise = theano.shared(numpy.float32(0.))
        # description string: #words x #samples
        x = tensor.matrix('x', dtype='int64')
        x.tag.test_value = self.x_tv
        mask = tensor.matrix('mask', dtype='float32')
        mask.tag.test_value = self.mask_tv
        # context: #samples x #annotations x dim
        ctx = tensor.tensor3('ctx', dtype='float32')
        ctx.tag.test_value = self.ctx_tv
        mask_ctx = tensor.matrix('mask_ctx', dtype='float32')
        mask_ctx.tag.test_value = self.ctx_mask_tv
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # index into the word embedding matrix, shift it forward in time
        emb = tparams['Wemb'][x.flatten()].reshape(
            [n_timesteps, n_samples, options['dim_word']])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted
        counts = mask_ctx.sum(-1).dimshuffle(0,'x')

        ctx_ = ctx

        if options['encoder'] == 'lstm_bi':
            # encoder
            ctx_fwd = self.get_layer('lstm')[1](tparams, ctx_.dimshuffle(1,0,2),
                                           options, mask=mask_ctx.dimshuffle(1,0),
                                           prefix='encoder')[0]
            ctx_rev = self.get_layer('lstm')[1](tparams, ctx_.dimshuffle(1,0,2)[::-1],
                                                options, mask=mask_ctx.dimshuffle(1,0)[::-1],
                                                prefix='encoder_rev')[0]
            ctx0 = concatenate((ctx_fwd, ctx_rev[::-1]), axis=2)
            ctx0 = ctx0.dimshuffle(1,0,2)
            ctx0 = concatenate((ctx_, ctx0), axis=2)
            ctx_mean = ctx0.sum(1)/counts

        elif options['encoder'] == 'lstm_uni':
            ctx0 = self.get_layer('lstm')[1](tparams, ctx_.dimshuffle(1,0,2),
                                           options,
                                           mask=mask_ctx.dimshuffle(1,0),
                                           prefix='encoder')[0]
            ctx0 = ctx0.dimshuffle(1,0,2)
            ctx_mean = ctx0.sum(1)/counts

        else:
            ctx0 = ctx_
            ctx_mean = ctx0.sum(1)/counts
        # initial state/cell
        for lidx in xrange(options['n_layers_init']):
            ctx_mean = self.get_layer('ff')[1](
                tparams, ctx_mean, options, prefix='ff_init_%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                ctx_mean = dropout_layer(ctx_mean, use_noise, trng)

        init_state = self.get_layer('ff')[1](
            tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
        init_memory = self.get_layer('ff')[1](
            tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')
        # decoder
        proj = self.get_layer('lstm_cond')[1](tparams, emb, options,
                                         prefix='decoder',
                                         mask=mask, context=ctx0,
                                         one_step=False,
                                         init_state=init_state,
                                         init_memory=init_memory,
                                         trng=trng,
                                         use_noise=use_noise)

        proj_h = proj[0]
        alphas = proj[2]
        ctxs = proj[3]
        if options['use_dropout']:
            proj_h = dropout_layer(proj_h, use_noise, trng)
        # compute word probabilities
        logit = self.get_layer('ff')[1](
            tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
        if options['prev2out']:
            logit += emb
        if options['ctx2out']:
            logit += self.get_layer('ff')[1](
                tparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')
        logit = tanh(logit)
        if options['use_dropout']:
            logit = dropout_layer(logit, use_noise, trng)
        if options['n_layers_out'] > 1:
            for lidx in xrange(1, options['n_layers_out']):
                logit = self.get_layer('ff')[1](
                    tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
                if options['use_dropout']:
                    logit = dropout_layer(logit, use_noise, trng)
        # (t,m,n_words)
        logit = self.get_layer('ff')[1](
            tparams, logit, options, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape
        # (t*m, n_words)
        probs = tensor.nnet.softmax(
            logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))
        # cost
        x_flat = x.flatten() # (t*m,)
        cost = -tensor.log(probs[T.arange(x_flat.shape[0]), x_flat] + 1e-8)

        cost = cost.reshape([x.shape[0], x.shape[1]])
        cost = (cost * mask).sum(0)
        extra = [probs, alphas]
        return trng, use_noise, x, mask, ctx, mask_ctx, alphas, cost, extra

    def build_sampler(self, tparams, options, use_noise, trng, mode=None):
        # context: #annotations x dim
        ctx0 = tensor.matrix('ctx_sampler', dtype='float32')
        #ctx0.tag.test_value = numpy.random.uniform(size=(50,1024)).astype('float32')
        ctx_mask = tensor.vector('ctx_mask', dtype='float32')
        #ctx_mask.tag.test_value = numpy.random.binomial(n=1,p=0.5,size=(50,)).astype('float32')


        ctx_ = ctx0
        counts = ctx_mask.sum(-1)

        if options['encoder'] == 'lstm_bi':
            # encoder
            ctx_fwd = self.get_layer('lstm')[1](tparams, ctx_,
                                           options, mask=ctx_mask,
                                           prefix='encoder',forget=False)[0]
            ctx_rev = self.get_layer('lstm')[1](tparams, ctx_[::-1],
                                           options, mask=ctx_mask[::-1],
                                           forget=False,
                                           prefix='encoder_rev')[0]
            ctx = concatenate((ctx_fwd, ctx_rev[::-1]), axis=1)
            ctx = concatenate((ctx_, ctx),axis=1)
            ctx_mean = ctx.sum(0)/counts
            ctx = ctx.dimshuffle('x',0,1)
        elif options['encoder'] == 'lstm_uni':
            ctx = self.get_layer('lstm')[1](tparams, ctx_,
                                           options,
                                           mask=ctx_mask,
                                           prefix='encoder')[0]
            ctx_mean = ctx.sum(0)/counts
            ctx = ctx.dimshuffle('x',0,1)
        else:
            # do not use RNN encoder
            ctx = ctx_
            ctx_mean = ctx.sum(0)/counts
            #ctx_mean = ctx.mean(0)
            ctx = ctx.dimshuffle('x',0,1)
        # initial state/cell
        for lidx in xrange(options['n_layers_init']):
            ctx_mean = self.get_layer('ff')[1](
                tparams, ctx_mean, options, prefix='ff_init_%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                ctx_mean = dropout_layer(ctx_mean, use_noise, trng)
        init_state = [self.get_layer('ff')[1](
            tparams, ctx_mean, options, prefix='ff_state', activ='tanh')]
        init_memory = [self.get_layer('ff')[1](
            tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')]
        

        print 'Building f_init...',
        f_init = theano.function(
            [ctx0, ctx_mask],
            [ctx0]+init_state+init_memory, name='f_init',
            on_unused_input='ignore',
            profile=False, mode=mode)
        print 'Done'

        x = tensor.vector('x_sampler', dtype='int64')
        init_state = [tensor.matrix('init_state', dtype='float32')]
        init_memory = [tensor.matrix('init_memory', dtype='float32')]
        

        # if it's the first word, emb should be all zero
        emb = tensor.switch(x[:,None] < 0, tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                            tparams['Wemb'][x])

        proj = self.get_layer('lstm_cond')[1](tparams, emb, options,
                                         prefix='decoder',
                                         mask=None, context=ctx,
                                         one_step=True,
                                         init_state=init_state[0],
                                         init_memory=init_memory[0],
                                         trng=trng,
                                         use_noise=use_noise,
                                         mode=mode)
        next_state, next_memory, ctxs = [proj[0]], [proj[1]], [proj[3]]

        if options['use_dropout']:
            proj_h = dropout_layer(proj[0], use_noise, trng)
        else:
            proj_h = proj[0]
        logit = self.get_layer('ff')[1](
            tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
        if options['prev2out']:
            logit += emb
        if options['ctx2out']:
            logit += self.get_layer('ff')[1](
                tparams, ctxs[-1], options, prefix='ff_logit_ctx', activ='linear')
        logit = tanh(logit)
        if options['use_dropout']:
            logit = dropout_layer(logit, use_noise, trng)
        if options['n_layers_out'] > 1:
            for lidx in xrange(1, options['n_layers_out']):
                logit = self.get_layer('ff')[1](
                    tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
                if options['use_dropout']:
                    logit = dropout_layer(logit, use_noise, trng)
        logit = self.get_layer('ff')[1](
            tparams, logit, options, prefix='ff_logit', activ='linear')
        logit_shp = logit.shape
        next_probs = tensor.nnet.softmax(logit)
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # next word probability
        print 'building f_next...'
        f_next = theano.function(
            [x, ctx0, ctx_mask]+init_state+init_memory,
            [next_probs, next_sample]+next_state+next_memory,
            name='f_next', profile=False, mode=mode, on_unused_input='ignore')
        print 'Done'
        return f_init, f_next

    def gen_sample(self, tparams, f_init, f_next, ctx0, ctx_mask, options,
                   trng=None, k=1, maxlen=30, stochastic=False,
                   restrict_voc=False):
        '''
        ctx0: (26,1024)
        ctx_mask: (26,)

        restrict_voc: set the probability of outofvoc words with 0, renormalize
        '''

        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling'

        sample = []
        sample_score = []
        if stochastic:
            sample_score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = []
        hyp_memories = []

        # [(26,1024),(512,),(512,)]
        rval = f_init(ctx0, ctx_mask)
        ctx0 = rval[0]

        next_state = []
        next_memory = []
        n_layers_lstm = 1
        
        for lidx in xrange(n_layers_lstm):
            next_state.append(rval[1+lidx])
            next_state[-1] = next_state[-1].reshape([live_k, next_state[-1].shape[0]])
        for lidx in xrange(n_layers_lstm):
            next_memory.append(rval[1+n_layers_lstm+lidx])
            next_memory[-1] = next_memory[-1].reshape([live_k, next_memory[-1].shape[0]])
        next_w = -1 * numpy.ones((1,)).astype('int64')
        # next_state: [(1,512)]
        # next_memory: [(1,512)]
        for ii in xrange(maxlen):
            # return [(1, 50000), (1,), (1, 512), (1, 512)]
            # next_w: vector
            # ctx: matrix
            # ctx_mask: vector
            # next_state: [matrix]
            # next_memory: [matrix]
            rval = f_next(*([next_w, ctx0, ctx_mask]+next_state+next_memory))
            next_p = rval[0]
            if restrict_voc:
                raise NotImplementedError()
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

        return sample, sample_score, next_state, next_memory

    def pred_probs(self, whichset, f_log_probs, verbose=True):
        
        probs = []
        n_done = 0
        NLL = []
        L = []
        if whichset == 'train':
            tags = self.engine.train
            iterator = self.engine.kf_train
        elif whichset == 'valid':
            tags = self.engine.valid
            iterator = self.engine.kf_valid
        elif whichset == 'test':
            tags = self.engine.test
            iterator = self.engine.kf_test
        else:
            raise NotImplementedError()
        n_samples = numpy.sum([len(index) for index in iterator])
        for index in iterator:
            tag = [tags[i] for i in index]
            x, mask, ctx, ctx_mask = data_engine.prepare_data(
                self.engine, tag)
            pred_probs = f_log_probs(x, mask, ctx, ctx_mask)
            L.append(mask.sum(0).tolist())
            NLL.append((-1 * pred_probs).tolist())
            probs.append(pred_probs.tolist())
            n_done += len(tag)
            if verbose:
                sys.stdout.write('\rComputing LL on %d/%d examples'%(
                             n_done, n_samples))
                sys.stdout.flush()
        print
        probs = common.flatten_list_of_list(probs)
        NLL = common.flatten_list_of_list(NLL)
        L = common.flatten_list_of_list(L)
        perp = 2**(numpy.sum(NLL) / numpy.sum(L) / numpy.log(2))
        return -1 * numpy.mean(probs), perp

    def train(self,
              random_seed=1234,
              dim_word=256, # word vector dimensionality
              ctx_dim=-1, # context vector dimensionality, auto set
              dim=1000, # the number of LSTM units
              n_layers_out=1,
              n_layers_init=1,
              encoder='none',
              encoder_dim=100,
              prev2out=False,
              ctx2out=False,
              patience=10,
              max_epochs=5000,
              dispFreq=100,
              decay_c=0.,
              alpha_c=0.,
              alpha_entropy_r=0.,
              lrate=0.01,
              selector=False,
              n_words=100000,
              maxlen=100, # maximum length of the description
              optimizer='adadelta',
              clip_c=2.,
              batch_size = 64,
              valid_batch_size = 64,
              save_model_dir='/data/lisatmp3/yaoli/exp/capgen_vid/attention/test/',
              validFreq=10,
              saveFreq=10, # save the parameters after every saveFreq updates
              sampleFreq=10, # generate some samples after every sampleFreq updates
              metric='blue',
              dataset='youtube2text',
              video_feature='googlenet',
              use_dropout=False,
              reload_=False,
              from_dir=None,
              K=10,
              OutOf=240,
              verbose=True,
              debug=True
              ):
        self.rng_numpy, self.rng_theano = common.get_two_rngs()

        model_options = locals().copy()
        if 'self' in model_options:
            del model_options['self']
        model_options = validate_options(model_options)
        with open('%smodel_options.pkl'%save_model_dir, 'wb') as f:
            pkl.dump(model_options, f)

        print 'Loading data'
        self.engine = data_engine.Movie2Caption('attention', dataset,
                                           video_feature,
                                           batch_size, valid_batch_size,
                                           maxlen, n_words,
                                           K, OutOf)
        model_options['ctx_dim'] = self.engine.ctx_dim

        # set test values, for debugging
        idx = self.engine.kf_train[0]
        [self.x_tv, self.mask_tv,
         self.ctx_tv, self.ctx_mask_tv] = data_engine.prepare_data(
            self.engine, [self.engine.train[index] for index in idx])

        print 'init params'
        t0 = time.time()
        params = self.init_params(model_options)

        # reloading
        if reload_:
            model_saved = from_dir+'/model_best_so_far.npz'
            assert os.path.isfile(model_saved)
            print "Reloading model params..."
            params = load_params(model_saved, params)

        tparams = init_tparams(params)

        trng, use_noise, \
              x, mask, ctx, mask_ctx, alphas, \
              cost, extra = \
              self.build_model(tparams, model_options)

        print 'buliding sampler'
        f_init, f_next = self.build_sampler(tparams, model_options, use_noise, trng)
        # before any regularizer
        print 'building f_log_probs'
        f_log_probs = theano.function([x, mask, ctx, mask_ctx], -cost,
                                      profile=False, on_unused_input='ignore')

        cost = cost.mean()
        if decay_c > 0.:
            decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for kk, vv in tparams.iteritems():
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            cost += weight_decay

        if alpha_c > 0.:
            alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
            alpha_reg = alpha_c * ((1.-alphas.sum(0))**2).sum(0).mean()
            cost += alpha_reg

        if alpha_entropy_r > 0:
            alpha_entropy_r = theano.shared(numpy.float32(alpha_entropy_r),
                                            name='alpha_entropy_r')
            alpha_reg_2 = alpha_entropy_r * (-tensor.sum(alphas *
                        tensor.log(alphas+1e-8),axis=-1)).sum(0).mean()
            cost += alpha_reg_2
        else:
            alpha_reg_2 = tensor.zeros_like(cost)
        print 'building f_alpha'
        f_alpha = theano.function([x, mask, ctx, mask_ctx],
                                  [alphas, alpha_reg_2],
                                  name='f_alpha',
                                  on_unused_input='ignore')

        print 'compute grad'
        grads = tensor.grad(cost, wrt=itemlist(tparams))
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g**2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(tensor.switch(g2 > (clip_c**2),
                                               g / tensor.sqrt(g2) * clip_c,
                                               g))
            grads = new_grads

        lr = tensor.scalar(name='lr')
        print 'build train fns'
        f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads,
                                                  [x, mask, ctx, mask_ctx], cost,
                                                  extra + grads)

        print 'compilation took %.4f sec'%(time.time()-t0)
        print 'Optimization'

        history_errs = []
        # reload history
        if reload_:
            print 'loading history error...'
            history_errs = numpy.load(
                from_dir+'model_best_so_far.npz')['history_errs'].tolist()

        bad_counter = 0

        processes = None
        queue = None
        rqueue = None
        shared_params = None

        uidx = 0
        uidx_best_blue = 0
        uidx_best_valid_err = 0
        estop = False
        best_p = unzip(tparams)
        best_blue_valid = 0
        best_valid_err = 999
        alphas_ratio = []
        for eidx in xrange(max_epochs):
            n_samples = 0
            train_costs = []
            grads_record = []
            print 'Epoch ', eidx
            for idx in self.engine.kf_train:
                tags = [self.engine.train[index] for index in idx]
                n_samples += len(tags)
                uidx += 1
                use_noise.set_value(1.)

                pd_start = time.time()
                x, mask, ctx, ctx_mask = data_engine.prepare_data(
                    self.engine, tags)
                pd_duration = time.time() - pd_start
                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    continue

                ud_start = time.time()
                rvals = f_grad_shared(x, mask, ctx, ctx_mask)
                cost = rvals[0]
                probs = rvals[1]
                alphas = rvals[2]
                grads = rvals[3:]
                grads, NaN_keys = grad_nan_report(grads, tparams)
                if len(grads_record) >= 5:
                    del grads_record[0]
                grads_record.append(grads)
                if NaN_keys != []:
                    print 'grads contain NaN'
                    import pdb; pdb.set_trace()
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected in cost'
                    import pdb; pdb.set_trace()
                # update params
                f_update(lrate)
                ud_duration = time.time() - ud_start

                if eidx == 0:
                    train_error = cost
                else:
                    train_error = train_error * 0.95 + cost * 0.05
                train_costs.append(cost)

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Train cost mean so far', \
                      train_error, 'fetching data time spent (sec)', pd_duration, \
                      'update time spent (sec)', ud_duration, 'save_dir', save_model_dir
                    alphas,reg = f_alpha(x,mask,ctx,ctx_mask)
                    print 'alpha ratio %.3f, reg %.3f'%(
                        alphas.min(-1).mean() / (alphas.max(-1)).mean(), reg)
                if numpy.mod(uidx, saveFreq) == 0:
                    pass

                if numpy.mod(uidx, sampleFreq) == 0:
                    use_noise.set_value(0.)
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

                        stochastic = False
                        for jj in xrange(numpy.minimum(10, x_s.shape[1])):
                            sample, score, _, _ = self.gen_sample(
                                tparams, f_init, f_next, ctx_s[jj], ctx_mask_s[jj],
                                model_options,
                                trng=trng, k=5, maxlen=30, stochastic=stochastic)
                            if not stochastic:
                                best_one = numpy.argmin(score)
                                sample = sample[0]
                            else:
                                sample = sample
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
                    sample_execute(from_which='valid')

                if validFreq != -1 and numpy.mod(uidx, validFreq) == 0:
                    t0_valid = time.time()
                    alphas,_ = f_alpha(x, mask, ctx, ctx_mask)
                    ratio = alphas.min(-1).mean()/(alphas.max(-1)).mean()
                    alphas_ratio.append(ratio)
                    numpy.savetxt(save_model_dir+'alpha_ratio.txt',alphas_ratio)

                    current_params = unzip(tparams)
                    numpy.savez(
                             save_model_dir+'model_current.npz',
                             history_errs=history_errs, **current_params)

                    use_noise.set_value(0.)
                    train_err = -1
                    train_perp = -1
                    valid_err = -1
                    valid_perp = -1
                    test_err = -1
                    test_perp = -1
                    if not debug:
                        # first compute train cost
                        if 1:
                            print 'computing cost on trainset'
                            train_err, train_perp = self.pred_probs(
                                    'train', f_log_probs,
                                    verbose=model_options['verbose'])
                        else:
                            train_err = 0.
                            train_perp = 0.
                        if 1:
                            print 'validating...'
                            valid_err, valid_perp = self.pred_probs(
                                'valid', f_log_probs,
                                verbose=model_options['verbose'],
                                )
                        else:
                            valid_err = 0.
                            valid_perp = 0.
                        if 1:
                            print 'testing...'
                            test_err, test_perp = self.pred_probs(
                                'test', f_log_probs,
                                verbose=model_options['verbose']
                                )
                        else:
                            test_err = 0.
                            test_perp = 0.

                    mean_ranking = 0
                    blue_t0 = time.time()
                    scores, processes, queue, rqueue, shared_params = \
                        metrics.compute_score(
                        model_type='attention',
                        model_archive=current_params,
                        options=model_options,
                        engine=self.engine,
                        save_dir=save_model_dir,
                        beam=5, n_process=5,
                        whichset='both',
                        on_cpu=False,
                        processes=processes, queue=queue, rqueue=rqueue,
                        shared_params=shared_params, metric=metric,
                        one_time=False,
                        f_init=f_init, f_next=f_next, model=self
                        )
                    '''
                     {'blue': {'test': [-1], 'valid': [77.7, 60.5, 48.7, 38.5, 38.3]},
                     'alternative_valid': {'Bleu_3': 0.40702270203174923,
                     'Bleu_4': 0.29276570520368456,
                     'CIDEr': 0.25247168210607884,
                     'Bleu_2': 0.529069629270047,
                     'Bleu_1': 0.6804308797115253,
                     'ROUGE_L': 0.51083584331688392},
                     'meteor': {'test': [-1], 'valid': [0.282787550236724]}}
                    '''

                    valid_B1 = scores['valid']['Bleu_1']
                    valid_B2 = scores['valid']['Bleu_2']
                    valid_B3 = scores['valid']['Bleu_3']
                    valid_B4 = scores['valid']['Bleu_4']
                    valid_Rouge = scores['valid']['ROUGE_L']
                    valid_Cider = scores['valid']['CIDEr']
                    valid_meteor = scores['valid']['METEOR']
                    test_B1 = scores['test']['Bleu_1']
                    test_B2 = scores['test']['Bleu_2']
                    test_B3 = scores['test']['Bleu_3']
                    test_B4 = scores['test']['Bleu_4']
                    test_Rouge = scores['test']['ROUGE_L']
                    test_Cider = scores['test']['CIDEr']
                    test_meteor = scores['test']['METEOR']
                    print 'computing meteor/blue score used %.4f sec, '\
                      'blue score: %.1f, meteor score: %.1f'%(
                    time.time()-blue_t0, valid_B4, valid_meteor)
                    history_errs.append([eidx, uidx, train_err, train_perp,
                                         valid_perp, test_perp,
                                         valid_err, test_err,
                                         valid_B1, valid_B2, valid_B3,
                                         valid_B4, valid_meteor, valid_Rouge, valid_Cider,
                                         test_B1, test_B2, test_B3,
                                         test_B4, test_meteor, test_Rouge, test_Cider])
                    numpy.savetxt(save_model_dir+'train_valid_test.txt',
                                  history_errs, fmt='%.3f')
                    print 'save validation results to %s'%save_model_dir
                    # save best model according to the best blue or meteor
                    if len(history_errs) > 1 and \
                      valid_B4 > numpy.array(history_errs)[:-1,11].max():
                        print 'Saving to %s...'%save_model_dir,
                        numpy.savez(
                            save_model_dir+'model_best_blue_or_meteor.npz',
                            history_errs=history_errs, **best_p)
                    if len(history_errs) > 1 and \
                      valid_err < numpy.array(history_errs)[:-1,6].min():
                        best_p = unzip(tparams)
                        bad_counter = 0
                        best_valid_err = valid_err
                        uidx_best_valid_err = uidx

                        print 'Saving to %s...'%save_model_dir,
                        numpy.savez(
                            save_model_dir+'model_best_so_far.npz',
                            history_errs=history_errs, **best_p)
                        with open('%smodel_options.pkl'%save_model_dir, 'wb') as f:
                            pkl.dump(model_options, f)
                        print 'Done'
                    elif len(history_errs) > 1 and \
                        valid_err >= numpy.array(history_errs)[:-1,6].min():
                        bad_counter += 1
                        print 'history best ',numpy.array(history_errs)[:,6].min()
                        print 'bad_counter ',bad_counter
                        print 'patience ',patience
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    if self.channel:
                        self.channel.save()

                    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err, \
                      'best valid err so far',best_valid_err
                    print 'valid took %.2f sec'%(time.time() - t0_valid)
                    # end of validatioin
                if debug:
                    break
            if estop:
                break
            if debug:
                break

            # end for loop over minibatches
            print 'This epoch has seen %d samples, train cost %.2f'%(
                n_samples, numpy.mean(train_costs))
        # end for loop over epochs
        print 'Optimization ended.'
        if best_p is not None:
            zipp(best_p, tparams)

        use_noise.set_value(0.)
        valid_err = 0
        test_err = 0
        if not debug:
            #if valid:
            valid_err, valid_perp = self.pred_probs(
                'valid', f_log_probs,
                verbose=model_options['verbose'])
            #if test:
            #test_err, test_perp = self.pred_probs(
            #    'test', f_log_probs,
            #    verbose=model_options['verbose'])


        print 'stopped at epoch %d, minibatch %d, '\
          'curent Train %.2f, current Valid %.2f, current Test %.2f '%(
              eidx,uidx,numpy.mean(train_err),numpy.mean(valid_err),numpy.mean(test_err))
        params = copy.copy(best_p)
        numpy.savez(save_model_dir+'model_best.npz',
                    train_err=train_err,
                    valid_err=valid_err, test_err=test_err, history_errs=history_errs,
                    **params)

        if history_errs != []:
            history = numpy.asarray(history_errs)
            best_valid_idx = history[:,6].argmin()
            numpy.savetxt(save_model_dir+'train_valid_test.txt', history, fmt='%.4f')
            print 'final best exp ', history[best_valid_idx]

        return train_err, valid_err, test_err

def train_from_scratch(state, channel):
    t0 = time.time()
    print 'training an attention model'
    model = Attention(channel)
    model.train(**state.attention)
    print 'training time in total %.4f sec'%(time.time()-t0)


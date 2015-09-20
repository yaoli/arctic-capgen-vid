import cPickle, os
    
import numpy
from collections import OrderedDict
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

# the dir where there should be a subdir named 'youtube2text_iccv15'
RAB_DATASET_BASE_PATH = '/data/lisatmp3/yaoli/datasets/'
# the dir where all the experiment data is dumped.
RAB_EXP_PATH = '/data/lisatmp3/yaoli/exp/'

def get_two_rngs(seed=None):
    if seed is None:
        seed = 1234
    else:
        seed = seed
    rng_numpy = numpy.random.RandomState(seed)
    rng_theano = MRG_RandomStreams(seed)
    return rng_numpy, rng_theano

rng_numpy, rng_theano = get_two_rngs()

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out
'''
Theano shared variables require GPUs, so to
make this code more portable, these two functions
push and pull variables between a shared
variable dictionary and a regular numpy 
dictionary
'''
# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise, 
                         state_before * 
                         trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                         state_before * 0.5)
    return proj


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are 
    orthogonal. 
    """
    W = rng_numpy.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng_numpy.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)

def rectifier(x):
    return tensor.maximum(0., x)

def linear(x):
    return x

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive'%kk)
        params[kk] = pp[kk]
    return params

def grad_nan_report(grads, tparams):
    numpy.set_printoptions(precision=3)
    D = OrderedDict()
    i = 0
    NaN_keys = []
    magnitude = []
    assert len(grads) == len(tparams)
    for k, v in tparams.iteritems():
        grad = grads[i]
        magnitude.append(numpy.abs(grad).mean())
        if numpy.isnan(grad.sum()):
            NaN_keys.append(k)
        #assert v.get_value().shape == grad.shape
        D[k] = grad
        i += 1
    #norm = [numpy.sqrt(numpy.sum(grad**2)) for grad in grads]
    #print '\tgrad mean(abs(x))', numpy.array(magnitude)
    return D, NaN_keys

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adadelta(lr, tparams, grads, inp, cost, extra):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, [cost] + extra, updates=zgup+rg2up,
                                    profile=False, on_unused_input='ignore')
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=False)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=False)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=False)

    return f_grad_shared, f_update

def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()


def generate_minibatch_idx(dataset_size, minibatch_size):
    # generate idx for minibatches SGD
    # output [m1, m2, m3, ..., mk] where mk is a list of indices
    assert dataset_size >= minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = numpy.split(numpy.asarray(idx), n_minibatches)
    else:
        print 'uneven minibath chunking, overall %d, last one %d'%(minibatch_size, leftover)
        minibatch_idx = numpy.split(numpy.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [numpy.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx

def get_rab_dataset_base_path():
    return RAB_DATASET_BASE_PATH

def get_rab_exp_path():
    return RAB_EXP_PATH

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        print 'creating directory %s'%directory
        os.makedirs(directory)
    else:
        print "%s already exists!"%directory

def flatten_list_of_list(l):
    # l is a list of list
    return [item for sublist in l for item in sublist]

def load_txt_file(path):
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    return lines

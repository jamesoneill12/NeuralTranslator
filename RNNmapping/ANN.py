# -*- coding: utf-8 -*-

from theano import tensor as T
import theano, copy, os, warnings, sys, time
import numpy as np
from collections import OrderedDict, defaultdict
from gensim.models import Word2Vec
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pkl
from keras.utils.np_utils import to_categorical
from helpers import *


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameteres from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk,vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def set_params(x, h_dim, **kwargs):
    params = OrderedDict()
    i = 0
    for arg in kwargs:
        if i == 0:
            params[arg] = theano.shared(name=arg, value=0.1 * np.random.uniform(-1.0, 1.0, (x.shape, h_dim))
                                        .astype(theano.config.floatX))
            i += 1
        else:
            params[arg] = theano.shared(name=arg, value=0.1 * np.random.uniform(-1.0, 1.0, (h_dim, h_dim))
                                        .astype(theano.config.floatX))
    return params

def ortho_weight(ndim):

    W = np.random.randn(ndim,ndim)
    s, u, v = np.linalg.svd(W)
    return v

# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')

def embedding_format(emb):
    # description string: #words x #samples
    x = T.matrix('x', dtype='int64')
    x_mask = T.matrix('x_mask', dtype='float32')
    x_mask = emb
    n_timesteps = emb[0].shape[0]
    n_samples = emb.shape[0]
    print (n_timesteps,n_samples,emb[0].shape[1])
    emb = x_mask.flatten()
    x = x_mask.reshape([n_timesteps, n_samples, x_mask[0].shape[1]])
    emb_shifted = np.zeros_like(x_mask)
    emb_shifted = T.set_subtensor(emb_shifted[1:], x_mask[:-1])
    return emb_shifted

def prepare_data(**kwargs):
    googleVecs = "C:/Users/1/James/grctc/GRCTC_Project/Classification/Data/" \
                 "Embeddings/word2vec/GoogleNews-vectors-negative300.bin"
    sentences = [clean_str(line.decode('utf-8').strip()).split() for line in open(filename, "r").readlines()]
    vecs = Word2Vec.load_word2vec_format(googleVecs, binary=True)  # C binary format
    v = get_embeddings(sentences, vecs)
    y = to_categorical(np.random.randint(2, size=len(v)))
    print y
    return (v,y)

'''
# test data prep is correct
root = "C:\\Users\\1\\James\\grctc\\GRCTC_Project\\Classification\\Word2Vec\\annotated_data"
emb = prepare_data(filename = root+'annotated_data\EU.AML2015_new.txt')
emb = embedding_format(emb)
'''

def tanh(x):
    return T.nnet.tanh(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def relu(x):
    return np.max(0,x, axis =1)

def norm_weights(ndim,x=None,bias=True,name=''):
    if bias:
        # ndim for the hidden layer but at softmax it should be (num_class,1)
        b = theano.shared(name=name, value=0.1 * np.random.uniform(-2.0, 2.0,(ndim,)).astype(theano.config.floatX))
        return b
    if x is not None:
        w = theano.shared(name=name, value=0.1 * np.random.uniform(-2.0, 2.0,(x,ndim)).astype(theano.config.floatX))
        return w
    w= theano.shared(name=name, value=0.1 * np.random.uniform(-2.0, 2.0,(ndim,ndim)).astype(theano.config.floatX))
    return w

def concatenate(tensor_list, axis=0):
    concat_size = sum(tt.shape[axis] for tt in tensor_list)
    output_shape = ()

    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)
    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)
        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]
    return out

def get_embeddings(document, model, pad_len=100):
    max_words, max_length = model.syn0.shape[0], model.syn0.shape[1]
    arr = np.zeros(max_length, dtype='float32')
    B = []
    for sentence in document:
        A = []
        for word in sentence:
            try:
                emb = model[word]
                A.append(emb)
            except:
                A.append(arr)
        A = np.array(A)
        difference = pad_len - len(A)
        A = np.resize(A, (pad_len, max_length))
        B.append(A)
    sequences = np.array(B)
    return sequences

class RNN(object):

    def __init__(self,**kwargs):
        '''
        h_dim :: dimension of the first hidden layer, h_dim >1 is a list of h_dims
        h_num :: number of hidden layers
        nc :: number of classes
        '''

        self.params = {
            "X": X,
            "y": y,
            "h_dim": 2,
            "h_num": 1,
            "arc": "lstm",
            "num_class" : 0,
            "loadvecs" : None,
            "pad" : 100
        }

        for (prop, default) in self.params.iteritems():
            setattr(self, prop, kwargs.get(prop, default))
        unique, counts = np.unique(self.params['y'], return_counts=True)
        self.params['num_classes'] = len(unique) + 1
        dh_dim = (self.params['h_dim'])*2

        if self.params['loadvecs'] is not None:
            self.params['loadvecs'] = Word2Vec.load_word2vec_format(self.params['loadvecs'], binary=True)  # C binary format

        # I think wx & ux should be a 100 (embded_dim) x 20 (hidden_dim)
        # always make sure the first arguement is input weights wx
        # think this is suppose to be the last set of weights ?

        # here I should be tieing the weights for each layer

        self.wx = norm_weights(self.params['h_dim'],self.params['X'][0].shape[1],name='wx')
        self.w0 = norm_weights(self.params['h_dim'], self.params['num_class'],name='w0')
        self.wi = norm_weights(self.params['h_dim'],name='wi')
        self.wc = norm_weights(self.params['h_dim'], name='wc')
        self.w  = norm_weights(self.params['num_class'],name='w')
        self.bf = norm_weights(self.params['h_dim'],bias=True, name='bf')
        self.bi = norm_weights(self.params['h_dim'],bias=True,name='bi')
        self.bc = norm_weights(self.params['h_dim'],bias=True,name='bc')
        self.b  = norm_weights(self.params['num_class'],bias=True,name='b')

        self.Ux= norm_weights(self.params['h_dim'],self.params['X'][0].shape[1],name='Ux')
        self.U = norm_weights(self.params['h_dim'], self.params['num_class'],name='U')
        self.Uf= norm_weights(self.params['h_dim'],name='Uf')
        self.Ui= norm_weights(self.params['h_dim'],name='Ui')
        self.Uc= norm_weights(self.params['h_dim'],name='Uc')
        self.hf= norm_weights(self.params['h_dim'],bias=True,name='hf')
        self.hi= norm_weights(self.params['h_dim'],bias=True,name='hi')
        self.hc= norm_weights(self.params['h_dim'],bias=True,name='hc')
        self.h = norm_weights(self.params['num_class'],bias=True,name='h')

        # not sure if U's need a bias unit for any of the gates ?
        self.idxs = T.imatrix()
        # [self.idxs].reshape((self.idxs.shape[0], self.embed_dim * self.context_win))
        self.x = self.params['X']
        self.y = self.params['y']
        self.y_sentence = T.ivector('y_sentence')  # labels

        if self.params['arc'] is 'language':
            self.Wr = norm_weights(self.params['h_dim'],name='W')
            self.Wz = norm_weights(self.params['h_dim'],name='Wz')
            self.W = norm_weights(self.params['h_dim'],self.params['X'][0].shape[1],name='W')
            self.Ur = norm_weights(self.params['h_dim'],self.params['X'][0].shape[1],name='Ur')
            self.Uz = norm_weights(self.params['h_dim'],self.params['X'][0].shape[1],name='Uz')
            self.U = norm_weights(self.params['h_dim'],self.params['X'][0].shape[1],name='Uz')
            self.Cr = norm_weights(self.params['h_dim'],name='Cr')
            self.Cz = norm_weights(self.params['h_dim'],name='Cr')
            self.C = norm_weights(self.params['h_dim'],name='C')
            self.h = norm_weights(self.params['h_dim'],name='h')

        if self.params['arc'] == 'gers':
            self.C0 = norm_weights(self.params['h_dim'],self.params['X'][0].shape[1],name='C0')
            self.Cx = norm_weights(self.params['h_dim'],self.params['X'][0].shape[1],name='Cx')
            self.Cf = norm_weights(self.params['h_dim'],name='Cf')
            self.Ci = norm_weights(self.params['h_dim'],name='Ci')
            self.Cc = norm_weights(self.params['h_dim'],name='Cc')

    #  encoder/decoder implemented according to NMT paper: https://arxiv.org/pdf/1506.06726v1.pdf
    # not sure if below should be using T.elemwise.Prod since I need to component wise Product
    def encoder(self, x_t,h_tm1, c_tm1):
        r_t = T.nnet.sigmoid(T.dot(self.Wr,x_t)+ T.dot(self.Ur,h_tm1))
        z_t = T.nnet.sigmoid(T.dot(self.Wz,x_t) + T.dot(self.Uz,h_tm1))
        hest_t = T.tanh(T.dot(self.Wx,x_t)+ T.dot(self.U,T.elemwise.Prod(r_t,h_tm1, axis=1)))
        h_t = T.elemwise.Prod((1- z_t),h_tm1) + T.elemwise.Prod(z_t, hest_t,axis=1)
        return [r_t, z_t, hest_t, h_t]

    def decoder(self, x_tm1, h_t, h_tm1, h_i, hest_t,r_t, z_t):
        # come back to this and figure what h(t+1) should be
        r_t = T.nnet.sigmoid(T.dot(self.Wr, x_tm1) + T.dot(self.Ur, h_tm1)+ T.elemwise.Prod(self.Cr, h_i))
        z_t = T.nnet.sigmoid(T.dot(self.Wz,x_tm1) + T.dot(self.Uz,h_tm1) + T.elemwise.Prod(self.Cz, h_i))
        hest_t = T.tanh(T.dot(self.Wx,x_tm1)+ T.dot(self.U,T.elemwise.Prod(r_t,h_tm1, axis=1)) + T.elemwise.Prod(self.Cz, h_i))
        h_tip1 = T.elemwise.Prod((1- z_t),h_tm1) + T.elemwise.Prod(z_t, hest_t,axis=1)
        s_t = T.nnet.softmax(T.dot(h_tip1, self.w) + self.b)
        return [s_t, r_t, z_t, hest_t, h_tip1]

    def recurrence(self, x_t, h_tm1, c_tm1):

        f_t = T.nnet.sigmoid(T.dot(np.append(self.Uf,self.wf), np.append(h_tm1,x_t))+ np.append(self.bf,self.bf))
        arg = T.dot(np.append(self.Ui,self.wi),np.append(h_tm1,x_t)) + np.append(self.bi,self.bi)
        i_t = T.nnet.sigmoid(arg)
        ctilda_t = T.tanh(T.dot(np.append(self.Uc,self.wc), np.append(h_tm1, x_t)) + np.append(self.bc,self.bc))
        c_t = f_t * c_tm1 + i_t * ctilda_t
        o_t = T.nnet.sigmoid(T.dot(np.append(self.U0,self.w0) * np.append(h_tm1, x_t)) + np.append(self.b0,self.b0))
        h_t = o_t * T.tanh(c_t)
        s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
        return [c_t,h_t,s_t]

    def gers_recurrence(self, x_t, h_tm1, c_tm1, alt=False):

        f_t = T.nnet.sigmoid(T.dot(np.append(self.Cf,self.wf,self.Uf) , np.append(c_tm1,x_t,h_tm1)) + np.append(self.bf,self.bf,self.bf))
        i_t = T.nnet.sigmoid(T.dot(np.append(self.Ci,self.wi,self.Ui) , np.append(c_tm1,h_tm1, x_t)) + np.append(self.bi,self.bi,self.bi))
        ctilda_t = T.tanh(T.dot(np.append(self.Cc,self.wc), np.append(h_tm1, x_t)) + np.append(self.bc,self.bc))

        if alt:
            c_t = f_t * c_tm1 + (1- f_t) * ctilda_t
        else:
            c_t = f_t * c_tm1 + i_t * ctilda_t

        o_t = T.nnet.sigmoid(T.dot(np.append(self.C0,self.U0,self.w0) , np.append(c_t,h_tm1, x_t)) + np.append(self.b0,self.b0,self.b0))
        h_t = o_t * T.tanh(c_t)
        s_t = T.nnet.softmax(T.dot(h_t, self.w) + np.append(self.b,self.b))

        self.states = {'ft':f_t,
                       'it': i_t,
                       'ct': c_t,
                       'ot': o_t,
                       'h_t': h_t,
                       's_t': s_t}
        return [c_t,h_t,s_t]

    def gru_recurrence(self, x_t, h_tm1):

        z_t = T.nnet.sigmoid(T.dot(np.append(self.wz,self.wz) , np.append(h_tm1,x_t)))
        r_t = T.nnet.sigmoid(T.dot(np.append(self.wr,self.wr) ,np.append(h_tm1,x_t)))
        htilda_t = T.tanh(self.w0 * (T.dot(r_t* h_tm1,x_t)))
        h_t = np.convolve((1 - z_t), h_tm1) + np.convolve(z_t , htilda_t)
        s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
        return [h_t,s_t]

    def forward_prop(self,x):
        # this needs to be completed and first working for lstm, then try other variations
        if self.arc is 'lstm':
            # try out gers and gru here aswel
            [h, s], _ = theano.scan(fn=self.recurrence, sequences=x,outputs_info=[self.c0,self.h0, None],n_steps=x[0].shape[0])
            self.p_y_given_x_sentence = s[:, 0, :]
            self.y_pred = T.argmax(self.p_y_given_x_sentence, axis=1)
        elif self.arc is 'gru':
            # try out gers and gru here aswel
            [h, s], _ = theano.scan(fn=self.gru_recurrence, sequences=x,outputs_info=[self.c0,self.h0, None],n_steps=x[0].shape[0])
            self.p_y_given_x_sentence = s[:, 0, :]
            self.y_pred = T.argmax(self.p_y_given_x_sentence, axis=1)
        elif self.arc is 'gers':
            # try out gers and gru here aswel
            [h, s], _ = theano.scan(fn=self.gers_recurrence, sequences=x,outputs_info=[self.c0,self.h0, None],n_steps=x[0].shape[0])
            self.p_y_given_x_sentence = s[:, 0, :]
            self.y_pred = T.argmax(self.p_y_given_x_sentence, axis=1)

        # recurrence needs to happen for current and previous sentence here
        # One decoder is used for the next sentence s(i+1) while a second decoder is used for the previous sentence s(i−1)
        elif self.arc is 'language':
            # try out gers and gru here aswel make sure to get outputs and inputs straight
            [r_t, z_t, hest_t, h_t], _ = theano.scan(fn=self.encoder, sequences=x,outputs_info=[],n_steps=x[0].shape[0])
            # pass output activations to decoders output info s(i+1)
            [s,r_t, z_t, hest_t, h_tip1], _ = theano.scan(fn=self.decoder, sequences=h_t,outputs_info=[],n_steps=x[0].shape[0])
            # pass output activations to decoders output info s(i-1)
            [s,r_t, z_t, hest_t, h_tip1], _ = theano.scan(fn=self.decoder, sequences=h_t,outputs_info=[],n_steps=x[0].shape[0])
            # set softmax prediction s_t to class variable
            self.p_y_given_x_sentence = s[:, 0, :]
            self.y_pred = T.argmax(self.p_y_given_x_sentence, axis=1)
        else:
            print ("Please specify the type of network you wish to use when making an instance.")

        return self.y_pred

    def lstm_backprop(self,):

        '''
        self.delta_it =
        self.delta_ot = np.append(self.delta_ht,T.tanh(self.ct))
        self.delta_ft =
        self.delta_ctm1 =
        cat_loss = (self.y - self.y_pred)
        '''

    def negative_loglikelihood(self,targets=None):

        if targets:
            self.y_sentence = targets

        if self.y_sentence:

            lr = T.scalar('lr')
            sentence_nll = -T.mean(T.log(self.p_y_given_x_sentence)[T.arange(self.x.shape[0]), self.y_sentence])
            sentence_gradients = T.grad(sentence_nll, self.params)
            sentence_updates = OrderedDict((p, p - lr * g) for p, g in zip(self.params, sentence_gradients))

            self.classify = theano.function(inputs=[self.idxs], outputs=self.y_pred)
            self.sentence_train = theano.function(inputs=[self.idxs, self.y_sentence, lr],outputs=sentence_nll,
                                                  updates=sentence_updates)
            self.normalize = theano.function(inputs=[],updates={self.emb:self.
                                             emb/T.sqrt((self.emb ** 2).sum(axis=1)).dimshuffle(0, 'x')})
        else:
            print "Pass target sentences before attempting weight updates"

    def cross_entropy(self,targets=None):

        if targets:
            self.y_sentence = targets

        return 1

    def get_minibatches_idx(n, minibatch_size, shuffle=False):
        """
        Used to shuffle the dataset at each iteration.
        """

        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
            minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

    def momentum(self,dw_tm1,eps,e_tdif):
        v_t = self.alpha * dw_tm1 - eps * e_tdif
        return v_t

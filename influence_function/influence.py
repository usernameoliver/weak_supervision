'''
Implementation of the calculation of influence function
@Auther: Degan Hao
@Date: 02/03/2020

'''

import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable
import os.path

def layer_wise_influence(params, loss, nth_layer, verbose, params_test, loss_test):
    '''
    The intuition is to calculate the influence of upweighting image A on the model's decision making on predicting image B.
    loss comes from the image A we want to calculate the influence with
    loss_test comes from the image B we want to test with
    '''
    count = 0
    param = params[nth_layer]
    Hessian = None
    grads = autograd.grad(loss, param, create_graph = True)
    param_test = params[nth_layer]
    grads_test = autograd.grad(loss_test, param_test, create_graph = True)

    grads_flatten = torch.cat([g.reshape(-1) for g in grads if g is not None])
    grads_flatten_test = torch.cat([g.reshape(-1) for g in grads_test if g is not None])
    if verbose:
        print('loss = ' + str(loss))
        print('parameters = ' + str(param)) 
        print('grads_flatten size = ' + str(grads_flatten.size()))
        print('grads = ' + str(grads))

    for grad in grads_flatten:
        #print('-' * 10)
        second_grads = autograd.grad(grad, param, create_graph = True)
        second_grads_flatten = torch.cat([g.reshape(-1) for g in second_grads if g is not None])
        second_grads_flatten = second_grads_flatten.unsqueeze(0)
        if Hessian is None:
            Hessian = second_grads_flatten
        else:
            Hessian = torch.cat((Hessian, second_grads_flatten), dim = 0)

    if verbose:
        print('Hessian = ' + str(Hessian))
        print('Hessian size = ' )
        #add damping term lambda * diagnal matrix, lambda = 0.01
        print('size of hessian  = ' + str(len(Hessian)))
        print('size of hessian  = ' + str(Hessian.size()))
    ones_diag = torch.diag(torch.ones(len(Hessian))).cuda()
    lambda_diag = ones_diag * 0.01
    Hessian = Hessian + lambda_diag

    H_inverse = torch.inverse(Hessian)
    influence = grads_flatten_test.t() @ (H_inverse @ grads_flatten)
    #print('influence = ' + str(influence))
    return influence


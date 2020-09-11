# -*- coding: utf-8 -*-
import os
import torch
import random
from torch.utils.cpp_extension import load
PATH_CPP = os.getcwd() + '/common/LWE_based_PHE/matrix_op/matrix_op_cuda.cpp'
PATH_CU = os.getcwd() + '/common/LWE_based_PHE/matrix_op/matrix_op_cuda_kernel.cu'
matrix_op_cuda = load(
    'matrix_op_cuda', [PATH_CPP, PATH_CU], verbose=True)
from LWE_based_PHE.matrix_op.matrix_op import matmul, vecmul

n = 3008
s = 8
p_ = 46
q_ = 64
p = 2 ** p_ + 1
q = 2 ** q_
l = 2 ** 16
prec = 32

class PublicKey:
    def __init__(self, A, P, n, s):
        self.A = A
        self.P = P
        self.n = n
        self.s = s

    def __repr__(self):
        return 'PublicKey({}, {}, {}, {})'.format(self.A, self.P, self.n, self.s)

class Ciphertext:
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def __repr__(self):
        return 'Ciphertext({}, {})'.format(self.c1, self.c2)

    def __add__(self, c):
        c1 = self.c1 + c.c1
        c2 = self.c2 + c.c2
        return Ciphertext(c1, c2)

def get_uniform_random_matrix(row, col, seed):
    sample = []
    for i in range(row):
        row_sample = []
        for i in range(col):
            random.seed(seed)
            seed += 1
            row_sample.append(random.randint(-q // 2 + 1, q // 2))
        sample.append(row_sample)
    return torch.LongTensor(sample).cuda()

def KeyGen(seed):
    work_path = os.getcwd()
    key_path = work_path + '/key/p_%d_q_%d_n_%d_l_%d_s_%d.pth' % (p_, q_, n, l, seed)
    if os.path.exists(key_path):
        key = torch.load(key_path)
        pk = key['pk']
        sk = key['sk']
    else:
        # random.seed(seed)
        torch.manual_seed(seed)
        R = torch.clamp(torch.randn(n, l) * s, min=-3*s, max=3*s).long().cuda()
        torch.manual_seed(seed + 1)
        S = torch.clamp(torch.randn(n, l) * s, min=-3*s, max=3*s).long().cuda()
        A = get_uniform_random_matrix(n, n, seed)
        P = p * R - matmul(A, S)
        pk, sk = PublicKey(A, P, n, s), S
        torch.save({'pk': pk, 'sk': sk}, key_path)

    return pk, sk

def Enc(pk, m):
    e1 = torch.clamp(torch.randn(n) * s, min=-3*s, max=3*s).long().cuda()
    e2 = torch.clamp(torch.randn(n) * s, min=-3*s, max=3*s).long().cuda()
    e3 = torch.clamp(torch.randn(l) * s, min=-3*s, max=3*s).long().cuda()

    if m.size(0) < l:
        m = torch.cat((m, torch.zeros(l - m.size(0)).long().cuda()), 0)

    c1 = vecmul(e1, pk.A) + p * e2
    c2 = vecmul(e1, pk.P) + p * e3 + m
    return Ciphertext(c1, c2)

def Dec(sk, c):
    return (vecmul(c.c1, sk) + c.c2) % p


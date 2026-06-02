import math
import time

import torch
import torch.nn as nn
import transformers
# import matplotlib.pyplot as plt

from quant_baq import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        W_ori = W.clone()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        # del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H_ori = H.clone()
        H = torch.linalg.cholesky(H)

        # ============================================
        Y = H.clone()
        # ============================================
        
        # H = torch.cholesky_inverse(H)
        # H = torch.linalg.cholesky(H, upper=True)
        # Hinv = H

        # for i1 in range(0, self.columns, blocksize):
        #     i2 = min(i1 + blocksize, self.columns)
        #     count = i2 - i1

        #     W1 = W[:, i1:i2].clone()
        #     Q1 = torch.zeros_like(W1)
        #     Err1 = torch.zeros_like(W1)
        #     Losses1 = torch.zeros_like(W1)
        #     Hinv1 = Hinv[i1:i2, i1:i2]

        #     for i in range(count):
        #         w = W1[:, i]
        #         d = Hinv1[i, i]

        #         if groupsize != -1:
        #             if not static_groups:
        #                 if (i1 + i) % groupsize == 0:
        #                     self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
        #             else:
        #                 idx = i1 + i
        #                 if actorder:
        #                     idx = perm[idx]
        #                 self.quantizer = groups[idx // groupsize]

        #         q = quantize(
        #             w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
        #         ).flatten()
        #         Q1[:, i] = q
        #         Losses1[:, i] = (w - q) ** 2 / d ** 2

        #         err1 = (w - q) / d
        #         W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        #         Err1[:, i] = err1

        #     Q[:, i1:i2] = Q1
        #     Losses[:, i1:i2] = Losses1 / 2

        #     W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        #     if DEBUG:
        #         self.layer.weight.data[:, :i2] = Q[:, :i2]
        #         self.layer.weight.data[:, i2:] = W[:, i2:]
        #         print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        #         print(torch.sum(Losses))

        # torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))
        # print('error', torch.sum(Losses).item())

        # if actorder:
        #     Q = Q[:, invperm]

        # if isinstance(self.layer, transformers.Conv1D):
        #     Q = Q.t()
        # # self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # if DEBUG:
        #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        # E = Q - W_ori
        # trace_val = torch.trace(E @ H_ori @ E.T)
        # print('trace:', trace_val.item())

        # return W_ori, Q
        WY = W_ori @ Y
        return W_ori, WY, Y

    def fasterquant_svdlora(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, W_ori=None, WY=None, Y=None
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        W_tilde = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        # H = torch.linalg.cholesky(H)
        # H = torch.cholesky_inverse(H)
        # H = torch.linalg.cholesky(H, upper=True)
        # Hinv = H

        

        # ========================================
        wbit_calib_P = 4
        wbit_calib_Q = 4
        m, n = WY.shape
        r_truncate = max(1, int(0.158 * min(m, n)))
        Y_inv = torch.linalg.inv(Y)   # (N, N)
        nblock = 2
        num_keep = r_truncate // nblock   # floor
        print("r_truncate:", r_truncate)
        print("num_keep:", num_keep)
        # ========================================
        
        # ===============================================================================
        # block 1
        # SVD for WY
        U, S, Vh = torch.linalg.svd(WY, full_matrices=True)          # (M, r)(r, r)(r, N)
        Qc_inv = Vh @ Y.T         # (r, N)
        
        U_tilde = U[:, :num_keep]
        S_tilde = torch.diag(S[:num_keep])
        Vh_tilde = Vh[:num_keep, :]
        
        P = U_tilde @ S_tilde         # (M, r)
        Q = Vh_tilde @ Y_inv          # (r, N)
        
        # Initialize the quantized matrix
        P_hat = torch.zeros_like(P)         # (M, r)
        Q_hat = torch.zeros_like(Q)          # (r, N)
        
        # Iteration of element-wise quantization and compensation
        for i in range(num_keep):
            # quantize i-th column
            pi = P[:, i].unsqueeze(1)    # (M, 1)
            qi = Q[i, :].unsqueeze(0)    # (1, N)
        
            p_hat_i = uniform_quantize(pi, wbit_calib_P, dim=0)    # (M, 1)
            q_hat_i = uniform_quantize(qi, wbit_calib_Q, dim=1)      # (1, N)
        
            P_hat[:, i] = p_hat_i.view(-1)
            Q_hat[i, :] = q_hat_i.view(-1)
        
            # calculate the quantization error and expand on the rest basis direction
            e_i = q_hat_i - qi    # (1, N)
            beta_i = Qc_inv @ e_i.T        # (r, N)x(N, 1)=(r, 1)
            
            
            if i + 1 < num_keep:
                beta_slice = beta_i[i+1:num_keep, :]                  # (r-i-1, 1)
                comp = p_hat_i @ beta_slice.T    # (M, 1)x(1, r-i-1)=(M, r-i-1)
                P[:, i+1:num_keep] -= comp    # (M, r-i-1)

        W_hat1 = P_hat @ Q_hat
        # ===============================================================================
        # ===============================================================================
        # block 2
        E1 = W_ori - W_hat1      # (M, N)
        E1Y = E1 @ Y
        U_E1, S_E1, Vh_E1 = torch.linalg.svd(E1Y, full_matrices=True)          # (M, r)(r, r)(r, N)
        Qc_E1_inv = Vh_E1 @ Y.T         # (r, N)
        
        U_E1_tilde = U_E1[:, :num_keep]
        S_E1_tilde = torch.diag(S_E1[:num_keep])
        Vh_E1_tilde = Vh_E1[:num_keep, :]
        
        P_E1 = U_E1_tilde @ S_E1_tilde         # (M, r)
        Q_E1 = Vh_E1_tilde @ Y_inv          # (r, N)
        
        # Initialize the quantized matrix
        P_E1_hat = torch.zeros_like(P_E1)         # (M, r)
        Q_E1_hat = torch.zeros_like(Q_E1)          # (r, N)
        
        # Iteration of element-wise quantization and compensation
        for i in range(num_keep):
            # quantize i-th column
            pi_E1 = P_E1[:, i:i+1]    # (M, 1)
            qi_E1 = Q_E1[i:i+1, :]    # (1, N)
        
            p_E1_hat_i = uniform_quantize(pi_E1, wbit_calib_P, dim=0)    # (M, 1)
            q_E1_hat_i = uniform_quantize(qi_E1, wbit_calib_Q, dim=1)     # (1, N)
        
            P_E1_hat[:, i:i+1] = p_E1_hat_i
            Q_E1_hat[i:i+1, :] = q_E1_hat_i
        
            # calculate the quantization error and expand on the rest basis direction
            e_E1_i = q_E1_hat_i - qi_E1    # (1, N)
            beta_E1_i = Qc_E1_inv @ e_E1_i.T         # (r, N)x(N, 1)=(r, 1)
            # print("e_E1_i max:", e_E1_i.max().item())
            # print("e_E1_i min:", e_E1_i.min().item())
            
            
            if i + 1 < num_keep:
                beta_slice_E1 = beta_E1_i[i+1:num_keep, :]                  # (r-i-1, 1)
                comp_E1 = p_E1_hat_i @ beta_slice_E1.T    # (M, 1)x(1, r-i-1)=(M, r-i-1)
                P_E1[:, i+1:num_keep] -= comp_E1    # (M, r-i-1)

        E1_hat = P_E1_hat @ Q_E1_hat          # (M, r)x(r, N)=(M, N)
        
        W_hat2 = W_hat1 + E1_hat
        # ===============================================================================
        # # ===============================================================================
        # # block 3
        # E2 = W_ori - W_hat2      # (M, N)
        # E2Y = E2 @ Y
        # U_E2, S_E2, Vh_E2 = torch.linalg.svd(E2Y, full_matrices=True)          # (M, r)(r, r)(r, N)
        # Qc_E2_inv = Vh_E2 @ Y.T         # (r, N)
        
        # U_E2_tilde = U_E2[:, :num_keep]
        # S_E2_tilde = torch.diag(S_E2[:num_keep])
        # Vh_E2_tilde = Vh_E2[:num_keep, :]
        
        # P_E2 = U_E2_tilde @ S_E2_tilde         # (M, r)
        # Q_E2 = Vh_E2_tilde @ Y_inv          # (r, N)
        
        # # Initialize the quantized matrix
        # P_E2_hat = torch.zeros_like(P_E2)         # (M, r)
        # Q_E2_hat = torch.zeros_like(Q_E2)          # (r, N)
        
        # # Iteration of element-wise quantization and compensation
        # for i in range(num_keep):
        #     # quantize i-th column
        #     pi_E2 = P_E2[:, i:i+1]    # (M, 1)
        #     qi_E2 = Q_E2[i:i+1, :]    # (1, N)
        
        #     p_E2_hat_i = uniform_quantize(pi_E2, wbit_calib_P, dim=0)    # (M, 1)
        #     q_E2_hat_i = uniform_quantize(qi_E2, wbit_calib_Q, dim=1)     # (1, N)
        
        #     P_E2_hat[:, i:i+1] = p_E2_hat_i
        #     Q_E2_hat[i:i+1, :] = q_E2_hat_i
        
        #     # calculate the quantization error and expand on the rest basis direction
        #     e_E2_i = q_E2_hat_i - qi_E2    # (1, N)
        #     beta_E2_i = Qc_E2_inv @ e_E2_i.T         # (r, N)x(N, 1)=(r, 1)
        #     # print("e_E1_i max:", e_E1_i.max().item())
        #     # print("e_E1_i min:", e_E1_i.min().item())
            
            
        #     if i + 1 < num_keep:
        #         beta_slice_E2 = beta_E2_i[i+1:num_keep, :]                  # (r-i-1, 1)
        #         comp_E2 = p_E2_hat_i @ beta_slice_E2.T    # (M, 1)x(1, r-i-1)=(M, r-i-1)
        #         P_E2[:, i+1:num_keep] -= comp_E2    # (M, r-i-1)

        # E2_hat = P_E2_hat @ Q_E2_hat          # (M, r)x(r, N)=(M, N)
        
        # W_hat3 = W_hat1 + E1_hat + E2_hat
        # # ===============================================================================
        # # ===============================================================================
        # # block 4
        # E3 = W_ori - W_hat3      # (M, N)
        # E3Y = E3 @ Y
        # U_E3, S_E3, Vh_E3 = torch.linalg.svd(E3Y, full_matrices=True)          # (M, r)(r, r)(r, N)
        # Qc_E3_inv = Vh_E3 @ Y.T         # (r, N)
        
        # U_E3_tilde = U_E3[:, :num_keep]
        # S_E3_tilde = torch.diag(S_E3[:num_keep])
        # Vh_E3_tilde = Vh_E3[:num_keep, :]
        
        # P_E3 = U_E3_tilde @ S_E3_tilde         # (M, r)
        # Q_E3 = Vh_E3_tilde @ Y_inv          # (r, N)
        
        # # Initialize the quantized matrix
        # P_E3_hat = torch.zeros_like(P_E3)         # (M, r)
        # Q_E3_hat = torch.zeros_like(Q_E3)          # (r, N)
        
        # # Iteration of element-wise quantization and compensation
        # for i in range(num_keep):
        #     # quantize i-th column
        #     pi_E3 = P_E3[:, i:i+1]    # (M, 1)
        #     qi_E3 = Q_E3[i:i+1, :]    # (1, N)
        
        #     p_E3_hat_i = uniform_quantize(pi_E3, wbit_calib_P, dim=0)    # (M, 1)
        #     q_E3_hat_i = uniform_quantize(qi_E3, wbit_calib_Q, dim=1)     # (1, N)
        
        #     P_E3_hat[:, i:i+1] = p_E3_hat_i
        #     Q_E3_hat[i:i+1, :] = q_E3_hat_i
        
        #     # calculate the quantization error and expand on the rest basis direction
        #     e_E3_i = q_E3_hat_i - qi_E3    # (1, N)
        #     beta_E3_i = Qc_E3_inv @ e_E3_i.T         # (r, N)x(N, 1)=(r, 1)
        #     # print("e_E1_i max:", e_E1_i.max().item())
        #     # print("e_E1_i min:", e_E1_i.min().item())
            
            
        #     if i + 1 < num_keep:
        #         beta_slice_E3 = beta_E3_i[i+1:num_keep, :]                  # (r-i-1, 1)
        #         comp_E3 = p_E3_hat_i @ beta_slice_E3.T    # (M, 1)x(1, r-i-1)=(M, r-i-1)
        #         P_E3[:, i+1:num_keep] -= comp_E3    # (M, r-i-1)

        # E3_hat = P_E3_hat @ Q_E3_hat          # (M, r)x(r, N)=(M, N)
        
        # W_hat4 = W_hat + E1_hat + E2_hat + E3_hat
        # # ===============================================================================
        W_tilde = W_hat2

        Losses = torch.abs(W_tilde - W_ori)
        print('mean error', torch.mean(Losses).item())

        if actorder:
            W_tilde = W_tilde[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            W_tilde = W_tilde.t()
        self.layer.weight.data = W_tilde.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            # self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

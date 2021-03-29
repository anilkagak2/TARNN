
import torch
import torch.nn as nn
from torch.nn import Parameter, init
import torch.nn.functional as F

import math

class TARNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_timesteps, K=1, sigma='relu', alpha_val=None, b_variant=True):
        super(TARNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.n_timesteps = n_timesteps
        self.K = K
        self.b_variant = b_variant

        self.NL = torch.tanh
        if sigma=='relu':
            self.NL = F.relu
        elif sigma=='sigmoid':
            self.NL = torch.sigmoid
        elif sigma=='tanh':
            print(sigma)
            self.NL = torch.tanh
        else:
            print('Warning: non-linearity not supported.')
            print('Defaulting to torch.tanh ')

        self.b1 = Parameter(torch.Tensor(hidden_size))
        self.b2 = Parameter(torch.Tensor(hidden_size))

        self.B1 = Parameter(torch.Tensor(input_size, hidden_size))
        self.B2 = Parameter(torch.Tensor(hidden_size, hidden_size))
       
        self.W1 = Parameter(torch.Tensor(input_size, hidden_size))
        self.W2 = Parameter(torch.Tensor(input_size, hidden_size))

        self.U1 = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U2 = Parameter(torch.Tensor(hidden_size, hidden_size))

        self.Wg = Parameter(torch.Tensor(input_size, hidden_size))
        self.Ug = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bg = Parameter(torch.Tensor(hidden_size))
        
        alpha_init = 0.01 #0.001 #+3.0 #0.01
        if alpha_val is not None: alpha_init = alpha_val
        print('alpha_init = ', alpha_init)
        self.alpha_init = alpha_init
        
        self.A1 = torch.nn.Parameter(alpha_init * torch.ones(K,n_timesteps))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.normal_(weight, 0.0, 0.1)
            #init.uniform_(weight, -stdv, stdv)
        
        init.constant_(self.A1, self.alpha_init)
        init.constant_(self.b1, 0.0)
        init.constant_(self.b2, 0.0)
        if self.b_variant:
            init.constant_(self.bg, -3.0)
        else:
            init.constant_(self.bg, 0.0)
        
    def forward(self, inputs, h=None):
        #print('inputs = ', inputs.size())
        h = h.squeeze(dim=0)
        #print('h = ', h.size())
        
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_size)
        
        old_input = None
        t=0
        outputs = []
        for input in torch.unbind(inputs, dim=0):
            #print('t = ', t)
            if old_input is None: old_input = input
            
            old_h = h
            hg = torch.sigmoid( torch.mm(old_h, self.Ug) + torch.mm(input, self.Wg) + self.bg )

            arg = torch.mm( old_h, self.U2 ) + torch.mm(input, self.W1) + torch.mm( old_input, self.W2 ) + self.b1
            bx = torch.mm( input, self.B1 )
            bh = torch.mm( h, self.B2 )

            if self.b_variant:
                for k in range(self.K):
                    at1 = self.A1[k][t]
                    h = (1-at1)*h + at1 * ( self.NL(torch.mm(h, self.U1) + arg) - h - old_h + bx + bh )

                h = (1-hg) * old_h + hg * h
            else:
                for k in range(self.K):
                    at1 = self.A1[k][t]
                    h = h + at1 * hg * ( self.NL(torch.mm(h, self.U1) + arg) - h - old_h + bx + bh )
            
            outputs.append( h.unsqueeze(0) )
            old_input = input
            t += 1
        
        output = torch.cat( outputs, dim=0 )
        final_state = h.unsqueeze(0)
        #assert(1==2)
        return output, final_state


class B_TARNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_timesteps, K=1, cuda=True, alpha_val=None):
        super(B_TARNN_LSTM, self).__init__()
        
        self.CUDA = cuda
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.n_timesteps = n_timesteps
        self.K = K
        assert(K==1)
        
        self.weight_ih = Parameter(torch.randn(4 * 2 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * 2 * hidden_size, 2*hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * 2 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * 2 * hidden_size))
        
        self.weight_bx = Parameter(torch.Tensor(2 * hidden_size, input_size))
        self.weight_bh = Parameter(torch.Tensor(2 * hidden_size, 2*hidden_size))
        self.weight_tp_ih = Parameter(torch.Tensor(4 * 2 * hidden_size, input_size))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
        
    def forward(self, inputs, state=None):
        #print('inputs = ', inputs.size())
        
        if state is None:
            state = ( x.new_zeros(x.shape[0], self.hidden_size), 
                     x.new_zeros(x.shape[0], self.hidden_size) )

        h, c = state
        #print('h = ', h.size())
        h, c = h.squeeze(dim=0), c.squeeze(dim=0)
        #print('h = ', h.size())
        h = torch.cat([h, c], dim=1)
        #assert(1==2)
        
        old_input = None
        outputs = []
        for input in torch.unbind(inputs, dim=0):
            if old_input is None: old_input = input
            
            old_h = h
            
            bx_mm = torch.mm(input, self.weight_bx.t())
            input_mm = torch.mm(input, self.weight_ih.t())
            old_input_mm = torch.mm( old_input, self.weight_tp_ih.t() )
            
            gates = ( input_mm + self.bias_ih + torch.mm(h, self.weight_hh.t()) 
                    + self.bias_hh + old_input_mm)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            h = (forgetgate * h) + (ingate * (cellgate + bx_mm + torch.mm(h, self.weight_bh.t())))
            h = outgate * torch.tanh(h)
            
            #p= h
            p, _ = h.chunk(2, 1)
            outputs.append( p.unsqueeze(0) )
            old_input = input
        
        h, c = h.chunk(2, 1)
        
        output = torch.cat( outputs, dim=0 )
        final_state = (h.unsqueeze(0), c.unsqueeze(0))
        #assert(1==2)
        return output, final_state

class oldB_TARNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_timesteps, K=1, cuda=True, alpha_val=None):
        super(oldB_TARNN, self).__init__()
        
        self.CUDA = cuda
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.n_timesteps = n_timesteps
        self.K = K
        assert(K==1)
        
        self.weight_tp_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(6 * hidden_size, hidden_size))
        #self.U = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_b1 = Parameter(torch.Tensor(hidden_size))
        self.weight_b2 = Parameter(torch.Tensor(hidden_size))
        self.weight_outb = Parameter(torch.Tensor(hidden_size))
        self.weight_bg = Parameter(torch.Tensor(hidden_size))
        
        alpha_init = 0.001 #0.001 #+3.0 #0.01
        if alpha_val is not None: alpha_init = alpha_val
        print('alpha_init = ', alpha_init)
        self.alpha_init = alpha_init
        
        #self.alpha = torch.nn.Parameter(alpha_init * torch.ones(K,n_timesteps))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            #init.normal_(weight, 0.0, 0.1)
            init.uniform_(weight, -stdv, stdv)
        
        #init.constant_(self.alpha, self.alpha_init)
        init.constant_(self.weight_b2, 0.0)
        init.constant_(self.weight_b1, 0.0)
        init.constant_(self.weight_bg, 0.0)
        init.constant_(self.weight_outb, 0.0)
        
    def forward(self, inputs, h=None):
        #print('inputs = ', inputs.size())
        h = h.squeeze(dim=0)
        #print('h = ', h.size())
        
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_size)
        
        old_input = None
        t=0
        outputs = []
        for input in torch.unbind(inputs, dim=0):
            #print('t = ', t)
            if old_input is None: old_input = input
            
            old_h = h
            
            Uh = torch.mm( h, self.weight_hh.t() )
            Wx = torch.mm( input, self.weight_ih.t() )
            Wxtp = torch.mm( old_input, self.weight_tp_ih.t() )
            
            w1, wg, bx, outx = Wx.chunk(4,1)
            w2, bx2, outx2 = Wxtp.chunk(3,1)
            bh, ug, u1, u2, old_u1, out_uh = Uh.chunk(6,1)
            b1, bg = self.weight_b1, self.weight_bg
        
            at1 = torch.sigmoid( u2 + bx2 + self.weight_b2 )
            h = (1-at1) * h + at1 * ( torch.tanh( u1 + old_u1 + w1 + w2 + b1 ) - h - old_h + bx + bh )
            
            hg = torch.sigmoid( ug + wg + bg )
            #h = hg * torch.tanh( h )
            h = (1-hg) * torch.tanh(old_h) + hg * torch.tanh(h)
            #h = (1-hg) * old_h + hg * h
            #h = torch.sigmoid( out_uh + outx + outx2 + self.weight_outb ) * torch.tanh(h)
            
            o = torch.sigmoid( out_uh + outx + outx2 + self.weight_outb ) * h 
            outputs.append( o.unsqueeze(0) )
            old_input = input
            t += 1
        
        #print('t = ', t)
        output = torch.cat( outputs, dim=0 )
        final_state = h.unsqueeze(0)
        #assert(1==2)
        return output, final_state


import torch
import torch.nn as nn
from torch.nn import Parameter
from matrix_utils import power_series_matrix_logarithm_trace
from spectral_norm_fc import spectral_norm_fc


class Attention_gaussian(nn.Module):
    '''
    Dot product, inv
    '''
    def __init__(self, input_channel_num, convGamma=True):
        super(Attention_gaussian, self).__init__()
        self.c_in = input_channel_num
        self.value_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
        self.convGamma = convGamma
        if convGamma:
            self.gamma = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1),
                                             coeff=.9, n_power_iterations=5)
        else:
            self.gamma = Parameter(torch.zeros(1))
            self.nonlin_2 = nn.Tanh()

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = x.view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        proj_key = x.view(B, -1, H * W)  # [B, C//8, HW]
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix multiplication, [B, HW, HW]
        energy = torch.exp(energy)
        energy_sum = torch.sum(energy,dim=(1), keepdim=True)
        energy = energy / (1.5 * energy_sum) #hooray
        proj_value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        out = torch.bmm(proj_value, energy).view(B, C, H, W)
        if self.convGamma:
            out = self.gamma(out)
        else:
            out = self.nonlin_2(self.gamma) * out
        return out

class InvAttention_gaussian(nn.Module):
    def __init__(self, input_channel_num, numTraceSamples=1, numSeriesTerms=5, convGamma = True):
        super(InvAttention_gaussian, self).__init__()
        self.res_branch= Attention_gaussian(input_channel_num, convGamma=convGamma)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
    def forward(self, x, ignore_logdet=False):
        Fx = self.res_branch(x)
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        x = x + Fx
        return x, trace
    def inverse(self, y, maxIter=100):
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)
            x = y
            for iter_index in range(maxIter):
                summand = self.res_branch(x)
                x = y - summand
            return x





class Attention_embedded_gaussian(nn.Module):
    '''
    Embedded Gaussian
    '''
    def __init__(self, input_channel_num, k=4, convGamma=True):
        super(Attention_embedded_gaussian, self).__init__()
        self.c_in = input_channel_num
        self.query_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1),coeff=.9, n_power_iterations=5)
        self.key_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1), coeff=.9, n_power_iterations=5)
        self.value_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
        self.convGamma = convGamma
        if convGamma:
            self.gamma = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1),
                                             coeff=.9, n_power_iterations=5)
        else:
            self.gamma = Parameter(torch.zeros(1))
            self.nonlin_2 = nn.Tanh()

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        proj_key = self.key_conv(x).view(B, -1, H * W)  # [B, C//8, HW]
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix multiplication, [B, HW, HW]
        energy = torch.exp(energy)
        energy_sum = torch.sum(energy,dim=(1), keepdim=True)
        energy = energy / (1.5 * energy_sum) #hooray
        proj_value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        out = torch.bmm(proj_value, energy).view(B, C, H, W)
        if self.convGamma:
            out = self.gamma(out)
        else:
            out = self.nonlin_2(self.gamma) * out
        return out

class InvAttention_embedded_gaussian(nn.Module):
    def __init__(self, input_channel_num, k=4, numTraceSamples=1, numSeriesTerms=5, convGamma = True):
        super(InvAttention_embedded_gaussian, self).__init__()
        self.res_branch= Attention_embedded_gaussian(input_channel_num, k=k, convGamma=convGamma)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
    def forward(self, x, ignore_logdet=False):
        Fx = self.res_branch(x)
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        x = x + Fx
        return x, trace
    def inverse(self, y, maxIter=100):
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)
            x = y
            for iter_index in range(maxIter):
                summand = self.res_branch(x)
                x = y - summand
            return x





class Attention_dot(nn.Module):
    '''
    Dot product, inv
    '''
    def __init__(self, input_channel_num, k=4, convGamma=True):
        super(Attention_dot, self).__init__()
        self.c_in = input_channel_num
        self.query_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
        self.key_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1),
                                         coeff=.9, n_power_iterations=5)
        self.value_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1),
                                           coeff=.5, n_power_iterations=5)
        self.nonlin = nn.ELU()
        self.convGamma = convGamma
        if convGamma:
            self.gamma = spectral_norm_fc(nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1),
                                             coeff=.9, n_power_iterations=5)
        else:
            self.gamma = Parameter(torch.zeros(1))
            self.nonlin_2 = nn.Tanh()

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//8]
        proj_key = self.key_conv(x).view(B, -1, H * W)  # [B, C//8, HW]
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix multiplication, [B, HW, HW]
        energy = self.nonlin(energy)
        energy_sum = torch.sum(energy,dim=(1), keepdim=True)
        energy = energy / (1.5 * energy_sum) #hooray
        proj_value = self.value_conv(x).view(B, -1, H * W)  # [B, C, HW]
        out = torch.bmm(proj_value, energy).view(B, C, H, W)
        if self.convGamma:
            out = self.gamma(out)
        else:
            out = self.nonlin_2(self.gamma) * out
        return out

class InvAttention_dot(nn.Module):
    def __init__(self, input_channel_num, k=4, numTraceSamples=1, numSeriesTerms=5, convGamma = True):
        super(InvAttention_dot, self).__init__()
        self.res_branch= Attention_dot(input_channel_num, k=k, convGamma=convGamma)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
    def forward(self, x, ignore_logdet=False):
        Fx = self.res_branch(x)
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        x = x + Fx
        return x, trace
    def inverse(self, y, maxIter=100):
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)
            x = y
            for iter_index in range(maxIter):
                summand = self.res_branch(x)
                x = y - summand
            return x


class Attention_concat(nn.Module):
    '''
      Concatenation Style PAM, with turbulance
    '''

    def __init__(self, in_c, k=4, convGamma = True):
        super(Attention_concat, self).__init__()
        self.in_c = in_c
        self.inter_c = in_c // k
        self.query_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.in_c, out_channels=self.in_c // k, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
        self.key_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.in_c, out_channels=self.in_c // k, kernel_size=1),
                                         coeff=.9, n_power_iterations=5)
        self.concat_conv = spectral_norm_fc(nn.Conv2d(in_channels=self.inter_c * 2, out_channels=1, kernel_size=1, bias=False),
                                            coeff=.9, n_power_iterations=5)
        self.value_conv = spectral_norm_fc(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
        self.convGamma = convGamma
        if convGamma:
            self.gamma = spectral_norm_fc(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=1),
                                           coeff=.9, n_power_iterations=5)
        else:
            self.gamma = Parameter(torch.zeros(1))
            self.nonlin_2 = nn.Tanh()
        self.nonlin_1 = nn.ELU()


    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, self.inter_c, -1, 1)  # [B, inter_c, HW, 1]
        proj_key = self.key_conv(x).view(B, self.inter_c, 1, -1)  # [B, inter_c, 1, HW]
        proj_query = proj_query.repeat(1, 1, 1, H * W)
        proj_key = proj_key.repeat(1, 1, H * W, 1)
        concat_feature = torch.cat([proj_query, proj_key], dim=1)  # [B, 2*inter_c, HW, HW]
        energy = self.concat_conv(concat_feature).squeeze().reshape(B, H*W, H*W)  # [B,  HW, HW]
        energy = self.nonlin_1(energy)
        energy = energy / (1.5 * torch.sum(energy, dim=(1), keepdim=True))
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, energy).view(B, -1, H, W)
        if self.convGamma:
             out = self.gamma(out)
        else:
            tmp = self.nonlin_2(self.gamma)
            out = tmp * out
        return out




class InvAttention_concat(nn.Module):
    def __init__(self, in_c, k=4, numTraceSamples = 1, numSeriesTerms = 5, convGamma=True):
        super(InvAttention_concat, self).__init__()
        self.res_branch = Attention_concat(in_c, k=k, convGamma=convGamma)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
    def forward(self, x, ignore_logdet=False):
        Fx = self.res_branch(x)
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        x = x + Fx
        return x, trace
    def inverse(self, y, maxIter=100):
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)
            x = y
            for iter_index in range(maxIter):
                summand = self.res_branch(x)
                x = y - summand
            return x






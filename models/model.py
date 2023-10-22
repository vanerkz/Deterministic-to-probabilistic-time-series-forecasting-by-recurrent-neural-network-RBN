import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import DataEmbedding, TimeFeatureEmbedding
import numpy as np
import math

def vanilla_block(in_feat, out_feat, normalize=True, activation=None):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.LayerNorm(out_feat))
    # 0.2 was used in DCGAN, I experimented with other values like 0.5 didn't notice significant change
    layers.append(nn.LeakyReLU(0.2) if activation is None else activation)
    return layers

class NRUCell(nn.Module):

    def __init__(self, device, input_size, hidden_size, memory_size=64, k=4,
                 activation="tanh", use_relu=False, layer_norm=False):
        super(NRUCell, self).__init__()

        self._device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.memory_size = memory_size
        self.k = k
        self._use_relu =  use_relu
        self._layer_norm = layer_norm

        assert math.sqrt(self.memory_size*self.k).is_integer()
        sqrt_memk = int(math.sqrt(self.memory_size*self.k))
        self.hm2v_alpha = nn.Linear(self.memory_size + hidden_size, 2 * sqrt_memk)
        self.hm2v_beta = nn.Linear(self.memory_size + hidden_size, 2 * sqrt_memk)
        self.hm2alpha = nn.Linear(self.memory_size + hidden_size, self.k)
        self.hm2beta = nn.Linear(self.memory_size + hidden_size, self.k)

        if self._layer_norm:
            self._ln_h = nn.LayerNorm(hidden_size)

        self.hmi2h = nn.Linear(self.memory_size + hidden_size + self.input_size, hidden_size)

    def _opt_relu(self, x):
        if self._use_relu:
            return F.relu(x)
        else:
            return x

    def _opt_layernorm(self, x):
        if self._layer_norm:
            return self._ln_h(x)
        else:
            return x

    def forward(self, input, last_hidden):
        hidden = {}
        c_input = torch.cat((input, last_hidden["h"], last_hidden["memory"]), 1)

        h = F.relu(self._opt_layernorm(self.hmi2h(c_input)))

        # Flat memory equations
        alpha = self._opt_relu(self.hm2alpha(torch.cat((h,last_hidden["memory"]),1))).clone()
        beta = self._opt_relu(self.hm2beta(torch.cat((h,last_hidden["memory"]),1))).clone()

        u_alpha = self.hm2v_alpha(torch.cat((h,last_hidden["memory"]),1)).chunk(2,dim=1)
        v_alpha = torch.bmm(u_alpha[0].unsqueeze(2), u_alpha[1].unsqueeze(1)).view(-1, self.k, self.memory_size)
        v_alpha = self._opt_relu(v_alpha)
        v_alpha = torch.nn.functional.normalize(v_alpha, p=5, dim=2, eps=1e-12)
        add_memory = alpha.unsqueeze(2)*v_alpha

        u_beta = self.hm2v_beta(torch.cat((h,last_hidden["memory"]),1)).chunk(2, dim=1)
        v_beta = torch.bmm(u_beta[0].unsqueeze(2), u_beta[1].unsqueeze(1)).view(-1, self.k, self.memory_size)
        v_beta = self._opt_relu(v_beta)
        v_beta = torch.nn.functional.normalize(v_beta, p=5, dim=2, eps=1e-12)
        forget_memory = beta.unsqueeze(2)*v_beta

        hidden["memory"] = last_hidden["memory"] + torch.mean(add_memory-forget_memory, dim=1)
        hidden["h"] = h
        return hidden

    def reset_hidden(self, batch_size, hidden_init=None):
        hidden = {}
        if hidden_init is None:
            hidden["h"] = torch.Tensor(np.zeros((batch_size, self.hidden_size))).to(self._device)
        else:
            hidden["h"] = hidden_init.to(self._device)
        hidden["memory"] = torch.Tensor(np.zeros((batch_size, self.memory_size))).to(self._device)
        return hidden


class NRU_RBN(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, d_model=512, d_ff=512, 
                dropout=0.0, embed='fixed', freq='h', activation='gelu', 
                output_attention = False,likeloss=True,
                device=torch.device('cuda:0')):
        super(NRU_RBN, self).__init__()
        
        self.d_model=d_model
        self.pred_len = out_len
        self.output_attention = output_attention
        self.likeloss = likeloss
        self.label_len=label_len
        self.vaemodel=d_ff
        self.c_out = c_out
        self.enc_in=enc_in
        self.device=device
        freq_map = {'h':5, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        timefeatureNos = freq_map[freq]

        # Residual Boosting Network (RBN)
        self.enc_embedding = DataEmbedding(enc_in, d_ff, embed, freq, dropout)

        modules = []
        hidden_dims =None

        if hidden_dims is None:
            hidden_dims = [enc_in+timefeatureNos,d_ff*2,d_ff*2*2,d_ff*2,d_ff]

        # MLP increase dimensions
        modules.append(
            nn.Sequential(
                    *vanilla_block(hidden_dims[0], hidden_dims[1]),
                    *vanilla_block(hidden_dims[1], hidden_dims[2]),
                    *vanilla_block(hidden_dims[2], hidden_dims[3]),
                    *vanilla_block(hidden_dims[3], hidden_dims[4], activation=nn.Tanh())))

        self.encodervar = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.fc_var = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.layernormfnin=nn.LayerNorm(d_ff)
        self.layernormfnout=nn.LayerNorm(d_ff)
        self.layernormsum=nn.LayerNorm(d_model)
        
        # 1D Cnn along h dimenions
        self.decoder_input = nn.Conv1d(in_channels=seq_len-1, out_channels=label_len+out_len, stride=1, kernel_size=3,padding=1,padding_mode='circular')

        # Decreasing Dimensions
        modules2 = []


        hidden_dims2 = hidden_dims
        
        modules2.append(
            nn.Sequential(
                *vanilla_block(hidden_dims2[4], hidden_dims2[3]),
                *vanilla_block(hidden_dims2[3], hidden_dims2[2]),
                *vanilla_block(hidden_dims2[2], hidden_dims2[1]),
                nn.Linear(hidden_dims2[1],d_ff),
                )
        )

        self.decodervar = nn.Sequential(*modules2)
        self.varout=nn.Linear(d_ff, enc_in)
        self.varoutsigma=nn.Linear(d_ff, enc_in)
        self.distribution_ressigma = nn.Softplus()

        self.projection = nn.Linear(d_model, enc_in)
        self.projectionmainsigma = nn.Linear(d_model, enc_in)
        self.distribution_mainsigma = nn.Softplus()

        device="cuda:0"
        input_size=enc_in+timefeatureNos
        output_size=enc_in
        num_layers=3
        layer_size=[d_model,d_model,d_model]
        output_activation="linear"
        layer_norm=True
        use_relu=True
        memory_size=256
        k=4
        self._device = device
        self._input_size = input_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._layer_size = layer_size
        self._output_activation = output_activation
        self._layer_norm = layer_norm
        self._use_relu = use_relu
        self._memory_size = memory_size
        self._k = k

        self._Cells = []

        self._add_cell(self._input_size, self._layer_size[0])
        for i in range(1, num_layers):
            self._add_cell(self._layer_size[i-1], self._layer_size[i])

        self._list_of_modules = nn.ModuleList(self._Cells)

        if self._output_activation == "linear":
            self._output_activation_fn = None
        elif self._output_activation == "sigmoid":
            self._output_activation_fn = F.sigmoid
        elif self._output_activation == "LogSoftmax":
            self._output_activation_fn = F.LogSoftmax

        self._W_h2o = nn.Parameter(torch.Tensor(layer_size[-1], output_size))
        self._b_o = nn.Parameter(torch.Tensor(output_size))

        self._W_h2o_sigma = nn.Parameter(torch.Tensor(layer_size[-1], output_size))
        self._b_o_sigma = nn.Parameter(torch.Tensor(output_size))
        self._output_activation_fn_sigma = nn.Sigmoid()
        self._reset_parameters()
        self.print_num_parameters()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x_enc, x_mark_enc,x_dec, x_mark_dec,hidden_init=None):
        x_enccat=torch.concat((x_enc,x_mark_enc),2)
        input=x_enccat.permute(1,0,2)
        batch_size = input.shape[1]
        seq_len = input.shape[0]

        hidden_inits = None
        if hidden_init is not None:
            hidden_inits = torch.chunk(hidden_init, len(self._Cells), dim=0)
        self.reset_hidden(batch_size, hidden_inits)

        outputs = []
        sigmalist=[]
        out_h=[]
        for t in range(seq_len):
            mu,sigma,h=self.step(input[t])
            out_h.append(h)
            outputs.append(mu)
            sigmalist.append(sigma)
        
        #output contruction
        output = torch.stack(outputs)  
        #preds=[]
        mu= output[-1]
        x_mark_dec=x_mark_dec[:,-self.pred_len:,:]
        x_mark_dec=x_mark_dec.permute(1,0,2)
        for t in range(self.pred_len):
            mu,sigma,h=self.step(torch.concat((mu,x_mark_dec[t]),1))
            out_h.append(h)
            outputs.append(mu)
            sigmalist.append(sigma)
        output = torch.stack(outputs)
        sigma =torch.stack(sigmalist)
        out_h=torch.stack(out_h)
        output=output.permute(1,0,2)
        output=output[:,:-1,:]
        sigma=sigma.permute(1,0,2)
        sigma=sigma[:,:-1,:]
        out_h=out_h.permute(1,0,2)
        out_h=out_h[:,:-1,:]
        
        resin=torch.sub(output[:,:-self.pred_len,:],x_enc[:,1:,:])
        resin=torch.concat((resin,x_mark_enc[:,1:,:]),2)
        
        result = self.encodervar(resin)
        resarout = self.fc_mu(result)
        resaroutnew=self.decoder_input(resarout)
        resraw = self.decodervar(resaroutnew)

        res = self.varout(resraw)

        ressigma=self.varoutsigma(resraw)
        ressigma =self.distribution_ressigma(ressigma)


        sample=64
        if(self.enc_in !=1):
            outputsample=torch.distributions.normal.Normal(output,sigma).rsample((sample,)).permute(1,2,3,0)
            ressample=torch.distributions.normal.Normal(res,ressigma).rsample((sample,)).permute(1,2,3,0)
            ressamplestd,ressamplemu=torch.std_mean(ressample,3)
            squares=(outputsample-output.unsqueeze(3))*(ressample-ressamplemu.unsqueeze(3))
            correlationind=squares/(ressamplestd.unsqueeze(3)*sigma.unsqueeze(3))
            correlationvariance, correlation=torch.std_mean(correlationind,3)
        else:

            outputsample=torch.distributions.normal.Normal(output.squeeze(),sigma.squeeze()).rsample((sample,)).permute(1,2,0)
            ressample=torch.distributions.normal.Normal(res.squeeze(),ressigma.squeeze()).rsample((sample,)).permute(1,2,0)
            ressamplestd,ressamplemu=torch.std_mean(ressample,2)
            ressamplemu=ressamplemu.unsqueeze(2)
            ressamplestd=ressamplestd.unsqueeze(2)
            squares=(outputsample-output)*(ressample-ressamplemu)
            correlationind=squares/(ressamplestd*sigma)
            correlationvariance, correlation=torch.std_mean(correlationind,2)
            correlationvariance=correlationvariance.unsqueeze(2)
            correlation=correlation.unsqueeze(2)
        sigma=torch.sqrt(torch.square(ressamplestd)+torch.square(sigma)+(2*correlation*ressamplestd*sigma))
        output=output+ressamplemu
        return output,sigma,correlation,correlationvariance,res


    def step(self, input):
        """Implements forward computation of the model for a single recurrent step.
        Args:
            input of shape (batch, input_size): 
            tensor containing the features of the input sequence
        Returns:
            model output for current time step.
        """
        
        h = []
        h.append(self._Cells[0](input, self._h_prev[0]))
        for i, cell in enumerate(self._Cells):
            if i != 0:
                h.append(cell(h[i-1]["h"], self._h_prev[i]))
        #output = torch.add(torch.mm(h[-1]["h"], self._W_h2o), self._b_o)
        output=self.projection(h[-1]["h"])
        output_sigma=self.projectionmainsigma(h[-1]["h"])
        #output_sigma = torch.add(torch.mm(h[-1]["h"], self._W_h2o_sigma), self._b_o_sigma)

        output_sigma=self._output_activation_fn_sigma(output_sigma)
        if self._output_activation_fn is not None:
            output = self._output_activation_fn(output)
        self._h_prev = h
        return output,output_sigma,h[-1]["h"]

    def reset_hidden(self, batch_size, hidden_inits=None):
        """Resets the hidden state for truncating the dependency."""

        self._h_prev = []
        for i, cell in enumerate(self._Cells):
            self._h_prev.append(cell.reset_hidden(batch_size, hidden_inits[i] if hidden_inits is not None else None))

    def _reset_parameters(self):
        """Initializes the parameters."""

        nn.init.xavier_normal_(self._W_h2o, gain=nn.init.calculate_gain(self._output_activation))
        nn.init.constant_(self._b_o, 0)

        nn.init.xavier_normal_(self._W_h2o_sigma, gain=nn.init.calculate_gain(self._output_activation))
        nn.init.constant_(self._b_o_sigma, 0)

    def register_optimizer(self, optimizer):
        """Registers an optimizer for the model.
        Args:
            optimizer: optimizer object.
        """

        self.optimizer = optimizer

    def _add_cell(self, input_size, hidden_size):
        """Adds a cell to the stack of cells.
        Args:
            input_size: int, size of the input vector.
            hidden_size: int, hidden layer dimension.
        """

        self._Cells.append(NRUCell(self._device, input_size, hidden_size, 
                                              memory_size=self._memory_size, k=self._k,
                                              use_relu=self._use_relu, layer_norm=self._layer_norm))
                                    
    def print_num_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        print("Num_params : {} ".format(num_params))
        return num_params

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Gate(nn.Module):
    def __init__(self,dim,topk,num_experts):
        super().__init__()
        self.dim = dim
        self.topk = int(topk)
        self.weight = nn.Parameter(torch.empty(num_experts,dim))
        self.bias = nn.Parameter(torch.empty(num_experts))
        self.reset_parameters()

    def reset_parameters(self):
        # Router parameters are plain nn.Parameter; initialize explicitly.
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    def forward(self,x):
        # x: (T, dim)
        # probs: (T, E) in float32
        logits = F.linear(x, self.weight, self.bias)
        probs = F.softmax(logits.float(), dim=-1)
        indices = torch.topk(probs, self.topk, dim=-1)[1]
        weights = probs.gather(1, indices)
        # Normalize weights to sum to 1 for stable mixture.
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        return probs, indices, weights

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, shared_hidden, k=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.shared_hidden = shared_hidden
        self.k = k
        # Instantiate routed experts and a shared convolutional expert.
        self.experts = nn.ModuleList([Mlp(in_features=self.input_size, out_features=self.output_size, hidden_features=self.hidden_size) for i in range(self.num_experts)])
        self.shared = MixFFN(in_features=self.input_size, hidden_features=self.shared_hidden, out_features=self.output_size)
        self.adapter = Adapter(input_size,hidden_dim=8)
        self.gate = Gate(input_size,topk=k,num_experts=self.num_experts)
        assert(self.k <= self.num_experts)
        self.last_loss_imp = None
        self.last_loss_load = None

    def forward(self, x):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        B,C,H,W = x.shape
        x_shared = self.shared(x)
        x = x.flatten(2).permute(0,2,1).contiguous()
        L = x.shape[1]
        x = x.view(-1,C)
        probs, indices, weights = self.gate(x)
        weights = weights.to(dtype=x.dtype)

        y = torch.zeros_like(x)
        counts = torch.bincount(
            indices.flatten(), minlength=self.num_experts).tolist()
        for i in range(self.num_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, pos = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, pos, None]
            
        y = y.view(B, L, C)
        y = y.transpose(1, 2).reshape(B, C, H, W).contiguous()

        # Auxiliary balancing terms are cached for callers that use them.
        if self.training:
            eps = 1e-9
            importance = probs.mean(dim=0)  # (E,), float32
            load = F.one_hot(indices, num_classes=self.num_experts).float().mean(dim=(0, 1))  # (E,)

            def _cv2(v: torch.Tensor) -> torch.Tensor:
                mean = v.mean()
                var = v.var(unbiased=False)
                return var / (mean * mean + eps)

            self.last_loss_imp = _cv2(importance)
            self.last_loss_load = _cv2(load)
        else:
            # Prevent stale values being read after eval forwards.
            self.last_loss_imp = None
            self.last_loss_load = None
        
        y = self.adapter(x_shared, y) + x_shared
        
        return y
    
class Adapter(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.adapter_down = nn.Conv2d(dim, hidden_dim, 1)
        self.adapter_down2 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = DWConv(hidden_dim)
        self.adapter_up = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x1,x2):
        x1 = self.adapter_down(x1)
        x2 = self.adapter_down2(x2)
        x2 = self.dwconv(x2)
        x = x1*x2
        x = self.adapter_up(x)
        return x

class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# class ConvBN(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBN, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
#             norm_layer(out_channels)
#         )

def cc2(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    # N,C,img1.shape
    img1 = img1.permute(0,2,1)
    img2 = img2.permute(0,2,1)
    # img1 = img1.reshape(N, C, -1)
    # img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()



class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim
        print("self.num_headsself.num_headsself.num_heads", self.num_heads)
        print("self.embed_dim", self.embed_dim)
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)

        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim [1, 17821, 256]
            l (_type_): bs, n_text, dim[1, 4,256]
            attention_mask_v (_type_, optional): _description_. bs, n_img [1, 17821]
            attention_mask_l (_type_, optional): _description_. bs, n_text [1, 4]
        Returns:
            _type_: _description_
        """

        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size() # [1, 17821, 256]
        query_states = self.v_proj(v) * self.scale # [1, 17821, 1024]
        key_states = self._shape(self.l_proj(l), -1, bsz) # [1, 4, 4, 256]
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz) # [1, 4, 17821, 256]
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz) # [1, 4, 4, 256]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # [4, 17821, 256]
        key_states = key_states.view(*proj_shape) # [ 4, 4, 256]
        value_v_states = value_v_states.view(*proj_shape) # [4, 17821, 256]
        value_l_states = value_l_states.view(*proj_shape) # [ 4, 4, 256]
        
        src_len = key_states.size(1) # 4
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt [4, 17821, 4]
       
 
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range

        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2) # [4, 4, 17821]
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]

        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            # attention_mask_v [4, 1, 17821]
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        attn_weights_l = attn_weights_l.softmax(dim=-1)
        # print("attention_mask_v",attention_mask_v.shape)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        

        attn_weights_v = attn_weights.softmax(dim=-1) # [4, 4, 17821]

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states) # [4, 17821, 256]
        attn_output_l = torch.bmm(attn_probs_l, value_v_states) # [4, 4, 256]

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        img_dim,
        d_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()


        v_dim = img_dim
        l_dim = d_dim

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, img, d, attention_mask_v=None, attention_mask_l=None):
        v = img
        l = d
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        # v, l = v + delta_v, l + delta_l
        
        # self.gamma_v torch.Size([256])
        # self.gamma_l torch.Size([256])
        # delta_v [1, 17821, 256]
        # delta_l [1, 8, 256]

        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        v = v + l
        return v, l




if __name__ == "__main__":
    
    input_data = torch.rand([10, 1024, 64]).cuda()
    input_data_d = torch.rand([10, 1024, 64]).cuda()
    print("input",input_data.shape)
    model = BiAttentionBlock(img_dim=64, d_dim=64, embed_dim=768, num_heads=4).cuda()
    for i in range(10000):
        print(i)
        select_feature = model(input_data, input_data_d)
    

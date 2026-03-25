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

class BiMultiHeadAttention(nn.Module):
    def __init__(self, img_dim, d_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.img_dim = img_dim
        self.d_dim = d_dim
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)

        self.dropout = dropout

        self.img_proj = nn.Linear(self.img_dim, self.embed_dim)
        self.d_proj = nn.Linear(self.d_dim, self.embed_dim)
        # self.values_img_proj = nn.Linear(self.img_dim, self.embed_dim)
        self.values_d_proj = nn.Linear(self.d_dim, self.embed_dim)

        self.out_img_proj = nn.Linear(self.embed_dim, self.img_dim)
        # self.out_d_proj = nn.Linear(self.embed_dim, self.d_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.img_proj.weight)
        self.img_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.d_proj.weight)
        self.d_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.values_img_proj.weight)
        # self.values_img_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_d_proj.weight)
        self.values_d_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_img_proj.weight)
        self.out_img_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.out_d_proj.weight)
        # self.out_d_proj.bias.data.fill_(0)

    def forward(self, img, d, attention_mask_v=None, attention_mask_l=None):
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

        bsz, tgt_len, _ = img.size() # [1, 17821, 256]
        query_states = self.img_proj(img) * self.scale # [1, 17821, 1024]
        key_states = self._shape(self.d_proj(d), -1, bsz) # [1, 4, 4, 256]
        # value_img_states = self._shape(self.values_img_proj(img), -1, bsz) # [1, 4, 17821, 256]
        value_d_states = self._shape(self.values_d_proj(d), -1, bsz) # [1, 4, 4, 256]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # [4, 17821, 256]
        key_states = key_states.view(*proj_shape) # [ 4, 4, 256]
        # value_img_states = value_img_states.view(*proj_shape) # [4, 17821, 256]
        value_d_states = value_d_states.view(*proj_shape) # [ 4, 4, 256]
        
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


        # attn_weights_T = attn_weights.transpose(1, 2) # [4, 4, 17821]
        # attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]

        # if self.clamp_min_for_underflow:
        #     attn_weights_l = torch.clamp(
        #         attn_weights_l, min=-50000
        #     )  # Do not increase -50000, data type half has quite limited range
        # if self.clamp_max_for_overflow:
        #     attn_weights_l = torch.clamp(
        #         attn_weights_l, max=50000
        #     )  # Do not increase 50000, data type half has quite limited range

        
        # # mask vison for language
        # if attention_mask_v is not None:
        #     attention_mask_v = (
        #         attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        #     )
        #     # attention_mask_v [4, 1, 17821]
        #     attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        # attn_weights_l = attn_weights_l.softmax(dim=-1)
        # # print("attention_mask_v",attention_mask_v.shape)

        # # mask language for vision
        # if attention_mask_l is not None:
        #     attention_mask_l = (
        #         attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        #     )
        #     attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        
        # print("attn_weights",attn_weights.shape)
        # print("attention_mask_l",attention_mask_l.shape)

        attn_weights_v = attn_weights.softmax(dim=-1) # [4, 4, 17821]

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        # attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_d_states) # [4, 17821, 256]
        # attn_output_l = torch.bmm(attn_probs_l, value_v_states) # [4, 4, 256]

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        # if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
        #     )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        # attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        # attn_output_l = attn_output_l.transpose(1, 2)
        # attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_img_proj(attn_output_v)
        # attn_output_l = self.out_d_proj(attn_output_l)

        return attn_output_v, attn_probs_v


# Bi-Direction MHA (text->image, image->text)
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


        # pre layer norm
        self.layer_norm_img = nn.LayerNorm(img_dim)
        self.layer_norm_d = nn.LayerNorm(d_dim)
        self.attn = BiMultiHeadAttention(
            img_dim=img_dim, d_dim=d_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((img_dim)), requires_grad=True)
        # self.gamma_l = nn.Parameter(init_values * torch.ones((d_dim)), requires_grad=True)


    def forward(self, img, d, attention_mask_v=None, attention_mask_l=None):

        img = self.layer_norm_img(img)
        d = self.layer_norm_d(d)
        d_select, atten_img2d = self.attn(
            img, d, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        # v, l = v + delta_v, l + delta_l
        
        # self.gamma_v torch.Size([256])
        # self.gamma_l torch.Size([256])
        # delta_v [1, 17821, 256]
        # delta_l [1, 8, 256]

        d_select = img + self.drop_path(self.gamma_v * d_select)
        # l = l + self.drop_path(self.gamma_l * delta_l)
        return d_select, atten_img2d

if __name__ == "__main__":
    
    input_data = torch.rand([10, 1024, 64]).cuda()
    input_data_d = torch.rand([10, 1024, 64]).cuda()
    print("input",input_data.shape)
    model = BiAttentionBlock(img_dim=64, d_dim=64, embed_dim=768, num_heads=4).cuda()
    for i in range(10000):
        print(i)
        select_feature = model(input_data, input_data_d)
    

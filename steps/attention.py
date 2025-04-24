import torch
import torch.nn as nn
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

# query = inputs[1]
# attn_scores_2 = torch.empty(inputs.shape[0])

# for i, x_i in enumerate(inputs):
#     attn_scores_2[i] = torch.dot(query, x_i)

# print(attn_scores_2)
# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())
# attn_scores_2.argmax()

# attention_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights:", attention_weights_2)
# print("Sum:", attention_weights_2.sum())

# context_vec_2 = torch.zeros(inputs.shape[1])

# for i, x_i in enumerate(inputs):
#     context_vec_2 += attention_weights_2[i] * x_i

# print("Context vector:", context_vec_2)

# attn_scores = torch.empty(6, 6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)

# print(attn_scores)
# attn_scores = inputs @ inputs.T
# attn_weights = torch.softmax(attn_scores, dim=-1)
# print(attn_weights)

# row_2_sum = attn_weights[1].sum()
# print("Row 2 sum:", row_2_sum)
# print("All rows sum:", attn_weights.sum(dim=-1))

# all_context_vecs = attn_weights @ inputs
# print(all_context_vecs)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ w_query
# key_2 = x_2 @ w_key
# value_2 = x_2 @ w_value

# print(query_2)

keys = inputs @ w_key
values = inputs @ w_value

# print("Keys shape:", keys.shape)
# print("Values shape:", values.shape)

# keys_2 = keys[1]
# attn_scores_22 = query_2.dot(keys_2)
# print(attn_scores_22)

attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)

# d_k = keys.shape[-1]
# attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)

# context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec
    

# torch.manual_seed(123)
# sa_v1 = SelfAttention_v1(d_in, d_out)

# print(sa_v1(inputs))

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec
    
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# print(sa_v2(inputs))

# Excercise 3.1
# sa1_check = SelfAttention_v1(d_in, d_out)
# sa2_check = SelfAttention_v2(d_in, d_out)
# sa1_check.W_query.data = sa2_check.W_query.weight.T
# sa1_check.W_key.data = sa2_check.W_key.weight.T
# sa1_check.W_value.data = sa2_check.W_value.weight.T

# print(sa1_check(inputs))
# print(sa2_check(inputs))

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

context_length = attn_weights.shape[1]
# mask_simple = torch.tril(torch.ones(context_length, context_length))
# # print(mask_simple)
# masked_simple = attn_weights * mask_simple
# print(masked_simple)

# rows_sum = masked_simple.sum(dim=-1, keepdim=True)
# masked_simple_normalized = masked_simple / rows_sum
# print(masked_simple_normalized)


mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), float("-inf"))
# print(masked)
# 
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

# torch.manual_seed(123)
# dropout = nn.Dropout(0.5)
# example = torch.ones(context_length, context_length)
# print(dropout(example))
# print(dropout(attn_weights))

batch = torch.stack([inputs, inputs], dim=0)
# print(batch.shape)

# sa = SelfAttention_v2(d_in, d_out)
# print(sa(batch))


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vec = ca(batch)
# print(context_vec)

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# torch.manual_seed(123)
# context_length = batch.shape[1]
# d_in, d_out = 3, 2
# mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, num_heads=2, dropout=0.0)
# context_vecs = mha(batch)
# print(context_vecs)
# print(context_vecs.shape)

# Excercise 3.2
# mha = MultiHeadAttentionWrapper(d_in, 1, context_length, num_heads=2, dropout=0.0)
# context_vecs = mha(batch)
# print(context_vecs)
# print(context_vecs.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.d_head = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.d_head)
        queries = queries.view(b, num_tokens, self.num_heads, self.d_head)
        values = values.view(b, num_tokens, self.num_heads, self.d_head)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  
        context_vec = self.out_proj(context_vec)

        return context_vec
    

# a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
# [0.8993, 0.0390, 0.9268, 0.7388],
# [0.7179, 0.7058, 0.9156, 0.4340]],
# [[0.0772, 0.3565, 0.1479, 0.5331],
# [0.4066, 0.2318, 0.4545, 0.9737],
# [0.4606, 0.5159, 0.4220, 0.5786]]]])
# print(a @ a.transpose(2, 3))
# first_head = a[0, 0, :, :]
# first_res = first_head @ first_head.T
# print("First head:\n", first_res)
# second_head = a[0, 1, :, :]
# second_res = second_head @ second_head.T
# print("\nSecond head:\n", second_res)

# torch.manual_seed(123)
# batch_size, context_length, d_in = batch.shape
# d_out = 2
# mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
# context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)

# Excercise 3.3
mha_gpt2 = MultiHeadAttention(d_in=768, d_out=768, context_length=1024, num_heads=12, dropout=0.1)
c = 9
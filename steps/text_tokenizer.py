import re
import string
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
# with open('data/the-verdict.txt', 'r') as file:
#     text = file.read()

# print("Total characters:", len(text))
# print(text[:100])

# text = "Hello, world. Is this-- a test?"
# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# preprocessed = [item for item in preprocessed if item.strip()]

# all_words = sorted(set(preprocessed))
# vocab_size = len(all_words)

# vocab = {token:integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break

# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {integer:token for token, integer in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         ids = [self.str_to_int[item] for item in preprocessed]
#         return ids
    
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[id] for id in ids])
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text
    
# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know", Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))

# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# vocab = {token:integer for integer, token in enumerate(all_tokens)}

# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {integer:token for token, integer in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
#         ids = [self.str_to_int[item] for item in preprocessed]
#         return ids
    
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[id] for id in ids])
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text
    
# tokenizer = SimpleTokenizerV2(vocab)
# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# ids = tokenizer.encode(text)
# print(tokenizer.decode(ids))

# tokenizer = tiktoken.get_encoding("gpt2")
# text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
# tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# strings = tokenizer.decode(tokens)
# print(strings)

# Excercise 1
# text = "Akwirw ier"
# tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# strings = tokenizer.decode(tokens)
# print(strings)

# with open('data/the-verdict.txt', 'r', encoding='utf-8') as file:
#     raw_text = file.read()

# enc_text = tokenizer.encode(raw_text)
# enc_sample = enc_text[50:]
# context_size = 4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     target = enc_sample[i]
#     print(tokenizer.decode(context), "---->", tokenizer.decode([target]))
    
# with open("data/the-verdict.txt", "r", encoding="utf-8") as file:
#     raw_text = file.read()

# dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)

# print(inputs)
# print(targets)

# input_ids = torch.tensor([2, 3, 5, 1])
# vocab_size = 6
# output_dim = 3

# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# print(embedding_layer.weight)
# print(embedding_layer(input_ids))


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader
import torch
from palm_rlhf_pytorch import PaLM

palm = PaLM(
    num_tokens = 20000,
    dim = 512,
    depth = 12,
    flash_attn = True # https://arxiv.org/abs/2205.14135
).cuda()

seq = torch.randint(0, 20000, (1, 2048)).cuda()

loss = palm(seq, return_loss = True)
loss.backward()

# after much training, you can now generate sequences

generated = palm.generate(2048) # (1, 2048)
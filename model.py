import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Loading the necessary classes from task files
from task1 import CausalSelfAttention, FeedforwardModule, LayerNorm
from task2 import RoFormerSinusoidalPositionalEmbedding, GroupQueryAttention

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# First, create a transformer block combining self attention, layernorm and feedforward network
class Block(nn.Module):

    def __init__(self, embed_size, num_heads, use_GQA=False, bias=True) -> None:
        super().__init__()
        # Initialise the attention heads, layernorms and the feedforward network
        # Check for GQA
        if use_GQA:
            self.attn = GroupQueryAttention(embed_size, num_heads, num_groups=4, bias=bias)
        else:
            self.attn = CausalSelfAttention(embed_size, num_heads, bias=bias)
        self.mlp = FeedforwardModule(embed_size)
        self.ln_1 = LayerNorm(embed_size, bias=bias)
        self.ln_2 = LayerNorm(embed_size, bias=bias)

    def forward(self, x):
        # Also, Add residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self,
                vocab_size,
                embed_size, 
                context_len, 
                num_blocks, 
                num_heads, 
                dropout, 
                bias=True,
                use_roformer=False,
                use_GQA=False):
        """
        Args:
        vocab_size -> Size of the vocabulary
        embed_size -> The dimensions of each embedding vector
        context_len -> maximum context length / sentence length
        num_blocks -> Number of Blocks to be considered
        num_heads -> number of heads to be considered
        dropout -> p_dropout for the dropout layer
        bias -> bias to be considered for layernorm
        use_roformer -> whether to use rotary positional embeddings
        use_GQA -> whether to use GroupQUeryAttention
        """
        super().__init__()
        self.block_size = context_len

        # Initialise the transformer
        if use_roformer:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(vocab_size, embed_size), # token embedding table
                wpe = RoFormerSinusoidalPositionalEmbedding(context_len, embed_size), # position embedding table
                drop = nn.Dropout(dropout),
                h = nn.ModuleList([Block(embed_size, num_heads, use_GQA, bias=bias) for _ in range(num_blocks)]),
                ln_f = LayerNorm(embed_size, bias=bias), # final layernorm
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(vocab_size, embed_size), # token embedding table
                wpe = nn.Embedding(context_len, embed_size), # position embedding table
                drop = nn.Dropout(dropout),
                h = nn.ModuleList([Block(embed_size, num_heads, use_GQA, bias=bias) for _ in range(num_blocks)]),
                ln_f = LayerNorm(embed_size, bias=bias), # final layernorm
            ))

        # The final linear layer to map embed_size -> vocab_size
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight
        # init all weights (Reference -> Andrei karpathy's nanogpt repository)
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blocks))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    
    def forward(self, token_idxs, targets=None):
        batch_size, seq_len = token_idxs.shape

        # Get the token and position embeddings and add them
        token_embeddings = self.transformer.wte(token_idxs) # B, T, C
        positional_embeddings = self.transformer.wpe(torch.arange(seq_len, device=device)) # T, C 
        x = token_embeddings + positional_embeddings # shape -> B, T, C
        x = self.transformer.drop(x)
        
        # Forward through the Transformer blocks and then through the final linear layer to obtain logits
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Compute the loss
        if targets is None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # Reference -> nanogpt (Andrei Karpathy)
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            B, T, vocab_size = logits.shape
            loss = F.cross_entropy(logits.view(B*T, vocab_size), targets.view(-1), ignore_index=-1)

        return logits, loss
    

    @torch.no_grad()
    def generate(self, idx, max_new_tokens,
                temperature=1.0, top_k=None):
        # Reference -> nanogpt (Andrei Karpathy)
        # idx.shape -> (B, T) where B is batch size and T is max context length
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # Method to print the num_parameters for testing
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    # Method to initialise weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    
    # Method to load the GPT2-small model, Reference -> Andrei Karpathy's nanogpt repository
    @classmethod
    def from_pretrained(cls, use_roformer=False, use_GQA=False):

        # Get the GPT2 model hyperparameters
        vocab_size = 50257
        embed_size = 768
        context_len = 1024
        num_blocks = 12
        num_heads = 12
        bias = True
        dropout = 0.0 # Not sure about this though
        
        # Download the pretrained GpT2 weights
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % 'gpt2')

        # create a from-scratch initialized minGPT model
        model = GPT(vocab_size=vocab_size,
                    embed_size=embed_size,
                    context_len=context_len,
                    num_blocks=num_blocks,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                    use_roformer=use_roformer,
                    use_GQA=use_GQA)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not ('attn' in k and 'bias' in k)] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not ('attn' in k and 'masked_bias' in k)] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not ('attn' in k and 'bias' in k)] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # for key in sd_keys_hf:
        #     if key not in sd_keys:
        #         print(key)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                if sd_hf[k].shape != sd[k].shape:
                    print(k, sd_hf[k].shape, sd[k].shape)
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


if __name__ == '__main__':

    #Validate Task1: Load the unaltered GPT2
    try:
        print('----Testing for Task1')
        model = GPT.from_pretrained()
        # Get sample predictions
        tst_input = torch.tensor([[0, 1, 3, 5, 223, 13849]], dtype=torch.long)
        out = model.generate(tst_input, max_new_tokens=10)
        print('GPT2-small loaded successfully')
        print('Sample generated tokens: \n', out, '\n')
    except:
        print('Unable to load the model properly.\n')
    
    # Validate Task2, part1: Load GPT2 into model with Roformer
    try:
        print('----Testing for Task2 part1: Use Roformer')
        model = GPT.from_pretrained(use_roformer=True)
        # Get sample predictions
        tst_input = torch.tensor([[0, 1, 3, 5, 223, 13849]], dtype=torch.long)
        out = model.generate(tst_input, max_new_tokens=10)
        print('GPT2-small loaded successfully')
        print('Sample generated tokens: \n', out, '\n')
    except:
        print('Unable to load the model properly.\n')
    
    # Validate Task2, part2: Load GPT2 into model with GQA
    try:
        print('----Testing for Task2 part2: Use GroupQueryAttention')
        model = GPT.from_pretrained(use_GQA=True)
        # Get sample predictions
        tst_input = torch.tensor([[0, 1, 3, 5, 223, 13849]], dtype=torch.long)
        out = model.generate(tst_input, max_new_tokens=10)
        print('GPT2-small loaded successfully')
        print('Sample generated tokens: \n', out)
    except:
        print('Unable to load the model properly.\n')
    

    
    

# SuperAGIsubmission
This is my submission for the SuperAGI Recruitment Assignment.

## Setup
```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
Dependencies:

* pytorch <3
* numpy <3
* transformers for huggingface transformers <3 (to load GPT-2 checkpoints)
* datasets for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
* tiktoken for OpenAI's fast BPE code <3
* wandb for optional logging <3
* tqdm for progress bars <3

Also, I will train this on a small Shakespearan text, referenced from Andrei Karpathy's makemore series.
To download the data run, `python data/shakespeare_char/prepare.py`. This will download the dataset.

The repository contains the following files for solutions:

## task1.py
It has code for implementation of Multi-Head Self Attention (MHA), The Feed-forward module and the LayerNorm layer as per
the original GPT2 architecture proposed in the paper. Andrei Karpathy's nanogpt repository helped me as the major reference in 
implementation

## task2.py
It has code for the following:
  * **Rotary Positional Embeddings**,  Su et. al. RoFormer. The code has been heavily borrowed from https://github.com/JunnYu/RoFormer_pytorch.
    I have slightly modified the class to blend it with my codebase.
  ```
  class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""
    .......
  ```

  * **Group Query Attention, Ainslie et. al. GQA**: Training Generalized Multi-Query Transformer. I refereed to the paper,
    as well as a couple of medium blogs for this. I implemented it myself by making appropriate changes to the MHA class.
    GQA is an interpolation between MHA and MQA. If I set number of groups = number of heads, then it is MHA, and if number of groups = 1,
    then it is MQA (Multi-Query Attention)

  ```
  class GroupQueryAttention(nn.Module):
    """The idea is to group the n_heads queries into G groups and each group will have a single head and value."""
    .......
  ```
Due to time constraints, I was unable to implement the 3rd part of this task, the **Sliding Window Attention Mechanism**. However, I found the paper quite interesting
and I will definitely look into it, especially if it improves computational efficiency.

## model.py
This file contains the code for clubbing all the modules together along with testing. The `Block` class combines the Feedforward Module,
the Attention Module and the Layernorm to form a block of decoder. These blocks are then stacked along with a Linear layer and a LayerNorm layer to 
form the `GPT` class.

```
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
```
At the bottom of the file is the  inference script. The `GPT` class contains a `from_pretrained` function
that loads pre-trained GPT2-small weights into the model. Here is what the inference script looks like:
```
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
```
Run the script with `python model.py`. It will first download the pre-trained weights which
might take some time and then load them into the model.

**Note: The number of parameters remain the same for all 3 cases.**

## train.py

This script contains the training loop for the first two tasks. The one with single GPU and the DDP.
Due to time constraints, I was unable to implement the third option. The code has been heavily borrowed from
Andrei Karpathy's nanogpt repository alongwith a few simplifications for my specific codebase.

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

First

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)





  

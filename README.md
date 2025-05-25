# Encoder-decoder transformer in JAX

This repo implements the encoder-decoder transformer from the attention is all you need paper. The implementation uses some of the flax nnx primitives like nnx.Embed and nnx.Einsum. I didn't use the out-of-the-box layers such as nnx.MultiheadAttention as that would defeat the purpose (of learning).

The training code is directly lifted from https://docs.jaxstack.ai/en/latest/JAX_machine_translation.html . The implemented transformer achieves the same loss and generation quality as the jaxstack implementation which is actually pretty decent ! 

I was able to train a 13.75M parameter model on a 3080 in 4-5mins.
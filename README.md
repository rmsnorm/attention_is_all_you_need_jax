# Encoder-decoder transformer in JAX

This repo implements the encoder-decoder transformer from the attention is all you need paper. The implementation uses some of the flax nnx primitives like nnx.Embed and nnx.Einsum. I didn't use the out-of-the-box layers such as nnx.MultiheadAttention as that would defeat the purpose (of learning).

The jupyter notebook trains the transformer on an English to Spanish translation dataset. The training code is directly lifted from https://docs.jaxstack.ai/en/latest/JAX_machine_translation.html . The implemented transformer achieves the same loss and generation quality as the jaxstack implementation which is actually pretty decent ! 

I was able to train a 2-block, 13.7M parameter model on a 3080 in 4-5mins. This uses tiktoken's cl100k_base tokenizer. Funnily enough, 13M params are from the embeddings itself.

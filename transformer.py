"""Implements the Transformer architecture"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

# Einsum notation shorthand
# b: batch
# n: num_heads
# t: num_query_tokens
# s: num_key_tokens
# h: head dim
# d: d_model
# f: d_ff

@dataclass
class BlockConfig:
    """Block-config for a transformer block"""
    d_model: int
    d_ff: int
    num_heads: int
    head_dim: int
    attn_dropout_rate: float
    mlp_dropout_rate: float
    attn_weights_dtype: jnp.dtype
    mlp_weights_dtype: jnp.dtype

@dataclass
class TransformerConfig:
    """Transformer-config"""
    num_layers: int
    vocab_size: int
    max_seq_len: int
    embedding_dtype: jnp.dtype
    block_cfg: BlockConfig


class Attention(nnx.Module):
    """Attention sub-block"""
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 head_dim: int,
                 dtype: jnp.dtype,
                 dropout_rate: float,
                 rngs: nnx.Rngs):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.w_q = nnx.Einsum(einsum_str="btd, ndh -> btnh",
                              kernel_shape=(num_heads,d_model,head_dim),
                              param_dtype=dtype,
                              rngs=rngs)
        self.w_k = nnx.Einsum(einsum_str="bsd, ndh -> bsnh",
                              kernel_shape=(num_heads,d_model,head_dim),
                              param_dtype=dtype,
                              rngs=rngs)
        self.w_v = nnx.Einsum(einsum_str="bsd, ndh -> bsnh",
                              kernel_shape=(num_heads,d_model,head_dim),
                              param_dtype=dtype,
                              rngs=rngs)
        self.w_o = nnx.Einsum(einsum_str="btnh, ndh -> btd",
                              kernel_shape=(num_heads,d_model,head_dim),
                              param_dtype=dtype,
                              rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, q_btd, k_bsd, v_bsd, attention_mask=None):
        q = self.w_q(q_btd)
        # jax.debug.print("[attn] q={q}", q=q)
        k = self.w_k(k_bsd)
        # jax.debug.print("[attn] k={k}", k=k)
        v = self.w_v(v_bsd)

        attn = jnp.einsum("btnh, bsnh -> bnts", q, k)
        # jax.debug.print("[attn] qk: {attn}", attn=attn)
        attn = attn/jnp.sqrt(self.head_dim)
        # jax.debug.print("[attn] before masking: {attn}", attn=attn)
        if attention_mask is not None:
            attn = jnp.where(attention_mask, attn, jnp.finfo(attn.dtype).min / 2)
        attn = nnx.softmax(attn, axis=-1)
        # jax.debug.print("[attn] after softmax: {attn}", attn=attn)

        o = jnp.einsum("bnts, bsnh -> btnh", attn, v)
        out = self.w_o(o)
        out = self.dropout(out)
        return self.ln(q_btd + out)


class MLP(nnx.Module):
    """MLP sub-block"""
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dtype: jnp.dtype,
                 dropout_rate: float,
                 rngs: nnx.Rngs):
        self.w_up = nnx.Einsum('btd, df -> btf',
                               kernel_shape=(d_model, d_ff),
                               param_dtype=dtype,
                               rngs=rngs)
        self.w_down = nnx.Einsum('btf, fd -> btd',
                                 kernel_shape=(d_ff, d_model),
                                 param_dtype=dtype,
                                 rngs=rngs)
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x_btd):
        x_btf = self.w_up(x_btd)
        x_btf = nnx.relu(x_btf)
        out = self.w_down(x_btf)
        out = self.dropout(out)
        return self.ln(x_btd + out)


class EncoderBlock(nnx.Module):
    """Encoder block."""
    def __init__(self, cfg: BlockConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        self.attn = Attention(d_model=cfg.d_model,
                              num_heads=cfg.num_heads,
                              head_dim=cfg.head_dim,
                              dtype=cfg.attn_weights_dtype,
                              dropout_rate=cfg.attn_dropout_rate,
                              rngs=rngs)
        self.mlp = MLP(d_model=cfg.d_model,
                       d_ff=cfg.d_ff,
                       dtype=cfg.mlp_weights_dtype,
                       dropout_rate=cfg.mlp_dropout_rate,
                       rngs=rngs)

    def __call__(self, x_btd):
        # jax.debug.print("[encoder] x_btd before self attn: {x_btd}", x_btd=x_btd)
        x_btd = self.attn(q_btd=x_btd,
                          k_bsd=x_btd,
                          v_bsd=x_btd,
                          attention_mask=None)
        # jax.debug.print("[encoder] x_btd after self attn: {x_btd}", x_btd=x_btd)
        x_btd = self.mlp(x_btd)
        # jax.debug.print("[encoder] x_btd after mlp: {x_btd}", x_btd=x_btd)
        return x_btd

class DecoderBlock(nnx.Module):
    """Decoder block."""
    def __init__(self, cfg: BlockConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        self.self_attn = Attention(d_model=cfg.d_model,
                              num_heads=cfg.num_heads,
                              head_dim=cfg.head_dim,
                              dtype=cfg.attn_weights_dtype,
                              dropout_rate=cfg.attn_dropout_rate,
                              rngs=rngs)
        self.cross_attn = Attention(d_model=cfg.d_model,
                              num_heads=cfg.num_heads,
                              head_dim=cfg.head_dim,
                              dtype=cfg.attn_weights_dtype,
                              dropout_rate=cfg.attn_dropout_rate,
                              rngs=rngs)
        self.mlp = MLP(d_model=cfg.d_model,
                       d_ff=cfg.d_ff,
                       dtype=cfg.mlp_weights_dtype,
                       dropout_rate=cfg.mlp_dropout_rate,
                       rngs=rngs)

    def get_causal_attn_mask(self, x_btd):
        bsz, seq_len = x_btd.shape[0], x_btd.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = mask[jnp.newaxis, jnp.newaxis, ...]
        mask = jnp.tile(mask, reps=[bsz, self.cfg.num_heads, 1, 1])
        return mask

    def __call__(self, x_btd, encoder_bsd, mask=None):
        causal_mask = self.get_causal_attn_mask(x_btd)
        padding_mask = None
        if mask is not None:
            padding_mask = mask[jnp.newaxis, jnp.newaxis, ...]
            padding_mask = jnp.minimum(padding_mask, causal_mask)

        # jax.debug.print("[decoder] x_btd before self attn: {x_btd}", x_btd=x_btd)
        x_btd = self.self_attn(q_btd=x_btd,
                               k_bsd=x_btd,
                               v_bsd=x_btd,
                               attention_mask=causal_mask)
        # jax.debug.print("[decoder] x_btd after self attn: {x_btd}", x_btd=x_btd)
        x_btd = self.cross_attn(q_btd=x_btd,
                                k_bsd=encoder_bsd,
                                v_bsd=encoder_bsd,
                                attention_mask=padding_mask)
        # jax.debug.print("[decoder] x_btd after cross attn: {x_btd}", x_btd=x_btd)
        x_btd = self.mlp(x_btd)
        # jax.debug.print("[decoder] x_btd after mlp: {x_btd}", x_btd=x_btd)
        return x_btd


class EncDecTransformer(nnx.Module):
    """Encoder-decoder transformer"""
    def __init__(self, cfg: TransformerConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        block_cfg = self.cfg.block_cfg

        self.emb = nnx.Embed(num_embeddings=cfg.vocab_size,
                             features=block_cfg.d_model,
                             dtype=cfg.embedding_dtype,
                             rngs=rngs)

        self.pos_emb = nnx.Embed(num_embeddings=cfg.max_seq_len,
                                 features=block_cfg.d_model,
                                 dtype=cfg.embedding_dtype,
                                 rngs=rngs)

        self.encoder_blocks = [EncoderBlock(block_cfg, rngs) \
                        for _ in range(cfg.num_layers)]

        self.decoder_blocks = [DecoderBlock(block_cfg, rngs) \
                        for _ in range(cfg.num_layers)]

        # self.dense = nnx.Einsum(einsum_str='btd, dv -> btv',
        #                         kernel_shape=(block_cfg.d_model, cfg.vocab_size),
        #                         param_dtype=cfg.embedding_dtype,
        #                         rngs=rngs)

    def __call__(self, encoder_bs, decoder_bt, mask = None):

        encoder_pos = jnp.arange(0, encoder_bs.shape[1])
        decoder_pos = jnp.arange(0, decoder_bt.shape[1])

        encoder_bsd = self.emb(encoder_bs) + self.pos_emb(encoder_pos[jnp.newaxis, ...])
        encoder_bsd *= jnp.sqrt(self.cfg.block_cfg.d_model)
        decoder_btd = self.emb(decoder_bt) + self.pos_emb(decoder_pos[jnp.newaxis, ...])
        decoder_btd *= jnp.sqrt(self.cfg.block_cfg.d_model)

        # jax.debug.print("[Transformer] encoder_bsd: {encoder_bsd}", encoder_bsd=encoder_bsd)
        # jax.debug.print("[Transformer] decoder_btd: {decoder_btd}", decoder_btd=decoder_btd)

        for i in range(self.cfg.num_layers):
            encoder = self.encoder_blocks[i]
            encoder_bsd = encoder(encoder_bsd)

            decoder = self.decoder_blocks[i]
            decoder_btd = decoder(decoder_btd, encoder_bsd, mask)

        logits = self.emb.attend(decoder_btd)
        # logits = self.dense(decoder_btd)
        # jax.debug.print("[Transformer] logits: {logits}", logits=logits)
        return logits

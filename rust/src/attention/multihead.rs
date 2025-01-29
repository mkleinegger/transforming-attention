use candle_core::{Result, Tensor};
use candle_nn::{linear, ops::softmax, Linear, Module, VarBuilder};

use crate::config::Config;

pub struct MultiHeadAttentionBlock {
    n_heads: usize,
    d_model: usize,
    head_size: usize,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
}

impl MultiHeadAttentionBlock {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<MultiHeadAttentionBlock> {
        // TODO: no bias for these linear layers
        let wq = linear(config.d_model, config.d_model, vb.pp("wq"))?;
        let wk = linear(config.d_model, config.d_model, vb.pp("wk"))?;
        let wv = linear(config.d_model, config.d_model, vb.pp("wv"))?;
        let wo = linear(config.d_model, config.d_model, vb.pp("wo"))?;

        assert!(config.d_model % config.n_heads == 0);

        Ok(Self {
            n_heads: config.n_heads,
            d_model: config.d_model,
            head_size: config.d_model / config.n_heads,
            wq,
            wk,
            wv,
            wo,
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // println!("q: {q:?}, \nk: {k:?}, \nmask: {mask:?}");
        let head_size = self.d_model / self.n_heads;
        // (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        let q_prime = self.wq.forward(q)?;
        let k_prime = self.wk.forward(k)?;
        let v_prime = self.wv.forward(v)?;

        let (batch_size, seq_len_decoder, _) = q.dims3()?;

        let query = q_prime
            .reshape((batch_size, seq_len_decoder, self.n_heads, head_size))?
            .transpose(1, 2)?;

        let (_, seq_len_encoder, _) = k.dims3()?;

        let key = k_prime
            .reshape((batch_size, seq_len_encoder, self.n_heads, head_size))?
            .transpose(1, 2)?;

        let value = v_prime
            .reshape((batch_size, seq_len_encoder, self.n_heads, head_size))?
            .transpose(1, 2)?;

        // (batch, n_heads, seq_decoder, headsize)
        let (attention_scores, _raw_attention_scores) =
            self.compute_attention_scores(query, key, value, mask)?;
        // (batch, seq_decoder, d_model), where d_model = n_heads * headsize
        let res = attention_scores.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len_decoder,
            self.d_model,
        ))?;
        self.wo.forward(&res)
    }

    fn compute_attention_scores(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let divisor = Tensor::new((self.head_size as f32).sqrt(), query.device())?;
        let mut attention_scores = query // (batch, head, seq_len_decoder, seq_len_encoder)
            .contiguous()?
            .matmul(&key.t()?.contiguous()?)?
            .broadcast_div(&divisor)?;

        let (batch, head, seq_len_decoder, seq_len_encoder) = attention_scores.dims4()?;

        // println!("att scores: {}", attention_scores);
        attention_scores = match mask {
            Some(m) => masked_fill(
                -1e9f32,                                                           // f32::NEG_INFINITY,
                &m.broadcast_as((batch, head, seq_len_decoder, seq_len_encoder))?, // &m.broadcast_left((query.dim(0)?, self.n_heads))?,
                &attention_scores,
            )?,
            None => attention_scores,
        };
        // println!("filled: {attention_scores}");

        let last_dim = attention_scores.dims().len();
        attention_scores = softmax(&attention_scores, last_dim - 1)?; // last_dim should be 4
        let final_scores = attention_scores // (batch, head, seq_decoder, headsize)
            .contiguous()?
            .matmul(&value.contiguous()?)?;
        // println!("encoder dims: {:?}, decoder dims: {:?}, value dims: {:?}", key.shape(), query.shape());
        // println!("attention_scores {:?}", final_scores.shape());
        Ok((final_scores, attention_scores))
    }
}

fn masked_fill(on_false: f32, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    //
    // 1 0 0 0 0
    // 1 1 0 0 0
    // 1 1 1 0 0
    // 0 0 0 0 0
    // 0 0 0 0 0
    //
    // println!("Masked mask: {}", mask);
    // println!("masked on_true: {}", on_true);

    let shape = mask.shape(); // (batch, 1, seq_len, seq_len)
    let on_false = Tensor::new(on_false, on_true.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(on_true, &on_false)?;
    Ok(m)
}

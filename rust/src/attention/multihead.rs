use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, ops::softmax_last_dim, Linear, Module, VarBuilder};

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
        let wq = linear_no_bias(config.d_model, config.d_model, vb.pp("wq"))?;
        let wk = linear_no_bias(config.d_model, config.d_model, vb.pp("wk"))?;
        let wv = linear_no_bias(config.d_model, config.d_model, vb.pp("wv"))?;
        let wo = linear_no_bias(config.d_model, config.d_model, vb.pp("wo"))?;

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

    /// Calculate multi-head attention scores. Expects input of dimensions
    /// (batch, seq_len, d_model) and returns output of dimensions
    /// (batch, seq_len_q, d_model).
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // println!("q: {q:?}, \nk: {k:?}, \nmask: {mask:?}");
        // (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        let q_prime = self.wq.forward(q)?;
        let k_prime = self.wk.forward(k)?;
        let v_prime = self.wv.forward(v)?;

        let (batch_size, seq_len_q, _d_model) = q.dims3()?;
        let seq_len_k = k.dim(1)?;

        // (batch, n_heads, seq_len, head_size)
        let query = q_prime
            .reshape((batch_size, seq_len_q, self.n_heads, self.head_size))?
            .transpose(1, 2)?;
        let key = k_prime
            .reshape((batch_size, seq_len_k, self.n_heads, self.head_size))?
            .transpose(1, 2)?;
        let value = v_prime
            .reshape((batch_size, seq_len_k, self.n_heads, self.head_size))?
            .transpose(1, 2)?;

        // (batch, n_heads, seq_q, head_size)
        let (attention_scores, _raw_attention_scores) =
            self.compute_attention_scores(query, key, value, mask)?;

        // (batch, seq_q, d_model), where d_model = n_heads * headsize
        let res = attention_scores.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len_q,
            self.d_model,
        ))?;

        // (batch, seq_q, d_model)
        self.wo.forward(&res)
    }

    fn compute_attention_scores(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // (batch, n_heads, seq_q, seq_k)
        let attention_raw = Self::calc_attention(&query, &key, self.head_size as f32)?;
        let (batch, head, seq_len_q, seq_len_k) = attention_raw.dims4()?;

        // println!("att scores: {}", attention_scores);
        // TODO: what happens to tokens of query axis when they are padding tokens? Setting
        // all to -1e9 should result in all getting a value of 1/seq_len_k??? Which results in
        // giving attention to padding tokens.
        let attention = match mask {
            Some(m) => masked_fill(
                -1e9f32,                                               // f32::NEG_INFINITY,
                &m.broadcast_as((batch, head, seq_len_q, seq_len_k))?, // &m.broadcast_left((query.dim(0)?, self.n_heads))?,
                &attention_raw,
            )?,
            None => attention_raw,
        };
        // println!("filled: {attention_scores}");

        let attention_scores = softmax_last_dim(&attention)?;
        let final_scores = attention_scores // (batch, head, seq_decoder, headsize)
            .contiguous()?
            .matmul(&value.contiguous()?)?;
        // println!("encoder dims: {:?}, decoder dims: {:?}, value dims: {:?}", key.shape(), query.shape());
        // println!("attention_scores {:?}", final_scores.shape());
        Ok((final_scores, attention_scores))
    }

    fn calc_attention(query: &Tensor, key: &Tensor, head_size: f32) -> Result<Tensor> {
        let divisor = Tensor::new(head_size.sqrt(), query.device())?;
        query // (batch, head, seq_len_decoder, seq_len_encoder)
            .contiguous()?
            .matmul(&key.t()?.contiguous()?)?
            .broadcast_div(&divisor)
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

    let shape = mask.shape(); // (batch, 1, seq_len_q, seq_len_k)
    let on_false = Tensor::new(on_false, on_true.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(on_true, &on_false)?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result};
    use candle_nn::{
        ops::{softmax, softmax_last_dim},
        VarMap,
    };

    use super::*;

    #[test]
    fn test_compute_attention_scores() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, candle_core::DType::F32, &device);
        let config = Config::default();
        let mha = MultiHeadAttentionBlock::new(vb, &config)?;

        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.], [0., 0., 0.]], &device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let mask = Tensor::new(&[[1u8, 0, 0], [1, 1, 0], [1, 1, 1]], &device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        println!("x: \n{x}\nmask: \n{mask}");
        let (att_scores, att_scores_raw) =
            mha.compute_attention_scores(x.clone(), x.clone(), x.clone(), Some(&mask))?;
        println!("Att scores \n{:}", att_scores_raw);
        println!("Final scores \n{:}", att_scores);

        Ok(())
    }

    #[test]
    fn test_masked_fill() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, candle_core::DType::F32, &device);
        let config = Config::default();
        let mha = MultiHeadAttentionBlock::new(vb, &config)?;

        // (batch, head, seq_len, head_size)
        let scores = Tensor::new(
            &[[1f32, 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]],
            &device,
        )?
        .unsqueeze(0)?
        .unsqueeze(0)?;
        let mask = Tensor::new(&[[1u8, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], &device)?
            .unsqueeze(0)? // head
            .unsqueeze(0)?; // batch

        let on_false = -1e9;
        let masked_scores = masked_fill(on_false, &mask, &scores)?;
        assert_eq!(
            masked_scores.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?,
            &[
                [1f32, 2., 3.],
                [4., 5., on_false],
                [7., on_false, on_false],
                [on_false, on_false, on_false]
            ]
        );
        println!("Masked scores: \n{}", masked_scores);

        Ok(())
    }
}

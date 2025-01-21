use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, Dropout, Linear, Module, VarBuilder};

use crate::config::Config;

pub struct MultiHeadAttention {
    n_heads: usize,
    d_model: usize,
    dropout: Dropout,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<MultiHeadAttention> {
        let wq = linear(config.d_model, config.d_model, vb.pp("wq"))?;
        let wk = linear(config.d_model, config.d_model, vb.pp("wk"))?;
        let wv = linear(config.d_model, config.d_model, vb.pp("wv"))?;
        let wo = linear(config.d_model, config.d_model, vb.pp("wo"))?;

        let dropout = Dropout::new(config.attention_dropout);

        assert!(config.d_model % config.n_heads == 0);

        Ok(Self {
            n_heads: config.n_heads,
            d_model: config.d_model,
            dropout,
            wq,
            wk,
            wv,
            wo,
        })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: bool) -> Result<Tensor> {
        let head_size = self.d_model / self.n_heads;

        // (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        let q_prime = self.wq.forward(q)?;
        let k_prime = self.wk.forward(k)?;
        let v_prime = self.wv.forward(v)?;

        let (batch_size, seq_len, _) = q.dims3()?;

        let query = q_prime
            .reshape((batch_size, seq_len, self.n_heads, head_size))?
            .transpose(1, 2)?;

        let key = k_prime
            .reshape((batch_size, seq_len, self.n_heads, head_size))?
            .transpose(1, 2)?;

        let value = v_prime
            .reshape((batch_size, seq_len, self.n_heads, head_size))?
            .transpose(1, 2)?;

        // calculate mask
        let mask = match mask {
            true => Some(get_mask(seq_len, q.device())?),
            false => None,
        };

        // TODO: implement attention_scores method
        let (attention_scores, raw_attention_scores) =
            self.compute_attention_scores(query, key, value, mask)?;
        let res = attention_scores.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len,
            self.d_model,
        ))?;

        Ok((self.wo.forward(&res))?)
    }

    fn compute_attention_scores(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        todo!("Implement attention scores")
    }
}

fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();

    Tensor::from_slice(&mask, (size, size), device)
}

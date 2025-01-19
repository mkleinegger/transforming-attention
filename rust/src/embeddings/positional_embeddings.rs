use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::Dropout;

pub struct PositionalEmbeddings {
    positional_embeddings: Tensor,
    seq_len: usize,
    d_model: usize,
    dropout: Dropout,
}

impl PositionalEmbeddings {
    pub fn new(seq_len: usize, d_model: usize, dropout: Dropout, device: &Device) -> Result<Self> {
        let positions = Tensor::arange(0f32, seq_len as f32, device)?;
        let denom = ((Tensor::arange(0f32, d_model as f32, device)?
            * (-10_000.0f64.ln() / d_model as f64))?)
            .exp()?;

        let expanded_positions = positions.unsqueeze(1)?;
        let expanded_denom = denom.unsqueeze(0)?;

        let product = (expanded_positions.matmul(&expanded_denom))?;
        println!("Positional Product: {}", product);

        let embeddings_even = product.sin()?;
        let embeddings_odd = product.cos()?;

        // TODO: find better way to initialize Tensor instead of loop unrolling
        let col_first_even = embeddings_even.get_on_dim(1, 0)?;
        let col_first_odd = embeddings_odd.get_on_dim(1, 0)?;
        let mut positional_embeddings: Tensor = Tensor::cat(&[&col_first_even, &col_first_odd], 0)?;

        // iterate over all values in d_model and apply positional embeddings
        for col in 1..(d_model / 2) {
            let col_even = embeddings_even.get_on_dim(1, col * 2)?;
            let col_odd = embeddings_odd.get_on_dim(1, col * 2 + 1)?;

            positional_embeddings = Tensor::cat(&[&positional_embeddings, &col_even], 0)?;
            positional_embeddings = Tensor::cat(&[&positional_embeddings, &col_odd], 0)?;
        }

        positional_embeddings = positional_embeddings
            .reshape((d_model, seq_len))?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        Ok(PositionalEmbeddings {
            positional_embeddings,
            seq_len,
            d_model,
            dropout,
        })
    }

    pub fn forward(&mut self, tensor: Tensor) -> Result<Tensor> {
        let result = (&self.positional_embeddings.i(0)? + tensor)?;
        self.positional_embeddings = result.unsqueeze(0)?;
        self.dropout.forward(&self.positional_embeddings, false)
    }
}

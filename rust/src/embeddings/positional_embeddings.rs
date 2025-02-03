use candle_core::{Device, Result, Tensor};
use candle_nn::Dropout;

use crate::config::Config;

pub struct PositionalEmbeddings {
    positional_embeddings: Tensor,
    // seq_len: usize,
    // d_model: usize,
    dropout: Dropout,
}

impl PositionalEmbeddings {
    pub fn new(config: &Config, device: &Device) -> Result<Self> {
        let positions = Tensor::arange(0f32, config.max_seq_len as f32, device)?;
        let denom = ((Tensor::arange_step(0f32, config.d_model as f32, 1f32, device)?
            * (-(10_000.0f64.ln()) / config.d_model as f64))?)
            .exp()?;

        let expanded_positions = positions.unsqueeze(1)?; // (seq_len, 1)
        let expanded_denom = denom.unsqueeze(0)?; // (1, d_model / 2)

        let product = (expanded_positions.matmul(&expanded_denom))?; // (seq_len, d_model)
        // println!("Positional Product: {:?}", product);

        let embeddings_even = product.sin()?;
        let embeddings_odd = product.cos()?;

        // TODO: find better way to initialize Tensor instead of loop unrolling
        let col_first_even = embeddings_even.get_on_dim(1, 0)?;
        let col_first_odd = embeddings_odd.get_on_dim(1, 1)?;
        let mut positional_embeddings: Tensor = Tensor::cat(&[&col_first_even, &col_first_odd], 0)?;

        // iterate over all values in d_model and apply positional embeddings
        for col in 1..(config.d_model / 2) {
            let col_even = embeddings_even.get_on_dim(1, col * 2)?;
            let col_odd = embeddings_odd.get_on_dim(1, col * 2 + 1)?;

            positional_embeddings = Tensor::cat(&[&positional_embeddings, &col_even], 0)?;
            positional_embeddings = Tensor::cat(&[&positional_embeddings, &col_odd], 0)?;
        }

        // (1, seq_len, d_model)
        positional_embeddings = positional_embeddings
            .reshape((config.d_model, config.max_seq_len))?
            .transpose(0, 1)?;
        // println!("Positional Embeddings: {}", positional_embeddings);

        Ok(PositionalEmbeddings {
            positional_embeddings,
            // seq_len: config.seq_len,
            // d_model: config.d_model,
            dropout: Dropout::new(config.pos_dropout),
        })
    }

    /// Apply positional embeddings to input tensor of shape (batch, seq_len, n_model)
    /// to each batch.
    pub fn forward(&mut self, tensor: Tensor, train: bool) -> Result<Tensor> {
        let result = (&self
            .positional_embeddings
            .narrow(0, 0, tensor.dim(1)?)?
            .unsqueeze(0)?
            .repeat((tensor.dim(0)?, 1, 1))?
            + tensor)?;
        // self.positional_embeddings = result.unsqueeze(0)?;
        self.dropout.forward(&result, train)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::input_embeddings::InputEmbeddings;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};
    use tokenizers::Tokenizer;

    #[test]
    fn verify_pos_embeddings_new() {
        println!("------ test pos_embeddings_new ----------");
        let device = Device::cuda_if_available(0).unwrap();
        let config = Config::default();

        let pe = PositionalEmbeddings::new(&config, &device).unwrap();
        println!("positional embeddings: {}\n", pe.positional_embeddings);
    }

    #[test]
    fn test_pos_embeddings_forward() -> Result<()> {
        let device = Device::cuda_if_available(0).unwrap();
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let config = Config::default();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let sentences = ["The black cat sits outside", "A man is playing guitar"];
        let encodings_batch = tokenizer.encode_batch(sentences.to_vec(), true).unwrap();
        let token_ids_batch = encodings_batch
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let token_ids_batch = Tensor::stack(&token_ids_batch, 0)?; // shape: (2, 7)
        let vocab_size = tokenizer.get_vocab_size(true);
        println!("Token Ids: {}", token_ids_batch);

        let input_embeds = InputEmbeddings::new(vocab_size, &config, vb)?;
        let embeddings = input_embeds.forward(&token_ids_batch, false)?;
        println!("vector embeddings: \n{}\n", embeddings);
        let mut pe = PositionalEmbeddings::new(&config, &device)?;
        println!("pos_embeddings main: \n{}\n", pe.positional_embeddings);
        let encoder_input = pe.forward(embeddings, false)?;
        println!("encoder_input: \n{}\n", encoder_input);

        Ok(())
    }
}

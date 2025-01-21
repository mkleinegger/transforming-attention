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
        let denom = ((Tensor::arange_step(0f32, d_model as f32, 1f32, device)?
            * (-(10_000.0f64.ln()) / d_model as f64))?)
            .exp()?;

        // let pe = Tensor::zeros((seq_len, d_model), candle_core::DType::F64, device)?;

        let expanded_positions = positions.unsqueeze(1)?;
        let expanded_denom = denom.unsqueeze(0)?;

        let product = (expanded_positions.matmul(&expanded_denom))?;
        println!("Positional Product: {:?}", product);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::input_embeddings::InputEmbeddings;
    use candle_core::Device;
    use tokenizers::Tokenizer;

    #[test]
    fn verify_pos_embeddings_new() {
        println!("------ test pos_embeddings_new ----------");
        let device = Device::cuda_if_available(0).unwrap();
        let pe = PositionalEmbeddings::new(8, 512, Dropout::new(0.3), &device).unwrap();
        println!("positional embeddings: {}\n", pe.positional_embeddings);
    }

    #[test]
    fn test_pos_embeddings_forward() {
        let device = Device::cuda_if_available(0).unwrap();
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();

        let encoding = tokenizer
            .encode(("Welcome to the library. ", "test this out"), true)
            .unwrap();
        println!("tok:  {:?}", encoding.get_tokens());
        // tok:  ["welcome", "to", "the", "library", ".", "test", "this", "out"]
        println!("ids:  {:?}", encoding.get_ids());
        // ids:  [5807, 11, 5, 1509, 7, 681, 48, 92]

        let vocab_size = tokenizer.get_vocab_size(true);
        let token_ids = encoding.get_ids();

        let input_embeds = InputEmbeddings::new(vocab_size, 512, &device).unwrap();
        let embeddings = input_embeds.forward(&token_ids, &device).unwrap();
        println!("vector embeddings: \n{}\n", embeddings);
        let mut pe = PositionalEmbeddings::new(8, 512, Dropout::new(0.3), &device).unwrap();
        println!("pos_embeddings main: \n{}\n", pe.positional_embeddings);
        let encoder_input = pe.forward(embeddings).unwrap();
        println!("encoder_input: \n{}\n", encoder_input);
    }
}

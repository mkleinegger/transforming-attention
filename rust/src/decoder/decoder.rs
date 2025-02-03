use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::Config;
use crate::decoder::decoder_block::DecoderBlock;
use crate::normalization::NormalizationLayer;

pub struct Decoder {
    layers: Vec<DecoderBlock>,
    norm: NormalizationLayer,
}

impl Decoder {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.n_decoders)
            .map(|index| DecoderBlock::new(config, vb.pp(format!("{index}"))))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            norm: NormalizationLayer::new(config, vb.device())?,
            layers,
        })
    }

    pub fn forward(
        &self,
        mut xs: Tensor,
        encoder_output: &Tensor,
        tgt_mask: Option<&Tensor>,
        src_mask: Option<&Tensor>,
        train: bool
    ) -> Result<Tensor> {
        for block in self.layers.iter() {
            xs = block.forward(xs, encoder_output, src_mask, tgt_mask, train)?
        }
        self.norm.forward(&xs)
    }
}

// #[cfg(test)]
// mod tests {
//     use core::num;
//
//     use candle_core::Device;
//     use candle_nn::Dropout;
//     use tokenizers::Tokenizer;
//
//     use crate::{
//         embeddings::{input_embeddings::InputEmbeddings, positional_embeddings::PositionalEmbeddings},
//         feed_forward::FeedForwardBlock,
//     };
//
//     use super::*;
//
//     #[test]
//     fn test_decoder() {
//         let d_model = 512usize;
//         let d_ff = 2048usize;
//         let num_heads = 4usize;
//         let dropout = 0.3;
//
//         let device = Device::Cpu;
//         let tokenizer = Tokenizer::from_file("./src/tokenizer/wordlevel-wiki.json").unwrap();
//         let encoding = tokenizer
//             .encode(("Welcome to the library. ", "test this out"), true)
//             .unwrap();
//         println!("tok:  {:?}", encoding.get_tokens());
//         // tok:  ["welcome", "to", "the", "library", ".", "test", "this", "out"]
//         println!("ids:  {:?}", encoding.get_ids());
//         // ids:  [5807, 11, 5, 1509, 7, 681, 48, 92]
//
//         let vocab_size = tokenizer.get_vocab_size(true);
//         let token_ids = encoding.get_ids();
//         let seq_len = encoding.get_ids().len();
//
//         let input_embeds = InputEmbeddings::new(vocab_size, d_model, &device).unwrap();
//         let embeddings = input_embeds.forward(&token_ids, &device).unwrap();
//         println!("vector embeddings: \n{}\n", embeddings);
//         let mut pe = PosEmbeddings::new(seq_len, d_model, Dropout::new(dropout), &device).unwrap();
//         println!("pos_embeddings main: \n{}\n", pe.pos_embeddings);
//         let decoder_input = pe.forward(embeddings).unwrap();
//         println!("Decoder_input: \n{}\n", decoder_input);
//
//         let mut layers = Vec::with_capacity(10);
//         for layer in 0..layers.capacity() {
//             layers.push(
//                 DecoderBlock::new(
//                     MultiHeadAttnBlock::new(d_model, num_heads, dropout, &device).unwrap(),
//                     MultiHeadAttnBlock::new(d_model, num_heads, dropout, &device).unwrap(),
//                     FeedForwardBlock::new(d_model, dropout, d_ff, &device).unwrap(),
//                     dropout,
//                 )
//                 .unwrap(),
//             )
//         }
//         let dummy_encoder_output = decoder_input.clone();
//
//         let decoder = Decoder::new(layers).unwrap();
//         let t = decoder
//             .forward(decoder_input, &dummy_encoder_output, false, true)
//             .unwrap();
//         println!("Decoder_output: \n{}\n", t);
//     }
// }

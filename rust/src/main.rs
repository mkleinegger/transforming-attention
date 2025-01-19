mod embeddings;

use crate::embeddings::input_embeddings::InputEmbeddings;
use crate::embeddings::positional_embeddings::PositionalEmbeddings;
use candle_core::{Device, IndexOp};
use candle_nn::Dropout;
use tokenizers::tokenizer::{Result, Tokenizer};

const D_MODEL: usize = 512;

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    let encoding = tokenizer.encode("Hello Tokenizer! How are you?", true)?;

    println!("tokens: {:?}", encoding.get_tokens());
    println!("ids: {:?}", encoding.get_ids());

    let vocab_size = tokenizer.get_vocab_size(true);
    let device = Device::new_cuda(0)?;

    let token_ids = encoding.get_ids();

    let input_embeddings = InputEmbeddings::new(vocab_size, D_MODEL, &device)?;
    let word_embeddings = input_embeddings.forward(&token_ids, &device)?;
    println!("word embeddings: {word_embeddings:?}");

    let mut positional_embeddings =
        PositionalEmbeddings::new(8, D_MODEL, Dropout::new(0.3), &device)?;
    let encoder_input = positional_embeddings.forward(word_embeddings.i(..8)?)?;

    println!("encoder input: {encoder_input}");

    Ok(())
}

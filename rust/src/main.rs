use candle_core::{DType, Device, IndexOp, Tensor};
use ta::embeddings::positional_embeddings::PositionalEmbeddings;
use ta::{config::Config, embeddings::input_embeddings::InputEmbeddings};

use candle_nn::{Dropout, VarBuilder};
use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;

    let config = Config::default();

    let vocab_size = tokenizer.get_vocab_size(true);
    let device = Device::cuda_if_available(0)?;
    println!("Using GPU: {:?}", !device.is_cpu());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F64, dev);

    // Load Data & Encode directly
    let encoding = tokenizer.encode("Hello Tokenizer! How are you?", true)?;
    let sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ];
    let encodings_batch = tokenizer.encode_batch(sentences.to_vec(), true)?;
    let token_ids_batch = encodings_batch
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &device)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let token_ids = encoding.get_ids();
    println!("tokens: {:?}", encoding.get_tokens());
    println!("ids: {:?}", encoding.get_ids());

    let input_embeddings = InputEmbeddings::new(vocab_size, D_MODEL, &device)?;
    let word_embeddings = input_embeddings.forward(&token_ids, &device)?;
    println!("word embeddings: {word_embeddings:?}");

    let mut positional_embeddings =
        PositionalEmbeddings::new(8, D_MODEL, Dropout::new(0.3), &device)?;
    let encoder_input = positional_embeddings.forward(word_embeddings.i(..8)?)?;

    println!("encoder input: {encoder_input}");
    // TODO: Create Encoders & Decoders

    Ok(())
}

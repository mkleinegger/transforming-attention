// use std::path::Path;

use candle_core::{DType, Device, Tensor};
// use ta::embeddings::positional_embeddings::PositionalEmbeddings;
// use ta::{config::Config, embeddings::input_embeddings::InputEmbeddings};

use candle_nn::{VarBuilder, VarMap};
use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;

    // let config = Config::default();

    let vocab_size = tokenizer.get_vocab_size(true);
    println!("Tokenizer uses vocab of size: {:?}", vocab_size);
    let device = Device::cuda_if_available(0)?;
    println!("Using GPU: {:?}", !device.is_cpu());

    let varmap = VarMap::new();
    let _vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Load Data & Encode directly
    // let encoding = tokenizer.encode("Hello Tokenizer! How are you?", true)?;
    let sentences = [
        "The black cat sits outside",
        "A man is playing guitar",
        "I love pasta today",
        // "The new movie is awesome",
        // "The cat plays in the garden",
        // "A woman watches TV",
        // "The new movie is so great",
        // "Do you like pizza?",
    ];
    let encodings_batch = tokenizer.encode_batch(sentences.to_vec(), true)?;
    let token_ids_batch = encodings_batch
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &device)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let token_ids_batch = Tensor::stack(&token_ids_batch, 0)?;

    // let token_ids = encoding.get_ids();
    // println!("tokens: {:?}", encoding.get_tokens());
    // println!("ids: {:?}", encoding.get_ids());
    println!("Token ids: {}", token_ids_batch);

    // let input_embeddings = InputEmbeddings::new(vocab_size, &config, vb.pp("input_embeddings"))?;
    // let word_embeddings = input_embeddings.forward(&token_ids_batch)?;
    // println!("word embeddings: {word_embeddings}");
    //
    // let mut positional_embeddings = PositionalEmbeddings::new(&config, &device)?;
    // let encoder_input = positional_embeddings.forward(word_embeddings, false)?;
    //
    // println!("encoder input: {encoder_input}");
    //
    // let save_path =
    //     Path::new("/home/lukas/Programming/uni/transforming-attention/rust/tmp/weights");
    // println!("Saving weights at {}", save_path.display());
    // varmap.save(save_path)?;
    //
    // let loaded = candle_core::safetensors::load(save_path, &device)?;
    // // let loaded_varmap = VarBuilder::from_mmaped_safetensors(save_path, DType::F32, &device);
    // // let loaded_vb = VarBuilder::from_tensors(loaded.clone(), DType::F32, &device);
    // println!("loaded: {:?}", loaded);

    Ok(())
}

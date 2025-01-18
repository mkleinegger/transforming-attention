use candle_core::Device;
use tokenizers::tokenizer::{Result, Tokenizer};
mod embeddings;
use embeddings::input_embeddings::InputEmbeddings;

const D_MODEL: usize = 512;

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    let encoding = tokenizer.encode(("Hello Tokenizer!", "Testing out if this works."), true)?;

    println!("tokens: {:?}", encoding.get_tokens());
    println!("ids: {:?}", encoding.get_ids());

    let vocab_size = tokenizer.get_vocab_size(true);
    let device = Device::Cpu;

    let input_embeddings = InputEmbeddings(vocab_size, D_MODEL, &device);

    Ok(())
}

// fn test_candle() -> Result<(), Box<dyn std::error::Error>> {
//     println!("Hello, world!");
//
//     let device = Device::Cpu;
//     let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
//     let b = Tensor::randn(0f32, 1., (3, 4), &device)?;
//
//     let c = a.matmul(&b)?;
//     println!("{c}");
//     Ok(())
// }

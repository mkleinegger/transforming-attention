use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    let encoding = tokenizer.encode(("Hello Tokenizer!", "Testing out if this works."), true)?;

    println!("tokens: {:?}", encoding.get_tokens());
    println!("ids: {:?}", encoding.get_ids());

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

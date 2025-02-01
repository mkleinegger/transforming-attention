use candle_core::{DType, Device, Tensor, IndexOp};
use candle_nn::VarBuilder;
use ta::transformer::Transformer;
use ta::config::Config;
use ta::data::TranslationDataset;
use polars::prelude::*;
use std::path::PathBuf;
use anyhow::Result;
use clap::builder::ValueParser;
use clap::{value_parser, Arg};
use std::fs::File;

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {:?}", device);

    let config = Config::default();

    let matches = clap::Command::new("predict")
    .about("Inference on bilingual candle transformer")
    .bin_name("predict")
    .styles(Default::default())
    .arg_required_else_help(true)
    .arg(
        Arg::new("data")
            .value_name("FILE")
            .help("Parquet dataset containing source and target tokens.")
            .short('d')
            .long("data")
            .required(true)
            .value_parser(value_parser!(PathBuf)),
    )
    .arg(
        Arg::new("model")
            .value_name("FILE")
            .help("Path to the models weights")
            .short('m')
            .long("model")
            .required(true)
            .value_parser(ValueParser::path_buf()),
    )
    .get_matches();

    let data_file = matches.get_one::<PathBuf>("data").unwrap();
    let model_file = matches.get_one::<PathBuf>("model").unwrap();


    // load weights
    let loaded = candle_core::safetensors::load(model_file, &device)?;
    let vb = VarBuilder::from_tensors(loaded.clone(), DType::F32, &device);
    println!("vb: {:?}", vb.get((33709, 512), "encode_embeddings"));

    // oad transformer
    let mut transformer = Transformer::new(vb, &config, 33709)?;

    // load dataset
    let dataset = TranslationDataset::new(data_file, config.max_seq_len)?;

    let src = dataset.src.iter().cloned().collect::<Vec<_>>();
    let tgt = dataset.tgt.iter().cloned().collect::<Vec<_>>();

    let mut generated_data: Vec<Vec<i64>> = Vec::new();
    let mut target_data: Vec<Vec<i64>> = Vec::new();

    let num_samples = 10;
    println!("Starting inference on {} samples...", num_samples);

    for (i, input_tensor) in src.iter().take(num_samples).enumerate() {
        let input_tensor = input_tensor.to_device(&device)?;

        let mut input_tensor = input_tensor.unsqueeze(0)?; // (1, seq_len)

        let target_tensor = tgt[i].to_device(&device)?;
        target_data.push(target_tensor.to_vec1()?);

        let max_length = 50;
        let eos_token_id = 33708; // eos

        // start decoding
        let mut output_tokens = vec![eos_token_id];
        let mut next_token = Tensor::from_vec(output_tokens.clone(), (1, 1), &device)?;
        let max_seq_len = 512;

        for _ in 0..max_length {
            let predictions = transformer.forward(&input_tensor, &next_token, true, false, false)?;
            let last_token_logits = predictions.i((0, predictions.dim(1)? - 1))?;

            let next_token_id = last_token_logits.argmax(0)?.to_dtype(candle_core::DType::I64)?.to_scalar::<i64>()?;

            output_tokens.push(next_token_id);

            if next_token_id == eos_token_id {
                break;
            }

            // Append new token to the input sequence
            next_token = Tensor::from_vec(vec![next_token_id], (1, 1), &device)?;

            input_tensor = Tensor::cat(&[&input_tensor, &next_token], 1)?;

            // Ensure `input_tensor` does not exceed `max_seq_len`
            if input_tensor.dim(1)? > max_seq_len {
                input_tensor = input_tensor.narrow(1, input_tensor.dim(1)? - max_seq_len, max_seq_len)?;
            }
        }

        generated_data.push(output_tokens);
    }


    // let generated_series: Series = generated_data
    //     .iter()
    //     .map(|v| Series::new("", v.clone()))
    //     .collect::<ListChunked>()
    //     .into_series();

    // let target_series: Series = target_data
    //     .iter()
    //     .map(|v| Series::new("", v.clone()))
    //     .collect::<ListChunked>()
    //     .into_series();

    let s1 = Column::new("generated".into(), &generated_data.iter().map(|l| l));
    let s2 = Column::new("target".into(), &target_data);

    let df = DataFrame::new(vec![s1, s2])?;

    // Save DataFrame as a Parquet file
    let file = File::create("output.parquet")?;
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(&mut df)?;


    Ok(())
}

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use clap::builder::ValueParser;
use clap::{value_parser, Arg};
use core::num;
use polars::prelude::*;
use std::fs::File;
use std::path::PathBuf;
use ta::config::Config;
use ta::data::{TranslationDataset, Vocabulary};
use ta::transformer::Transformer;
use ta::util::predict_next_token;

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    // let device = Device::Cpu;
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
        .arg(
            Arg::new("vocab")
                .value_name("FILE")
                .help("Path to the Vocabulary")
                .short('t')
                .long("vocab")
                .required(true)
                .value_parser(ValueParser::path_buf()),
        )
        .get_matches();

    let data_file = matches.get_one::<PathBuf>("data").unwrap();
    let model_file = matches.get_one::<PathBuf>("model").unwrap();
    let vocab_file = matches.get_one::<PathBuf>("vocab").unwrap();

    // load weights
    let loaded = candle_core::safetensors::load(model_file, &device)?;
    let vb = VarBuilder::from_tensors(loaded.clone(), DType::F32, &device);
    // println!("vb: {:?}", vb.get((33709, 512), "encode_embeddings"));

    // load transformer
    let mut transformer = Transformer::new(vb, &config, 33709)?;

    // load dataset
    let dataset = TranslationDataset::new(data_file, config.max_seq_len)?;

    let src = dataset.src.to_vec();
    let tgt = dataset.tgt.to_vec();

    let mut generated_data: Vec<Vec<i64>> = Vec::new();
    let mut target_data: Vec<Vec<i64>> = Vec::new();

    let num_samples = 50;
    let num_skip = 50;
    println!("Starting inference on {} samples...", num_samples);

    for (i, input_tensor) in src.iter().take(num_samples).enumerate() {
        let input_tensor = input_tensor.to_device(&device)?;

        let mut input_tensor = input_tensor.unsqueeze(0)?; // (1, seq_len)

        let target_tensor = tgt[i].to_device(&device)?;
        target_data.push(target_tensor.to_vec1()?);

        let max_length = 50;
        let eos_token_id = 1; // eos
        let sos_token_id = 33708;

        // start decoding
        let mut output_tokens = vec![sos_token_id];
        let mut next_token = Tensor::from_vec(output_tokens.clone(), (1, 1), &device)?;
        let max_seq_len = 512;

        for _ in 0..max_length {
            let predictions = transformer.forward(&input_tensor, &next_token, true, true, false)?;
            let last_token_logits = predictions.i((0, predictions.dim(1)? - 1))?;

            let next_predictions = predict_next_token(&last_token_logits, &output_tokens);
            let next_token_id = *next_predictions?.first().unwrap();

            output_tokens.push(next_token_id);

            if next_token_id == eos_token_id {
                break;
            }

            // Append new token to the input sequence
            next_token = Tensor::from_vec(vec![next_token_id], (1, 1), &device)?;
            input_tensor = Tensor::cat(&[&input_tensor, &next_token], 1)?;

            // Ensure `input_tensor` does not exceed `max_seq_len`
            if input_tensor.dim(1)? > max_seq_len {
                input_tensor =
                    input_tensor.narrow(1, input_tensor.dim(1)? - max_seq_len, max_seq_len)?;
            }
        }

        generated_data.push(output_tokens);
    }

    let vocabulary = Vocabulary::new(vocab_file.to_str().unwrap());
    let decoded: Vec<String> = generated_data
        .iter()
        .map(|v| v.iter().map(|i| *i as usize).collect::<Vec<usize>>())
        .map(|v| vocabulary.decode(v[1..v.len()-1].to_vec()).unwrap())
        .collect();

    let target_decoded: Vec<String> = target_data
        .iter()
        // .skip(num_skip)
        .map(|v| v.iter().map(|i| *i as usize).collect::<Vec<usize>>())
        .map(|v| vocabulary.decode(v).unwrap())
        .collect();

    let src_decoded: Vec<String> = src
        .iter()
        // .skip(num_skip)
        .map(|t| t.to_vec1::<i64>().unwrap())
        .map(|v| v.iter().map(|u| *u as usize).collect())
        .map(|s| vocabulary.decode(s).unwrap())
        .collect::<Vec<String>>();

    let combined = decoded
        .iter()
        .zip(target_decoded.iter())
        .zip(src_decoded.iter())
        .map(|((p, t), s)| (p, t, s))
        .map(|(p, t, s)| format!("Source: {}\nPrediction: {}\nOriginal: {}", s, p, t))
        .for_each(|s| println!("{s}\n"));

    println!("{:?}", combined);

    // println!("Source: \n{}", decoded.join("\n"));
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

    // let s1 = Column::new("generated".into(), &generated_data.iter().map(|l| l));
    // let s2 = Column::new("target".into(), &target_data);
    //
    // let df = DataFrame::new(vec![s1, s2])?;
    //
    // // Save DataFrame as a Parquet file
    // let file = File::create("output.parquet")?;
    // ParquetWriter::new(file)
    //     .with_compression(ParquetCompression::Zstd(None))
    //     .finish(&mut df)?;

    Ok(())
}

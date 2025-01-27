use candle_core::{DType, Device, IndexOp, Tensor};
use candle_datasets::nlp::tinystories::{Dataset, DatasetRandomIter};
use candle_optimisers::adam::{Adam, ParamsAdam};
use polars::prelude::*;
use polars::prelude::{col, LazyFrame};
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;
use ta::data::{Batch, BatchSampler, DataLoader, HuggingfaceDataset, TranslationDataset};
use ta::embeddings::positional_embeddings::PositionalEmbeddings;
use ta::transformer::Transformer;
use ta::{config::Config, embeddings::input_embeddings::InputEmbeddings};

use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let config = Config::default();
    let mut tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    // tokenizer.with_padding(Some(PaddingParams {
    //     strategy: PaddingStrategy::Fixed(config.seq_len),
    //     ..Default::default()
    // }));
    let vocab_size = tokenizer.get_vocab_size(true);
    // println!("Tokenizer uses vocab of size: {:?}", vocab_size);
    let device = Device::cuda_if_available(0)?;
    println!("Using GPU: {:?}", !device.is_cpu());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // let dataset = HuggingfaceDataset::new("wmt/wmt14")?;
    // let dataloader = DataLoader::new(
    //     dataset.train.collect()?.head(Some(16)).lazy(),
    //     &tokenizer,
    //     4,
    //     256,
    // )?;

    let dataset = TranslationDataset::new(
        "/home/lukas/Programming/uni/transforming-attention/data/translate_ende_small.parquet",
    )?;
    let src = dataset.src.iter().cloned().collect();
    let tgt = dataset.tgt.iter().cloned().collect();
    let batch_sampler = BatchSampler::new(dataset, 512)?;

    let batch_samples: Vec<_> = batch_sampler.into_iter().collect();
    let batches: Vec<Batch> = batch_samples
        .iter()
        .map(|s| BatchSampler::collate(s, &src, &tgt))
        .collect();
    println!("Batches: {:?}", batches);

    let mut transformer = Transformer::new(vb, &config, vocab_size)?;
    let mut optimizer = Adam::new(
        varmap.all_vars(),
        ParamsAdam {
            lr: 0.004,
            ..Default::default()
        },
    )?;

    let n_epochs = 200;
    for _epoch in 0..n_epochs {
        let mut total_loss = 0f32;
        for batch in &batches {
            let src = batch.source.to_device(&device)?; // (batch, seq_len)
            let tgt = batch.target.to_device(&device)?; // (batch, seq_len)

            // Teacher forcing, input to decoder is everything except last token,
            // labels are everything except first token.
            let tgt_in = tgt.narrow(1, 0, tgt.dim(1)? - 1)?; // (batch, seq_len-1)
            let tgt_out = tgt.narrow(1, 1, tgt.dim(1)? - 1)?; // (batch, seq_len-1)

            // TODO: correct masking
            let encoded = transformer.encode(&src, false, true)?;
            let decoded = transformer.decode(&tgt_in, &encoded, false, false, true)?;
            let mut predictions = transformer.project(&decoded)?; // (batch, seq_len - 1, vocab_size)

            // Reshape predictions and labels
            predictions = predictions.reshape(((), predictions.dim(2)?))?; // (batch * (seq_len - 1), vocab_size)
            let tgt_out = tgt_out.reshape(((),))?; // (batch * (seq_len - 1))

            let loss = loss::cross_entropy(&predictions, &tgt_out)?;
            optimizer.backward_step(&loss)?;
            total_loss += loss.to_scalar::<f32>()?;
        }
        println!(
            "=== Epoch {_epoch}/{n_epochs} Total loss: {} ===",
            total_loss
        );
    }

    // TODO: calculate metrics
    let save_path =
        Path::new("/home/lukas/Programming/uni/transforming-attention/rust/tmp/weights");
    println!("Saving weights at {}", save_path.display());
    varmap.save(save_path)?;

    // let loaded = candle_core::safetensors::load(save_path, &device)?;
    // let loaded_varmap = VarBuilder::from_mmaped_safetensors(save_path, DType::F32, &device);
    // let loaded_vb = VarBuilder::from_tensors(loaded.clone(), DType::F32, &device);
    // println!("loaded: {:?}", loaded);

    Ok(())
}

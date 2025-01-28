use candle_core::{DType, Device};
use candle_optimisers::adam::{Adam, ParamsAdam};
use std::path::Path;
use ta::config::Config;
use ta::data::{Batch, BatchSampler, TranslationDataset};
use ta::transformer::Transformer;

use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let config = Config::default();
    let device = Device::cuda_if_available(0)?;
    println!("Using GPU: {:?}", !device.is_cpu());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let dataset = TranslationDataset::new(
        "/home/lukas/Programming/uni/transforming-attention/data/translate_ende_small.parquet",
        config.max_seq_len,
    )?;

    println!("Loading src");
    let src = dataset.src.iter().cloned().collect();
    println!("Loading tgt");
    let tgt = dataset.tgt.iter().cloned().collect();
    let batch_sampler = BatchSampler::new(dataset, config.batch_size)?;

    println!("Creating Batches");
    let batch_samples: Vec<_> = batch_sampler.into_iter().collect();
    println!("Collected batch_samples");
    let batches: Vec<Batch> = batch_samples
        .iter()
        .map(|s| BatchSampler::collate(s, &src, &tgt))
        .collect();

    println!("Create Transformer");
    let mut transformer = Transformer::new(vb, &config, 33709)?;
    let mut optimizer = Adam::new(
        varmap.all_vars(),
        ParamsAdam {
            lr: 0.004,
            ..Default::default()
        },
    )?;

    let mut total_batches = 0;
    let num_batches = batch_samples.len();
    for epoch in 0..i32::MAX {
        let mut total_loss = 0f32;
        for (i, batch) in batches.iter().enumerate() {
            if total_batches >= config.max_steps {
                break;
            }

            let src = batch.source.to_device(&device)?; // (batch, seq_len)
            let tgt = batch.target.to_device(&device)?; // (batch, seq_len)

            // Teacher forcing, input to decoder is everything except last token,
            // labels are everything except first token.
            let tgt_in = tgt.narrow(1, 0, tgt.dim(1)? - 1)?; // (batch, seq_len-1)
            let tgt_out = tgt.narrow(1, 1, tgt.dim(1)? - 1)?; // (batch, seq_len-1)

            let mut predictions = transformer.forward(&src, &tgt_in, true, true, true)?;
            // let encoded = transformer.encode(&src, true, true)?;
            // let decoded = transformer.decode(&tgt_in, &encoded, true, true, true)?;
            // let mut predictions = transformer.project(&decoded)?; // (batch, seq_len - 1, vocab_size)

            // Reshape predictions and labels
            predictions = predictions.reshape(((), predictions.dim(2)?))?; // (batch * (seq_len - 1), vocab_size)
            let tgt_out = tgt_out.reshape(((),))?; // (batch * (seq_len - 1))

            let loss = loss::cross_entropy(&predictions, &tgt_out)?;
            optimizer.backward_step(&loss)?;
            let loss = loss.to_scalar::<f32>()?;

            if total_batches % config.log_x_steps == 0 {
                println!("--- Epoch {epoch} Step {i}/{num_batches} Total Steps {total_batches}/{} loss: {loss} ---", config.max_steps);
                let save_path = Path::new(
                    "/home/lukas/Programming/uni/transforming-attention/rust/tmp/weights",
                );
                println!("Saving weights at {}", save_path.display());
                varmap.save(save_path)?;
            }

            // TODO: set correct learning rate asd
            optimizer.set_learning_rate(optimizer.learning_rate() * 0.9);

            total_loss += loss;
            total_batches += 1;
        }
        if total_batches >= config.max_steps {
            break;
        }
        println!("=== Epoch {epoch} Total loss: {} ===", total_loss);
    }

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

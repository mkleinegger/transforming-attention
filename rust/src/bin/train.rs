use candle_core::{DType, Device, Tensor};
use candle_optimisers::adam::{Adam, ParamsAdam};
use clap::builder::ValueParser;
use clap::{value_parser, Arg};
use indicatif::{MultiProgress, ProgressBar};
use indicatif_log_bridge::LogWrapper;
use log::info;
use std::path::PathBuf;
use ta::config::Config;
use ta::data::{Batch, BatchSampler, TranslationDataset};
use ta::transformer::Transformer;
use ta::util::progress_bar_style;

use candle_core::Result;
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};

use env_logger::Env;

fn main() -> Result<()> {
    let env = Env::default().default_filter_or("info");
    let logger = env_logger::Builder::from_env(env).build();
    let level = logger.filter();
    let multi = MultiProgress::new();
    LogWrapper::new(multi.clone(), logger).try_init().unwrap();
    log::set_max_level(level);

    let config = Config::default();
    let device = Device::cuda_if_available(0)?;

    let matches = clap::Command::new("train")
        .about("Train candle transformer on bilingual data")
        .bin_name("train")
        .styles(Default::default())
        .arg_required_else_help(true)
        .arg(
            Arg::new("train")
                .value_name("FILE")
                .help("Training file of bilingual dataset used for training the transformer")
                .short('t')
                .long("train")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("checkpoint")
                .value_name("FILE")
                .help("Checkpoint dir where model files will be written to")
                .short('c')
                .long("check")
                .required(true)
                .value_parser(ValueParser::path_buf()),
        )
        .get_matches();
    let train_file = matches.get_one::<PathBuf>("train").unwrap();
    let check_dir = matches.get_one::<PathBuf>("checkpoint").unwrap();

    info!("Using GPU: {:?}", !device.is_cpu());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let dataset = TranslationDataset::new(train_file, config.max_seq_len).unwrap();

    info!("Loading src");
    let src = dataset.src.to_vec();
    info!("Loading tgt");
    let tgt = dataset.tgt.to_vec();
    let batch_sampler = BatchSampler::new(dataset, config.batch_size)?;

    info!("Creating Batches");
    let batch_samples: Vec<_> = batch_sampler.into_iter().collect();

    info!("Collating batches");
    let progress_batches = multi.add(ProgressBar::new(batch_samples.len() as u64));
    progress_batches.set_style(progress_bar_style("Creating Batches"));
    let batches: Vec<Batch> = batch_samples
        .iter()
        .map(|s| {
            progress_batches.inc(1);
            BatchSampler::collate(s, &src, &tgt)
        })
        .collect();

    info!("Create Transformer");
    let mut transformer = Transformer::new(vb, &config, 33709)?;
    let mut optimizer = Adam::new(
        varmap.all_vars(),
        ParamsAdam {
            beta_1: 0.9,
            beta_2: 0.98,
            eps: 10e-9,
            ..Default::default()
        },
    )?;

    let mut total_batches = 0;
    let num_batches = batch_samples.len();
    let progress_steps = multi.add(ProgressBar::new(config.max_steps as u64));
    progress_steps.set_style(progress_bar_style("Steps"));
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

            let loss = calculate_loss(&predictions, &tgt_out, 0)?;
            optimizer.set_learning_rate(
                (config.d_model as f64).powf(-0.5)
                    * f64::min(
                        (total_batches as f64).powf(-0.5),
                        total_batches as f64 * 4000f64.powf(-1.5),
                    ),
            );
            optimizer.backward_step(&loss)?;
            let loss = loss.to_scalar::<f32>()?;

            if total_batches % config.log_x_steps == 0 {
                info!("--- Epoch {epoch} Step {i}/{num_batches} Total Steps {total_batches}/{} loss: {loss} ---", config.max_steps);
                let save_path = check_dir.join(format!("step_{total_batches}.safetensor"));
                info!("Saving weights at {}", save_path.display());
                varmap.save(save_path)?;
            }

            total_loss += loss;
            total_batches += 1;
            progress_steps.inc(1);
        }
        if total_batches >= config.max_steps {
            break;
        }
        info!("=== Epoch {epoch} Total loss: {} ===", total_loss);
    }

    fn calculate_loss(
        predictions: &Tensor,
        target: &Tensor,
        padding_token: usize,
    ) -> Result<Tensor> {
        let vec: Vec<i64> = target.to_vec1()?;
        let filtered: Vec<_> = vec
            .iter()
            .enumerate()
            .filter(|(_i, index)| **index != (padding_token as i64))
            .map(|(i, _index)| i as u32)
            .collect();

        let indices: Tensor = Tensor::new(filtered, predictions.device())?;

        let target_filtered = target.index_select(&indices, 0)?;
        let predictions_filtered = predictions.index_select(&indices, 0)?;
        loss::cross_entropy(&predictions_filtered, &target_filtered)
    }

    let save_path = check_dir.join("final.safetensor");
    info!("Saving weights at {}", save_path.display());
    varmap.save(&save_path)?;

    let loaded = candle_core::safetensors::load(&save_path, &device)?;
    // let loaded_varmap = VarBuilder::from_mmaped_safetensors(save_path, DType::F32, &device);
    let loaded_vb = VarBuilder::from_tensors(loaded.clone(), DType::F32, &device);
    println!("loaded: {:?}", loaded);

    Ok(())
}

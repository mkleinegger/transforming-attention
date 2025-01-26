use std::path::{Path, PathBuf};

use candle_core::{Device, Result, Tensor};
use candle_hf_hub::api::sync::Api;
use polars::prelude::*;
use tokenizers::Tokenizer;

pub struct HuggingfaceDataset {
    pub dataset: String,
    pub train: LazyFrame,
    pub test: LazyFrame,
    pub validation: LazyFrame,
}

impl HuggingfaceDataset {
    pub fn new(dataset: &str) -> anyhow::Result<Self> {
        let api = Api::new()?;
        let repo = api.dataset(String::from(dataset));
        let data = repo.info();

        let mut train_files: Vec<PathBuf> = Vec::new();
        let mut test_file: Option<PathBuf> = None;
        let mut validation_file: Option<PathBuf> = None;

        for sibling in data?.siblings {
            if sibling.rfilename.contains("de-en") {
                let path = repo.get(&sibling.rfilename)?;
                println!("{:?}", &sibling.rfilename);
                if sibling.rfilename.contains("train") {
                    train_files.push(path);
                } else if sibling.rfilename.contains("test") {
                    test_file = Some(path);
                } else if sibling.rfilename.contains("validation") {
                    validation_file = Some(path);
                }
            }
        }

        let train_frames: Vec<LazyFrame> = train_files
            .iter()
            .map(|path| Self::load(path).unwrap())
            .collect();
        let train = concat(train_frames, UnionArgs::default())?;

        let test = Self::load(&test_file.unwrap())?;
        let validation = Self::load(&validation_file.unwrap())?;
        let dataset = String::from(dataset);

        Ok(Self {
            dataset,
            test,
            validation,
            train,
        })
    }

    fn load(path: &PathBuf) -> PolarsResult<LazyFrame> {
        // Load from hf directly:
        // LazyFrame::scan_parquet(
        //     "hf://datasets/wmt/wmt14/de-en/test-00000-of-00001.parquet",
        //     Default::default(),
        // )

        // Load from local cache:
        LazyFrame::scan_parquet(
            path,
            // "/home/lukas/Programming/uni/transforming-attention/rust/resource/validation*.parquet",
            Default::default(),
        )
    }
}

#[derive(Debug)]
pub struct Batch {
    // (batch, seq_len, d_model)
    pub source: Tensor,
    pub target: Tensor,
}

/// Load a Dataset from a polars LazyFrame, tokenizes it and transforms it into batches, which
/// can be iterated over to pass into a ML model.
pub struct DataLoader {
    de_tokens: Vec<Tensor>,
    en_tokens: Vec<Tensor>,
}

// TODO: implement batching by loading dataset as vec, tokenizing each string and creating a
// tensor out of it. Then randomly select a batch of those tensors and stack them together,
// resulting in a batch tensor of (batch, seq_len) which can be used directly as input for
// the transformer.
impl DataLoader {
    pub fn new(
        data: LazyFrame,
        tokenizer: &Tokenizer,
        batch_size: usize,
        max_seq_len: usize,
    ) -> anyhow::Result<DataLoader> {
        let device = Device::Cpu;
        let unnested = data.unnest(["translation"]).collect()?; // [de (str), en (str)]
        let de: Vec<&str> = unnested.column("de")?.str()?.into_no_null_iter().collect();
        let en: Vec<&str> = unnested.column("en")?.str()?.into_no_null_iter().collect();
        let chunk_size = 1024;

        println!("Tokenizing DE");
        let de_tokens = DataLoader::tokenize_chunked(de, tokenizer, chunk_size, &device);

        println!("Tokenizing EN");
        let en_tokens = DataLoader::tokenize_chunked(en, tokenizer, chunk_size, &device);

        Ok(Self {
            de_tokens,
            en_tokens,
        })
    }

    fn tokenize_chunked(
        data: Vec<&str>,
        tokenizer: &Tokenizer,
        chunk_size: usize,
        device: &Device,
    ) -> Vec<Tensor> {
        let n_chunks = data.len() / chunk_size;
        let mut tokens: Vec<Tensor> = Vec::with_capacity(data.len());
        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            if i % 100 == 0 {
                println!("Chunk: {}/{}", i, n_chunks);
            }
            let mut tensors = tokenizer
                .encode_batch_fast(chunk.to_vec(), true)
                .unwrap()
                .iter()
                .map(|tokens| {
                    let tokens = tokens.get_ids().to_vec();
                    Tensor::new(tokens.as_slice(), device).unwrap()
                })
                .collect::<Vec<_>>();
            tokens.append(&mut tensors);
        }
        tokens
    }

    pub fn load(path: &Path, batch_size: usize, max_seq_len: usize) {}

    pub fn epoch_iterable(&self) -> Result<Vec<Batch>> {
        let de_stacked = Tensor::stack(&self.de_tokens, 0)?;
        let en_stacked = Tensor::stack(&self.en_tokens, 0)?;

        let mut res: Vec<Batch> = Vec::new();
        res.push(Batch {
            source: de_stacked,
            target: en_stacked,
        });
        Ok(res)
    }
}

use candle_core::{Device, Result, Tensor};
use candle_hf_hub::api::sync::Api;
use polars::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};
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

/// implement batching by loading dataset as vec, tokenizing each string and creating a
/// tensor out of it. Then randomly select a batch of those tensors and stack them together,
/// resulting in a batch tensor of (batch, seq_len) which can be used directly as input for
/// the transformer.
impl DataLoader {
    pub fn new(
        data: LazyFrame,
        tokenizer: &Tokenizer,
        _batch_size: usize,
        _max_seq_len: usize,
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

pub struct TranslationDataset {
    pub src: Vec<Tensor>, // (seq_len)
    pub tgt: Vec<Tensor>, // (seq_len)
}

impl TranslationDataset {
    pub fn new(path: &PathBuf, max_len: usize) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let data = LazyFrame::scan_parquet(path, Default::default())?.collect()?;

        let data = data.head(Some(100));

        fn to_tensor(
            data: &DataFrame,
            col: &str,
            device: &Device,
            max_len: usize,
        ) -> Result<Vec<Tensor>> {
            let mut src: Vec<Vec<i64>> = data
                .column(col)
                .unwrap()
                .list()
                .unwrap()
                .into_no_null_iter()
                .map(|s| s.i64().unwrap().into_no_null_iter().collect())
                .collect();
            src.iter_mut().for_each(|v| v.insert(0, 33708));
            let t: Vec<_> = src
                .iter()
                .map(|t| Tensor::new(t.as_slice(), device).unwrap())
                .map(|t| t.narrow(0, 0, t.dims1().unwrap().min(max_len)).unwrap())
                .collect();

            Ok(t)
        }

        Ok(Self {
            src: to_tensor(&data, "inputs", &device, max_len)?,
            tgt: to_tensor(&data, "targets", &device, max_len)?,
        })
    }

    pub fn get(&self, index: usize) -> Result<(&Tensor, &Tensor)> {
        Ok((self.src.get(index).unwrap(), self.tgt.get(index).unwrap()))
    }

    pub fn len(&self) -> usize {
        self.src.len()
    }
}

pub struct BatchSampler {
    dataset: TranslationDataset,
    max_tokens: usize,
    pub num_batches: usize,
    curr: usize,
}

impl BatchSampler {
    pub fn new(dataset: TranslationDataset, max_tokens: usize) -> Result<Self> {
        let num_batches = Self::calculate_num_batches(&dataset, max_tokens)?;

        Ok(Self {
            dataset,
            max_tokens,
            num_batches,
            curr: 0,
        })
    }

    pub fn collate(samples: &Vec<usize>, src: &Vec<Tensor>, tgt: &Vec<Tensor>) -> Batch {
        let srcs: Vec<_> = samples
            .iter()
            .map(|s| src.get(*s).unwrap())
            // .map(|s| self.dataset.get(*s).unwrap().0)
            .collect();
        let tgts: Vec<_> = samples
            .iter()
            // .map(|s| self.dataset.get(*s).unwrap().1)
            .map(|s| tgt.get(*s).unwrap())
            .collect();

        let max_srcs: usize = srcs.iter().map(|t| t.dims1().unwrap()).max().unwrap();
        let max_tgts: usize = tgts.iter().map(|t| t.dims1().unwrap()).max().unwrap();

        let srcs = BatchSampler::pad_tensors(srcs, max_srcs).unwrap();
        let src_tensor = Tensor::stack(srcs.as_slice(), 0).unwrap(); // (batch, seq_len)
        let tgts = BatchSampler::pad_tensors(tgts, max_tgts).unwrap();
        let tgt_tensor = Tensor::stack(tgts.as_slice(), 0).unwrap(); // (batch, seq_len)

        Batch {
            source: src_tensor,
            target: tgt_tensor,
        }
    }

    fn pad_tensors(tensors: Vec<&Tensor>, len: usize) -> Result<Vec<Tensor>> {
        let res: Vec<_> = tensors
            .iter()
            .map(|t| t.pad_with_zeros(0, 0, len - t.dims1().unwrap()).unwrap())
            .collect();

        Ok(res)
    }

    fn calculate_num_batches(dataset: &TranslationDataset, max_tokens: usize) -> Result<usize> {
        // let mut num_batches = 0;
        // let mut src_tokens = 0;
        // let mut tgt_tokens = 0;

        // for idx in 0..dataset.len() {
        //     let (src, tgt) = dataset.get(idx)?;
        //
        //     if src_tokens + src.dims1()? > max_tokens || tgt_tokens + tgt.dims1()? > max_tokens {
        //         num_batches += 1;
        //         tgt_tokens = 0;
        //         src_tokens = 0;
        //     }
        //
        //     src_tokens += src.dims1()?;
        //     tgt_tokens += tgt.dims1()?;
        // }
        //
        // if src_tokens != 0 || tgt_tokens != 0 {
        //     num_batches += 1;
        // }
        let num_batches = dataset.len() / max_tokens;
        Ok(num_batches)
    }
}

// impl Iterator for BatchSampler {
//     type Item = Vec<usize>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         let mut batch = Vec::new();
//         let mut src_tokens = 0;
//         let mut tgt_tokens = 0;
//
//         // sleep(time::Duration::from_secs(10));
//
//         // println!(
//         //     "start {}/{} {} {} {}",
//         //     self.curr,
//         //     self.dataset.len(),
//         //     src_tokens,
//         //     tgt_tokens,
//         //     batch.len()
//         // );
//         for idx in self.curr..self.dataset.len() {
//             let (src, tgt) = self.dataset.get(idx).unwrap();
//
//             // println!("{idx} {:?} {:?}", src.dims(), tgt.dims());
//
//             if src_tokens + src.dims1().unwrap() > self.max_tokens
//                 || tgt_tokens + tgt.dims1().unwrap() > self.max_tokens
//             {
//                 self.curr = idx;
//                 println!("Batch {} finished", self.curr);
//                 return Some(batch);
//             }
//             batch.push(idx);
//             src_tokens += src.dims1().unwrap();
//             tgt_tokens += tgt.dims1().unwrap();
//         }
//         self.curr = self.dataset.len();
//         // println!("Last Batch");
//
//         if src_tokens != 0 {
//             return Some(batch);
//         }
//         self.curr = 0; // Iterator finished, do cleanup
//         None
//     }
// }

impl Iterator for BatchSampler {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::new();
        for idx in self.curr..self.dataset.len().min(self.curr + self.max_tokens) {
            batch.push(idx);
            self.curr = idx;
        }
        if batch.len() < self.max_tokens {
            self.curr = self.dataset.len()
        }
        if !batch.is_empty() {
            return Some(batch);
        }
        self.curr = 0; // Iterator finished, do cleanup
        None
    }
}

pub struct Vocabulary {
    pub idx2token: Vec<String>,
    pub token2idx: HashMap<String, usize>,
}

impl Vocabulary {
    pub fn new(vocab_path: &str) -> Self {
        let file = File::open(vocab_path).expect("Vocab file does not exist");
        let buf = BufReader::new(file);
        let idx2token = buf
            .lines()
            .map(|l| {
                let l = l.expect("Could not parse line");
                l.as_str()[1..l.len() - 1].into() // cut `'` characters
            })
            .collect::<Vec<String>>();

        let token2idx: HashMap<String, usize> = idx2token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        Vocabulary {
            idx2token,
            token2idx,
        }
    }

    pub fn decode(&self, token_idx: Vec<usize>) -> Result<String> {
        let default = "na".to_string();
        let tokens: Vec<_> = token_idx
            .iter()
            .map(|t| self.idx2token.get(*t).unwrap_or(&default).to_string())
            .collect();
        let joined: String = tokens.join("").replace("_", " ");
        Ok(joined)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_vocab_loading() -> anyhow::Result<()> {
        let path = "../data/vocab.ende";
        let vocabulary = Vocabulary::new(path);

        // println!("Token2idx: {:?}", vocabulary.token2idx);
        println!("Ki: {:?}", vocabulary.token2idx.get("Ki").unwrap());
        // println!("idx2token: {:?}", vocabulary.idx2token);

        Ok(())
    }

    #[test]
    fn test_decode_vec() -> anyhow::Result<()> {
        let path = "../data/vocab.ende";
        let vocabulary = Vocabulary::new(path);
        let tokens = vec![1, 2, 3, 4, 5, 6, 300, 300, 300];

        let decoded = vocabulary.decode(tokens)?;

        println!("Decoded: {}", decoded);

        Ok(())
    }
}

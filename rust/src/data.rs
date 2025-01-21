use candle_core::Result;
use std::{fs::File, path::PathBuf};

use candle_hf_hub::api::sync::Api;
use polars::{frame::DataFrame, prelude::*};

pub struct HuggingfaceDataset {
    pub dataset: String,
    pub train: LazyFrame,
    pub test: LazyFrame,
    pub validation: LazyFrame,
}

impl HuggingfaceDataset {
    pub fn new(dataset: &str) -> Result<Self> {
        let api = Api::new().unwrap();
        let repo = api.dataset(String::from("wmt/wmt14"));
        let data = repo.info();

        let mut train_files: Vec<PathBuf> = Vec::new();
        let mut test_file: Option<PathBuf> = None;
        let mut validation_file: Option<PathBuf> = None;

        for sibling in data.unwrap().siblings {
            if sibling.rfilename.contains("de-en") {
                let path = repo.get(&sibling.rfilename).unwrap();
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

        // TODO: implement proper error handling of loaded LazyFrames
        let train_frames: Vec<LazyFrame> = train_files
            .iter()
            .map(|path| Self::load(path).unwrap())
            .collect();
        let train = concat(train_frames, UnionArgs::default()).unwrap();

        let test = Self::load(&test_file.unwrap()).unwrap();
        let validation = Self::load(&validation_file.unwrap()).unwrap();
        let dataset = String::from(dataset);

        Ok(Self {
            dataset,
            test,
            validation,
            train,
        })
    }

    fn load(path: &PathBuf) -> PolarsResult<LazyFrame> {
        // LazyFrame::scan_parquet(
        //     "hf://datasets/wmt/wmt14/de-en/test-00000-of-00001.parquet",
        //     Default::default(),
        // )
        LazyFrame::scan_parquet(
            path,
            // "/home/lukas/Programming/uni/transforming-attention/rust/resource/validation*.parquet",
            Default::default(),
        )
    }
}

pub struct Dataset {}

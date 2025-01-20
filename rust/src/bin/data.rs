use hf_hub::api::sync::Api;
use polars::prelude::*;

use ta::data::HuggingfaceDataset;

fn main() {
    println!("data");

    let dataset = HuggingfaceDataset::new("wmt14/de-en");
    // println!("{dataset:?}")
    let name = &dataset.dataset;
    println!("{name}");
    // println!("{:?}", dataset.train.collect().unwrap().head(None).get_columns());
    println!("{:?}", dataset.train.collect().unwrap().shape());
}

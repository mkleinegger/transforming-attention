fn main() -> anyhow::Result<()>{
    println!("data");

    // let dataset = HuggingfaceDataset::new("wmt14/de-en").unwrap();
    // println!("{dataset:?}")
    // let name = &dataset.dataset;
    // println!("{name}");
    // println!("{:?}", dataset.train.collect().unwrap().head(None).get_columns());
    // println!("{:?}", dataset.train.collect().unwrap().shape());

    // let dataset = TranslationDataset::new(
    //     "/home/lukas/Programming/uni/transforming-attention/data/translate_ende_small.parquet",
    // )?;

    // println!("Dataset: {:?}", dataset.src);
    Ok(())
}

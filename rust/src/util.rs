use polars::prelude::*;

pub fn load_data(path: &str) -> DataFrame {
    let data = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_quote_char(Some(b'"')))
        .try_into_reader_with_file_path(Some(
            "/home/lukas/Programming/uni/transforming-attention/rust/resource/sample.csv".into(),
        ))
        .unwrap()
        .finish()
        .unwrap();
    println!("{}", data);
    data
}

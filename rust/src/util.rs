use std::collections::HashMap;

use candle_core::{Result, Tensor};
use indicatif::{ProgressIterator, ProgressStyle};

pub fn progress_bar_style(name: &str) -> ProgressStyle {
    ProgressStyle::with_template(&format!(
        "{{spinner:.green}} {{elapsed_precise}} {}: {{wide_bar}} {{pos}}/{{len}}",
        name
    ))
    .unwrap()
}

pub fn predict_next_token(logits: &Tensor, previous_predictions: &Vec<i64>) -> Result<Vec<i64>> {
    let frequencies: HashMap<i64, i64> = previous_predictions.iter().fold(
        HashMap::with_capacity(previous_predictions.len()),
        |mut map, token| {
            *map.entry(*token).or_default() += 1;
            map
        },
    );

    let logits_adjusted = logits
        .to_vec1::<f32>()?
        .iter()
        .copied()
        .map(|prob| prob.exp())
        .enumerate()
        .take(logits.dims1()?)
        .map(|(i, log_prob)| {
            log_prob / ((*frequencies.get(&(i as i64)).unwrap_or(&0_i64) as f32) + 1.).powf(2.)
        })
        .collect::<Vec<_>>();

    // logits.to_vec1().iter().map()
    Ok([(logits_adjusted
        .iter()
        .enumerate()
        .max_by(|(_, &prob), (_, &prob_other)| prob.total_cmp(&prob_other))
        .map(|(idx, _)| idx)
        .unwrap() as i64)]
    .to_vec())
}

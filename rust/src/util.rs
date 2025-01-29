use indicatif::ProgressStyle;

pub fn progress_bar_style(name: &str) -> ProgressStyle {
    ProgressStyle::with_template(&format!(
        "{{spinner:.green}} {{elapsed_precise}} {}: {{wide_bar}} {{pos}}/{{len}}",
        name
    ))
    .unwrap()
}

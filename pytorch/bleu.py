import polars as pl
from nltk.translate.bleu_score import corpus_bleu

parquet_filename = "../data/beaminference.parquet"

df = pl.read_parquet(parquet_filename)
df = df.with_columns(
    pl.col("generated").map_elements(
        lambda xs: [33708] + list(xs),
        return_dtype=pl.List(pl.Int64)
    )
)
print(df)

target, generated = df.row(0)
references = [[tokens[1:]] for tokens in df["target"]]
hypotheses = df["generated"].to_list()

# compute BLEU score
bleu = corpus_bleu(references, hypotheses)

# output BLEU score
print(f"BLEU Score: {bleu:.4f}")

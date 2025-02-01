import polars


class T2TVocabulary:
    """
    A class that decodes 
    Tensor2Tensor's SubwordTextEncoded data.
    """

    def __init__(self, vocab_file):
        """Loads vocabulary into dictionaries in-memory.""" 
        with open(vocab_file, 'r', encoding='utf-8') as f:
            tokens = f.read().splitlines()
        tokens = [token.replace("'", "").replace("_", " ") for token in tokens]
        self.id_to_token = {idx: token for idx, token in enumerate(tokens)}
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}

    def decode(self, ids: list):
        return [self.id_to_token[_id] for _id in ids]


VOCAB_FILE = 'data/vocab.ende'
DATASET_FILE = 'data/dataset.parquet'


if __name__ == "__main__":

    df = polars.read_parquet(DATASET_FILE)
    vocabulary = T2TVocabulary(VOCAB_FILE)

    input, target = df.row(0)

    # decode input
    input_decoded = vocabulary.decode(input)
    print(input, '\n', "".join(input_decoded))

    # decode target
    target_decoded = vocabulary.decode(target)
    print(target, '\n', "".join(target_decoded))
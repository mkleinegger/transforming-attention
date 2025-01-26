use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::{
    config::Config,
    decoder::decoder::Decoder,
    embeddings::{input_embeddings::InputEmbeddings, positional_embeddings::PositionalEmbeddings},
    encoder::encoder::Encoder,
    projection::ProjectionLayer,
};
// TODO: create Transformer

pub struct Transformer {
    encode_embeddings: InputEmbeddings,
    encode_positional: PositionalEmbeddings,
    encoder: Encoder,
    decode_embeddings: InputEmbeddings,
    decode_positional: PositionalEmbeddings,
    decoder: Decoder,
    projection_layer: ProjectionLayer,
}

impl Transformer {
    pub fn new(vb: VarBuilder, config: &Config, vocab_size: usize) -> Result<Self> {
        let device = vb.device();
        Ok(Self {
            encode_embeddings: InputEmbeddings::new(
                vocab_size,
                config,
                vb.pp("encode_embeddings"),
            )?,
            encode_positional: PositionalEmbeddings::new(config, device)?,
            encoder: Encoder::new(vb.pp("encoder"), config)?,
            decode_embeddings: InputEmbeddings::new(
                vocab_size,
                config,
                vb.pp("decode_embeddings"),
            )?,
            decode_positional: PositionalEmbeddings::new(config, device)?,
            decoder: Decoder::new(vb.pp("decoder"), config)?,
            projection_layer: ProjectionLayer::new(vb.pp("projection"), config, vocab_size)?,
        })
    }

    /// Encode indices of shape (batch, seq_len), by creating embeddings, doing positional
    /// encoding and passing it to an encoder, resulting in an encoded tensor of shape
    /// (batch, seq_len, d_model).
    pub fn encode(&mut self, indices: &Tensor, src_mask: bool, train: bool) -> Result<Tensor> {
        let embedded = self.encode_embeddings.forward(indices)?;
        let embedded = self.encode_positional.forward(embedded, train)?;
        self.encoder.forward(embedded, src_mask)
    }

    /// Decode encoded tensor of shape (batch, seq_len, d_model) by passing it to a decoder,
    /// resulting in a tensor of shape (batch, seq_len, d_model).
    pub fn decode(
        &mut self,
        indices: &Tensor,
        encoder_output: &Tensor,
        src_mask: bool,
        tgt_mask: bool,
        train: bool
    ) -> Result<Tensor> {
        let embedded = self.decode_embeddings.forward(indices)?;
        let embedded = self.decode_positional.forward(embedded, train)?;

        self.decoder
            .forward(embedded, encoder_output, tgt_mask, src_mask)
    }

    pub fn project(&self, input: &Tensor) -> Result<Tensor> {
        self.projection_layer.forward(input)
    }
}

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::{
    config::Config,
    decoder::decoder::Decoder,
    embeddings::{input_embeddings::InputEmbeddings, positional_embeddings::PositionalEmbeddings},
    encoder::encoder::Encoder,
    projection::ProjectionLayer,
};

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

    pub fn forward(
        &mut self,
        src: &Tensor,
        tgt: &Tensor,
        source_mask: bool,
        target_mask: bool,
        train: bool,
    ) -> Result<Tensor> {
        let src_mask = match source_mask {
            true => Some(encoder_mask(src)?),
            false => None,
        };
        let tgt_mask = match target_mask {
            true => Some(decoder_mask(tgt)?),
            false => None,
        };
        let src_mask_decoder = match source_mask {
            true => Some(decoder_encoder_mask(tgt, src)?),
            false => None,
        };

        let encoded = self.encode(src, src_mask.as_ref(), train)?;
        let decoded = self.decode(
            tgt,
            &encoded,
            src_mask_decoder.as_ref(),
            tgt_mask.as_ref(),
            train,
        )?;
        self.project(&decoded)
    }

    /// Encode indices of shape (batch, seq_len), by creating embeddings, doing positional
    /// encoding and passing it to an encoder, resulting in an encoded tensor of shape
    /// (batch, seq_len, d_model).
    pub fn encode(
        &mut self,
        indices: &Tensor,
        src_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let embedded = self.encode_embeddings.forward(indices, train)?;
        let embedded = self.encode_positional.forward(embedded, train)?;
        self.encoder.forward(embedded, src_mask, train)
    }

    /// Decode encoded tensor of shape (batch, seq_len, d_model) by passing it to a decoder,
    /// resulting in a tensor of shape (batch, seq_len, d_model).
    pub fn decode(
        &mut self,
        indices: &Tensor,
        encoder_output: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let embedded = self.decode_embeddings.forward(indices, train)?;
        let embedded = self.decode_positional.forward(embedded, train)?;

        self.decoder
            .forward(embedded, encoder_output, tgt_mask, src_mask, train)
    }

    pub fn project(&self, input: &Tensor) -> Result<Tensor> {
        self.projection_layer.forward(input)
    }
}

// TODO: find better way to calculate masks
fn encoder_mask(xs_encoder: &Tensor) -> Result<Tensor> {
    // println!("Encoder in mask: {:?}", xs_encoder.shape());
    let (batch_size, seq_len) = xs_encoder.dims2()?;
    let zeros = Tensor::zeros(
        xs_encoder.shape(),
        candle_core::DType::I64,
        xs_encoder.device(),
    )?;

    let padded = xs_encoder
        .ne(&zeros)? // 1 = not padded, 0 = padded
        .unsqueeze(1)?
        .unsqueeze(3)?
        .broadcast_as((batch_size, 1, seq_len, seq_len))?; // (batch, 1 (head), seq_len, seq_len (seq_len))

    let twos = Tensor::new(&[2u8], xs_encoder.device())?;
    let padded = padded.add(&padded.t()?)?.broadcast_eq(&twos)?;
    // println!("Encoder mask : {:}", padded);
    Ok(padded)
}

fn decoder_mask(xs_decoder: &Tensor) -> Result<Tensor> {
    // println!("Decoder in mask: {:?}", xs_decoder.shape());

    let (batch_size, seq_len) = xs_decoder.dims2()?;
    let zeros = Tensor::zeros(
        xs_decoder.shape(),
        candle_core::DType::I64,
        xs_decoder.device(),
    )?;

    // (batch, 1, seq_len, seq_len)
    let padded = xs_decoder
        .ne(&zeros)?
        .unsqueeze(1)?
        .unsqueeze(3)?
        .broadcast_as((batch_size, 1, seq_len, seq_len))?;

    let padded = padded.add(&padded.t()?)?; // 2 in kept positions
                                            // println!("Padded decoder: {}", padded);

    let causal_mask_vec: Vec<_> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j <= i)))
        .collect();
    let causal_mask = // (batch (1), head (1), seq_len, seq_len
        Tensor::from_slice(&causal_mask_vec, (seq_len, seq_len), xs_decoder.device())?
            .unsqueeze(0)?.unsqueeze(0)?;
    let threes = Tensor::new(3u8, xs_decoder.device())?;
    let combined = causal_mask.broadcast_add(&padded)?.broadcast_eq(&threes)?;

    // println!("Causal mask: {:}", causal_mask);
    // println!("combined: {:}", combined);
    Ok(combined)
}

fn decoder_encoder_mask(xs_decoder: &Tensor, xs_encoder: &Tensor) -> Result<Tensor> {
    // println!("Encoder in mask: {:?}", xs_encoder.shape());
    let seq_len_decoder = xs_decoder.dim(1)?;
    let (batch_size, seq_len_encoder) = xs_encoder.dims2()?;
    let zeros = Tensor::zeros(
        xs_encoder.shape(),
        candle_core::DType::I64,
        xs_encoder.device(),
    )?;

    let padded_encoder = xs_encoder
        .ne(&zeros)? // 1 = not padded, 0 = padded
        .unsqueeze(1)?
        .unsqueeze(1)?
        // .unsqueeze(3)?
        .broadcast_as((batch_size, 1, seq_len_decoder, seq_len_encoder))?; // (batch, 1 (head), seq_len, seq_len (seq_len))

    let zeros = Tensor::zeros(
        xs_decoder.shape(),
        candle_core::DType::I64,
        xs_decoder.device(),
    )?;
    let padded_decoder = xs_decoder
        .ne(&zeros)?
        .unsqueeze(1)?
        .unsqueeze(3)?
        .broadcast_as((batch_size, 1, seq_len_decoder, seq_len_encoder))?;

    let twos = Tensor::new(&[2u8], xs_encoder.device())?;
    let padded = padded_encoder.add(&padded_decoder)?.broadcast_eq(&twos)?;
    Ok(padded)
}

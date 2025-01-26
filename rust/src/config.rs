use candle_core::Device;

pub struct Config {
    pub n_encoders: usize,
    pub n_decoders: usize,
    pub d_model: usize,
    pub d_feed_forward: usize,
    pub n_heads: usize,
    pub device: Device,
    pub attention_dropout: f32,
    pub pos_dropout: f32,
    pub feed_forward_dropout: f32,
    pub residual_dropout: f32,
    pub seq_len: usize,
}

impl Default for Config {
    fn default() -> Self {
        let device = Device::cuda_if_available(0).unwrap();

        Self {
            n_encoders: 8,
            n_decoders: 8,
            n_heads: 8,
            d_model: 256,
            d_feed_forward: 2048,
            device,
            attention_dropout: 0.3,
            pos_dropout: 0.3,
            residual_dropout: 0.3,
            feed_forward_dropout: 0.3,
            seq_len: 160,
        }
    }
}

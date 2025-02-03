use candle_core::Device;

pub struct Config {
    pub n_encoders: usize,
    pub n_decoders: usize,
    pub d_model: usize,
    pub d_feed_forward: usize,
    pub n_heads: usize,
    pub device: Device,
    pub attention_dropout: f32,
    pub embedding_dropout: f32,
    pub pos_dropout: f32,
    pub feed_forward_dropout: f32,
    pub residual_dropout: f32,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub max_steps: usize,
    pub log_x_steps: usize,
}

impl Default for Config {
    fn default() -> Self {
        let device = Device::cuda_if_available(0).unwrap();
        // let device = Device::Cpu;

        Self {
            n_encoders: 6,
            n_decoders: 6,
            n_heads: 8,
            d_model: 512,
            d_feed_forward: 2048,
            device,
            attention_dropout: 0.1,
            embedding_dropout: 0.0,
            pos_dropout: 0.1,
            residual_dropout: 0.1,
            feed_forward_dropout: 0.1,
            batch_size: 8,
            max_seq_len: 256,
            max_steps: 100_000,
            log_x_steps: 1_000,
        }
    }
}

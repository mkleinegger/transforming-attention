use candle_core::Device;

pub struct Config {
    pub n_encoders: usize,
    pub n_decoders: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub device: Device,
    pub attention_dropout: f32,
}

impl Default for Config {
    fn default() -> Self {
        let device = Device::cuda_if_available(0).unwrap();

        Self {
            n_encoders: 6,
            n_decoders: 6,
            n_heads: 6,
            d_model: 512,
            device,
            attention_dropout: 0.3,
        }
    }
}

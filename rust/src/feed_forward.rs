use candle_core::{Device, Result};
use candle_nn::{Dropout, Linear};

pub struct FeedForward {
    // layer1: Linear,
    // layer2: Linear,
    // dropout: Dropout,
}

impl FeedForward {
    fn new(d_model: usize, dropout: f32, d_ff: usize, device: &Device) -> Result<Self> {
        // create 2 linear layers according to the paper:
        // W1: [512 * 2048] and [512]
        // W2: [2048 * 512] and [2048]



        Ok(Self {})
    }
}

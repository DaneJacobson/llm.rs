pub const B: usize = 4;
pub const T: usize = 64;
pub const VOCAB_SIZE: usize = 50257;
pub const GENT: usize = 64; // number of steps of inference we will do

pub const HEADER_SIZE: usize = 256;
pub const BUFFER_SIZE: usize = B*T+1;
use std::fmt;
use std::fs::File;

use crate::constants::{B, T};
use crate::model::GPT2Config;
use crate::utils;

// ---------------------- //
// -- ParameterTensors -- //
// ---------------------- //

// V=vocab, C=embed dim, L=layers, maxT=context window
pub struct ParameterTensors {
    // encoding
    pub wte: Vec<f32>, // (V, C)
    pub wpe: Vec<f32>, // (maxT, C)

    // first linear
    pub ln1w: Vec<f32>, // (L, C)
    pub ln1b: Vec<f32>, // (L, C)
    // attention
    pub qkvw: Vec<f32>, // (L, 3*C, C) -> takes C to 3*C
    pub qkvb: Vec<f32>, // (L, 3*C)
    pub attprojw: Vec<f32>, // (L, C, C)
    pub attprojb: Vec<f32>, // (L, C)
    // second linear
    pub ln2w: Vec<f32>, // (L, C)
    pub ln2b: Vec<f32>, // (L, C)

    // multi-layer perceptron
    pub fcw: Vec<f32>, // (L, 4*C, C) -> takes C to 4*C
    pub fcb: Vec<f32>, // (L, 4*C)
    pub fcprojw: Vec<f32>, // (L, C, 4*C) -> takes 4C to C
    pub fcprojb: Vec<f32>, // (L, C)

    // final linear
    pub lnfw: Vec<f32>, // (C)
    pub lnfb: Vec<f32>, // (C)
}

impl ParameterTensors {
    pub fn new(model_file: &mut File, config: &GPT2Config) -> ParameterTensors {
        let vp: usize = config.padded_vocab_size;
        let c: usize = config.channels;
        let maxt: usize = config.max_seq_len;
        let l: usize = config.num_layers;

        return ParameterTensors {
            wte: utils::read_floats_into_vecf32(model_file, vp * c, &String::from("wte")),
            wpe: utils::read_floats_into_vecf32(model_file, maxt * c, &String::from("wpe")),
            ln1w: utils::read_floats_into_vecf32(model_file, l * c, &String::from("ln1w")),
            ln1b: utils::read_floats_into_vecf32(model_file, l * c, &String::from("ln1b")),
            qkvw: utils::read_floats_into_vecf32(model_file, l * 3*c * c, &String::from("qkvw")),
            qkvb: utils::read_floats_into_vecf32(model_file, l * 3*c, &String::from("qkvb")),
            attprojw: utils::read_floats_into_vecf32(model_file, l * c * c, &String::from("attprojw")),
            attprojb: utils::read_floats_into_vecf32(model_file, l * c, &String::from("attprojb")),
            ln2w: utils::read_floats_into_vecf32(model_file, l * c, &String::from("ln2w")),
            ln2b: utils::read_floats_into_vecf32(model_file, l * c, &String::from("ln2b")),
            fcw: utils::read_floats_into_vecf32(model_file, l * 4*c * c, &String::from("fcw")),
            fcb: utils::read_floats_into_vecf32(model_file, l * 4*c, &String::from("fcb")),
            fcprojw: utils::read_floats_into_vecf32(model_file, l * c * 4*c, &String::from("fcprojw")),
            fcprojb: utils::read_floats_into_vecf32(model_file, l * c, &String::from("fcprojb")),
            lnfw: utils::read_floats_into_vecf32(model_file, c, &String::from("lnfw")),
            lnfb: utils::read_floats_into_vecf32(model_file, c, &String::from("lnfb")),
        }
    }

    pub fn new_empty(config: &GPT2Config) -> ParameterTensors {
        let vp: usize = config.padded_vocab_size;
        let c: usize = config.channels;
        let maxt: usize = config.max_seq_len;
        let l: usize = config.num_layers;

        return ParameterTensors {
            wte: vec![0f32; vp*c], // (V, C)
            wpe: vec![0f32; maxt*c], // (maxT, C)
            ln1w: vec![0f32; l*c], // (L, C)
            ln1b: vec![0f32; l*c], // (L, C)
            qkvw: vec![0f32; l*3*c*c], // (L, 3*C, C) -> takes C to 3*C
            qkvb: vec![0f32; l*3*c], // (L, 3*C)
            attprojw: vec![0f32; l*c*c], // (L, C, C)
            attprojb: vec![0f32; l*c], // (L, C)
            ln2w: vec![0f32; l*c], // (L, C)
            ln2b: vec![0f32; l*c], // (L, C)
            fcw: vec![0f32; l*4*c*c], // (L, 4*C, C) -> takes C to 4*C
            fcb: vec![0f32; l*4*c], // (L, 4*C)
            fcprojw: vec![0f32; l*c*4*c], // (L, C, 4*C) -> takes 4C to C
            fcprojb: vec![0f32; l*c], // (L, C)
            lnfw: vec![0f32; c], // (C)
            lnfb: vec![0f32; c], // (C)
        }
    }

    pub fn fields(&mut self) -> Vec<&mut Vec<f32>> {
        return vec![
            &mut self.wte, &mut self.wpe, &mut self.ln1w, &mut self.ln1b,
            &mut self.qkvw, &mut self.qkvb, &mut self.attprojw, &mut self.attprojb,
            &mut self.ln2w, &mut self.ln2b, &mut self.fcw, &mut self.fcb,
            &mut self.fcprojw, &mut self.fcprojb, &mut self.lnfw, &mut self.lnfb,
        ]
    }
}

impl fmt::Display for ParameterTensors {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, 
            "ParameterTensors: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}", 
            self.wte.len(), // (V, C)
            self.wpe.len(), // (maxT, C)
            self.ln1w.len(), // (L, C)
            self.ln1b.len(), // (L, C)
            self.qkvw.len(), // (L, 3*C, C)
            self.qkvb.len(), // (3*C, C)
            self.attprojw.len(), // (L, C, C)
            self.attprojb.len(), // (L, C)
            self.ln2w.len(), // (L, C)
            self.ln2b.len(), // (L, C)
            self.fcw.len(), // (L, 4*C, C)
            self.fcb.len(), // (L, 4*C)
            self.fcprojw.len(), // (L, C, 4*C)
            self.fcprojb.len(), // (L, C)
            self.lnfw.len(), // (C)
            self.lnfb.len(), // (C)
        )
    }
}

// ----------------------- //
// -- ActivationTensors -- //
// ----------------------- //

// B=batch size, T=tokens, C=embed dim, L=layers
pub struct ActivationTensors {
    // first layer norm
    pub ln1: Vec<f32>, // (L, B, T, C)
    pub ln1_mean: Vec<f32>, // (L, B, T)
    pub ln1_rstd: Vec<f32>, // (L, B, T)
    // attention
    pub qkv: Vec<f32>, // (L, B, T, 3*C)
    pub atty: Vec<f32>, // (L, B, T, C)
    pub preatt: Vec<f32>, // (L, B, NH, T, T)
    pub att: Vec<f32>, // (L, B, NH, T, T)
    pub attproj: Vec<f32>, // (L, B, T, C)
    pub residual2: Vec<f32>, // (L, B, T, C)
    // second layer norm
    pub ln2: Vec<f32>, // (L, B, T, C)
    pub ln2_mean: Vec<f32>, // (L, B, T)
    pub ln2_rstd: Vec<f32>, // (L, B, T)
    // MLP
    pub fch: Vec<f32>, // (L, B, T, 4*C)
    pub fch_gelu: Vec<f32>, // (L, B, T, 4*C)
    pub fcproj: Vec<f32>, // (L, B, T, C)
    pub residual3: Vec<f32>, // (L, B, T, C)
    // final linear layer
    pub lnf: Vec<f32>, // (B, T, C)
    pub lnf_mean: Vec<f32>, // (B, T)
    pub lnf_rstd: Vec<f32>, // (B, T)
    // drop to token space
    pub logits: Vec<f32>, // (B, T, Vp)
    pub probs: Vec<f32>, // (B, T, Vp)
    pub losses: Vec<f32>, // (B, T)
}

impl ActivationTensors {
    pub fn new(config: &GPT2Config) -> ActivationTensors {
        let l: usize = config.num_layers; 
        let c: usize = config.channels; 
        let nh: usize = config.num_heads;
        let vp: usize = config.padded_vocab_size;
        
        return ActivationTensors {
            ln1: vec![0f32; l*B*T*c],
            ln1_mean: vec![0f32; l*B*T],
            ln1_rstd: vec![0f32; l*B*T],
            qkv: vec![0f32; l*B*T*3*c], 
            atty: vec![0f32; l*B*T*c], 
            preatt: vec![0f32; l*B*nh*T*T], 
            att: vec![0f32; l*B*nh*T*T], 
            attproj: vec![0f32; l*B*T*c], 
            residual2: vec![0f32; l*B*T*c], 
            ln2: vec![0f32; l*B*T*c], 
            ln2_mean: vec![0f32; l*B*T],
            ln2_rstd: vec![0f32; l*B*T],
            fch: vec![0f32; l*B*T*4*c], 
            fch_gelu: vec![0f32; l*B*T*4*c], 
            fcproj: vec![0f32; l*B*T*c], 
            residual3: vec![0f32; l*B*T*c], 
            lnf: vec![0f32; B*T*c],
            lnf_mean: vec![0f32; B*T],
            lnf_rstd: vec![0f32; B*T],
            logits: vec![0f32; B*T*vp],
            probs: vec![0f32; B*T*vp],
            losses: vec![0f32; B*T],
        }
    }

    pub fn fields(&mut self) -> Vec<&mut Vec<f32>> {
        return vec![
            &mut self.ln1, &mut self.ln1_mean, &mut self.ln1_rstd, &mut self.qkv,
            &mut self.atty, &mut self.preatt, &mut self.att, &mut self.attproj, 
            &mut self.residual2, &mut self.ln2, &mut self.ln2_mean, &mut self.ln2_rstd, 
            &mut self.fch, &mut self.fch_gelu, &mut self.fcproj, &mut self.residual3, 
            &mut self.lnf, &mut self.lnf_mean, &mut self.lnf_rstd, &mut self.logits, 
            &mut self.probs, &mut self.losses
        ]
    }
}
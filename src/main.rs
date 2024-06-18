use std::fmt;
use std::fs::File;
use std::process::exit;
// use std::time::Instant;

pub mod utils;
pub mod dataloader;
pub mod tokenizer;


pub const B: usize = 4;
pub const T: usize = 64;
pub const VOCAB_SIZE: usize = 50257;
pub const GENT: u32 = 64; // number of steps of inference we will do

pub const HEADER_SIZE: usize = 256;
pub const BUFFER_SIZE: usize = B*T+1;
pub const U8_BUFFER_SIZE: usize = 2 * BUFFER_SIZE;


struct GPT2Config {
    max_seq_len: usize, // maxT
    vocab_size: usize, // V
    padded_vocab_size: usize, // Vp
    num_layers: usize, // L
    num_heads: usize, // H
    channels: usize, // C
}

impl GPT2Config {
    fn new(model_file: &mut File) -> GPT2Config {
        let model_header: [u32; 256] = utils::read_header(model_file);

        if model_header[0] != 20240326 {
            println!("Bad magic model file\n");
            exit(1);
        }
        if model_header[1] != 3 {
            println!("Bad version in model file\n");
            exit(1);
        }

        let maxt: usize = model_header[2] as usize;
        let v: usize = model_header[3] as usize;
        let l: usize = model_header[4] as usize;
        let nh: usize = model_header[5] as usize;
        let c: usize = model_header[6] as usize;
        let vp: usize = model_header[7] as usize;

        println!("[GPT-2]");
        println!("max_seq_len: {}", maxt);
        println!("vocab_size: {}", v);
        println!("padded_vocab_size: {}", vp);
        println!("num_layers: {}", l);
        println!("num_heads: {}", nh);
        println!("channels: {}", c);
        println!("\n");

        return GPT2Config {
            max_seq_len: maxt,
            vocab_size: v,
            padded_vocab_size: vp,
            num_layers: l,
            num_heads: nh,
            channels: c, 
        };
    }
}

impl fmt::Display for GPT2Config {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, 
            "GPT2Config: {}, {}, {}, {}, {}, {}", 
            self.max_seq_len, 
            self.vocab_size, 
            self.padded_vocab_size, 
            self.num_layers, 
            self.num_heads, 
            self.channels
        )
    }
}

// V=vocab, C=embed dim, L=layers, maxT=context window
struct ParameterTensors {
    // encoding
    wte: Vec<f32>, // (V, C)
    wpe: Vec<f32>, // (maxT, C)

    // first linear
    ln1w: Vec<f32>, // (L, C)
    ln1b: Vec<f32>, // (L, C)
    // attention
    qkvw: Vec<f32>, // (L, 3*C, C) -> takes C to 3*C
    qkvb: Vec<f32>, // (L, 3*C)
    attprojw: Vec<f32>, // (L, C, C)
    attprojb: Vec<f32>, // (L, C)
    // second linear
    ln2w: Vec<f32>, // (L, C)
    ln2b: Vec<f32>, // (L, C)

    // multi-layer perceptron
    fcw: Vec<f32>, // (L, 4*C, C) -> takes C to 4*C
    fcb: Vec<f32>, // (L, 4*C)
    fcprojw: Vec<f32>, // (L, C, 4*C) -> takes 4C to C
    fcprojb: Vec<f32>, // (L, C)

    // final linear
    lnfw: Vec<f32>, // (C)
    lnfb: Vec<f32>, // (C)
}

impl ParameterTensors {
    fn new(model_file: &mut File, config: &GPT2Config) -> ParameterTensors {
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


// B=batch size, T=tokens, C=embed dim, L=layers
// const NUM_ACTIVATION_TENSORS: usize = 23;
struct ActivationTensors {
    // encoded: Vec<f32>, // (B, T, C)
    ln1: Vec<f32>, // (L, B, T, C)
    ln1_mean: Vec<f32>, // (L, B, T)
    ln1_rstd: Vec<f32>, // (L, B, T)
    qkv: Vec<f32>, // (L, B, T, 3*C)
    atty: Vec<f32>, // (L, B, T, C)
    preatt: Vec<f32>, // (L, B, NH, T, T)
    att: Vec<f32>, // (L, B, NH, T, T)
    attproj: Vec<f32>, // (L, B, T, C)
    residual2: Vec<f32>, // (L, B, T, C)
    ln2: Vec<f32>, // (L, B, T, C)
    ln2_mean: Vec<f32>, // (L, B, T)
    ln2_rstd: Vec<f32>, // (L, B, T)
    fch: Vec<f32>, // (L, B, T, 4*C)
    fch_gelu: Vec<f32>, // (L, B, T, 4*C)
    fcproj: Vec<f32>, // (L, B, T, C)
    residual3: Vec<f32>, // (L, B, T, C)
    lnf: Vec<f32>, // (B, T, C)
    lnf_mean: Vec<f32>, // (B, T)
    lnf_rstd: Vec<f32>, // (B, T)
    logits: Vec<f32>, // (B, T, Vp)
    probs: Vec<f32>, // (B, T, Vp)
    losses: Vec<f32>, // (B, T)
}

impl ActivationTensors {
    fn new(config: &GPT2Config) -> ActivationTensors {
        let l: usize = config.num_layers; 
        let c: usize = config.channels; 
        let nh: usize = config.num_heads;
        let vp: usize = config.padded_vocab_size;
        
        return ActivationTensors {
            // encoded: vec![0f32; B*T*c],
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
}

struct GPT2 {
    config: GPT2Config,
    params: ParameterTensors,
    acts: ActivationTensors,
    grads: ParameterTensors,
    grads_acts: ActivationTensors,
    // m_memory: Option<Vec<f32>>,
    // v_memory: Option<Vec<f32>>,
    // batch_size: u32, // the batch size (B) of current forward pass
    // seq_len: u32, // the sequence length (T) of current forward pass
    // inputs: &Vec<u32>, // the input tokens for the current forward pass
    // targets: &Vec<u32>, // the target tokens for the current forward pass
    mean_loss: f32, // after a forward pass with targets, will be populated with the mean loss
}

impl GPT2 {
    fn new(checkpoint_path: &String) -> GPT2 {
        // read in model from a checkpoint file
        let mut model_file: File = utils::fopen_check(checkpoint_path);
        let config: GPT2Config = GPT2Config::new(&mut model_file);
        let params: ParameterTensors = ParameterTensors::new(&mut model_file, &config);
        let acts: ActivationTensors = ActivationTensors::new(&config);
        let grads: ParameterTensors = ParameterTensors::new_empty(&config);
        let grads_acts: ActivationTensors = ActivationTensors::new(&config);

        return GPT2 {
            config: config,
            params: params,
            grads: grads,
            grads_acts: grads_acts,
            // m_memory: None,
            // v_memory: None,
            acts: acts,
            // batch_size: 0,
            // seq_len: 0,
            // inputs: &mut vec![0u32; B*T],
            mean_loss: -1.0f32,
        };
    }

    fn encoder_forward(&mut self, inputs: &Vec<u32>) {
        // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        // inp is (B,T) of integers, holding the token ids at each (b,t) position
        // wte is (V,C) of token embeddings, short for "weight token embeddings"
        // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"

        let c = self.config.channels;
        for b in 0..B {
            for t in 0..T {
                let idx = inputs[b*(T) + t] as usize;
                for i in 0..c {
                    let wte_val: f32 = self.params.wte[idx*c + i];
                    let wpe_val: f32 = self.params.wpe[t*c + i];
                    // pretend that the encoding is residual3[0], layer is 0
                    self.acts.residual3[0*(B*T*c) + b*(T*c) + t*(c) + i] = wte_val + wpe_val;
                }
            }
        }
    }

    pub fn forward(&mut self, is_train: bool, inputs: &Vec<u32>, targets: &Vec<u32>) {
        // convenience parameters (size_t to help prevent int overflow)
        let v = self.config.vocab_size;
        let vp = self.config.padded_vocab_size;
        let nl = self.config.num_layers;
        let nh = self.config.num_heads;
        let c = self.config.channels;

        // validate inputs, all indices must be in the range [0, V)
        for i in 0..B*T {
            // assert!(0 <= inputs[i]);
            assert!(inputs[i] < v as u32);
            // assert!(0 <= targets[i]);
            assert!(targets[i] < v as u32);
        }

        // forward pass
        println!("Encoder firing");
        self.encoder_forward(&inputs); // encoding goes into residual3[0] to bootstrap cycle
        for l in 0..nl {
            println!("Layer {} is running", l);
            // TODO: need to fix the residual stuff, it's a little confusing
            // Multi-Head Attention
            println!("lyrnrm1");
            lyrnrm_fwd(l, l, c, &self.acts.residual3, &self.params.ln1w, &self.params.ln1b, &mut self.acts.ln1, &mut self.acts.ln1_mean, &mut self.acts.ln1_rstd);
            println!("matmul1");
            matmul_fwd(l, c, 3*c, &self.acts.ln1, &self.params.qkvw, Some(&self.params.qkvb), &mut self.acts.qkv);
            println!("attent");
            attent_fwd(l, c, nh, &self.acts.qkv, &mut self.acts.preatt, &mut self.acts.att, &mut self.acts.atty);
            println!("matmul2");
            matmul_fwd(l, c, c, &self.acts.atty, &self.params.attprojw, Some(&self.params.attprojb), &mut self.acts.attproj);
            println!("residual");
            residual_fwd(l, c,&self.acts.residual3, &self.acts.attproj, &mut self.acts.residual2);
            println!("lyrnrm2");
            lyrnrm_fwd(l, l, c, &self.acts.residual2, &self.params.ln2w, &self.params.ln2b, &mut self.acts.ln2, &mut self.acts.ln2_mean, &mut self.acts.ln2_rstd);
            // MLP
            println!("mlp matmul");
            matmul_fwd(l, c, 4*c, &self.acts.ln2, &self.params.fcw, Some(&self.params.fcb), &mut self.acts.fch);
            println!("gelu");
            gelu_fwd(l, c, &self.acts.fch, &mut self.acts.fch_gelu);
            println!("mlp matmul2");
            matmul_fwd(l, 4*c, c, &self.acts.fch_gelu, &self.params.fcprojw, Some(&self.params.fcprojb), &mut self.acts.fcproj);
            println!("mlp residual");
            residual_fwd(l, c, &self.acts.residual2, &self.acts.residual3, &mut self.acts.fcproj);
            println!("Layer {} is done", l);
        }

        // last residual is in residual3[-1]
        println!("Last layernorm firing");
        lyrnrm_fwd(nl-1, 0, c, &self.acts.residual3, &self.params.lnfw, &self.params.lnfb, &mut self.acts.lnf, &mut self.acts.lnf_mean, &mut self.acts.lnf_rstd,);
        println!("Last matmul fwd naive firing");
        matmul_fwd( 0, c, vp, &self.acts.lnf, &self.params.wte, None, &mut self.acts.logits);
        println!("Softmax firing");
        softmax_fwd(v, vp, &self.acts.logits, &mut self.acts.probs);

        // also forward the cross-entropy loss function if we have the targets
        if is_train {
            println!("crossentropy firing");
            crossentropy_fwd(vp, &self.acts.probs, &targets, &mut self.acts.losses);
            // for convenience also evaluate the mean loss
            self.mean_loss = 0.0;
            for i in 0..B*T {self.mean_loss += self.acts.losses[i]};
            for i in 0..B*T {self.mean_loss += self.acts.losses[i]};
            self.mean_loss /= (B*T) as f32;
        } else {
            println!("no trainig");
            // if we don't have targets, we don't have a loss
            self.mean_loss = -1.0;
        }
    }

    // pub fn zero_grad(&mut self) {


    //     if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    //     if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
    // }

    pub fn backward(&mut self, inputs: &Vec<u32>, targets: &Vec<u32>) {
        println!("backward pass runs");
        // double check we forwarded previously, with targets
        // if self.mean_loss == -1.0 {
        //     println!("Error: must forward with targets before backward\n");
        //     exit(1);
        // }

        // convenience shortcuts (and size_t to help prevent int overflow)
        // size_t B = model->batch_size;
        // size_t T = model->seq_len;
        let v: usize = self.config.vocab_size;
        let vp: usize = self.config.padded_vocab_size;
        let nl: usize = self.config.num_layers;
        let nh: usize = self.config.num_heads;
        let c: usize = self.config.channels;

        // backward pass: go in the reverse order of the forward pass, and call backward() functions
        // ParameterTensors params = model->params; // for brevity
        // ParameterTensors grads = model->grads;
        // ActivationTensors acts = model->acts;
        // ActivationTensors grads_acts = model->grads_acts;

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        let dloss_mean: f32 = 1.0 / (B*T) as f32;
        for i in 0..B*T {self.grads_acts.losses[i] = dloss_mean}

        println!("backwards crossentropy firing");
        crossentropy_softmax_backward(v, vp, &mut self.grads_acts.logits, &self.grads_acts.losses, &self.acts.probs, &targets);
        // crossentropy_softmax_backward(v, vp, &targets, &self.grads_acts.losses, &self.acts.probs,&mut self.grads_acts.logits);
        println!("backwards matmul firing");
        matmul_backward(0, vp, c, &mut self.grads_acts.lnf, &mut self.grads.wte, None, &self.grads_acts.logits, &self.acts.lnf, &self.params.wte);
        println!("layernorm backwards firing");
        lyrnrm_bwd(nl-1, 0, c, &mut self.grads_acts.residual3, &mut self.grads.lnfw, &mut self.grads.lnfb, &mut self.grads_acts.lnf, &self.acts.residual3, &self.params.lnfw, &self.acts.lnf_mean, &self.acts.lnf_rstd);

        for l in (0..nl).rev() {

            // residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
            // dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;
    
            // // get the pointers of the weights for this layer
            // float* l_ln1w = params.ln1w + l * C;
            // float* l_qkvw = params.qkvw + l * 3*C * C;
            // float* l_attprojw = params.attprojw + l * C * C;
            // float* l_ln2w = params.ln2w + l * C;
            // float* l_fcw = params.fcw + l * 4*C * C;
            // float* l_fcprojw = params.fcprojw + l * C * 4*C;
            // // get the pointers of the gradients of the weights for this layer
            // float* dl_ln1w = grads.ln1w + l * C;
            // float* dl_ln1b = grads.ln1b + l * C;
            // float* dl_qkvw = grads.qkvw + l * 3*C * C;
            // float* dl_qkvb = grads.qkvb + l * 3*C;
            // float* dl_attprojw = grads.attprojw + l * C * C;
            // float* dl_attprojb = grads.attprojb + l * C;
            // float* dl_ln2w = grads.ln2w + l * C;
            // float* dl_ln2b = grads.ln2b + l * C;
            // float* dl_fcw = grads.fcw + l * 4*C * C;
            // float* dl_fcb = grads.fcb + l * 4*C;
            // float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
            // float* dl_fcprojb = grads.fcprojb + l * C;
            // // get the pointers of the activations for this layer
            // float* l_ln1 = acts.ln1 + l * B * T * C;
            // float* l_ln1_mean = acts.ln1_mean + l * B * T;
            // float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
            // float* l_qkv = acts.qkv + l * B * T * 3*C;
            // float* l_atty = acts.atty + l * B * T * C;
            // float* l_att = acts.att + l * B * NH * T * T;
            // float* l_residual2 = acts.residual2 + l * B * T * C;
            // float* l_ln2 = acts.ln2 + l * B * T * C;
            // float* l_ln2_mean = acts.ln2_mean + l * B * T;
            // float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            // float* l_fch = acts.fch + l * B * T * 4*C;
            // float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
            // // get the pointers of the gradients of the activations for this layer
            // float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
            // float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
            // float* dl_atty = grads_acts.atty + l * B * T * C;
            // float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
            // float* dl_att = grads_acts.att + l * B * NH * T * T;
            // float* dl_attproj = grads_acts.attproj + l * B * T * C;
            // float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
            // float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
            // float* dl_fch = grads_acts.fch + l * B * T * 4*C;
            // float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
            // float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
            // float* dl_residual3 = grads_acts.residual3 + l * B * T * C;
    
            // backprop this layer
            println!("res bwk");
            residual_backward(l, c, &mut self.grads_acts.residual2, &mut self.grads_acts.fcproj, &self.grads_acts.residual3);
            println!("matmul backward in layer loop");
            matmul_backward(l, c, 4*c, &mut self.grads_acts.fch_gelu, &mut self.grads.fcprojw, Some(&mut self.grads.fcprojb), &self.grads_acts.fcproj, &self.acts.fch_gelu, &self.params.fcprojw);
            println!("gelu backward in layer loop");
            gelu_backward(l, c, &mut self.grads_acts.fch, &mut self.acts.fch, &self.grads_acts.fch_gelu);
            println!("2 matmul backward in layer loop");
            matmul_backward(l, 4*c, c, &mut self.grads_acts.ln2, &mut self.grads.fcw, Some(&mut self.grads.fcb), &self.grads_acts.fch, &self.acts.ln2, &mut self.params.fcw);
            println!("layernorm backward in layer loop");
            lyrnrm_bwd(l, l, c, &mut self.grads_acts.residual2, &mut self.grads.ln2w, &mut self.grads.ln2b, &mut self.grads_acts.ln2, &self.acts.residual2, &self.params.ln2w, &self.acts.ln2_mean, &self.acts.ln2_rstd);
            println!("residual backward in layer loop");
            residual_backward(l, c, &mut self.grads_acts.residual3, &mut self.grads_acts.attproj, &self.grads_acts.residual2);
            println!("3 matmul backward in layer loop ");
            matmul_backward(l, c, c, &mut self.grads_acts.atty, &mut self.grads.attprojw, Some(&mut self.grads.attprojb), &self.grads_acts.attproj, &self.acts.atty, &self.params.attprojw);
            println!("attention bwd in layer loop");
            attention_backward(l, c, nh, &mut self.grads_acts.qkv, &mut self.grads_acts.preatt, &mut self.grads_acts.att, &self.grads_acts.atty, &self.acts.qkv, &self.acts.att);
            println!("4 matmul bwd in layer loop");
            matmul_backward(l, 3*c, c, &mut self.grads_acts.ln1, &mut self.grads.qkvw, Some(&mut self.grads.qkvb), &self.grads_acts.qkv, &self.acts.ln1, &self.params.qkvw);
            println!("2 layernorm backward in layer loop");
            lyrnrm_bwd(l, l, c, &mut self.grads_acts.residual3, &mut self.grads.ln1w, &mut self.grads.ln1b, &mut self.grads_acts.ln1, &self.acts.residual3, &self.params.ln1w, &self.acts.ln1_mean, &self.acts.ln1_rstd);
            println!("loop done");
        }
        println!("encoder backward");
        // dout is residual[3] for code readability
        encoder_backward(c, &mut self.grads.wte, &mut self.grads.wpe, &self.grads_acts.residual3, &inputs);
        println!("done");
    }
}

impl fmt::Display for GPT2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, 
            "GPT2:\n{}\n{}", 
            self.config,
            self.params
        )
    }
}

fn main() {
    // build the GPT-2 model from a checkpoint
    let path: String = String::from("./bin/gpt2_124M.bin");
    let mut model: GPT2 = GPT2::new(&path);
    println!("{}", model);

    // build the DataLoaders from tokens files. For now use tiny_shakespeare if available
    let tiny_shakespeare_train: String = String::from("data/tinyshakespeare/tiny_shakespeare_train.bin");
    let tiny_shakespeare_val: String = String::from("data/tinyshakespeare/tiny_shakespeare_val.bin");
    let mut train_loader = dataloader::DataLoader::new(&tiny_shakespeare_train, 0, 1);
    // let mut val_loader = dataloader::DataLoader::new(&tiny_shakespeare_val, 0, 1);
    println!("train dataset num_batches: {}", train_loader.num_tokens / (B*T));
    // println!("val dataset num_batches: {}\n", val_loader.num_tokens / (B*T));
    let val_num_batches: usize = 5;

    // build the Tokenizer
    let tokenizer_path: String = String::from("data/tokenizer/gpt2_tokenizer.bin");
    let tokenizer: tokenizer::Tokenizer = tokenizer::Tokenizer::new(&tokenizer_path);
    println!("{}", tokenizer);

    // some memory for generating samples from the model
    // let rng_state: u64 = 1337;
    // let gen_tokens: [u32; B*T*16] = [0u32; B*T*16];

    // train
    // let time_instant: Instant = Instant::now();
    // for step in 0..41 {
    //     // once in a while estimate the validation loss
    //     if step % 10 == 0 {
    //         let mut val_loss: f32 = 0.0;
    //         val_loader.reset();
    //         for val_batch in 0..val_num_batches {
    //             let (inputs, targets) = val_loader.next_batch();
    //             println!("Forward {} is running", val_batch);
    //             model.forward(true, &inputs, &targets);
    //             val_loss += model.mean_loss;
    //         }
    //         val_loss /= val_num_batches as f32;
    //         println!("val loss {}\n", val_loss);
    //     }
    // }

        // // once in a while do model inference to print generated text
        // if (step > 0) && (step % 20 == 0) {
        //     // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
        //     for i in 0..B*T {
        //         gen_tokens[i] = tokenizer.eot_token;
        //     }
        //     // now sample from the model autoregressively
        //     println!("generating:\n---\n");
        //     for t in 0..GENT {
        //         // note that inference is very wasteful here because for each token
        //         // we re-calculate the forward pass for all of (B,T) positions from scratch
        //         // but the inference here is just for sanity checking anyway
        //         // and we can maybe optimize a bit more later, with careful tests
        //         model.forward(gen_tokens, None, B, T);
        //         // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
        //         // we're in principle running B "inference streams" in parallel here
        //         // but only using position 0
        //         // get the Vp-dimensional vector probs[0, t-1, :]
        //         let probs: Vec<f32> = model.self.acts.probs + (t-1) * model.config.padded_vocab_size;
        //         let coin: f32 = random_f32(&rng_state);
        //         // note we're only sampling from the first V elements, ignoring padding
        //         // (the probabilities in the padded region should be zero anyway)
        //         let next_token: usize = sample_mult(probs, model.config.vocab_size, coin);
        //         gen_tokens[t] = next_token;
        //         // print the generated token, either using the Tokenizer or a fallback
        //         if tokenizer.init_ok {
        //             let token_str: String = tokenizer.decode(next_token);
        //             println!("{}", token_str);
        //         } else {
        //             // fall back to printing the token id
        //             println!("{}", next_token);
        //         }
        //         fflush(stdout);
        //     }
        //     println!("\n---\n");
        // }

        // do a training step
        // let start_time = time_instant.elapsed().as_secs();
        let (train_inputs, train_targets) = train_loader.next_batch();
        // model.forward(true, &train_inputs, &train_targets);
        // model.zero_grad();
        model.backward(&train_inputs, &train_targets);
        // model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step+1);
        // let end_time = time_instant.now.elapsed().as_secs();
        // // TODO: wait what the fuck is happening in this line
        // let time_elapsed_s: f32 = (end_time.now() - start_time.now()) + (end.tv_nsec - start.tv_nsec) / 1e9;
        // println!("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
}

// ----------------------- //
// -- Forward Functions -- //
// ----------------------- //

fn lyrnrm_fwd(
    il: usize,
    ol: usize,
    c: usize,
    inp: &Vec<f32>,
    weight: &Vec<f32>,
    bias: &Vec<f32>,
    out: &mut Vec<f32>,
    mean: &mut Vec<f32>,
    rstd: &mut Vec<f32>,
) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted

    let eps: f32 = 1.0e-5;
    for b in 0..B {
        for t in 0..T {
            // let lb: usize = l*(B*T*c) + b*(T*c);
            let ilbt: usize = il*(B*T*c) + b*(T*c) + t*(c);

            // calculate the mean
            let mut m: f32 = 0.0;
            for i in 0..c {
                m += inp[ilbt+i];
            }
            m = m / c as f32;

            // calculate the variance (without any bias correction)
            let mut v: f32 = 0.0;
            for i in 0..c {
                let xshift: f32 = inp[ilbt+i] - m;
                v += xshift * xshift;
            }
            v = v/c as f32;

            // calculate the rstd (reciprocal standard deviation)
            let s: f32 = 1.0 / (v + eps).sqrt();

            // perform calculations
            for i in 0..c {
                let n: f32 = s * (inp[ilbt+i] - m); // normalize
                let o: f32 = n * weight[ol*(c) + i] + bias[ol*(c) + i]; // scale and shift
                out[ol*(B*T*c) + b*(T*c) + t*(c) + i] = o; // write
            }

            // cache the mean and rstd for the backward pass later
            mean[ol*(B*T) + b*(T) + t] = m;
            rstd[ol*(B*T) + b*(T) + t] = s;
        }
    }
}

fn matmul_fwd_naive(
    l: usize,
    ic: usize,
    oc: usize,
    inp: &Vec<f32>,
    weight: &Vec<f32>,
    bias: Option<&Vec<f32>>,
    out: &mut Vec<f32>,
) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_fwd(), below.
    for b in 0..B {
        for t in 0..T {
            // println!("Running {} out of {}", b*T + t, B*T);
            for o in 0..oc {
                let mut val: f32 = match bias {
                    Some(bias_vec) => bias_vec[l*oc + o],
                    _ => 0.0,
                };
                for i in 0..ic {
                    let input_val: f32 = inp[l*(B*T*ic) + b*(T*ic) + t*(ic) + i];
                    let weight_val: f32 = weight[l*(oc*ic) + o*(ic) + i];
                    val += input_val * weight_val;
                }
                out[l*(B*T*oc) + b*(T*oc) + t*(oc) + o] = val;
            }
        }
    }
}

fn matmul_fwd(
    l: usize,
    ic: usize,
    oc: usize,
    inp: &Vec<f32>,
    weight: &Vec<f32>,
    bias: Option<&Vec<f32>>,
    out: &mut Vec<f32>,
) {
    // most of the running time is spent here and in matmul_bwd
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // oc is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,oc)

    // make sure the tiled loop will be correct or fallback to naive version
    // const LOOP_UNROLL: usize = 8;
    // if B*T % LOOP_UNROLL != 0 {
    matmul_fwd_naive(l, ic, oc, inp, weight, bias, out);
    return;
    // }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // #pragma omp parallel for
    // for obt in 0..B*T where obt += LOOP_UNROLL {
    //     for o in 0..oc {
    //         // we'll keep LOOP_UNROLL many results in registers
    //         let mut result: Vec<f32> = Vec::new(LOOP_UNROLL);
    //         // initialize the bias, if it exists
    //         for ibt in 0..LOOP_UNROLL {
    //             result[ibt] = (bias != None) ? bias[o] : 0.0;
    //         }
    //         // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
    //         // the value of weight[i + o * C] and reuse it.
    //         // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
    //         for i in 0..c {
    //             float w = weight[i + o * C];
    //             for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
    //                 int bt = obt + ibt;
    //                 result[ibt] += inp[bt * C + i] * w;
    //             }
    //         }
    //         // write back results to main memory
    //         for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
    //             int bt = obt + ibt;
    //             out[bt * OC + o] = result[ibt];
    //         }
    //     }
}

fn attent_fwd(
    l: usize,
    c: usize,
    nh: usize,
    qkv: &Vec<f32>,
    preatt: &mut Vec<f32>,
    att: &mut Vec<f32>,
    atty: &mut Vec<f32>,
) {
    // qkv is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    let c3: usize = 3*c;
    let hs: usize = c / nh; // head size
    let scale: f32 = 1.0 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..nh {
                let query_t: usize = l*(B*T*c3) + b*(T*c3) + t*(c3) + h*hs;
                let preatt_bth: usize = l*(B*nh*T*T) + b*(nh*T*T) + h*(T*T) + t*T;
                let att_bth: usize = l*(B*nh*T*T) + b*nh*T*T + h*(T*T) + t*T;

                // 1: calculate QK and maxval
                let mut maxval: f32 = -10000.0;
                for t2 in 0..t+1 {
                    let key_t2: usize = l*(B*T*c3) + b*T*c3 + t2*c3 + h*hs + c; // +C because it's key

                    // (query_t) dot (key_t2)
                    let mut val: f32 = 0.0;
                    for i in 0..hs {
                        val += qkv[query_t+i] * qkv[key_t2+i];
                    }
                    val *= scale;
                    if val > maxval {maxval = val};

                    preatt[preatt_bth+t2] = val;
                }

                // 2: calculate the exponentiation and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                let mut expsum: f32 = 0.0;
                for t2 in 0..t+1 {
                    let expv: f32 = (preatt[preatt_bth+t2] - maxval).exp();
                    expsum += expv;
                    att[att_bth+t2] = expv;
                }
                let expsum_inv: f32 = if expsum == 0.0 {0.0} else {1.0 / expsum};

                // pass 3: normalize to get the softmax
                for t2 in 0..T {
                    if t2 <= t {
                        att[att_bth+t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att[att_bth+t2] = 0.0;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                let atty_bth: usize = l*(B*T*c) + b*(T*c) + t*(c) + h*(hs);
                for i in 0..hs {atty[atty_bth+i] = 0.0}
                for t2 in 0..t+1 {
                    let value_t2: usize = l*(B*T*c3) + b*(T*c3) + t2*(c3) + h*(hs) + c*2; // +C*2 because it's value
                    let att_btht2: f32 = att[att_bth+t2];
                    for i in 0..hs {
                        atty[atty_bth+i] += att_btht2 * qkv[value_t2+i];
                    }
                }
            }
        }
    }
}

fn residual_fwd(
    l: usize,
    c: usize,
    inp1: &Vec<f32>,
    inp2: &Vec<f32>,
    out: &mut Vec<f32>,
) {
    let skip: usize = l*(B*T*c);
    for i in 0..(B*T*c) {
        out[skip+i] = inp1[skip+i] + inp2[skip+i];
    }
}

fn gelu_fwd(
    l: usize,
    c: usize,
    fch: &Vec<f32>,
    fch_gelu: &mut Vec<f32>,
) {
    const M_PI: f32 = std::f32::consts::PI;
    const GELU_SCALING_FACTOR: f32 = 2.0 / M_PI; // needs a square root
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    let skip: usize = l*(B*T*4*c);
    for i in 0..(B*T*4*c) {
        let x: f32 = fch[skip+i];
        let cube: f32 = 0.044715*x*x*x;
        fch_gelu[skip+i] = 0.5 * x * (1.0 + (GELU_SCALING_FACTOR.sqrt() * (x+cube)).sqrt());
    }
}

fn softmax_fwd(
    v: usize,
    vp: usize,
    logits: &Vec<f32>,
    probs: &mut Vec<f32>,
) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    // #pragma omp parallel for collapse(2)
    for b in 0..B {
        for t in 0..T {
            // probs <- softmax(logits)
            let logits_bt: usize = b*(T*vp) + t*(vp);
            let probs_bt: usize = b*(T*vp) + t*(vp);

            // maxval is only calculated and subtracted for numerical stability
            let mut maxval: f32 = -10000.0; // TODO something better
            for i in 0..v {
                if logits[logits_bt+i] > maxval {
                    maxval = logits[logits_bt+i];
                }
            }
            let mut sum: f32 = 0.0;
            for i in 0..v {
                probs[probs_bt+i] = (probs[probs_bt+i] - maxval).exp();
                sum += probs[probs_bt+i];
            }
            // note we only loop to V, leaving the padded dimensions
            for i in 0..v {
                probs[probs_bt+i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for i in 0..vp {
                probs[probs_bt+i] = 0.0;
            }
        }
    }
}

fn crossentropy_fwd(
    vp: usize,
    probs: &Vec<f32>,
    targets: &Vec<u32>,
    losses: &mut Vec<f32>,
) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for b in 0..B {
        for t in 0..T {
            // loss = -log(probs[target])
            let probs_bt: usize = b*(T*vp) + t*(vp);
            let ix: usize = targets[b*(T) + t] as usize;
            losses[b*(T) + t] = -1.0 * (probs[probs_bt+ix]).ln();
        }
    }
}

// ------------------------ //
// -- Backward Functions -- //
// ------------------------ //

fn encoder_backward(
    c: usize,
    dwte: &mut Vec<f32>, 
    dwpe: &mut Vec<f32>,
    dout: &Vec<f32>, 
    inp: &Vec<u32>,
) {
    for b in 0..B {
        for t in 0..T {
            let dout_bt: usize = b*(T*c) + t*(c);
            let ix: usize = inp[b*(T) + t] as usize;
            println!("{}", ix);
            let dwte_ix: usize = ix*(c);
            let dwpe_t: usize = t*(c);
            for i in 0..c {
                let d: f32 = dout[dout_bt+i];
                dwte[dwte_ix+i] += d;
                dwpe[dwpe_t+i] += d;
            }
        }
    }
}

fn attention_backward(
    l: usize,
    c: usize,
    nh: usize,
    dinp: &mut Vec<f32>, 
    dpreatt: &mut Vec<f32>, 
    datt: &mut Vec<f32>,
    dout: &Vec<f32>, 
    inp: &Vec<f32>, 
    att: &Vec<f32>,
) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    let c3: usize = c*3;
    let hs: usize = c / nh; // head size
    let scale: f32 = 1.0 / (hs as f32).sqrt();

    for b in 0..B {
        for t in 0..T {
            for h in 0..nh {
                let att_lbth: usize = l*(B*nh*T*T) + b*(nh*T*T) + h*(T*T) + t*(T);
                let datt_lbth: usize = l*(B*nh*T*T) + b*(nh*T*T) + h*(T*T) + t*(T);
                let dpreatt_lbth: usize = l*(B*nh*T*T) + b*(nh*T*T) + h*(T*T) + t*(T);
                let dquery_lbth: usize = l*(B*T*c3) + b*(T*c3) + t*(c3) + h*(hs);
                let query_lbth: usize = l*(B*T*c3) + b*(T*c3) + t*(c3) + h*(hs);

                // backward pass 4, through the value accumulation
                let dout_lbth: usize = l*(B*T*c) + b*(T*c) + t*(c) + h*(hs);
                for t2 in 0..t {
                    let value_lbthc: usize = l*(B*T*c3) + b*(T*c3) + t2*(c3) + h*(hs) + c*2; // +C*2 because it's value
                    let dvalue_lbthc: usize = l*(B*T*c3) + b*(T*c3) + t2*(c3) + h*(hs) + c*2;
                    for i in 0..hs {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt[datt_lbth+t2] += inp[value_lbthc+i] * dout[dout_lbth+i];
                        dinp[dvalue_lbthc+i] += att[att_lbth+t2] * dout[dout_lbth+i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in 0..t {
                    for t3 in 0..t {
                        let indicator: f32 = if t2 == t3 {1.0} else {0.0};
                        let local_derivative: f32 = att[att_lbth+t2] * (indicator - att[att_lbth+t3]);
                        dpreatt[dpreatt_lbth+t3] += local_derivative * datt[datt_lbth+t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for t2 in 0..t {
                    let key_lbthc: usize = l*(B*T*c3)+ b*(T*c3) + t2*(c3) + h*(hs) + c; // +C because it's key
                    let dkey_lbthc: usize = l*(B*T*c3) + b*(T*c3) + t2*(c3) + h*(hs) + c; // +C because it's key
                    for i in 0..hs {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dinp[dquery_lbth+i] += inp[key_lbthc+i] * dpreatt[dpreatt_lbth+t2] * scale;
                        dinp[dkey_lbthc+i] += inp[query_lbth+i] * dpreatt[dpreatt_lbth+t2] * scale;
                    }
                }
            }
        }
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
// #pragma float_control(precise, on, push)
// #if defined(__GNUC__) && !defined(__clang__)
// __attribute__((optimize("no-finite-math-only")))
// #endif
fn gelu_backward(
    l: usize,
    c: usize,
    dinp: &mut Vec<f32>, 
    inp: &mut Vec<f32>,
    dout: &Vec<f32>
) {
    const M_PI: f32 = std::f32::consts::PI;
    const GELU_SCALING_FACTOR: f32 = 2.0 / M_PI; // needs a square root

    let skip: usize = l*(B*T*4*c);
    for i in 0..(B*T*4*c) {
        let x: f32 = inp[skip+i];
        let cube: f32 = 0.044715 * x * x * x;
        let tanh_arg: f32 = GELU_SCALING_FACTOR * (x + cube);
        let tanh_out: f32 = tanh_arg.tanh();
        let coshf_out: f32 = tanh_arg.cosh();
        let sech_out: f32 = 1.0 / (coshf_out * coshf_out);
        let local_grad: f32 = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1.0 + 3.0 * 0.044715 * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
// #pragma float_control(pop)

fn residual_backward(
    l: usize, 
    c: usize, 
    dinp1: &mut Vec<f32>, 
    dinp2: &mut Vec<f32>, 
    dout: &Vec<f32>
) {
    let skip: usize = l*(B*T*c);
    for i in 0..(B*T*c) {
        dinp1[skip+i] += dout[skip+i];
        dinp2[skip+i] += dout[skip+i];
    }
}

fn lyrnrm_bwd(
    il: usize,
    ol: usize,
    c: usize,
    dinp: &mut Vec<f32>, 
    dweight: &mut Vec<f32>, 
    dbias: &mut Vec<f32>,
    dout: &mut Vec<f32>, 
    inp: &Vec<f32>, 
    weight: &Vec<f32>, 
    mean: &Vec<f32>, 
    rstd: &Vec<f32>,
) {
    for b in 0..B {
        for t in 0..T {
            let dout_lbt: usize = ol*(B*T*c) + b*(T*c) + t*(c);
            let inp_lbt: usize = il*(B*T*c) + b*(T*c) + t*(c);
            let dinp_lbt: usize = il*(B*T*c) + b*(T*c) + t*(c);
            let mean_val: f32 = mean[ol*(B*T) + b*(T) + t];
            let rstd_val: f32 = rstd[ol*(B*T) + b*(T) + t];

            // first: two reduce operations
            let mut dnorm_mean: f32 = 0.0;
            let mut dnorm_norm_mean: f32 = 0.0;
            for i in 0..c {
                let norm_bti: f32 = (inp[inp_lbt + i] - mean_val) * rstd_val;
                let dnorm_i: f32 = weight[i] * dout[dout_lbt + i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / c as f32;
            dnorm_norm_mean = dnorm_norm_mean / c as f32;

            // now iterate again and accumulate all the gradients
            for i in 0..c {
                let norm_bti: f32 = (inp[inp_lbt + i] - mean_val) * rstd_val;
                let dnorm_i: f32 = weight[i] * dout[dout_lbt + i];
                // gradient contribution to bias
                dbias[i] += dout[dout_lbt + i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout[dout_lbt + i];
                // gradient contribution to input
                let mut dval: f32 = 0.0;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_val; // final scale
                dinp[dinp_lbt + i] += dval;
            }
        }
    }
}

fn matmul_backward(
    l: usize,
    oc: usize,
    ic: usize,
    dinp: &mut Vec<f32>,
    dweight: &mut Vec<f32>,
    mut dbias: Option<&mut Vec<f32>>,
    dout: &Vec<f32>,
    inp: &Vec<f32>,
    weight: &Vec<f32>,
) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy
    // TODO: For CPU purposes I need to do this in a single "round" of loops like
    // it's done above

    for b in 0..B {
        for t in 0..T {
            println!("Backwards matmul {} out of {}", b*T+t, B*T);
            for o in 0..oc {
                let d: f32 = dout[l*(B*T*oc) + b*(T*oc) + t*(oc) + o];
                for i in 0..ic {
                    dinp[l*(B*T*ic) + b*(T*ic) + t*(ic) + i] += weight[l*(oc*ic) + o*(ic) + i] * d;
                }
                match dbias {
                    Some(ref mut dbias_vec) => dbias_vec[l*oc + o] += d,
                    _ => {},
                };
                for i in 0..ic {
                    dweight[l*(oc*ic) + o*(ic) + i] += inp[l*(B*T*ic) + b*(T*ic) + t*(ic) + i] * d;
                }
            }
            return;
        }
    }
}

fn crossentropy_softmax_backward(
    v: usize,
    vp: usize,
    dlogits: &mut Vec<f32>,
    dlosses: &Vec<f32>,
    probs: &Vec<f32>,
    targets: &Vec<u32>,
) {
    // backwards through both softmax and crossentropy
    for b in 0..B {
        for t in 0..T {
            let dloss: f32 = dlosses[b*T + t];
            let ix: usize = targets[b*T + t] as usize;
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for i in 0..v {
                let p: f32 = probs[b*(T*vp) + t*(vp) + i];
                let indicator: f32 = if i == ix {1.0} else {0.0};
                dlogits[b*(T*vp) + t*(vp) + i] += (p - indicator) * dloss;
            }
        }
    }
}
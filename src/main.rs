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
    qkvw: Vec<f32>, // (L, 3*C, C)
    qkvb: Vec<f32>, // (L, 3*C)
    attprojw: Vec<f32>, // (L, C, C)
    attprojb: Vec<f32>, // (L, C)
    // second linear
    ln2w: Vec<f32>, // (L, C)
    ln2b: Vec<f32>, // (L, C)

    // multi-layer perceptron
    fcw: Vec<f32>, // (L, 4*C, C)
    fcb: Vec<f32>, // (L, 4*C)
    fcprojw: Vec<f32>, // (L, C, 4*C)
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
            fcw: utils::read_floats_into_vecf32(model_file, l * 4*c, &String::from("fcw")),
            fcb: utils::read_floats_into_vecf32(model_file, l * 4*c, &String::from("fcb")),
            fcprojw: utils::read_floats_into_vecf32(model_file, l * c * 4*c, &String::from("fcprojw")),
            fcprojb: utils::read_floats_into_vecf32(model_file, l * c, &String::from("fcprojb")),
            lnfw: utils::read_floats_into_vecf32(model_file, c, &String::from("lnfw")),
            lnfb: utils::read_floats_into_vecf32(model_file, c, &String::from("lnfb")),
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
    // attproj: Vec<f32>, // (L, B, T, C)
    // residual2: Vec<f32>, // (L, B, T, C)
    // ln2: Vec<f32>, // (L, B, T, C)
    // ln2_mean: Vec<f32>, // (L, B, T)
    // ln2_rstd: Vec<f32>, // (L, B, T)
    // fch: Vec<f32>, // (L, B, T, 4*C)
    // fch_gelu: Vec<f32>, // (L, B, T, 4*C)
    // fcproj: Vec<f32>, // (L, B, T, C)
    residual3: Vec<f32>, // (L, B, T, C)
    // lnf: Vec<f32>, // (B, T, C)
    // lnf_mean: Vec<f32>, // (B, T)
    // lnf_rstd: Vec<f32>, // (B, T)
    // logits: Vec<f32>, // (B, T, V)
    // probs: Vec<f32>, // (B, T, V)
    // losses: Vec<f32>, // (B, T)
}

impl ActivationTensors {
    fn new(l: usize, c: usize, nh: usize) -> ActivationTensors {
        return ActivationTensors {
            // encoded: vec![0f32; B*T*c],
            ln1: vec![0f32; l*B*T*c],
            ln1_mean: vec![0f32; l*B*T*c],
            ln1_rstd: vec![0f32; l*B*T*c],
            qkv: vec![0f32; l*B*T*3*c], 
            atty: vec![0f32; l*B*T*c], 
            preatt: vec![0f32; l*B*nh*T*T], 
            att: vec![0f32; l*B*nh*T*T], 
            // attproj: vec![0f32; B*T*c], 
            // residual2: vec![0f32; B*T*c], 
            // ln2: vec![0f32; B*T*c], 
            // ln2_mean: vec![0f32; B*T*c],
            // ln2_rstd: vec![0f32; B*T*c],
            // fch: vec![0f32; B*T*c], 
            // fch_gelu: vec![0f32; B*T*c], 
            // fcproj: vec![0f32; B*T*c], 
            residual3: vec![0f32; l*B*T*c], 
            // lnf: vec![0f32; B*T*c],
            // lnf_mean: vec![0f32; B*T*c],
            // lnf_rstd: vec![0f32; B*T*c],
            // logits: vec![0f32; B*T*c],
            // probs: vec![0f32; B*T*c],
            // losses: vec![0f32; B*T*c],
        }
    }
}

struct GPT2 {
    config: GPT2Config,
    params: ParameterTensors,
    // grads: Option<ParameterTensors>,
    // m_memory: Option<Vec<f32>>,
    // v_memory: Option<Vec<f32>>,
    acts: ActivationTensors,
    // grads_acts: Option<ActivationTensors>,
    // batch_size: u32, // the batch size (B) of current forward pass
    // seq_len: u32, // the sequence length (T) of current forward pass
    inputs: Vec<usize>, // the input tokens for the current forward pass
    targets: Vec<usize>, // the target tokens for the current forward pass
    // mean_loss: f32, // after a forward pass with targets, will be populated with the mean loss
}

impl GPT2 {
    fn new(checkpoint_path: &String) -> GPT2 {
        // read in model from a checkpoint file
        let mut model_file: File = utils::fopen_check(checkpoint_path);
        let config: GPT2Config = GPT2Config::new(&mut model_file);
        let params: ParameterTensors = ParameterTensors::new(&mut model_file, &config);
        let acts: ActivationTensors = ActivationTensors::new(config.num_layers, config.channels, config.num_heads);

        return GPT2 {
            config: config,
            params: params,
            // grads: None,
            // m_memory: None,
            // v_memory: None,
            acts: acts,
            // grads_acts: None,
            // batch_size: 0,
            // seq_len: 0,
            inputs: vec![0; B*T],
            targets: vec![0; B*T],
            // mean_loss: 0f32,
        };
    }

    fn encoder_forward(&mut self) {
        // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        // inp is (B,T) of integers, holding the token ids at each (b,t) position
        // wte is (V,C) of token embeddings, short for "weight token embeddings"
        // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"

        let c = self.config.channels;
        for b in 0..B {
            for t in 0..T {
                let idx = self.inputs[b*(T) + t] as usize;
                for i in 0..c {
                    let wte_val: f32 = self.params.wte[idx*c + i];
                    let wpe_val: f32 = self.params.wpe[t*c + i];
                    // pretend that the encoding is residual3[0], layer is 0
                    self.acts.residual3[0*(B*T*c) + b*(T*c) + t*(c) + i] = wte_val + wpe_val;
                }
            }
        }
    }

    // fn residual_fwd(
    //     &mut self,
    //     l: usize,
    //     inp1: Vec<f32>,
    //     inp2: Vec<f32>,
    //     mut out: Vec<f32>,
    // ) {
    //     let c: usize = self.config.channels;
    //     let skip: usize = l*(B*T*c);
    //     for i in 0..(B*T*c) {
    //         out[skip+i] = inp1[skip+i] + inp2[skip+i];
    //     }
    // }

    // fn gelu_fwd(
    //     &mut self,
    //     l: usize,
    //     fch: Vec<f32>,
    //     mut fch_gelu: Vec<f32>,
    // ) {
    //     let c: usize = self.config.channels;
    //     const M_PI: f32 = std::f32::consts::PI;
    //     const GELU_SCALING_FACTOR: f32 = 2.0 / M_PI; // needs a square root
    //     // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    //     let skip: usize = l*(B*T*4*c);
    //     for i in 0..(B*T*4*c) {
    //         let x: f32 = fch[skip+i];
    //         let cube: f32 = 0.044715*x*x*x;
    //         fch_gelu[skip+i] = 0.5 * x * (1.0 + (GELU_SCALING_FACTOR.sqrt() * (x+cube)).sqrt());
    //     }
    // }

    // pub fn softmax_fwd(
    //     &mut self,
    //     v: usize,
    //     vp: usize,
    //     mut probs: Vec<f32>,
    //     logits: Vec<f32>,
    // ) {
    //     // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    //     // input: logits is (B,T,Vp) of the unnormalized log probabilities
    //     // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    //     // example: Vp is 50304 and V is 50257
    //     // #pragma omp parallel for collapse(2)
    //     for b in 0..B {
    //         for t in 0..T {
    //             // probs <- softmax(logits)
    //             let logits_bt: usize = b*(T*vp) + t*(vp);
    //             let probs_bt: usize = b*(T*vp) + t*(vp);

    //             // maxval is only calculated and subtracted for numerical stability
    //             let mut maxval: f32 = -10000.0; // TODO something better
    //             for i in 0..v {
    //                 if logits[logits_bt+i] > maxval {
    //                     maxval = logits[logits_bt+i];
    //                 }
    //             }
    //             let mut sum: f32 = 0.0;
    //             for i in 0..v {
    //                 probs[probs_bt+i] = (probs[probs_bt+i] - maxval).exp();
    //                 sum += probs[probs_bt+i];
    //             }
    //             // note we only loop to V, leaving the padded dimensions
    //             for i in 0..v {
    //                 probs[probs_bt+i] /= sum;
    //             }
    //             // for extra super safety we may wish to include this too,
    //             // forcing the probabilities here to be zero, but it shouldn't matter
    //             for i in 0..vp {
    //                 probs[probs_bt+i] = 0.0;
    //             }
    //         }
    //     }
    // }

    // fn crossentropy_fwd(
    //     &mut self,
    //     vp: usize,
    //     mut losses: Vec<f32>,
    //     probs: Vec<f32>,
    //     targets: Vec<usize>,
    // ) {
    //     // output: losses is (B,T) of the individual losses at each position
    //     // input: probs are (B,T,Vp) of the probabilities
    //     // input: targets is (B,T) of integers giving the correct index in logits
    //     for b in 0..B {
    //         for t in 0..T {
    //             // loss = -log(probs[target])
    //             let probs_bt: usize = b*(T*vp) + t*(vp);
    //             let ix: usize = targets[b*(T) + t];
    //             losses[b*(T) + t] = -1.0 * (probs[probs_bt+ix]).ln();
    //         }
    //     }
    // }

    pub fn forward(&mut self) {
        // convenience parameters (size_t to help prevent int overflow)
        let v = self.config.vocab_size;
        // let vp = self.config.padded_vocab_size;
        let nl = self.config.num_layers;
        let nh = self.config.num_heads;
        let c = self.config.channels;

        // validate inputs, all indices must be in the range [0, V)
        for i in 0..B*T {
            // assert!(0 <= self.inputs[i]);
            assert!(self.inputs[i] < v);
            // assert!(0 <= self.targets[i]);
            assert!(self.targets[i] < v);
        }

        // let acts: &mut ActivationTensors = &mut self.acts;
        // let prms: &ParameterTensors = &self.params;

        // forward pass
        // let mut residual: Vec<f32> = Vec::new(); // (L * C)
        self.encoder_forward(); // encoding goes into residual3[0]
        for l in 0..nl {
            println!("Layer {} is running", l);
            // TODO: need to fix the residual stuff, it's a little confusing
            // Multi-Head Attention
            // println!("{}", self.acts.ln1[0]);
            // println!("{}", self.acts.residual3.len());
            lyrnrm_fwd(l, c, &self.acts.residual3, &self.params.ln1w, &self.params.ln1b, &mut self.acts.ln1, &mut self.acts.ln1_mean, &mut self.acts.ln1_rstd);
            matmul_fwd(l, c, c, 3*c, &self.acts.ln1, &self.params.qkvw, Some(&self.params.qkvb), &mut self.acts.qkv);
            attent_fwd(l, c, nh, &self.acts.qkv, &mut self.acts.preatt, &mut self.acts.att, &mut self.acts.atty);
            // self.matmul_fwd(l, c, c, self.acts.atty, self.params.attprojw, self.params.attprojb, self.acts.attproj);
            // self.residual_fwd(l, self.acts.residual3, self.acts.attproj, self.acts.residual2);
            // self.lyrnrm_fwd(l, self.acts.residual2, self.params.ln2w, self.params.ln2b, self.acts.ln2, self.acts.ln2_mean, self.acts.ln2_rstd);
            // // MLP
            // self.matmul_fwd(l, c, 4*c, self.acts.ln2, self.params.fcw, self.params.fcb, self.acts.fch);
            // self.gelu_fwd(l, self.acts.fch, self.acts.fch_gelu);
            // self.matmul_fwd(l, 4*c, c, self.acts.fch_gelu, self.params.fcprojw, self.params.fcprojb, self.acts.fcproj);
            // self.residual_fwd(l, self.acts.residual2, self.acts.residual3, self.acts.fcproj);
        }

        // last residual is in residual3[-1]
        // self.lyrnrm_fwd(nl-1, self.acts.residual3, self.params.lnfw, self.params.lnfb, self.acts.lnf, self.acts.lnf_mean, self.acts.lnf_rstd,);
        // self.matmul_fwd(nl-1, c, vp, self.acts.logits, self.acts.lnf, self.params.wte, None);
        // self.softmax_fwd(v, vp, self.acts.probs, self.acts.logits);

        // // also forward the cross-entropy loss function if we have the targets
        // if self.targets != None {
        //     self.crossentropy_fwd(vp, self.acts.losses, self.acts.probs, self.targets);
        //     // for convenience also evaluate the mean loss
        //     self.mean_loss = 0.0;
        //     for i in 0..B*T {self.mean_loss += self.acts.losses[i]}
        //     for i in 0..B*T {self.mean_loss += self.acts.losses[i]};
        //     self.mean_loss /= B*T as f32;
        // } else {
        //     // if we don't have targets, we don't have a loss
        //     self.mean_loss = -1.0;
        // }
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
    let train_loader = dataloader::DataLoader::new(&tiny_shakespeare_train, 0, 1);
    let mut val_loader = dataloader::DataLoader::new(&tiny_shakespeare_val, 0, 1);
    println!("train dataset num_batches: {}", train_loader.num_tokens / (B*T));
    println!("val dataset num_batches: {}\n", val_loader.num_tokens / (B*T));
    // let val_num_batches: usize = 1; // TODO: This should be 5!

    // build the Tokenizer
    let tokenizer_path: String = String::from("data/tokenizer/gpt2_tokenizer.bin");
    let tokenizer: tokenizer::Tokenizer = tokenizer::Tokenizer::new(&tokenizer_path);
    println!("{}", tokenizer);

    // some memory for generating samples from the model
    // let rng_state: u64 = 1337;
    // let gen_tokens: [u32; B*T*16] = [0u32; B*T*16];

    // train
    // let time_instant: Instant = Instant::now();
    val_loader.next_batch();
    // println!("Forward {} is running", val_batch);
    model.forward();

    // for step in 0..40 {
    //     // once in a while estimate the validation loss
    //     if step % 10 == 0 {
    //         let mut _val_loss: f32 = 0.0;
    //         val_loader.reset();
    //         for val_batch in 0..val_num_batches {
    //             val_loader.next_batch();
    //             println!("Forward {} is running", val_batch);
    //             model.forward();
    //             _val_loss += model.mean_loss;
    //         }
    //     }
    // }

    // for step in 0..41 {
    //     if step % 10 == 0 {
    //         let mut val_loss: f32 = 0.0;
    //         val_loader.reset();
    //         for i in 0..val_num_batches {
    //             val_loader.next_batch();
    //             model.forward(&val_loader.inputs, &val_loader.targets);
    //             val_loss += model.mean_loss;
    //         }
        //     val_loss /= val_num_batches;
        //     println!("val loss {}\n", val_loss);
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

        // // do a training step
        // let start_time = time_instant.now.elapsed().as_secs();
        // train_loader.next_batch();
        // model.forward(&train_loader.inputs, &train_loader.targets, B, T);
        // model.zero_grad();
        // model.backward();
        // model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step+1);
        // let end_time = time_instant.now.elapsed().as_secs();
        // // TODO: wait what the fuck is happening in this line
        // let time_elapsed_s: f32 = (end_time.now() - start_time.now()) + (end.tv_nsec - start.tv_nsec) / 1e9;
        // println!("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
}






fn lyrnrm_fwd(
    l: usize,
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
    // let c: usize = self.config.channels;
    // println!("\n");
    // println!("LBTC: {}", 12*4*64*768);
    // println!("inp: {}", inp.len());
    // println!("LC: {}", 12*768);
    // println!("weight: {}", weight.len());
    // println!("LC: {}", 12*768);
    // println!("bias: {}", bias.len());
    // println!("LBTC: {}", 12*4*64*768);
    // println!("out: {}", out.len());
    // println!("LBT: {}", 12*4*64*768);
    // println!("mean: {}", mean.len());
    // println!("LBT: {}", 12*4*64*768);
    // println!("rstd: {}", rstd.len());

    let eps: f32 = 1.0e-5;
    for b in 0..B {
        for t in 0..T {
            let lb: usize = l*(B*T*c) + b*(T*c);
            let lbt: usize = lb + t*(c);

            // calculate the mean
            let mut m: f32 = 0.0;
            for i in 0..c {
                m += inp[lbt+i];
            }
            m = m / c as f32;

            // calculate the variance (without any bias correction)
            let mut v: f32 = 0.0;
            for i in 0..c {
                let xshift: f32 = inp[lbt+i] - m;
                v += xshift * xshift;
            }
            v = v/c as f32;

            // calculate the rstd (reciprocal standard deviation)
            let s: f32 = 1.0 / (v + eps).sqrt();

            // perform calculations
            for i in 0..c {
                let n: f32 = s * (inp[lbt+i] - m); // normalize
                let o: f32 = n * weight[l*(c) + i] + bias[l*(c) + i]; // scale and shift
                out[lbt+i] = o; // write
            }

            // cache the mean and rstd for the backward pass later
            mean[lb + t] = m;
            rstd[lb + t] = s;
        }
    }
}

fn matmul_fwd_naive(
    l: usize,
    c: usize,
    ic: usize,
    oc: usize,
    inp: &Vec<f32>,
    weight: &Vec<f32>,
    bias: Option<&Vec<f32>>,
    out: &mut Vec<f32>,
) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    for b in 0..B {
        for t in 0..T {
            let lbt: usize = l*(B*T) + b*(T) + t;
            for o in 0..oc {
                let mut val: f32 = match bias {
                    Some(bias_vec) => bias_vec[l*oc + o],
                    _ => 0.0,
                };
                for i in 0..ic { // TODO I think this is the only change required
                    // val += inp[lbt*c + i] * weight[l*(oc*c) + o*c + i];
                    val += inp[lbt*c + i] * weight[o*oc + i];
                }
                out[lbt*oc + o] = val;
            }
        }
    }
}

fn matmul_fwd(
    l: usize,
    c: usize,
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
    matmul_fwd_naive(l, c, ic, oc, inp, weight, bias, out);
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
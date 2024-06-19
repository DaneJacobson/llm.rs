use std::time::{Instant, Duration};

mod constants;
mod dataloader;
mod model;
mod tensors;
mod tokenizer;
mod utils;

use constants::{B, T, GENT, BUFFER_SIZE};
use dataloader::DataLoader;
use tokenizer::Tokenizer;

// -----------------------------------------------------------------------------

fn main() {
    // build the GPT-2 model from a checkpoint
    let path: String = String::from("./bin/gpt2_124M.bin");
    let mut model: model::GPT2 = model::GPT2::new(&path);
    println!("{}", model);

    // build the DataLoaders from tokens files. For now use tiny_shakespeare if available
    let tiny_shakespeare_train: String = String::from("data/tinyshakespeare/tiny_shakespeare_train.bin");
    let tiny_shakespeare_val: String = String::from("data/tinyshakespeare/tiny_shakespeare_val.bin");
    let mut train_loader: DataLoader = DataLoader::new(&tiny_shakespeare_train, 0, 1);
    let mut val_loader: DataLoader = DataLoader::new(&tiny_shakespeare_val, 0, 1);
    println!("train dataset num_batches: {}", train_loader.num_tokens / (B*T));
    println!("val dataset num_batches: {}\n", val_loader.num_tokens / (B*T));
    let val_num_batches: usize = 5;

    // build the Tokenizer
    let tokenizer_path: String = String::from("data/tokenizer/gpt2_tokenizer.bin");
    let tokenizer: Tokenizer = Tokenizer::new(&tokenizer_path);
    println!("{}", tokenizer);

    // some memory for generating samples from the model
    let mut gen_tokens: Vec<u32> = vec![0u32; B*T*16];

    // train
    for step in 0..41 {
        // once in a while estimate the validation loss
        if step % 10 == 0 {
            let mut val_loss: f32 = 0.0;
            val_loader.reset();
            for val_batch in 0..val_num_batches {
                let (inputs, targets) = val_loader.next_batch();
                println!("Forward batch {} is running", val_batch);
                model.forward(true, &inputs, &targets);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches as f32;
            println!("val loss {}\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0) && (step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for i in 0..B*T {
                gen_tokens[i] = tokenizer.eot_token as u32;
            }
            // now sample from the model autoregressively
            println!("generating:\n---\n");
            for t in 0..GENT {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                model.forward(false, &gen_tokens, &vec![0u32; BUFFER_SIZE]);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]

                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                let next_token: usize = model.sample_mult(t);
                gen_tokens[t] = next_token as u32;
                // print the generated token, either using the Tokenizer or a fallback
                if tokenizer.init_ok {
                    // Decode the token using the tokenizer
                    if let Some(decoded_str) = tokenizer.decode(next_token) {
                        println!("{}", decoded_str);
                    } else {
                        // Handle the case where no token could be decoded
                        println!("Error decoding token or token not found");
                    }
                } else {
                    // fall back to printing the token id
                    println!("{}", next_token);
                }
            }
            println!("\n---\n");
        }

        // do a training step
        let start: Instant = Instant::now();
        let (train_inputs, train_targets) = train_loader.next_batch();
        model.forward(true, &train_inputs, &train_targets);
        model.zero_grad();
        model.backward(&train_inputs, &train_targets);
        let step: usize = 0;
        model.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step+1);
        let end: Instant = Instant::now();
        let duration: Duration = end.duration_since(start);
        let time_elapsed_s: f64 = duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9;
        println!("step {}: train loss {} (took {} ms)\n", step, model.mean_loss, time_elapsed_s * 1000.0);
    }
}
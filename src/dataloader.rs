use std::fs::File;
use std::io::{BufReader, Read, Cursor};
use std::process::exit;

use glob::glob;

use crate::utils;

pub const B: usize = 4;
pub const T: usize = 64;
pub const VOCAB_SIZE: usize = 50257;
pub const GENT: u32 = 64; // number of steps of inference we will do

pub const HEADER_SIZE: usize = 256;
pub const BUFFER_SIZE: usize = B*T+1;
pub const U8_BUFFER_SIZE: usize = 2 * BUFFER_SIZE;

pub struct DataLoader {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    process_rank: usize,
    n_proc: usize,
    // input handling and its state
    gl_pathc: usize, // count of pattern-matched paths
    gl_pathv: Vec<String>, // vector list of matched paths of length gl_pathc
    // gl_offs: u32, // number of empty pointers in gl_pathv for convenience
    current_shard: usize, // the current shard we are reading from
    current_position: u64, // the current position in the file we are reading from
    // file_size: u64,
    // current_position: u64,
    buffer: [u16; BUFFER_SIZE], // we fread data from file into this buffer
    // public variables that could be accessed from outside
    pub num_tokens: usize, // total number of tokens
    pub inputs: Vec<u32>,  // input tokens into transformer
    pub targets: Vec<u32>, // target tokens for the transformer
}

impl DataLoader {
    pub fn new(
        filename_pattern: &str,
        process_rank: usize,
        n_proc: usize
    ) -> DataLoader {
        // Pattern to match, similar to "*.txt" in glob in C
        let mut gl_pathc: usize = 0;
        let mut gl_pathv: Vec<String> = Vec::new();
        for path in glob(filename_pattern).expect("Failed to read glob pattern") {
            match path {
                Ok(path) => {
                    println!("{:?}", path.display());
                    gl_pathc += 1;
                    gl_pathv.push(path.display().to_string());
                },
                Err(e) => println!("{:?}", e),
            }
        }

        // inspect and validate all shards so we don't get any runtime errors later
        // if too slow / too many shards, may wish to revisit later
        let mut ntok_total: usize = 0;
        for shard_index in 0..gl_pathc {
            let mut tokens_file: File = utils::fopen_check(&gl_pathv[shard_index]);
            
            // validate the header
            let shard_header: [u32; 256] = utils::read_header(&mut tokens_file);
            if shard_header[0] != 20240520 {
                println!("Bad magic in the data file, check file\n");
                println!("{}", shard_header[0]);
                exit(1);
            }
            if shard_header[1] != 1 {
                println!("Bad version in data file\n");
                exit(1);
            }
            let shard_ntok: u64 = u64::from(shard_header[2]);
            assert!(shard_ntok > 0);

            let file_size: u64 = tokens_file.metadata().unwrap().len();
            let expected_size: u64 = HEADER_SIZE as u64 * 4 + shard_ntok * 2;
            if file_size != expected_size {
                println!("Error: file size is not as expected\n");
                exit(1);
            }
            assert!(shard_ntok >= (n_proc * B * T + 1) as u64);

            ntok_total += shard_ntok as usize;
        }

        println!("DataLoader: filename_pattern: {}", filename_pattern);
        println!(
            "DataLoader: Found {} tokens across {} shards\n",
            ntok_total, 
            gl_pathc
        );
        
        return DataLoader {
            process_rank: process_rank,
            n_proc: n_proc,
            gl_pathc: gl_pathc,
            gl_pathv: gl_pathv,
            current_shard: 0,
            current_position: 0,
            buffer: [0u16; BUFFER_SIZE], // we will read data from the current file into this buffer
            // public variables that could be accessed from outside
            num_tokens: ntok_total, // total number of tokens
            inputs: vec![0; B*T],  // input tokens into transformer
            targets: vec![0; B*T], // target tokens for the transformer
        } 
    }

    pub fn advance(&mut self) {
        // advance the loader by loading the next data shard and resetting the position
        if self.gl_pathc > 1 {
            // if we have more than one shard, advance to the next one
            self.current_shard = (self.current_shard + 1) % self.gl_pathc;
            self.current_position = 0;
        }
        self.current_position = (HEADER_SIZE*16 + self.process_rank*B*T*16) as u64;

    }

    pub fn reset(&mut self) {
        // fully resets the DataLoader object to init configuration
        // each process starts at a different offset in the file
        let header_bytes: usize = HEADER_SIZE * 16;
        let token_bytes_offset: usize = self.process_rank * B * T * 16;
        self.current_shard = 0;
        self.current_position = (header_bytes + token_bytes_offset) as u64;
    }

    // Load the next set of inputs/targets
    pub fn next_batch(&mut self) -> (Vec<u32>, Vec<u32>) {
        let mut inputs: Vec<u32> = vec![0u32; B*T];
        let mut targets: Vec<u32> = vec![0u32; B*T];

        // read B*T+1 u16 tokens from the file into buffer
        let file = utils::fopen_check(&self.gl_pathv[self.current_shard]);
        let mut reader = BufReader::new(&file);
        let mut temp_buf = [0u8; 2 * BUFFER_SIZE];
        reader.read_exact(&mut temp_buf).unwrap();

        // Convert byte intelligences to u32 array of size 256
        let mut cursor: Cursor<[u8; 2 * BUFFER_SIZE]> = Cursor::new(temp_buf);
        for value in self.buffer.iter_mut() {
            let mut bytes: [u8; 2] = [0u8; 2];
            cursor.read_exact(&mut bytes).unwrap();
            *value = u16::from_ne_bytes(bytes);
        }
        
        // decode the buffer into inputs and targets (cast to int)
        println!("{}", self.inputs.len());
        for i in 0..B*T {
            inputs[i] = self.buffer[i] as u32;
            targets[i] = self.buffer[i + 1] as u32;
        }

        // advance the current position by B*T*n_proc integers
        // note: the "stride" of tokens by which we move each time is definitely B * T
        // we only load B * T + 1 tokens at each iteration because the targets are offset by 1
        self.current_position += ((self.n_proc*B*T)*16) as u64;
        // if the next batch would go past the end of the file, advance the loader
        let file_size = file.metadata().unwrap().len();
        if self.current_position + ((self.n_proc*B*T+1)*16) as u64 > file_size {
            self.advance();
        }

        return (inputs, targets);
    }

}
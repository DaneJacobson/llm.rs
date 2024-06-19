/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/
use std::fmt;
use std::io::Read;
use std::process::exit;

use crate::constants::VOCAB_SIZE;
use crate::utils;

// -----------------------------------------------------------------------------

pub struct Tokenizer {
    pub vocab_size: usize,
    pub token_table: Vec<Vec<u8>>,
    pub init_ok: bool,
    pub eot_token: usize, // <|endoftext|> token id
}

impl Tokenizer {
    pub fn new(filename: &str) -> Tokenizer {
        let mut file = utils::fopen_check(&filename);

        // Read in and validate header
        let header: [u32; 256] = utils::read_header(&mut file);
        let date: u32 = header[0];
        let version: u32 = header[1];
        let vocab_size: u32 = header[2];
        let mut eot_token: u32 = header[3];

        assert!(date == 20240328);
        if version == 1 {
            // version 1 didn't include the EOT token id
            // so we assume it is 50256, the EOT in GPT-2
            assert!(vocab_size == VOCAB_SIZE as u32);
            assert!(eot_token == 50256);
        } else if version == 2 {
            eot_token = vocab_size;
        } else {
            println!("Tokenizer model file {} has bad version: {}\n", filename, version);
            exit(1);
        }

        // read in all the tokens
        let mut token_table: Vec<Vec<u8>> = Vec::new();
        for _ in 0..vocab_size {
            let mut length_buffer = [0u8; 1];
            file.read_exact(&mut length_buffer).unwrap();
            let length: u8 = length_buffer[0];
            assert!(length > 0);
            let mut token_bytes = vec![0u8; length as usize];
            file.read_exact(&mut token_bytes).unwrap();
            token_bytes.push(b'\0');
            token_table.push(token_bytes);
        }

        return Tokenizer {
            vocab_size: VOCAB_SIZE,
            token_table: token_table,
            init_ok: true,
            eot_token: eot_token as usize,
        };
    }

    pub fn decode(&self, token_id: usize) -> Option<String> {
        if self.init_ok == false {
            return None;
        }
        if token_id < self.vocab_size {
            // Convert Vec<u8> to String
            let result = String::from_utf8(self.token_table[token_id].clone());
            match result {
                Ok(string) => {
                    return Some(string);
                },
                Err(e) => {
                    println!("Failed to convert: {:?}", e);
                    exit(1);
                }
            }
        } else {
            println!("invalid token id {}!", token_id);
            return None;
        }
    }

}

impl fmt::Display for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, 
            "Tokenizer: {} {} {}", 
            self.vocab_size,
            self.init_ok,
            self.eot_token,
        )
    }
}
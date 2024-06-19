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

use crate::utils;

pub const B: usize = 4;
pub const T: usize = 64;
pub const VOCAB_SIZE: usize = 50257;
pub const GENT: u32 = 64; // number of steps of inference we will do

pub const HEADER_SIZE: usize = 256;
pub const BUFFER_SIZE: usize = B*T+1;
pub const U8_BUFFER_SIZE: usize = 2 * BUFFER_SIZE;


// ----------------------------------------------------------------------------

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


// void safe_printf(const char *piece) {
//     // the tokens are raw bytes, and we we only want to print the printable ones
//     // many bytes can be various control codes, backspace, etc.
//     if (piece == NULL) { return; }
//     if (piece[0] == '\0') { return; }
//     // handle individual byte tokens
//     // every token is asserted to be at least one byte so doing piece[1] is ok
//     if (piece[1] == '\0') {
//         unsigned char byte_val = piece[0];
//         if (!(isprint(byte_val) || isspace(byte_val))) {
//             return; // weird byte, don't print it
//         }
//     }
//     printf("%s", piece);
// }



// const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
//     if (tokenizer->init_ok == 0) {
//         return NULL;
//     }
//     if (token_id < tokenizer->vocab_size) {
//         return tokenizer->token_table[token_id];
//     } else {
//         printf("invalid token id %d!\n", token_id);
//         return NULL;
//     }
// }

// void tokenizer_free(Tokenizer *tokenizer) {
//     if (tokenizer->init_ok) {
//         for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
//             free(tokenizer->token_table[i]);
//         }
//         free(tokenizer->token_table);
//     }
// }

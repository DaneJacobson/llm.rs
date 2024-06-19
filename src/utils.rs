/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of helper functions xxxCheck that call the 
 corresponding Rust standard library function and check its return code. If an 
 error was reported, the program prints some debug information and exits.
*/

use std::fs::File;
use std::io::{BufReader, Error, Read};
use std::process::exit;
use std::io::Cursor;

// ----------------------------------------------------------------------------
// open convenience utils, with nice handling of error checking using match
// simple replace open, read, close, seek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

pub fn fopen_check(path: &str) -> File {
    let model_file: Result<File, Error> = File::open(path);
    match model_file {
        Ok(model_file) => {
            println!("{} is loaded", path);
            return model_file;
        },
        Err(error) => {
            let file: &str = file!();
            let line: u32 = line!();
            println!("Error: {} Failed to open file '{}' at {}:{}\n", error, path, file, line);
            println!("Error details: {}\n", error);
            println!("{}  File: {}\n", error, file);
            println!("{}  Line: {}\n", error, line);
            println!("{}  Path: {}\n", error, path);
            println!("{}---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n", error);
            println!("{}---> HINT 2: possibly try to re-run `python train_gpt2.py`\n", error);
            exit(1);
        }
    }
}

// Read a specified number of data types from a stream
pub fn read_header(stream: &mut File) -> [u32; 256] {
    let mut buffer: [u8; 1024] = [0; 256 * 4]; // 4 bytes per u32
    stream.read_exact(&mut buffer).unwrap();

    // Convert byte intelligences to u32 array of size 256
    let mut array: [u32; 256] = [0u32; 256];
    let mut cursor: Cursor<[u8; 1024]> = Cursor::new(buffer);
    
    for value in array.iter_mut() {
        let mut bytes: [u8; 4] = [0u8; 4];
        cursor.read_exact(&mut bytes).unwrap();
        *value = u32::from_ne_bytes(bytes);
    }

    return array;
}

pub fn read_floats_into_vecf32(stream: &mut File, nmemb: usize, name: &String) -> Vec<f32> {
    println!("Loading: {} {}", name, nmemb);

    let nmemb_usize: usize = nmemb as usize;
    let mut reader: BufReader<&mut File> = BufReader::new(stream);
    let mut buffer: Vec<u8> = vec![0u8; 4 * nmemb_usize];  // Each f32 is 4 bytes
    let result: Result<(), Error> = reader.read_exact(&mut buffer);
    match result {
        Ok(result) => {result},
        Err(error) => {println!("{}", error)}
    }

    let f32s: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|chunk: &[u8]| {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(bytes)  // Assumes little endian byte order
        })
        .collect();

    return f32s;
}
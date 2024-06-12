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
            // TODO: Note to self, I removed mode error statement bc I didn't think it would be necessary
            println!("{}---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n", error);
            println!("{}---> HINT 2: possibly try to re-run `python train_gpt2.py`\n", error);
            exit(1);
        }
    }
}

// pub fn fseek_check(file: &File, ) -> {
//     if (fseek(fp, off, whence) != 0) {
//         fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, line);
//         fprintf(stderr, "Error details:\n");
//         fprintf(stderr, "  Offset: %ld\n", off);
//         fprintf(stderr, "  Whence: %d\n", whence);
//         fprintf(stderr, "  File:   %s\n", file);
//         fprintf(stderr, "  Line:   %d\n", line);
//         exit(EXIT_FAILURE);
//     }
// }

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


pub fn read_bytes(stream: &mut File, nmemb: usize) -> Vec<u8> {
    println!("Loading: {} {}", "token", nmemb);
    let mut reader = BufReader::new(stream);
    let mut buffer = vec![0u8; nmemb];
    reader.read_exact(&mut buffer).unwrap();
    return buffer;
}

// pub fn fread_check<T>(stream: File, nmemb: u32) -> Vec<T> {
    // let mut reader: BufReader<File> = BufReader::new(stream);
    // let mut buffer: [u8; 4 * 256] = [0; 4 * 256]; // 4 bytes per i32 TODO: need to make this more general, using nmemb and size instead of hack
    // reader.read_exact(&mut buffer);
    // let ratio: usize = size_of::<T>() / size_of::<u8>();
    // let buffer_size: usize = ratio * nmemb;
    // let mut buffer: Vec<T> = Vec::new();
    // reader.read_exact(&mut buffer);

    // Convert bytes to vector
    // let mut vec: Vec<T> = Vec::new();
    // let mut cursor: Cursor<[u8; 4 * 256]> = Cursor::new();
    
    // for value in array.iter_mut() {
    //     let mut bytes: [u8; size_of::<T>()] = [0u8; 4];
    //     cursor.read_exact(&mut bytes);
    //     *value = i32::from_ne_bytes(bytes);
    // }

    // return array;

    // let mut buffer: Vec<u8> = vec![0u8; nmemb * size_of::<T>()];
    // stream.read_exact(&mut buffer);

    // let mut result: Vec<T> = Vec::with_capacity(nmemb);
    // let t_slice: &[T] = unsafe {
    //     // SAFETY: The following unsafe block assumes that buffer contains valid T data.
    //     std::slice::from_raw_parts(buffer.as_ptr() as *const T, nmemb)
    // };

    // result.extend_from_slice(t_slice);
    // return result;
// }

// void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
//     size_t result = fread(ptr, size, nmemb, stream);
//     if (result != nmemb) {
//         if (feof(stream)) {
//             fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
//         } else if (ferror(stream)) {
//             fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
//         } else {
//             fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
//                     file, line, nmemb, result);
//         }
//         fprintf(stderr, "Error details:\n");
//         fprintf(stderr, "  File: %s\n", file);
//         fprintf(stderr, "  Line: %d\n", line);
//         fprintf(stderr, "  Expected elements: %zu\n", nmemb);
//         fprintf(stderr, "  Read elements: %zu\n", result);
//         exit(EXIT_FAILURE);
//     }
// }

// #define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// void fclose_check(FILE *fp, const char *file, int line) {
//     if (fclose(fp) != 0) {
//         fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
//         fprintf(stderr, "Error details:\n");
//         fprintf(stderr, "  File: %s\n", file);
//         fprintf(stderr, "  Line: %d\n", line);
//         exit(EXIT_FAILURE);
//     }
// }

// #define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

// void fseek_check(FILE *fp, long off, int whence, const char *file, int line) {
//     if (fseek(fp, off, whence) != 0) {
//         fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, line);
//         fprintf(stderr, "Error details:\n");
//         fprintf(stderr, "  Offset: %ld\n", off);
//         fprintf(stderr, "  Whence: %d\n", whence);
//         fprintf(stderr, "  File:   %s\n", file);
//         fprintf(stderr, "  Line:   %d\n", line);
//         exit(EXIT_FAILURE);
//     }
// }

// #define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

// // ----------------------------------------------------------------------------
// // malloc error-handling wrapper util

// void *malloc_check(size_t size, const char *file, int line) {
//     void *ptr = malloc(size);
//     if (ptr == NULL) {
//         fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
//         fprintf(stderr, "Error details:\n");
//         fprintf(stderr, "  File: %s\n", file);
//         fprintf(stderr, "  Line: %d\n", line);
//         fprintf(stderr, "  Size: %zu bytes\n", size);
//         exit(EXIT_FAILURE);
//     }
//     return ptr;
// }

// #define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// #endif
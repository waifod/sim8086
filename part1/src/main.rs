use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
mod decoder;
mod executor;
mod instruction;

use decoder::Decoder;
use executor::Executor;
use instruction::Instruction;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <input_binary_file_path>", args[0]);
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let stem = input_path
        .file_stem()
        .expect("Input filepath has no stem")
        .to_str()
        .expect("Stem is not valid UTF-8");
    let output_file = format!("{}_decoded.asm", stem);
    let image_file = format!("{}_memory.data", stem);

    println!("Starting 8086 simulator...");

    let original_bytes = read_file(input_path).expect("Failed to read input file");

    let decoded_instructions = Decoder::new(&original_bytes).decode_all();

    write_decoding(&decoded_instructions, output_file)
        .expect("Failed to write decoded instructions");

    let memory = Executor::new(&original_bytes).run();

    write_memory(&memory[..16384], image_file).expect("Failed to write memory");

    println!("Simulation complete!");
}

fn read_file<P: AsRef<Path>>(filepath: P) -> io::Result<Vec<u8>> {
    let mut file = File::open(filepath)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn write_decoding<P: AsRef<Path>>(
    instructions: &HashMap<usize, Instruction>,
    output_filepath: P,
) -> io::Result<()> {
    let mut output_file = File::create(output_filepath)?;
    let mut keys: Vec<&usize> = instructions.keys().collect();
    keys.sort();

    for key in keys {
        let instruction = instructions.get(key).unwrap();
        writeln!(output_file, "{}", instruction)?;
    }
    Ok(())
}

fn write_memory<P: AsRef<Path>>(bytes: &[u8], output_filepath: P) -> io::Result<()> {
    let mut output_file = File::create(output_filepath)?;
    output_file
        .write_all(&bytes[..16384])
        .expect("Failed to write image data");
    Ok(())
}

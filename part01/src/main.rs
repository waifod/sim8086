use core::panic;
use std::env; // Import the env module for command-line arguments
use std::fmt;
use std::fmt::write;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path; // Import the fmt module for Display trait

/// Represents the 16-bit general-purpose registers of the 8086.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Register16 {
    AX, // Accumulator Register
    CX, // Count Register
    DX, // Data Register
    BX, // Base Register
    SP, // Stack Pointer
    BP, // Base Pointer
    SI, // Source Index
    DI, // Destination Index,
}

impl Register16 {
    /// Converts a 3-bit register encoding to a `Register16` enum variant.
    /// Panics if the encoding is not a valid 16-bit register.
    fn from_encoding(encoding: u8) -> Self {
        match encoding {
            0b000 => Register16::AX,
            0b001 => Register16::CX,
            0b010 => Register16::DX,
            0b011 => Register16::BX,
            0b100 => Register16::SP,
            0b101 => Register16::BP,
            0b110 => Register16::SI,
            0b111 => Register16::DI,
            _ => panic!("Invalid 3-bit encoding for 16-bit register: {}", encoding),
        }
    }
}

// Implement fmt::Display for Register16
impl fmt::Display for Register16 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use the Debug implementation for a simple string representation
        write!(f, "{:?}", self)
    }
}

/// Represents the 8-bit general-purpose registers of the 8086.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Register8 {
    AL, // Accumulator Low
    CL, // Count Low
    DL, // Data Low
    BL, // Base Low
    AH, // Accumulator High
    CH, // Count High
    DH, // Data High
    BH, // Base High
}

impl Register8 {
    /// Converts a 3-bit register encoding to a `Register8` enum variant.
    ///
    /// The 8086 uses the same 3-bit encoding for both low (AL, CL, etc.) and high (AH, CH, etc.)
    /// byte registers. The `W` bit in the instruction's opcode determines which set is used.
    /// This function maps the 3-bit encoding directly to the correct 8-bit register.
    ///
    /// Panics if the encoding is not a valid 3-bit register encoding.
    fn from_encoding(encoding: u8) -> Self {
        match encoding {
            0b000 => Register8::AL,
            0b001 => Register8::CL,
            0b010 => Register8::DL,
            0b011 => Register8::BL,
            0b100 => Register8::AH,
            0b101 => Register8::CH,
            0b110 => Register8::DH,
            0b111 => Register8::BH,
            _ => panic!("Invalid 3-bit encoding for 8-bit register: {}", encoding),
        }
    }
}

// Implement fmt::Display for Register8
impl fmt::Display for Register8 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use the Debug implementation for a simple string representation
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MemoryAddress {
    R(Register16),
    RO8(Register16, u8),
    RO16(Register16, u16),
    RR(Register16, Register16),
    RRO8(Register16, Register16, u8),
    RRO16(Register16, Register16, u16),
    DA(u16),
}

impl MemoryAddress {
    fn from_encoding(md: u8, rm: u8, bytes: &[u8]) -> Self {
        match md {
            0b00 => match rm {
                0b000 => MemoryAddress::RR(Register16::BX, Register16::SI),
                0b001 => MemoryAddress::RR(Register16::BX, Register16::DI),
                0b010 => MemoryAddress::RR(Register16::BP, Register16::SI),
                0b011 => MemoryAddress::RR(Register16::BP, Register16::DI),
                0b100 => MemoryAddress::R(Register16::SI),
                0b101 => MemoryAddress::R(Register16::DI),
                0b110 => {
                    if bytes.len() < 2 {
                        panic!("Not enough bytes")
                    } else {
                        MemoryAddress::DA((bytes[0] as u16) + 256 * (bytes[1] as u16))
                    }
                }
                0b111 => MemoryAddress::R(Register16::BX),
                _ => panic!("Unsupported byte sequence"),
            },
            0b01 => {
                if bytes.len() == 0 {
                    panic!("not enough bytes")
                } else {
                    let displacement = bytes[0];
                    match rm {
                        0b000 => MemoryAddress::RRO8(Register16::BX, Register16::SI, displacement),
                        0b001 => MemoryAddress::RRO8(Register16::BX, Register16::DI, displacement),
                        0b010 => MemoryAddress::RRO8(Register16::BP, Register16::SI, displacement),
                        0b011 => MemoryAddress::RRO8(Register16::BP, Register16::DI, displacement),
                        0b100 => MemoryAddress::RO8(Register16::SI, displacement),
                        0b101 => MemoryAddress::RO8(Register16::DI, displacement),
                        0b110 => MemoryAddress::RO8(Register16::BP, displacement),
                        0b111 => MemoryAddress::RO8(Register16::BX, displacement),
                        _ => panic!("Unsupported byte sequence"),
                    }
                }
            }
            0b10 => {
                if bytes.len() < 2 {
                    panic!("not enough bytes")
                } else {
                    let displacement = (bytes[0] as u16) + 256 * (bytes[1] as u16);
                    match rm {
                        0b000 => MemoryAddress::RRO16(Register16::BX, Register16::SI, displacement),
                        0b001 => MemoryAddress::RRO16(Register16::BX, Register16::DI, displacement),
                        0b010 => MemoryAddress::RRO16(Register16::BP, Register16::SI, displacement),
                        0b011 => MemoryAddress::RRO16(Register16::BP, Register16::DI, displacement),
                        0b100 => MemoryAddress::RO16(Register16::SI, displacement),
                        0b101 => MemoryAddress::RO16(Register16::DI, displacement),
                        0b110 => MemoryAddress::RO16(Register16::BP, displacement),
                        0b111 => MemoryAddress::RO16(Register16::BX, displacement),
                        _ => panic!("Unsupported byte sequence"),
                    }
                }
            }
            _ => panic!("Unsupported byte sequence"),
        }
    }
}

impl fmt::Display for MemoryAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use the Debug implementation for a simple string representation
        match self {
            MemoryAddress::R(reg) => write!(f, "[{:?}]", reg),
            MemoryAddress::RO8(reg, offset) => write!(f, "[{:?} + {:?}]", reg, offset),
            MemoryAddress::RO16(reg, offset) => write!(f, "[{:?} + {:?}]", reg, offset),
            MemoryAddress::RR(reg1, reg2) => write!(f, "[{:?} + {:?}]", reg1, reg2),
            MemoryAddress::RRO8(reg1, reg2, offset) => {
                write!(f, "[{:?} + {:?} + {:?}]", reg1, reg2, *offset as i8)
            }
            MemoryAddress::RRO16(reg1, reg2, offset) => {
                write!(f, "[{:?} + {:?} + {:?}]", reg1, reg2, *offset as i16)
            }
            MemoryAddress::DA(addr) => write!(f, "[{}]", addr),
        }
    }
}

/// Represents an operand, which can be either an 8-bit or a 16-bit register.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Operand {
    R8(Register8),
    R16(Register16),
    MA(MemoryAddress),
    I8(u8),
    I16(u16),
}

// Implement fmt::Display for Operand
impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Operand::R8(reg) => write!(f, "{}", reg),
            Operand::R16(reg) => write!(f, "{}", reg),
            Operand::MA(addr) => write!(f, "{}", addr),
            Operand::I8(val) => write!(f, "byte {}", *val as i8),
            Operand::I16(val) => write!(f, "word {}", *val as i16),
        }
    }
}

/// Represents a decoded 8086 instruction.
#[derive(Debug, PartialEq, Eq)] // Added PartialEq and Eq for easier testing of Instruction Vec
pub enum Instruction {
    /// A MOV instruction moving data from a source register to a destination register.
    Mov {
        destination: Operand,
        source: Operand,
    },
    // Future instructions can be added here.
}

// Implement fmt::Display for Instruction
impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Instruction::Mov {
                destination,
                source,
            } => write!(f, "MOV {}, {}", destination, source),
        }
    }
}

fn decode_mov_operands(
    d_bit: u8,
    w_bit: u8,
    md: u8,
    reg: u8,
    rm: u8,
    bytes: &[u8],
) -> io::Result<(Operand, Operand, usize)> {
    let op1 = if w_bit == 0 {
        Operand::R8(Register8::from_encoding(reg))
    } else {
        Operand::R16(Register16::from_encoding(reg))
    };
    let (op2, offset) = match md {
        0b00 => (
            Operand::MA(MemoryAddress::from_encoding(md, rm, bytes)),
            if rm == 0b110 { 2 } else { 0 },
        ),
        0b01 => (Operand::MA(MemoryAddress::from_encoding(md, rm, bytes)), 1),
        0b10 => (Operand::MA(MemoryAddress::from_encoding(md, rm, bytes)), 2),
        0b11 => (
            if w_bit == 0 {
                Operand::R8(Register8::from_encoding(rm))
            } else {
                Operand::R16(Register16::from_encoding(rm))
            },
            0,
        ),
        _ => panic!("Unsupported byte sequence"),
    };
    if d_bit == 1 {
        Ok((op1, op2, offset))
    } else {
        Ok((op2, op1, offset))
    }
}

/// Decodes a register-to-register MOV instruction from a byte slice.
///
/// This function expects to receive the opcode byte (which must have the 100010 prefix)
/// and the ModR/M byte. It uses the `d_bit` and `w_bit` extracted from the opcode
/// to correctly determine the direction and width of the operation.
///
/// # Arguments
/// * `bytes` - A slice of bytes containing the instruction.
/// * `d_bit` - The direction bit (bit 1 of the opcode). 1 means REG is destination, RM is source.
///             0 means RM is destination, REG is source.
/// * `w_bit` - The word bit (bit 0 of the opcode). 1 means 16-bit operation, 0 means 8-bit operation.
///
/// # Returns
/// * `Ok((Instruction, usize))` if decoding is successful, where `usize` is the instruction length.
/// * `Err(io::Error)` if there are not enough bytes for the minimal instruction or
///   an unsupported MOD field is found (which will panic as per requirement).
pub fn decode_mov_instruction(
    d_bit: u8,
    w_bit: u8,
    bytes: &[u8],
) -> io::Result<(Instruction, usize)> {
    // MOV RegToReg instructions are always 2 bytes.
    const MIN_BYTES: usize = 1;

    if bytes.len() < MIN_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Not enough bytes to decode a MOV instruction. Expected at least {}, got {}.",
                MIN_BYTES,
                bytes.len()
            ),
        ));
    }

    let mod_rm_byte = bytes[0];

    // Extract MOD, REG, and RM fields from the ModR/M byte
    let md = (mod_rm_byte >> 6) & 0b11; // Bits 7-6
    let reg = (mod_rm_byte >> 3) & 0b111; // Bits 5-3
    let rm = mod_rm_byte & 0b111; // Bits 2-0

    // For register-to-register MOV, MOD must be 0b11 (3)
    let (destination, source, offset) =
        decode_mov_operands(d_bit, w_bit, md, reg, rm, &bytes[1..])?;

    Ok((
        Instruction::Mov {
            destination,
            source,
        },
        MIN_BYTES + offset,
    )) // Return instruction and its length
}

fn decode_mov_immediate_instruction(
    w_bit: u8,
    reg: u8,
    bytes: &[u8],
) -> io::Result<(Instruction, usize)> {
    if (w_bit as usize) + 1 > bytes.len() {
        panic!("Not enough bytes");
    }
    let (op1, op2) = if w_bit == 0 {
        (
            Operand::R8(Register8::from_encoding(reg)),
            Operand::I8(bytes[0]),
        )
    } else {
        (
            Operand::R16(Register16::from_encoding(reg)),
            Operand::I16((bytes[0] as u16) + 256 * (bytes[1] as u16)),
        )
    };
    Ok((
        Instruction::Mov {
            destination: op1,
            source: op2,
        },
        (w_bit + 1) as usize,
    ))
}

/// Decodes a single 8086 instruction from a byte slice and returns the instruction
/// along with the number of bytes it consumed.
///
/// This function currently only supports MOV instructions with register-to-register
/// addressing, which are always 2 bytes long. It extracts the D (direction) and W (width)
/// bits from the opcode to correctly interpret the instruction.
///
/// # Arguments
/// * `bytes` - A slice of bytes containing the instruction.
///
/// # Returns
/// * `Ok((Instruction, usize))` if decoding is successful, where `usize` is the instruction length.
/// * `Err(io::Error)` if there are not enough bytes for the minimal instruction or
///   an unsupported instruction is found (which will panic as per requirement).
pub fn decode_instruction(bytes: &[u8]) -> io::Result<(Instruction, usize)> {
    // For MOV RegToReg instructions, they are always 2 bytes.
    // This check ensures we have enough bytes for this specific instruction type.
    const MIN_INSTRUCTION_LENGTH: usize = 2; // Minimum bytes needed for any instruction we support

    if bytes.len() < MIN_INSTRUCTION_LENGTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Not enough bytes to decode a minimal instruction. Expected at least {}, got {}.",
                MIN_INSTRUCTION_LENGTH,
                bytes.len()
            ),
        ));
    }

    let opcode = bytes[0];

    // Check for MOV instruction (opcode prefix 100010DW)
    // Mask out the D (bit 1) and W (bit 0) bits to check the common prefix.
    let (instruction, offset) = if (opcode & 0b11111100) == 0b10001000 {
        // 0x88 in binary is 10001000
        let d_bit = (opcode >> 1) & 0b1; // Extract D bit (bit 1)
        let w_bit = opcode & 0b1; // Extract W bit (bit 0)
        decode_mov_instruction(d_bit, w_bit, &bytes[1..])?
    } else if (opcode & 0b11110000) == 0b10110000 {
        let w_bit = (opcode >> 3) & 0b1;
        let reg = opcode & 0b111;
        decode_mov_immediate_instruction(w_bit, reg, &bytes[1..])?
    } else {
        // Panic as requested for unsupported scenarios
        // TODO: add support for "Immediate to register/memory"
        panic!(
            "Unsupported opcode: 0x{:02X}. Only MOV reg/mem to/from register (100010DW) and immediate MOV (1011WREG) are currently supported.",
            opcode
        );
    };
    Ok((instruction, offset + 1))
}

/// Decodes a byte slice into a vector of 8086 instructions.
///
/// This function iterates through the byte stream, decoding instructions
/// until the end of the stream or an incomplete instruction is encountered.
/// Unsupported opcodes or MOD fields will cause a panic as per requirement.
///
/// # Arguments
/// * `bytes` - The slice of bytes containing the binary data to decode.
///
/// # Returns
/// * `Vec<Instruction>` containing all successfully decoded instructions.
pub fn decode(bytes: &[u8]) -> Vec<Instruction> {
    let mut instructions = Vec::new();
    let mut current_offset = 0;

    while current_offset < bytes.len() {
        match decode_instruction(&bytes[current_offset..]) {
            Ok((instruction, bytes_consumed)) => {
                current_offset += bytes_consumed;
                println!("Decoded: {:?}, offset: {:?}", instruction, current_offset);
                instructions.push(instruction);
            }
            Err(e) => {
                if e.kind() == io::ErrorKind::InvalidInput {
                    eprintln!("Warning: Incomplete instruction at offset {}: {}. Skipping remaining bytes.", current_offset, e);
                    break; // Stop decoding if we hit an incomplete instruction at the end
                } else {
                    // This branch should ideally not be hit if decode_instruction only returns
                    // InvalidInput for length issues and panics for unsupported opcodes/MOD.
                    panic!("Decoding error at offset {}: {}", current_offset, e);
                }
            }
        }
    }
    instructions
}

/// Reads the entire content of a file into a `Vec<u8>`.
///
/// # Arguments
/// * `filepath` - The path to the input binary file.
///
/// # Returns
/// * `Ok(Vec<u8>)` if reading is successful.
/// * `Err(io::Error)` if an I/O error occurs.
pub fn read_file<P: AsRef<Path>>(filepath: P) -> io::Result<Vec<u8>> {
    let mut file = File::open(filepath)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

/// Writes a vector of decoded 8086 instructions to a specified output file.
///
/// Each instruction is formatted into a human-readable string and written on a new line.
///
/// # Arguments
/// * `instructions` - A slice of `Instruction` enums to write.
/// * `output_filepath` - The path to the output text file.
///
/// # Errors
/// * `Err(io::Error)` if an I/O error occurs during file writing.
pub fn write_decoding<P: AsRef<Path>>(
    instructions: &[Instruction],
    output_filepath: P,
) -> io::Result<()> {
    let mut output_file = File::create(output_filepath)?;
    for instruction in instructions {
        // Use the Display trait implementation for formatting
        writeln!(output_file, "{}", instruction)?;
    }
    Ok(())
}

fn main() {
    // Collect command-line arguments
    let args: Vec<String> = env::args().collect();

    // Expect 3 arguments: program name, input file, output file
    if args.len() != 3 {
        eprintln!("Usage: {} <input_binary_file> <output_text_file>", args[0]);
        // Exit with a non-zero status code to indicate an error
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];

    // --- Create a dummy input.bin for testing if it doesn't exist ---
    // This is helpful for initial testing without manually creating a file.
    // In a real scenario, you might remove this or make it an optional feature.
    if !Path::new(input_file).exists() {
        println!(
            "Input file '{}' not found. Creating a dummy file for demonstration.",
            input_file
        );
        // Example instructions:
        // 0x8B C3: MOV AX, BX (D=1, W=1)
        // 0x89 C3: MOV BX, AX (D=0, W=1)
        // 0x8A C3: MOV AL, BL (D=1, W=0)
        // 0x88 C3: MOV BL, AL (D=0, W=0)
        // 0x8B CA: MOV CX, DX (D=1, W=1)
        // 0x8A F2: MOV DH, CL (D=1, W=0, DH is 100, CL is 001)
        let dummy_bytes = vec![
            0x8B, 0xC3, 0x89, 0xC3, 0x8A, 0xC3, 0x88, 0xC3, 0x8B, 0xCA, 0x8A, 0xF2,
        ];
        File::create(input_file)
            .expect("Could not create dummy input file")
            .write_all(&dummy_bytes)
            .expect("Could not write to dummy input file");
    }
    // --- End of dummy file creation logic ---

    println!("Starting 8086 decoder...");
    match read_file(input_file) {
        Ok(buffer) => {
            let decoded_instructions = decode(&buffer);
            match write_decoding(&decoded_instructions, output_file) {
                Ok(_) => println!("Decoding complete! Output written to '{}'.", output_file),
                Err(e) => eprintln!("Error writing decoded instructions: {}", e),
            }
        }
        Err(e) => eprintln!("Error reading input file: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register16_from_encoding() {
        assert_eq!(Register16::from_encoding(0b000), Register16::AX);
        assert_eq!(Register16::from_encoding(0b001), Register16::CX);
        assert_eq!(Register16::from_encoding(0b010), Register16::DX);
        assert_eq!(Register16::from_encoding(0b011), Register16::BX);
        assert_eq!(Register16::from_encoding(0b100), Register16::SP);
        assert_eq!(Register16::from_encoding(0b101), Register16::BP);
        assert_eq!(Register16::from_encoding(0b110), Register16::SI);
        assert_eq!(Register16::from_encoding(0b111), Register16::DI);
    }

    #[test]
    #[should_panic(expected = "Invalid 3-bit encoding for 16-bit register: 8")]
    fn test_register16_from_encoding_panic() {
        Register16::from_encoding(0b1000);
    }

    #[test]
    fn test_register8_from_encoding() {
        assert_eq!(Register8::from_encoding(0b000), Register8::AL);
        assert_eq!(Register8::from_encoding(0b001), Register8::CL);
        assert_eq!(Register8::from_encoding(0b010), Register8::DL);
        assert_eq!(Register8::from_encoding(0b011), Register8::BL);
        assert_eq!(Register8::from_encoding(0b100), Register8::AH);
        assert_eq!(Register8::from_encoding(0b101), Register8::CH);
        assert_eq!(Register8::from_encoding(0b110), Register8::DH);
        assert_eq!(Register8::from_encoding(0b111), Register8::BH);
    }

    #[test]
    #[should_panic(expected = "Invalid 3-bit encoding for 8-bit register: 8")]
    fn test_register8_from_encoding_panic() {
        Register8::from_encoding(0b1000);
    }

    #[test]
    fn test_decode_mov_instruction_16bit_d1_w1() {
        // MOV AX, BX (0x8B C3) - D=1, W=1
        let bytes = [0x8B, 0xC3];
        let (instruction, length) = decode_mov_instruction(&bytes, 1, 1).unwrap();
        assert_eq!(length, 2);
        assert!(matches!(
            instruction,
            Instruction::Mov {
                destination: Operand::R16(Register16::AX),
                source: Operand::R16(Register16::BX)
            }
        ));
        assert_eq!(format!("{}", instruction), "MOV AX, BX");
    }

    #[test]
    fn test_decode_mov_instruction_16bit_d0_w1() {
        // MOV BX, AX (0x89 C3) - D=0, W=1
        let bytes = [0x89, 0xC3];
        let (instruction, length) = decode_mov_instruction(&bytes, 0, 1).unwrap();
        assert_eq!(length, 2);
        assert!(matches!(
            instruction,
            Instruction::Mov {
                destination: Operand::R16(Register16::BX),
                source: Operand::R16(Register16::AX)
            }
        ));
        assert_eq!(format!("{}", instruction), "MOV BX, AX");
    }

    #[test]
    fn test_decode_mov_instruction_8bit_d1_w0() {
        // MOV AL, BL (0x8A C3) - D=1, W=0
        let bytes = [0x8A, 0xC3];
        let (instruction, length) = decode_mov_instruction(&bytes, 1, 0).unwrap();
        assert_eq!(length, 2);
        assert!(matches!(
            instruction,
            Instruction::Mov {
                destination: Operand::R8(Register8::AL),
                source: Operand::R8(Register8::BL)
            }
        ));
        assert_eq!(format!("{}", instruction), "MOV AL, BL");
    }

    #[test]
    fn test_decode_mov_instruction_8bit_d0_w0() {
        // MOV BL, AL (0x88 C3) - D=0, W=0
        let bytes = [0x88, 0xC3];
        let (instruction, length) = decode_mov_instruction(&bytes, 0, 0).unwrap();
        assert_eq!(length, 2);
        assert!(matches!(
            instruction,
            Instruction::Mov {
                destination: Operand::R8(Register8::BL),
                source: Operand::R8(Register8::AL)
            }
        ));
        assert_eq!(format!("{}", instruction), "MOV BL, AL");
    }

    #[test]
    fn test_decode_mov_instruction_8bit_d1_w0_high_low() {
        // MOV DH, CL (0x8A F2) - D=1, W=0
        // REG = 110 (DH), RM = 010 (CL)
        let bytes = [0x8A, 0xF2];
        let (instruction, length) = decode_mov_instruction(&bytes, 1, 0).unwrap();
        assert_eq!(length, 2);
        assert!(matches!(
            instruction,
            Instruction::Mov {
                destination: Operand::R8(Register8::DH),
                source: Operand::R8(Register8::CL)
            }
        ));
        assert_eq!(format!("{}", instruction), "MOV DH, CL");
    }

    #[test]
    #[should_panic(
        expected = "Unsupported MOD field for MOV instruction: 00. Only register-to-register (11) is supported."
    )]
    fn test_decode_mov_instruction_unsupported_mod() {
        // MOV with MOD 00 (memory mode)
        let bytes = [0x8B, 0x00]; // 0x8B is MOV, 0x00 means MOD 00, REG 000, RM 000
        let _ = decode_mov_instruction(&bytes, 1, 1).unwrap(); // d_bit and w_bit don't matter for this panic
    }

    #[test]
    fn test_decode_instruction_16bit_mov() {
        // MOV AX, BX (0x8B C3)
        let bytes = [0x8B, 0xC3];
        let (instruction, length) = decode_instruction(&bytes).unwrap();
        assert_eq!(length, 2);
        assert!(matches!(
            instruction,
            Instruction::Mov {
                destination: Operand::R16(Register16::AX),
                source: Operand::R16(Register16::BX)
            }
        ));
        assert_eq!(format!("{}", instruction), "MOV AX, BX");
    }

    #[test]
    fn test_decode_instruction_8bit_mov() {
        // MOV AL, BL (0x8A C3)
        let bytes = [0x8A, 0xC3];
        let (instruction, length) = decode_instruction(&bytes).unwrap();
        assert_eq!(length, 2);
        assert!(matches!(
            instruction,
            Instruction::Mov {
                destination: Operand::R8(Register8::AL),
                source: Operand::R8(Register8::BL)
            }
        ));
        assert_eq!(format!("{}", instruction), "MOV AL, BL");
    }

    #[test]
    #[should_panic(
        expected = "Unsupported opcode: 0x00. Only MOV reg/mem to/from register (100010DW) is currently supported."
    )]
    fn test_decode_instruction_unsupported_opcode() {
        let bytes = [0x00, 0x00]; // Not a MOV instruction
        let _ = decode_instruction(&bytes).unwrap();
    }

    #[test]
    fn test_decode_instruction_not_enough_bytes() {
        let bytes = [0x8B]; // Only one byte
        let result = decode_instruction(&bytes);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_read_file() -> io::Result<()> {
        let test_file = "test_read.bin";
        let dummy_bytes = vec![0x01, 0x02, 0x03];
        File::create(test_file)?.write_all(&dummy_bytes)?;
        let read_bytes = read_file(test_file)?;
        assert_eq!(read_bytes, dummy_bytes);
        std::fs::remove_file(test_file)?;
        Ok(())
    }

    #[test]
    fn test_decode_function_multiple_instructions() {
        // MOV AX, BX (0x8B C3)
        // MOV CX, DX (0x8B CA)
        // MOV AL, BL (0x8A C3)
        let bytes = vec![0x8B, 0xC3, 0x8B, 0xCA, 0x8A, 0xC3];
        let instructions = decode(&bytes);
        assert_eq!(instructions.len(), 3);
        assert_eq!(format!("{}", instructions[0]), "MOV AX, BX");
        assert_eq!(format!("{}", instructions[1]), "MOV CX, DX");
        assert_eq!(format!("{}", instructions[2]), "MOV AL, BL");
    }

    #[test]
    fn test_decode_function_with_incomplete_last_instruction() {
        let bytes = vec![0x8B, 0xC3, 0x8B]; // MOV AX, BX; then an incomplete instruction
        let instructions = decode(&bytes);
        assert_eq!(instructions.len(), 1);
        assert_eq!(format!("{}", instructions[0]), "MOV AX, BX");
    }

    #[test]
    fn test_write_decoding() -> io::Result<()> {
        let output_file = "test_write.txt";
        let instructions = vec![
            Instruction::Mov {
                destination: Operand::R16(Register16::AX),
                source: Operand::R16(Register16::BX),
            },
            Instruction::Mov {
                destination: Operand::R8(Register8::CL),
                source: Operand::R8(Register8::DH),
            },
        ];
        write_decoding(&instructions, output_file)?;
        let mut content = String::new();
        File::open(output_file)?.read_to_string(&mut content)?;
        assert_eq!(content.trim(), "MOV AX, BX\nMOV CL, DH");
        std::fs::remove_file(output_file)?;
        Ok(())
    }

    #[test]
    fn test_full_flow_integration() -> io::Result<()> {
        let input_file = "test_full_input.bin";
        let output_file = "test_full_output.txt";

        // MOV AX, BX (0x8B C3)
        // MOV SI, DI (0x8B FE)
        // MOV AH, BL (0x8A E3) - D=1, W=0, REG=100 (AH), RM=011 (BL)
        let dummy_bytes = vec![0x8B, 0xC3, 0x8B, 0xFE, 0x8A, 0xE3];
        File::create(input_file)?.write_all(&dummy_bytes)?;

        // Read
        let buffer = read_file(input_file)?;
        // Decode
        let decoded_instructions = decode(&buffer);
        // Write
        write_decoding(&decoded_instructions, output_file)?;

        let mut output_content = String::new();
        File::open(output_file)?.read_to_string(&mut output_content)?;

        assert_eq!(output_content.trim(), "MOV AX, BX\nMOV SI, DI\nMOV AH, BL");

        // Clean up
        std::fs::remove_file(input_file)?;
        std::fs::remove_file(output_file)?;

        Ok(())
    }
}

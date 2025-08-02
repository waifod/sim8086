use core::panic;
use std::env; // Import the env module for command-line arguments
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path; // Import the fmt module for Display trait

// --- Main public API ---

/// Decodes a byte slice into a vector of 8086 instructions.
///
/// It will process bytes sequentially until the slice is exhausted.
///
/// # Panics
/// Panics if it encounters an unsupported opcode or an invalid byte sequence
/// for a supported instruction.
pub fn decode(bytes: &[u8]) -> Vec<Instruction> {
    let mut decoder = Decoder::new(bytes);
    let mut instructions = Vec::new();
    while !decoder.is_eof() {
        instructions.push(decoder.decode_next_instruction());
    }
    instructions
}

// --- Instruction and Operand Enums ---

/// Represents the 16-bit general-purpose registers of the 8086.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Register16 {
    AX,
    CX,
    DX,
    BX,
    SP,
    BP,
    SI,
    DI,
}

impl Register16 {
    fn from_encoding(encoding: u8) -> Self {
        match encoding {
            0b000 => Self::AX,
            0b001 => Self::CX,
            0b010 => Self::DX,
            0b011 => Self::BX,
            0b100 => Self::SP,
            0b101 => Self::BP,
            0b110 => Self::SI,
            0b111 => Self::DI,
            _ => panic!("Invalid 3-bit encoding for 16-bit register: {}", encoding),
        }
    }
}

impl fmt::Display for Register16 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use a more direct mapping for lowercase assembly convention
        write!(
            f,
            "{}",
            match self {
                Self::AX => "ax",
                Self::CX => "cx",
                Self::DX => "dx",
                Self::BX => "bx",
                Self::SP => "sp",
                Self::BP => "bp",
                Self::SI => "si",
                Self::DI => "di",
            }
        )
    }
}

/// Represents the 8-bit general-purpose registers of the 8086.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Register8 {
    AL,
    CL,
    DL,
    BL,
    AH,
    CH,
    DH,
    BH,
}

impl Register8 {
    fn from_encoding(encoding: u8) -> Self {
        match encoding {
            0b000 => Self::AL,
            0b001 => Self::CL,
            0b010 => Self::DL,
            0b011 => Self::BL,
            0b100 => Self::AH,
            0b101 => Self::CH,
            0b110 => Self::DH,
            0b111 => Self::BH,
            _ => panic!("Invalid 3-bit encoding for 8-bit register: {}", encoding),
        }
    }
}

impl fmt::Display for Register8 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::AL => "al",
                Self::CL => "cl",
                Self::DL => "dl",
                Self::BL => "bl",
                Self::AH => "ah",
                Self::CH => "ch",
                Self::DH => "dh",
                Self::BH => "bh",
            }
        )
    }
}

/// Represents a memory address calculation.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MemoryAddress {
    R(Register16),
    RD8(Register16, u8),
    RD16(Register16, u16),
    RR(Register16, Register16),
    RRD8(Register16, Register16, u8),
    RRD16(Register16, Register16, u16),
    DA(u16),
}

impl fmt::Display for MemoryAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::R(reg) => write!(f, "[{}]", reg),
            Self::RR(r1, r2) => write!(f, "[{} + {}]", r1, r2),
            Self::DA(addr) => write!(f, "[{}]", addr),
            Self::RD8(reg, offset) => {
                let disp = *offset as i8;
                if disp >= 0 {
                    write!(f, "[{} + {}]", reg, disp)
                } else {
                    write!(f, "[{} - {}]", reg, -disp)
                }
            }
            Self::RD16(reg, offset) => {
                let disp = *offset as i16;
                if disp >= 0 {
                    write!(f, "[{} + {}]", reg, disp)
                } else {
                    write!(f, "[{} - {}]", reg, -disp)
                }
            }
            Self::RRD8(r1, r2, offset) => {
                let disp = *offset as i8;
                if disp >= 0 {
                    write!(f, "[{} + {} + {}]", r1, r2, disp)
                } else {
                    write!(f, "[{} + {} - {}]", r1, r2, -disp)
                }
            }
            Self::RRD16(r1, r2, offset) => {
                let disp = *offset as i16;
                if disp >= 0 {
                    write!(f, "[{} + {} + {}]", r1, r2, disp)
                } else {
                    write!(f, "[{} + {} - {}]", r1, r2, -disp)
                }
            }
        }
    }
}

/// Represents an operand for an instruction.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Operand {
    R8(Register8),
    R16(Register16),
    MA(MemoryAddress),
    I8(u8),
    I16(u16),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::R8(reg) => write!(f, "{}", reg),
            Self::R16(reg) => write!(f, "{}", reg),
            Self::MA(addr) => write!(f, "{}", addr),
            Self::I8(val) => write!(f, "byte {}", val),
            Self::I16(val) => write!(f, "word {}", val),
        }
    }
}

/// Represents a decoded 8086 instruction.
#[derive(Debug, PartialEq, Eq)]
pub enum Instruction {
    Mov {
        destination: Operand,
        source: Operand,
    },
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Mov {
                destination,
                source,
            } => write!(f, "mov {}, {}", destination, source),
        }
    }
}

// --- Private Decoder Implementation ---

/// Manages the byte stream and decoding process.
struct Decoder<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    /// Creates a new Decoder for the given byte slice.
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Checks if all bytes have been consumed.
    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// Reads a single byte and advances the position. Panics if out of bounds.
    fn read_u8(&mut self) -> u8 {
        if self.pos >= self.bytes.len() {
            panic!("Unexpected end of byte stream trying to read a byte");
        }
        let byte = self.bytes[self.pos];
        self.pos += 1;
        byte
    }

    /// Reads a 16-bit little-endian word. Panics if out of bounds.
    fn read_u16_le(&mut self) -> u16 {
        let low = self.read_u8() as u16;
        let high = self.read_u8() as u16;
        low | (high << 8)
    }

    /// Main instruction decoding loop.
    fn decode_next_instruction(&mut self) -> Instruction {
        let opcode = self.read_u8();
        match opcode {
            // MOV: Register/memory to/from register
            0x88..=0x8B => self.decode_mov_reg_mem((opcode & 0b10) != 0, (opcode & 0b1) != 0),
            // MOV: Immediate to register
            0xB0..=0xBF => self.decode_mov_imm_to_reg((opcode & 0b1000) != 0, opcode & 0b111),
            // MOV: Immediate to register/memory
            0xC6 | 0xC7 => self.decode_mov_imm_to_rm((opcode & 0b1) != 0),
            // MOV: Memory to/from accumulator
            0xA0..=0xA3 => {
                self.decode_mov_mem_accumulator((opcode & 0b10) == 0, (opcode & 0b1) != 0)
            }
            _ => panic!("Unsupported opcode: {:#04x}", opcode),
        }
    }

    // --- MOV Instruction Decoders ---

    fn decode_mov_reg_mem(&mut self, d: bool, w: bool) -> Instruction {
        let mod_rm_byte = self.read_u8();
        let reg_code = (mod_rm_byte >> 3) & 0b111;

        let reg_op = if w {
            Operand::R16(Register16::from_encoding(reg_code))
        } else {
            Operand::R8(Register8::from_encoding(reg_code))
        };

        let rm_op = self.decode_rm_operand(mod_rm_byte, w);

        let (destination, source) = if d { (reg_op, rm_op) } else { (rm_op, reg_op) };
        Instruction::Mov {
            destination,
            source,
        }
    }

    fn decode_mov_imm_to_reg(&mut self, w: bool, reg_code: u8) -> Instruction {
        let destination = if w {
            Operand::R16(Register16::from_encoding(reg_code))
        } else {
            Operand::R8(Register8::from_encoding(reg_code))
        };
        let source = if w {
            Operand::I16(self.read_u16_le())
        } else {
            Operand::I8(self.read_u8())
        };
        Instruction::Mov {
            destination,
            source,
        }
    }

    fn decode_mov_imm_to_rm(&mut self, w: bool) -> Instruction {
        let mod_rm_byte = self.read_u8();
        let destination = self.decode_rm_operand(mod_rm_byte, w);
        let source = if w {
            Operand::I16(self.read_u16_le())
        } else {
            Operand::I8(self.read_u8())
        };
        Instruction::Mov {
            destination,
            source,
        }
    }

    fn decode_mov_mem_accumulator(&mut self, d: bool, w: bool) -> Instruction {
        let acc_op = if w {
            Operand::R16(Register16::AX)
        } else {
            Operand::R8(Register8::AL)
        };
        let mem_op = Operand::MA(MemoryAddress::DA(self.read_u16_le()));
        let (destination, source) = if d {
            (acc_op, mem_op)
        } else {
            (mem_op, acc_op)
        };
        Instruction::Mov {
            destination,
            source,
        }
    }

    /// Decodes a ModR/M byte to get an operand. Handles register or memory addressing.
    fn decode_rm_operand(&mut self, mod_rm_byte: u8, w: bool) -> Operand {
        let md = mod_rm_byte >> 6;
        let rm = mod_rm_byte & 0b111;

        if md == 0b11 {
            // Register-direct addressing
            return if w {
                Operand::R16(Register16::from_encoding(rm))
            } else {
                Operand::R8(Register8::from_encoding(rm))
            };
        }

        // Memory addressing
        let addr = match md {
            0b00 => match rm {
                0b110 => MemoryAddress::DA(self.read_u16_le()),
                _ => Self::effective_address_no_disp(rm),
            },
            0b01 => Self::effective_address_disp8(rm, self.read_u8()),
            0b10 => Self::effective_address_disp16(rm, self.read_u16_le()),
            _ => unreachable!(), // Should not happen with 2-bit md
        };
        Operand::MA(addr)
    }

    // --- Static Helpers for Memory Address Calculation ---

    fn effective_address_no_disp(rm: u8) -> MemoryAddress {
        match rm {
            0b000 => MemoryAddress::RR(Register16::BX, Register16::SI),
            0b001 => MemoryAddress::RR(Register16::BX, Register16::DI),
            0b010 => MemoryAddress::RR(Register16::BP, Register16::SI),
            0b011 => MemoryAddress::RR(Register16::BP, Register16::DI),
            0b100 => MemoryAddress::R(Register16::SI),
            0b101 => MemoryAddress::R(Register16::DI),
            0b111 => MemoryAddress::R(Register16::BX),
            _ => panic!("Invalid R/M encoding for MOD=00: {}", rm),
        }
    }

    fn effective_address_disp8(rm: u8, disp: u8) -> MemoryAddress {
        match rm {
            0b000 => MemoryAddress::RRD8(Register16::BX, Register16::SI, disp),
            0b001 => MemoryAddress::RRD8(Register16::BX, Register16::DI, disp),
            0b010 => MemoryAddress::RRD8(Register16::BP, Register16::SI, disp),
            0b011 => MemoryAddress::RRD8(Register16::BP, Register16::DI, disp),
            0b100 => MemoryAddress::RD8(Register16::SI, disp),
            0b101 => MemoryAddress::RD8(Register16::DI, disp),
            0b110 => MemoryAddress::RD8(Register16::BP, disp),
            0b111 => MemoryAddress::RD8(Register16::BX, disp),
            _ => unreachable!(),
        }
    }

    fn effective_address_disp16(rm: u8, disp: u16) -> MemoryAddress {
        match rm {
            0b000 => MemoryAddress::RRD16(Register16::BX, Register16::SI, disp),
            0b001 => MemoryAddress::RRD16(Register16::BX, Register16::DI, disp),
            0b010 => MemoryAddress::RRD16(Register16::BP, Register16::SI, disp),
            0b011 => MemoryAddress::RRD16(Register16::BP, Register16::DI, disp),
            0b100 => MemoryAddress::RD16(Register16::SI, disp),
            0b101 => MemoryAddress::RD16(Register16::DI, disp),
            0b110 => MemoryAddress::RD16(Register16::BP, disp),
            0b111 => MemoryAddress::RD16(Register16::BX, disp),
            _ => unreachable!(),
        }
    }
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

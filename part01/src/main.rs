use num_enum::TryFromPrimitive;
use std::env;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use strum_macros::Display;

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
// Derived traits automatically handle Display and conversion from u8.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Display, TryFromPrimitive)]
#[repr(u8)]
pub enum Register16 {
    #[strum(serialize = "ax")]
    AX = 0b000,
    #[strum(serialize = "cx")]
    CX = 0b001,
    #[strum(serialize = "dx")]
    DX = 0b010,
    #[strum(serialize = "bx")]
    BX = 0b011,
    #[strum(serialize = "sp")]
    SP = 0b100,
    #[strum(serialize = "bp")]
    BP = 0b101,
    #[strum(serialize = "si")]
    SI = 0b110,
    #[strum(serialize = "di")]
    DI = 0b111,
}

/// Represents the 8-bit general-purpose registers of the 8086.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Display, TryFromPrimitive)]
#[repr(u8)]
pub enum Register8 {
    #[strum(serialize = "al")]
    AL = 0b000,
    #[strum(serialize = "cl")]
    CL = 0b001,
    #[strum(serialize = "dl")]
    DL = 0b010,
    #[strum(serialize = "bl")]
    BL = 0b011,
    #[strum(serialize = "ah")]
    AH = 0b100,
    #[strum(serialize = "ch")]
    CH = 0b101,
    #[strum(serialize = "dh")]
    DH = 0b110,
    #[strum(serialize = "bh")]
    BH = 0b111,
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

/// Represents a jump condition.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Display, TryFromPrimitive)]
#[repr(u8)]
pub enum JumpCondition {
    #[strum(serialize = "jo")]
    JO = 0x0,
    #[strum(serialize = "jno")]
    JNO = 0x1,
    #[strum(serialize = "jb")]
    JB = 0x2,
    #[strum(serialize = "jnb")]
    JNB = 0x3,
    #[strum(serialize = "jz")]
    JZ = 0x4,
    #[strum(serialize = "jnz")]
    JNZ = 0x5,
    #[strum(serialize = "jbe")]
    JBE = 0x6,
    #[strum(serialize = "ja")]
    JA = 0x7,
    #[strum(serialize = "js")]
    JS = 0x8,
    #[strum(serialize = "jns")]
    JNS = 0x9,
    #[strum(serialize = "jp")]
    JP = 0xA,
    #[strum(serialize = "jnp")]
    JNP = 0xB,
    #[strum(serialize = "jl")]
    JL = 0xC,
    #[strum(serialize = "jge")]
    JGE = 0xD,
    #[strum(serialize = "jle")]
    JLE = 0xE,
    #[strum(serialize = "jg")]
    JG = 0xF,
}

/// Represents the type of a LOOP or JCXZ instruction.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Display, TryFromPrimitive)]
#[repr(u8)]
pub enum LoopCondition {
    #[strum(serialize = "loopnz")]
    LOOPNZ = 0xE0,
    #[strum(serialize = "loopz")]
    LOOPZ = 0xE1,
    #[strum(serialize = "loop")]
    LOOP = 0xE2,
    #[strum(serialize = "jcxz")]
    JCXZ = 0xE3,
}

/// Represents a binary arithmetic or move operation.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Display)]
pub enum BinaryOp {
    #[strum(serialize = "mov")]
    MOV,
    #[strum(serialize = "add")]
    ADD,
    #[strum(serialize = "sub")]
    SUB,
    #[strum(serialize = "cmp")]
    CMP,
}

impl BinaryOp {
    // This function is kept because only some variants map to an encoding.
    fn arithmetic_op_from_encoding(op: u8) -> Self {
        match op {
            0b000 => BinaryOp::ADD,
            0b101 => BinaryOp::SUB,
            0b111 => BinaryOp::CMP,
            _ => unreachable!(),
        }
    }
}

/// Represents a decoded 8086 instruction.
#[derive(Debug, PartialEq, Eq)]
pub enum Instruction {
    BO(BinaryOp, Operand, Operand),
    JMP(JumpCondition, i8),
    LOOP(LoopCondition, i8),
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BO(op, dest, src) => write!(f, "{} {}, {}", op, dest, src),
            Self::JMP(cond, disp) => {
                let target_offset = *disp as i16 + 2;
                write!(f, "{} short ${:+}", cond, target_offset)
            }
            Self::LOOP(cond, disp) => {
                let target_offset = *disp as i16 + 2;
                write!(f, "{} ${:+}", cond, target_offset)
            }
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
        let start_pos = self.pos;
        let opcode = self.read_u8();
        let instr = match opcode {
            // Arithmetic: Register/memory with register
            0x00..=0x03 | 0x28..=0x2B | 0x38..=0x3B => self.decode_arithmetic_reg_mem(opcode),
            // Arithmetic: Immediate to accumulator
            0x04 | 0x05 | 0x2C | 0x2D | 0x3C | 0x3D => self.decode_arithmetic_imm_to_acc(opcode),
            // Arithmetic: Immediate to Register/memory
            0x80..=0x83 => self.decode_arithmetic_imm_to_rm(opcode),
            // MOV: Register/memory to/from register
            0x88..=0x8B => self.decode_mov_reg_mem(opcode),
            // MOV: Memory to/from accumulator
            0xA0..=0xA3 => self.decode_mov_mem_accumulator(opcode),
            // MOV: Immediate to register
            0xB0..=0xBF => self.decode_mov_imm_to_reg(opcode),
            // MOV: Immediate to register/memory
            0xC6 | 0xC7 => self.decode_mov_imm_to_rm(opcode),
            // Conditional Jumps
            0x70..=0x7F => self.decode_jump(opcode),
            // LOOP and JCXZ instructions
            0xE0..=0xE3 => self.decode_loop(opcode),
            _ => panic!("Unsupported opcode: {:#04x}", opcode),
        };

        // --- Printing for debugging purposes ---
        println!(
            "Starting position: {}\nProcessed: {} bytes\nBytes: {}\n{}\n",
            start_pos,
            self.pos - start_pos,
            self.bytes[start_pos..self.pos]
                .iter()
                .map(|n| format!("{:08b}", n))
                .collect::<Vec<_>>()
                .join(", "),
            instr
        );
        // ---------------------------------------

        instr
    }

    // --- Instruction Format Decoders ---

    /// Decodes a LOOP, LOOPZ, LOOPNZ, or JCXZ instruction.
    fn decode_loop(&mut self, opcode: u8) -> Instruction {
        let cond = LoopCondition::try_from(opcode).expect("Invalid loop condition encoding");
        let disp = self.read_u8() as i8;
        Instruction::LOOP(cond, disp)
    }

    /// Decodes a conditional jump instruction.
    fn decode_jump(&mut self, opcode: u8) -> Instruction {
        let cond = JumpCondition::try_from(opcode & 0x0F).expect("Invalid jump condition encoding");
        let disp = self.read_u8() as i8;
        Instruction::JMP(cond, disp)
    }

    fn decode_arithmetic_reg_mem(&mut self, opcode: u8) -> Instruction {
        let op = BinaryOp::arithmetic_op_from_encoding((opcode >> 3) & 0b111);
        let d = (opcode & 0b10) != 0;
        let w = (opcode & 0b1) != 0;
        let mod_rm_byte = self.read_u8();
        let reg_op = if w {
            let reg_encoding = (mod_rm_byte >> 3) & 0b111;
            let reg = Register16::try_from(reg_encoding).expect("Invalid 16-bit reg encoding");
            Operand::R16(reg)
        } else {
            let reg_encoding = (mod_rm_byte >> 3) & 0b111;
            let reg = Register8::try_from(reg_encoding).expect("Invalid 8-bit reg encoding");
            Operand::R8(reg)
        };
        let rm_op = self.decode_rm_operand(mod_rm_byte, w);
        let (dest, src) = if d { (reg_op, rm_op) } else { (rm_op, reg_op) };
        Instruction::BO(op, dest, src)
    }

    fn decode_arithmetic_imm_to_rm(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1) != 0;
        let mod_rm_byte = self.read_u8();
        let op = BinaryOp::arithmetic_op_from_encoding((mod_rm_byte >> 3) & 0b111);
        let dest = self.decode_rm_operand(mod_rm_byte, w);
        let src = if w {
            if opcode == 0x83 {
                Operand::I16(self.read_u8() as i8 as u16)
            } else {
                Operand::I16(self.read_u16_le())
            }
        } else {
            Operand::I8(self.read_u8())
        };
        Instruction::BO(op, dest, src)
    }

    fn decode_arithmetic_imm_to_acc(&mut self, opcode: u8) -> Instruction {
        let op = BinaryOp::arithmetic_op_from_encoding((opcode >> 3) & 0b111);
        let w = (opcode & 0b1) != 0;
        let dest = if w {
            Operand::R16(Register16::AX)
        } else {
            Operand::R8(Register8::AL)
        };
        let src = if w {
            Operand::I16(self.read_u16_le())
        } else {
            Operand::I8(self.read_u8())
        };
        Instruction::BO(op, dest, src)
    }

    fn decode_mov_reg_mem(&mut self, opcode: u8) -> Instruction {
        let d = (opcode & 0b10) != 0;
        let w = (opcode & 0b1) != 0;
        let mod_rm_byte = self.read_u8();
        let reg_op = if w {
            let reg_encoding = (mod_rm_byte >> 3) & 0b111;
            let reg = Register16::try_from(reg_encoding).expect("Invalid 16-bit reg encoding");
            Operand::R16(reg)
        } else {
            let reg_encoding = (mod_rm_byte >> 3) & 0b111;
            let reg = Register8::try_from(reg_encoding).expect("Invalid 8-bit reg encoding");
            Operand::R8(reg)
        };
        let rm_op = self.decode_rm_operand(mod_rm_byte, w);
        let (dest, src) = if d { (reg_op, rm_op) } else { (rm_op, reg_op) };
        Instruction::BO(BinaryOp::MOV, dest, src)
    }

    fn decode_mov_imm_to_reg(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1000) != 0;
        let dest = if w {
            let reg = Register16::try_from(opcode & 0b111).expect("Invalid 16-bit reg encoding");
            Operand::R16(reg)
        } else {
            let reg = Register8::try_from(opcode & 0b111).expect("Invalid 8-bit reg encoding");
            Operand::R8(reg)
        };
        let src = if w {
            Operand::I16(self.read_u16_le())
        } else {
            Operand::I8(self.read_u8())
        };
        Instruction::BO(BinaryOp::MOV, dest, src)
    }

    fn decode_mov_imm_to_rm(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1) != 0;
        let mod_rm_byte = self.read_u8();
        let dest = self.decode_rm_operand(mod_rm_byte, w);
        let src = if w {
            Operand::I16(self.read_u16_le())
        } else {
            Operand::I8(self.read_u8())
        };
        Instruction::BO(BinaryOp::MOV, dest, src)
    }

    fn decode_mov_mem_accumulator(&mut self, opcode: u8) -> Instruction {
        let d = (opcode & 0b10) == 0;
        let w = (opcode & 0b1) != 0;
        let acc_op = if w {
            Operand::R16(Register16::AX)
        } else {
            Operand::R8(Register8::AL)
        };
        let mem_op = Operand::MA(MemoryAddress::DA(self.read_u16_le()));
        let (dest, src) = if d {
            (acc_op, mem_op)
        } else {
            (mem_op, acc_op)
        };
        Instruction::BO(BinaryOp::MOV, dest, src)
    }

    /// Decodes a ModR/M byte to get an operand. Handles register or memory addressing.
    fn decode_rm_operand(&mut self, mod_rm_byte: u8, w: bool) -> Operand {
        let md = mod_rm_byte >> 6;
        let rm = mod_rm_byte & 0b111;
        if md == 0b11 {
            return if w {
                let reg = Register16::try_from(rm).expect("Invalid 16-bit reg encoding");
                Operand::R16(reg)
            } else {
                let reg = Register8::try_from(rm).expect("Invalid 8-bit reg encoding");
                Operand::R8(reg)
            };
        }
        let addr = match md {
            0b00 => {
                if rm == 0b110 {
                    MemoryAddress::DA(self.read_u16_le())
                } else {
                    Self::effective_address_no_disp(rm)
                }
            }
            0b01 => Self::effective_address_disp8(rm, self.read_u8()),
            0b10 => Self::effective_address_disp16(rm, self.read_u16_le()),
            _ => unreachable!(),
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

// --- File I/O and Main Function ---

pub fn read_file<P: AsRef<Path>>(filepath: P) -> io::Result<Vec<u8>> {
    let mut file = File::open(filepath)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

pub fn write_decoding<P: AsRef<Path>>(
    instructions: &[Instruction],
    output_filepath: P,
) -> io::Result<()> {
    let mut output_file = File::create(output_filepath)?;
    for instruction in instructions {
        writeln!(output_file, "{}", instruction)?;
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input_binary_file> <output_text_file>", args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];

    println!("Starting 8086 decoder...\n");
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

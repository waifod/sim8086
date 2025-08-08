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

// --- Instruction and Operand Enums and Constants ---

/// Represents a single 8-bit or 16-bit general-purpose register of the 8086.
/// This replaces the separate Register8 and Register16 enums.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Display)]
pub enum Register {
    // 16-bit registers (w=1)
    #[strum(serialize = "ax")]
    AX,
    #[strum(serialize = "cx")]
    CX,
    #[strum(serialize = "dx")]
    DX,
    #[strum(serialize = "bx")]
    BX,
    #[strum(serialize = "sp")]
    SP,
    #[strum(serialize = "bp")]
    BP,
    #[strum(serialize = "si")]
    SI,
    #[strum(serialize = "di")]
    DI,
    // 8-bit registers (w=0)
    #[strum(serialize = "al")]
    AL,
    #[strum(serialize = "cl")]
    CL,
    #[strum(serialize = "dl")]
    DL,
    #[strum(serialize = "bl")]
    BL,
    #[strum(serialize = "ah")]
    AH,
    #[strum(serialize = "ch")]
    CH,
    #[strum(serialize = "dh")]
    DH,
    #[strum(serialize = "bh")]
    BH,
}

/// A lookup table for all general-purpose registers, indexed by the W-bit
/// and the 3-bit register field (REG).
///
/// `REGISTERS[w][reg]` will yield the correct register.
/// `w=0` is for 8-bit registers, `w=1` is for 16-bit registers.
const REGISTERS: [[Register; 8]; 2] = [
    // w=0: 8-bit registers
    [
        Register::AL,
        Register::CL,
        Register::DL,
        Register::BL,
        Register::AH,
        Register::CH,
        Register::DH,
        Register::BH,
    ],
    // w=1: 16-bit registers
    [
        Register::AX,
        Register::CX,
        Register::DX,
        Register::BX,
        Register::SP,
        Register::BP,
        Register::SI,
        Register::DI,
    ],
];

/// Represents a memory address calculation.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MemoryAddress {
    R(Register),
    RD8(Register, u8),
    RD16(Register, u16),
    RR(Register, Register),
    RRD8(Register, Register, u8),
    RRD16(Register, Register, u16),
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
    R(Register),
    MA(MemoryAddress),
    IMM8(u8),
    IMM16(u16),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::R(reg) => write!(f, "{}", reg),
            Self::MA(addr) => write!(f, "{}", addr),
            Self::IMM8(val) => write!(f, "byte {}", val),
            Self::IMM16(val) => write!(f, "word {}", val),
        }
    }
}

/// Represents the condition for a conditional jump instruction.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Display)]
pub enum JumpCondition {
    #[strum(serialize = "jo")]
    JO,
    #[strum(serialize = "jno")]
    JNO,
    #[strum(serialize = "jb")]
    JB,
    #[strum(serialize = "jnb")]
    JNB,
    #[strum(serialize = "jz")]
    JZ,
    #[strum(serialize = "jnz")]
    JNZ,
    #[strum(serialize = "jbe")]
    JBE,
    #[strum(serialize = "ja")]
    JA,
    #[strum(serialize = "js")]
    JS,
    #[strum(serialize = "jns")]
    JNS,
    #[strum(serialize = "jp")]
    JP,
    #[strum(serialize = "jnp")]
    JNP,
    #[strum(serialize = "jl")]
    JL,
    #[strum(serialize = "jge")]
    JGE,
    #[strum(serialize = "jle")]
    JLE,
    #[strum(serialize = "jg")]
    JG,
}

const JUMP_CONDITIONS: [JumpCondition; 16] = [
    JumpCondition::JO,
    JumpCondition::JNO,
    JumpCondition::JB,
    JumpCondition::JNB,
    JumpCondition::JZ,
    JumpCondition::JNZ,
    JumpCondition::JBE,
    JumpCondition::JA,
    JumpCondition::JS,
    JumpCondition::JNS,
    JumpCondition::JP,
    JumpCondition::JNP,
    JumpCondition::JL,
    JumpCondition::JGE,
    JumpCondition::JLE,
    JumpCondition::JG,
];

/// Represents the condition for a loop instruction.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Display)]
pub enum LoopCondition {
    #[strum(serialize = "loopnz")]
    LOOPNZ,
    #[strum(serialize = "loopz")]
    LOOPZ,
    #[strum(serialize = "loop")]
    LOOP,
    #[strum(serialize = "jcxz")]
    JCXZ,
}

const LOOP_CONDITIONS: [LoopCondition; 4] = [
    LoopCondition::LOOPNZ,
    LoopCondition::LOOPZ,
    LoopCondition::LOOP,
    LoopCondition::JCXZ,
];

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
        #[cfg(debug_assertions)]
        {
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
        }
        // ---------------------------------------

        instr
    }

    // --- Instruction Format Decoders ---

    /// Decodes a LOOP, LOOPZ, LOOPNZ, or JCXZ instruction.
    fn decode_loop(&mut self, opcode: u8) -> Instruction {
        // Look up the LoopCondition from the LOOP_CONDITIONS array
        let cond_index = (opcode - 0xE0) as usize;
        let cond = *LOOP_CONDITIONS
            .get(cond_index)
            .expect("Invalid loop condition opcode");
        let disp = self.read_u8() as i8;
        Instruction::LOOP(cond, disp)
    }

    /// Decodes a conditional jump instruction.
    fn decode_jump(&mut self, opcode: u8) -> Instruction {
        // Look up the JumpCondition from the JUMP_CONDITIONS array
        let cond_index = (opcode & 0x0F) as usize;
        let cond = *JUMP_CONDITIONS
            .get(cond_index)
            .expect("Invalid jump condition opcode");
        let disp = self.read_u8() as i8;
        Instruction::JMP(cond, disp)
    }

    fn decode_arithmetic_reg_mem(&mut self, opcode: u8) -> Instruction {
        let op = BinaryOp::arithmetic_op_from_encoding((opcode >> 3) & 0b111);
        let d = (opcode & 0b10) != 0;
        let w = (opcode & 0b1) != 0;
        let (dest, src) = self.decode_reg_and_rm_operands(d, w);
        Instruction::BO(op, dest, src)
    }

    fn decode_mov_reg_mem(&mut self, opcode: u8) -> Instruction {
        let d = (opcode & 0b10) != 0;
        let w = (opcode & 0b1) != 0;
        let (dest, src) = self.decode_reg_and_rm_operands(d, w);
        Instruction::BO(BinaryOp::MOV, dest, src)
    }

    fn decode_arithmetic_imm_to_rm(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1) != 0;
        let mod_rm_byte = self.read_u8();
        let op = BinaryOp::arithmetic_op_from_encoding((mod_rm_byte >> 3) & 0b111);
        let dest = self.decode_rm_operand(mod_rm_byte, w);
        let src = if w {
            if opcode == 0x83 {
                Operand::IMM16(self.read_u8() as i8 as u16)
            } else {
                Operand::IMM16(self.read_u16_le())
            }
        } else {
            Operand::IMM8(self.read_u8())
        };
        Instruction::BO(op, dest, src)
    }

    fn decode_arithmetic_imm_to_acc(&mut self, opcode: u8) -> Instruction {
        let op = BinaryOp::arithmetic_op_from_encoding((opcode >> 3) & 0b111);
        let w = (opcode & 0b1) != 0;
        // The accumulator registers (AL, AX) are always at index 0
        let dest_reg = REGISTERS
            .get(w as usize)
            .expect("Invalid W bit for accumulator register")
            .get(0)
            .expect("Missing accumulator register");
        let dest = Operand::R(*dest_reg);
        let src = if w {
            Operand::IMM16(self.read_u16_le())
        } else {
            Operand::IMM8(self.read_u8())
        };
        Instruction::BO(op, dest, src)
    }

    fn decode_mov_imm_to_reg(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1000) != 0;
        let reg_encoding = (opcode & 0b111) as usize;
        let dest_reg = REGISTERS
            .get(w as usize)
            .expect("Invalid W bit")
            .get(reg_encoding)
            .expect("Invalid REG encoding");
        let dest = Operand::R(*dest_reg);
        let src = if w {
            Operand::IMM16(self.read_u16_le())
        } else {
            Operand::IMM8(self.read_u8())
        };
        Instruction::BO(BinaryOp::MOV, dest, src)
    }

    fn decode_mov_imm_to_rm(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1) != 0;
        let mod_rm_byte = self.read_u8();
        let dest = self.decode_rm_operand(mod_rm_byte, w);
        let src = if w {
            Operand::IMM16(self.read_u16_le())
        } else {
            Operand::IMM8(self.read_u8())
        };
        Instruction::BO(BinaryOp::MOV, dest, src)
    }

    fn decode_mov_mem_accumulator(&mut self, opcode: u8) -> Instruction {
        let d = (opcode & 0b10) == 0;
        let w = (opcode & 0b1) != 0;
        let acc_reg = REGISTERS
            .get(w as usize)
            .expect("Invalid W bit for accumulator register")
            .get(0)
            .expect("Missing accumulator register");
        let acc_op = Operand::R(*acc_reg);
        let mem_op = Operand::MA(MemoryAddress::DA(self.read_u16_le()));
        let (dest, src) = if d {
            (acc_op, mem_op)
        } else {
            (mem_op, acc_op)
        };
        Instruction::BO(BinaryOp::MOV, dest, src)
    }

    /// Decodes a ModR/M byte to get a register and a register/memory operand.
    /// Returns a tuple of (destination, source) based on the d flag.
    fn decode_reg_and_rm_operands(&mut self, d: bool, w: bool) -> (Operand, Operand) {
        let mod_rm_byte = self.read_u8();
        let reg_encoding = (mod_rm_byte >> 3) & 0b111;
        let reg_op = Operand::R(
            *REGISTERS
                .get(w as usize)
                .expect("Invalid W bit")
                .get(reg_encoding as usize)
                .expect("Invalid REG encoding"),
        );
        let rm_op = self.decode_rm_operand(mod_rm_byte, w);

        if d {
            (reg_op, rm_op)
        } else {
            (rm_op, reg_op)
        }
    }

    /// Decodes a ModR/M byte to get an operand. Handles register or memory addressing.
    fn decode_rm_operand(&mut self, mod_rm_byte: u8, w: bool) -> Operand {
        let md = mod_rm_byte >> 6;
        let rm = (mod_rm_byte & 0b111) as usize;
        if md == 0b11 {
            let reg = REGISTERS
                .get(w as usize)
                .expect("Invalid W bit")
                .get(rm)
                .expect("Invalid R/M encoding for register");
            return Operand::R(*reg);
        }
        let addr = match md {
            0b00 => {
                if rm == 0b110 {
                    MemoryAddress::DA(self.read_u16_le())
                } else {
                    Self::decode_effective_address_no_disp(rm as u8)
                }
            }
            0b01 => Self::decode_effective_address_disp8(rm as u8, self.read_u8()),
            0b10 => Self::decode_effective_address_disp16(rm as u8, self.read_u16_le()),
            _ => unreachable!(),
        };
        Operand::MA(addr)
    }

    // --- Static Helpers for Memory Address Calculation ---

    fn decode_effective_address_no_disp(rm: u8) -> MemoryAddress {
        match rm {
            0b000 => MemoryAddress::RR(REGISTERS[1][3], REGISTERS[1][6]),
            0b001 => MemoryAddress::RR(REGISTERS[1][3], REGISTERS[1][7]),
            0b010 => MemoryAddress::RR(REGISTERS[1][5], REGISTERS[1][6]),
            0b011 => MemoryAddress::RR(REGISTERS[1][5], REGISTERS[1][7]),
            0b100 => MemoryAddress::R(REGISTERS[1][6]),
            0b101 => MemoryAddress::R(REGISTERS[1][7]),
            0b111 => MemoryAddress::R(REGISTERS[1][3]),
            _ => panic!("Invalid R/M encoding for MOD=00: {}", rm),
        }
    }

    fn decode_effective_address_disp8(rm: u8, disp: u8) -> MemoryAddress {
        match rm {
            0b000 => MemoryAddress::RRD8(REGISTERS[1][3], REGISTERS[1][6], disp),
            0b001 => MemoryAddress::RRD8(REGISTERS[1][3], REGISTERS[1][7], disp),
            0b010 => MemoryAddress::RRD8(REGISTERS[1][5], REGISTERS[1][6], disp),
            0b011 => MemoryAddress::RRD8(REGISTERS[1][5], REGISTERS[1][7], disp),
            0b100 => MemoryAddress::RD8(REGISTERS[1][6], disp),
            0b101 => MemoryAddress::RD8(REGISTERS[1][7], disp),
            0b110 => MemoryAddress::RD8(REGISTERS[1][5], disp),
            0b111 => MemoryAddress::RD8(REGISTERS[1][3], disp),
            _ => unreachable!(),
        }
    }

    fn decode_effective_address_disp16(rm: u8, disp: u16) -> MemoryAddress {
        match rm {
            0b000 => MemoryAddress::RRD16(REGISTERS[1][3], REGISTERS[1][6], disp),
            0b001 => MemoryAddress::RRD16(REGISTERS[1][3], REGISTERS[1][7], disp),
            0b010 => MemoryAddress::RRD16(REGISTERS[1][5], REGISTERS[1][6], disp),
            0b011 => MemoryAddress::RRD16(REGISTERS[1][5], REGISTERS[1][7], disp),
            0b100 => MemoryAddress::RD16(REGISTERS[1][6], disp),
            0b101 => MemoryAddress::RD16(REGISTERS[1][7], disp),
            0b110 => MemoryAddress::RD16(REGISTERS[1][5], disp),
            0b111 => MemoryAddress::RRD16(REGISTERS[1][3], REGISTERS[1][7], disp),
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

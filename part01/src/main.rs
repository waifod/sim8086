use std::env;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

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

/// Represents a jump condition.
#[derive(Debug, PartialEq, Eq)]
pub enum JumpCondition {
    JO,
    JNO,
    JB,
    JNB,
    JZ,
    JNZ,
    JBE,
    JA,
    JS,
    JNS,
    JP,
    JNP,
    JL,
    JGE,
    JLE,
    JG,
}

impl JumpCondition {
    /// Creates a JumpCondition from the 4-bit encoding in the opcode.
    fn from_encoding(encoding: u8) -> Self {
        match encoding {
            0b0000 => Self::JO,
            0b0001 => Self::JNO,
            0b0010 => Self::JB,
            0b0011 => Self::JNB,
            0b0100 => Self::JZ,
            0b0101 => Self::JNZ,
            0b0110 => Self::JBE,
            0b0111 => Self::JA,
            0b1000 => Self::JS,
            0b1001 => Self::JNS,
            0b1010 => Self::JP,
            0b1011 => Self::JNP,
            0b1100 => Self::JL,
            0b1101 => Self::JGE,
            0b1110 => Self::JLE,
            0b1111 => Self::JG,
            _ => panic!("Invalid 4-bit encoding for JumpCondition: {}", encoding),
        }
    }
}

impl fmt::Display for JumpCondition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::JO => "jo",
                Self::JNO => "jno",
                Self::JB => "jb",
                Self::JNB => "jnb",
                Self::JZ => "jz",
                Self::JNZ => "jnz",
                Self::JBE => "jbe",
                Self::JA => "ja",
                Self::JS => "js",
                Self::JNS => "jns",
                Self::JP => "jp",
                Self::JNP => "jnp",
                Self::JL => "jl",
                Self::JGE => "jge",
                Self::JLE => "jle",
                Self::JG => "jg",
            }
        )
    }
}

/// Represents the type of a LOOP or JCXZ instruction.
#[derive(Debug, PartialEq, Eq)]
pub enum LoopType {
    LOOPNZ,
    LOOPZ,
    LOOP,
    JCXZ,
}

impl fmt::Display for LoopType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::LOOPNZ => "loopnz",
                Self::LOOPZ => "loopz",
                Self::LOOP => "loop",
                Self::JCXZ => "jcxz",
            }
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum BinaryOp {
    MOV,
    ADD,
    SUB,
    CMP,
}

impl BinaryOp {
    fn arithmetic_op_from_encoding(op: u8) -> Self {
        match op {
            0b000 => BinaryOp::ADD,
            0b101 => BinaryOp::SUB,
            0b111 => BinaryOp::CMP,
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::MOV => "mov",
                Self::ADD => "add",
                Self::SUB => "sub",
                Self::CMP => "cmp",
            }
        )
    }
}

/// Represents a decoded 8086 instruction.
#[derive(Debug, PartialEq, Eq)]
pub enum Instruction {
    BO(BinaryOp, Operand, Operand),
    JMP(JumpCondition, i8),
    LOOP(LoopType, i8),
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BO(op, dest, src) => write!(f, "{} {}, {}", op, dest, src),
            Self::JMP(cond, disp) => {
                let target_offset = *disp as i16 + 2;
                write!(f, "{} short ${:+}", cond, target_offset)
            }
            Self::LOOP(loop_type, disp) => {
                let target_offset = *disp as i16 + 2;
                // The only change is removing "short" from this line
                write!(f, "{} ${:+}", loop_type, target_offset)
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
    /// Creates a Decoder for the given byte slice.
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
        println!("{}", instr); // Uncomment for debugging
        instr
    }

    // --- Instruction Format Decoders ---

    /// Decodes a LOOP, LOOPZ, LOOPNZ, or JCXZ instruction.
    fn decode_loop(&mut self, opcode: u8) -> Instruction {
        let loop_type = match opcode {
            0xE0 => LoopType::LOOPNZ,
            0xE1 => LoopType::LOOPZ,
            0xE2 => LoopType::LOOP,
            0xE3 => LoopType::JCXZ,
            _ => unreachable!(), // Guarded by the call site match statement
        };
        let disp = self.read_u8() as i8;
        Instruction::LOOP(loop_type, disp)
    }

    /// Decodes a conditional jump instruction.
    fn decode_jump(&mut self, opcode: u8) -> Instruction {
        let cond = JumpCondition::from_encoding(opcode & 0x0F);
        let disp = self.read_u8() as i8;
        Instruction::JMP(cond, disp)
    }

    fn decode_arithmetic_reg_mem(&mut self, opcode: u8) -> Instruction {
        let op = BinaryOp::arithmetic_op_from_encoding((opcode >> 3) & 0b111);
        let d = (opcode & 0b10) != 0;
        let w = (opcode & 0b1) != 0;
        let mod_rm_byte = self.read_u8();
        let reg_op = if w {
            Operand::R16(Register16::from_encoding((mod_rm_byte >> 3) & 0b111))
        } else {
            Operand::R8(Register8::from_encoding((mod_rm_byte >> 3) & 0b111))
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
        let reg_code = (mod_rm_byte >> 3) & 0b111;
        let reg_op = if w {
            Operand::R16(Register16::from_encoding(reg_code))
        } else {
            Operand::R8(Register8::from_encoding(reg_code))
        };
        let rm_op = self.decode_rm_operand(mod_rm_byte, w);
        let (dest, src) = if d { (reg_op, rm_op) } else { (rm_op, reg_op) };
        Instruction::BO(BinaryOp::MOV, dest, src)
    }

    fn decode_mov_imm_to_reg(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1000) != 0;
        let reg_code = opcode & 0b111;
        let dest = if w {
            Operand::R16(Register16::from_encoding(reg_code))
        } else {
            Operand::R8(Register8::from_encoding(reg_code))
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
                Operand::R16(Register16::from_encoding(rm))
            } else {
                Operand::R8(Register8::from_encoding(rm))
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

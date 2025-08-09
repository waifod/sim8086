use std::env;
use std::fmt;
use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use strum_macros::Display;

pub fn execute(bytes: &[u8]) {
    let mut executor = Executor::new(bytes);
    println!("\nStart of execution log:");
    println!("------------------------");
    while !executor.is_eof() {
        executor.execute_next_instruction();
    }
    println!("------------------------");

    println!("\nFinal registers:");
    println!(
        "      bx: 0x{:04x} ({})",
        executor.get_16bit_register_value(Register::BX),
        executor.get_16bit_register_value(Register::BX)
    );
    println!(
        "      cx: 0x{:04x} ({})",
        executor.get_16bit_register_value(Register::CX),
        executor.get_16bit_register_value(Register::CX)
    );
    println!("      ip: 0x{:04x} ({})", executor.ip, executor.ip);

    // Print the final flags in a more readable format
    print!("    flags:");
    executor.flags.print_active();
    println!();

    println!("done");
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct RegisterRow {
    low: u8,
    high: u8,
}

impl Display for RegisterRow {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:02x} {:02x}", self.high, self.low)
    }
}

// A new struct to manage the 8086 flags by name.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Flags {
    parity: bool,
    zero: bool,
    sign: bool,
    carry: bool,
    auxiliary_carry: bool,
}

impl Flags {
    // A helper method to get the flag state as a compact string
    fn to_string(&self) -> String {
        let mut s = String::new();
        if self.parity {
            s.push('P');
        }
        if self.zero {
            s.push('Z');
        }
        if self.sign {
            s.push('S');
        }
        if self.carry {
            s.push('C');
        }
        if self.auxiliary_carry {
            s.push('A');
        }
        s
    }

    // Prints the currently active flags
    fn print_active(&self) {
        if self.parity {
            print!("P");
        }
        if self.zero {
            print!("Z");
        }
        if self.sign {
            print!("S");
        }
        if self.carry {
            print!("C");
        }
        if self.auxiliary_carry {
            print!("A");
        }
    }
}

struct Executor<'a> {
    bytes: &'a [u8],
    register_rows: [RegisterRow; 8],
    flags: Flags,
    ip: usize,
    decoder: Decoder<'a>,
}

impl<'a> Executor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            register_rows: [RegisterRow { low: 0, high: 0 }; 8],
            flags: Flags {
                parity: false,
                zero: false,
                sign: false,
                carry: false,
                auxiliary_carry: false,
            },
            ip: 0,
            decoder: Decoder::new(bytes),
        }
    }

    fn is_eof(&self) -> bool {
        self.ip >= self.bytes.len()
    }

    fn get_register_row(&mut self, reg: Register) -> &mut RegisterRow {
        let idx = reg.get_index().1 as usize;
        self.register_rows
            .get_mut(idx)
            .expect("Fatal error: Attempted to get register row with invalid index.")
    }

    fn get_16bit_register_value(&self, reg: Register) -> u16 {
        let idx = reg.get_index().1 as usize;
        let row = &self.register_rows[idx];
        (row.high as u16) << 8 | (row.low as u16)
    }

    fn get_8bit_register_value(&self, reg: Register) -> u8 {
        let (w, i) = reg.get_index();
        if w == 1 {
            panic!(
                "Fatal error: Attempted to get 8-bit value from a 16-bit register: {}",
                reg
            );
        }
        let idx = i & 0b11;
        let row = &self.register_rows[idx as usize];
        if (i >> 2) != 0 {
            row.high
        } else {
            row.low
        }
    }

    fn get_register_display_value(&self, reg: Register) -> u16 {
        let (w, i) = reg.get_index();
        if w == 1 {
            self.get_16bit_register_value(reg)
        } else {
            self.get_8bit_register_value(reg) as u16
        }
    }

    fn execute_next_instruction(&mut self) {
        // Capture initial state for logging
        let start_ip = self.ip;
        let start_flags = self.flags;

        // Decode the instruction from the current position
        self.decoder.pos = self.ip;
        let instruction = self.decoder.decode_next_instruction();
        let instruction_len = self.decoder.pos - self.ip;
        self.ip = self.decoder.pos;

        // Capture initial state of operands for logging
        let initial_state = self.get_operand_values_for_log(&instruction);

        // Execute the instruction
        self.execute_instruction(&instruction);

        // Capture final state for logging
        let final_state = self.get_operand_values_for_log(&instruction);
        let end_ip = self.ip;
        let end_flags = self.flags;

        // Print the log line
        self.log_instruction_execution(
            instruction,
            start_ip,
            end_ip,
            initial_state,
            final_state,
            start_flags,
            end_flags,
        );
    }

    // A new function to handle all the logging logic
    fn log_instruction_execution(
        &self,
        instruction: Instruction,
        start_ip: usize,
        end_ip: usize,
        initial_state: Vec<(Operand, u16)>,
        final_state: Vec<(Operand, u16)>,
        start_flags: Flags,
        end_flags: Flags,
    ) {
        let mut log_line = format!("{}", instruction);
        let mut state_changes = Vec::new();

        // Log operand value changes
        for i in 0..initial_state.len() {
            let (op, initial_val) = &initial_state[i];
            let (_, final_val) = &final_state[i];
            if initial_val != final_val {
                if let Operand::R(reg) = op {
                    state_changes.push(format!("{}:0x{:x}->0x{:x}", reg, initial_val, final_val));
                }
            }
        }

        // Log IP change
        if start_ip != end_ip {
            state_changes.push(format!("ip:0x{:x}->0x{:x}", start_ip, end_ip));
        }

        // Log flag changes
        if start_flags != end_flags {
            let start_flags_str = start_flags.to_string();
            let end_flags_str = end_flags.to_string();
            state_changes.push(format!("flags:{}->{}", start_flags_str, end_flags_str));
        }

        if !state_changes.is_empty() {
            log_line.push_str(&format!(" ; {}", state_changes.join(" ")));
        }

        println!("{}", log_line);
    }

    // Helper function to get the values of operands for logging
    fn get_operand_values_for_log(&self, instruction: &Instruction) -> Vec<(Operand, u16)> {
        let mut values = Vec::new();
        match instruction {
            Instruction::BO(_, dest, src) => {
                if let Operand::R(reg) = dest {
                    values.push((*dest, self.get_register_display_value(*reg)));
                }
            }
            _ => (),
        }
        values
    }

    fn execute_instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::BO(op, arg1, arg2) => self.execute_binary_instruction(*op, *arg1, *arg2),
            Instruction::JMP(op, arg) => self.execute_jump_instruction(*op, *arg),
            _ => panic!("Unsupported instruction type: {:?}", instruction),
        }
    }

    /// Executes a conditional jump instruction.
    /// The displacement is an `i8` which can be positive or negative.
    /// It's relative to the END of the jump instruction, so we add it to `self.ip`.
    fn execute_jump_instruction(&mut self, op: JumpCondition, arg: i8) {
        let should_jump = match op {
            // JZ/JNZ check the Zero Flag, not the Sign Flag.
            JumpCondition::JZ => self.flags.zero,
            JumpCondition::JNZ => !self.flags.zero,
            _ => panic!("no other jump conditions supported yet"),
        };
        // The jump displacement is relative to the start of the next instruction.
        // Since self.ip is already at the start of the next instruction, we can
        // just add the displacement. We cast to isize to handle negative jumps correctly.
        if should_jump {
            self.ip = (self.ip as isize + arg as isize) as usize;
        }
    }

    fn execute_binary_instruction(&mut self, op: BinaryOp, arg1: Operand, arg2: Operand) {
        match op {
            BinaryOp::MOV => self.execute_mov_instruction(arg1, arg2),
            BinaryOp::ADD | BinaryOp::SUB | BinaryOp::CMP => {
                self.execute_arithmetic_instruction(op, arg1, arg2)
            }
            _ => panic!("Unsupported binary operation: {:?}", op),
        }
    }

    fn execute_mov_instruction(&mut self, arg1: Operand, arg2: Operand) {
        self.set_value(arg1, arg2);
    }

    fn execute_arithmetic_instruction(&mut self, op: BinaryOp, arg1: Operand, arg2: Operand) {
        let val1 = self.get_operand_value(arg1);
        let val2 = self.get_operand_value(arg2);

        // We'll perform the arithmetic and then update the flags.
        let (result, _carry_out) = match op {
            BinaryOp::ADD => {
                let (res, overflow) = (val1 as i16).overflowing_add(val2 as i16);
                let carry_flag = overflow;
                (res, carry_flag)
            }
            BinaryOp::SUB | BinaryOp::CMP => {
                let (res, overflow) = (val1 as i16).overflowing_sub(val2 as i16);
                let carry_flag = overflow;
                (res, carry_flag)
            }
            _ => unreachable!("This should have been caught by the outer match statement."),
        };

        // Update all flags based on the result.
        self.update_sign_flag(result);
        self.update_zero_flag(result);
        self.update_carry_flag(val1 as u16, val2 as u16, op);
        self.update_auxiliary_carry_flag(val1 as u16, val2 as u16, op);
        self.update_parity_flag(result);

        if op == BinaryOp::ADD || op == BinaryOp::SUB {
            self.set_value(arg1, Operand::IMM16(result as u16));
        }
    }

    // Correct parity check: checks the number of set bits in the lower 8 bits.
    fn update_parity_flag(&mut self, val: i16) {
        let low_byte = (val as u8).count_ones();
        self.flags.parity = low_byte % 2 == 0;
    }

    fn update_zero_flag(&mut self, val: i16) {
        self.flags.zero = val == 0;
    }

    fn update_sign_flag(&mut self, val: i16) {
        self.flags.sign = val < 0;
    }

    // Update the Carry flag based on the operation.
    fn update_carry_flag(&mut self, val1: u16, val2: u16, op: BinaryOp) {
        match op {
            BinaryOp::ADD => {
                let (_res, overflow) = val1.overflowing_add(val2);
                self.flags.carry = overflow;
            }
            BinaryOp::SUB | BinaryOp::CMP => {
                let (_res, underflow) = val1.overflowing_sub(val2);
                self.flags.carry = underflow;
            }
            _ => (),
        }
    }

    // Update the Auxiliary Carry flag based on the lower nibble of the operation.
    fn update_auxiliary_carry_flag(&mut self, val1: u16, val2: u16, op: BinaryOp) {
        match op {
            BinaryOp::ADD => {
                let res = (val1 & 0xF) + (val2 & 0xF);
                self.flags.auxiliary_carry = res > 0xF;
            }
            BinaryOp::SUB | BinaryOp::CMP => {
                let res = (val1 & 0xF) as i16 - (val2 & 0xF) as i16;
                self.flags.auxiliary_carry = res < 0;
            }
            _ => (),
        }
    }

    fn get_operand_value(&self, arg: Operand) -> u16 {
        match arg {
            Operand::IMM8(val) => val as u16,
            Operand::IMM16(val) => val,
            Operand::R(reg) => {
                let (w, i) = reg.get_index();
                if w == 1 {
                    let row = self
                        .register_rows
                        .get(i as usize)
                        .expect("Fatal error: Invalid register index for 16-bit register access.");
                    (row.low as u16) | ((row.high as u16) << 8)
                } else {
                    let high = (i >> 2) != 0;
                    let idx = i & 0b11;
                    let row = self
                        .register_rows
                        .get(idx as usize)
                        .expect("Fatal error: Invalid register index for 8-bit register access.");
                    if high {
                        row.high as u16
                    } else {
                        row.low as u16
                    }
                }
            }
            _ => panic!("Fatal error: Memory addressing mode not yet supported."),
        }
    }

    fn set_value(&mut self, arg1: Operand, arg2: Operand) {
        let val = self.get_operand_value(arg2);
        if let Operand::R(reg) = arg1 {
            let row = self.get_register_row(reg);
            let (w, i) = reg.get_index();
            let low = val as u8;
            let high = (val >> 8) as u8;
            if w == 1 {
                row.low = low;
                row.high = high;
            } else if i < 4 {
                row.low = low;
            } else {
                row.high = low;
            }
        }
    }
}

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
}

impl Register {
    /// Returns the (w_bit, reg_index) for the register.
    /// This corresponds to its position in the `REGISTERS` lookup table.
    pub fn get_index(&self) -> (u8, u8) {
        match self {
            Self::AL => (0, 0),
            Self::CL => (0, 1),
            Self::DL => (0, 2),
            Self::BL => (0, 3),
            Self::AH => (0, 4),
            Self::CH => (0, 5),
            Self::DH => (0, 6),
            Self::BH => (0, 7),
            Self::AX => (1, 0),
            Self::CX => (1, 1),
            Self::DX => (1, 2),
            Self::BX => (1, 3),
            Self::SP => (1, 4),
            Self::BP => (1, 5),
            Self::SI => (1, 6),
            Self::DI => (1, 7),
        }
    }
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
            _ => unreachable!("Invalid arithmetic op encoding: {:b}", op),
        }
    }
}

/// Represents a decoded 8086 instruction.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
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
            panic!("Fatal error: Unexpected end of byte stream while trying to read a byte at position {}", self.pos);
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
            _ => panic!(
                "Fatal error: Unsupported opcode {:#04x} at position {}",
                opcode, start_pos
            ),
        };

        instr
    }

    // --- Instruction Format Decoders ---

    /// Decodes a LOOP, LOOPZ, LOOPNZ, or JCXZ instruction.
    fn decode_loop(&mut self, opcode: u8) -> Instruction {
        let cond_index = (opcode - 0xE0) as usize;
        let cond = *LOOP_CONDITIONS
            .get(cond_index)
            .expect("Fatal error: Invalid loop condition opcode during decoding.");
        let disp = self.read_u8() as i8;
        Instruction::LOOP(cond, disp)
    }

    /// Decodes a conditional jump instruction.
    fn decode_jump(&mut self, opcode: u8) -> Instruction {
        let cond_index = (opcode & 0x0F) as usize;
        let cond = *JUMP_CONDITIONS
            .get(cond_index)
            .expect("Fatal error: Invalid jump condition opcode during decoding.");
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
        let op_code = (mod_rm_byte >> 3) & 0b111;
        let op = BinaryOp::arithmetic_op_from_encoding(op_code);
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
        let dest_reg = REGISTERS
            .get(w as usize)
            .unwrap_or_else(|| {
                panic!(
                    "Fatal error: Invalid W bit ({}) for accumulator register during decoding.",
                    w
                )
            })
            .first()
            .expect("Fatal error: Missing accumulator register in lookup table.");
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
            .unwrap_or_else(|| {
                panic!(
                    "Fatal error: Invalid W bit ({}) for MOV immediate to register.",
                    w
                )
            })
            .get(reg_encoding)
            .unwrap_or_else(|| {
                panic!(
                    "Fatal error: Invalid REG encoding ({}) for MOV immediate to register.",
                    reg_encoding
                )
            });
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
            .unwrap_or_else(|| {
                panic!(
                    "Fatal error: Invalid W bit ({}) for MOV memory to accumulator.",
                    w
                )
            })
            .first()
            .expect("Fatal error: Missing accumulator register in lookup table.");
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
                .unwrap_or_else(|| {
                    panic!("Fatal error: Invalid W bit ({}) during ModR/M decoding.", w)
                })
                .get(reg_encoding as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "Fatal error: Invalid REG encoding ({}) during ModR/M decoding.",
                        reg_encoding
                    )
                }),
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
                .unwrap_or_else(|| {
                    panic!(
                        "Fatal error: Invalid W bit ({}) for R/M register decoding.",
                        w
                    )
                })
                .get(rm)
                .unwrap_or_else(|| {
                    panic!("Fatal error: Invalid R/M encoding ({}) for register.", rm)
                });
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
            _ => unreachable!("Invalid MOD field during ModR/M decoding: {:b}", md),
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
            _ => panic!(
                "Fatal error: Invalid R/M encoding for MOD=00 with no displacement: {}",
                rm
            ),
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
            _ => unreachable!("Invalid R/M encoding for MOD=01 with 8-bit displacement."),
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
            _ => unreachable!("Invalid R/M encoding for MOD=10 with 16-bit displacement."),
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
            execute(&buffer);
            let decoded_instructions = decode(&buffer);
            match write_decoding(&decoded_instructions, output_file) {
                Ok(_) => println!("Decoding complete! Output written to '{}'.", output_file),
                Err(e) => eprintln!("Error writing decoded instructions: {}", e),
            }
        }
        Err(e) => eprintln!("Error reading input file: {}", e),
    }
}


use std::fmt;
use strum_macros::Display;

pub const REGISTERS: [[Register; 8]; 2] = [
    // w=0: 8-bit registers
    [
        Register::Al,
        Register::Cl,
        Register::Dl,
        Register::Bl,
        Register::Ah,
        Register::Ch,
        Register::Dh,
        Register::Bh,
    ],
    // w=1: 16-bit registers
    [
        Register::Ax,
        Register::Cx,
        Register::Dx,
        Register::Bx,
        Register::Sp,
        Register::Bp,
        Register::Si,
        Register::Di,
    ],
];

#[derive(Debug, PartialEq, Eq, Copy, Clone, Display, Hash)]
pub enum Register {
    // 8-bit registers (w=0)
    #[strum(serialize = "al")]
    Al,
    #[strum(serialize = "cl")]
    Cl,
    #[strum(serialize = "dl")]
    Dl,
    #[strum(serialize = "bl")]
    Bl,
    #[strum(serialize = "ah")]
    Ah,
    #[strum(serialize = "ch")]
    Ch,
    #[strum(serialize = "dh")]
    Dh,
    #[strum(serialize = "bh")]
    Bh,
    // 16-bit registers (w=1)
    #[strum(serialize = "ax")]
    Ax,
    #[strum(serialize = "cx")]
    Cx,
    #[strum(serialize = "dx")]
    Dx,
    #[strum(serialize = "bx")]
    Bx,
    #[strum(serialize = "sp")]
    Sp,
    #[strum(serialize = "bp")]
    Bp,
    #[strum(serialize = "si")]
    Si,
    #[strum(serialize = "di")]
    Di,
}

impl Register {
    pub fn get_index(&self) -> (u8, u8) {
        match self {
            Self::Al => (0, 0),
            Self::Cl => (0, 1),
            Self::Dl => (0, 2),
            Self::Bl => (0, 3),
            Self::Ah => (0, 4),
            Self::Ch => (0, 5),
            Self::Dh => (0, 6),
            Self::Bh => (0, 7),
            Self::Ax => (1, 0),
            Self::Cx => (1, 1),
            Self::Dx => (1, 2),
            Self::Bx => (1, 3),
            Self::Sp => (1, 4),
            Self::Bp => (1, 5),
            Self::Si => (1, 6),
            Self::Di => (1, 7),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MemoryAddress {
    R(Register),
    Rd8(Register, u8),
    Rd16(Register, u16),
    Rr(Register, Register),
    Rrd8(Register, Register, u8),
    Rrd16(Register, Register, u16),
    Da(u16),
}

impl fmt::Display for MemoryAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::R(reg) => write!(f, "[{}]", reg),
            Self::Rr(r1, r2) => write!(f, "[{} + {}]", r1, r2),
            Self::Da(addr) => write!(f, "[{}]", addr),
            Self::Rd8(reg, offset) => {
                let disp = *offset as i8;
                if disp >= 0 {
                    write!(f, "[{} + {}]", reg, disp)
                } else {
                    write!(f, "[{} - {}]", reg, -disp)
                }
            }
            Self::Rd16(reg, offset) => {
                let disp = *offset as i16;
                if disp >= 0 {
                    write!(f, "[{} + {}]", reg, disp)
                } else {
                    write!(f, "[{} - {}]", reg, -disp)
                }
            }
            Self::Rrd8(r1, r2, offset) => {
                let disp = *offset as i8;
                if disp >= 0 {
                    write!(f, "[{} + {} + {}]", r1, r2, disp)
                } else {
                    write!(f, "[{} + {} - {}]", r1, r2, -disp)
                }
            }
            Self::Rrd16(r1, r2, offset) => {
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

impl MemoryAddress {
    /// Calculates the Effective Address (EA) clock cycles for the given memory addressing mode.
    /// Based on Intel 8086 User's Manual, Table 2-10.
    pub fn calculate_ea_clocks(&self) -> u32 {
        match self {
            MemoryAddress::R(reg) => {
                // [BX], [BP], [SI], [DI]
                match reg {
                    Register::Bx | Register::Bp | Register::Si | Register::Di => 5,
                    _ => {
                        // This case should ideally not be reachable if MemoryAddress::R is only used for valid base/index registers.
                        // For robustness, return 0.
                        panic!(
                            "Warning: Invalid register {:?} used in MemoryAddress::R for EA calculation.",
                            reg,
                        );
                    }
                }
            }
            MemoryAddress::Rr(r1, r2) => {
                // [BX + SI], [BP + DI], [BX + DI], [BP + SI]
                match (r1, r2) {
                    (Register::Bx, Register::Si) | (Register::Bp, Register::Di) => 7,
                    (Register::Bx, Register::Di) | (Register::Bp, Register::Si) => 8,
                    _ => {
                        eprintln!(
                            "Warning: Invalid register combination {:?}, {:?} used in MemoryAddress::Rr for EA calculation.",
                            r1, r2
                        );
                        0
                    }
                }
            }
            MemoryAddress::Da(_) => 6, // [disp16]
            MemoryAddress::Rd8(reg, _) | MemoryAddress::Rd16(reg, _) => {
                // [BX + disp8/16], [BP + disp8/16], [SI + disp8/16], [DI + disp8/16]
                match reg {
                    Register::Bx | Register::Bp | Register::Si | Register::Di => 9,
                    _ => {
                        panic!(
                            "Warning: Invalid register {:?} used in MemoryAddress::Rd8/Rd16 for EA calculation.",
                            reg,
                        );
                    }
                }
            }
            MemoryAddress::Rrd8(r1, r2, _) | MemoryAddress::Rrd16(r1, r2, _) => {
                // [BX + SI + disp8/16], [BP + DI + disp8/16], [BX + DI + disp8/16], [BP + SI + disp8/16]
                match (r1, r2) {
                    (Register::Bx, Register::Si) | (Register::Bp, Register::Di) => 11,
                    (Register::Bx, Register::Di) | (Register::Bp, Register::Si) => 12,
                    _ => {
                        panic!(
                            "Warning: Invalid register combination {:?}, {:?} used in MemoryAddress::Rrd8/Rrd16 for EA calculation.",
                            r1, r2,
                        );
                    }
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Operand {
    R(Register),
    Ma(MemoryAddress),
    Imm8(u8),
    Imm16(u16),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::R(reg) => write!(f, "{}", reg),
            Self::Ma(addr) => write!(f, "{}", addr),
            Self::Imm8(val) => write!(f, "byte {}", val),
            Self::Imm16(val) => write!(f, "word {}", val),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Display)]
pub enum JumpCondition {
    #[strum(serialize = "jo")]
    Jo,
    #[strum(serialize = "jno")]
    Jno,
    #[strum(serialize = "jb")]
    Jb,
    #[strum(serialize = "jnb")]
    Jnb,
    #[strum(serialize = "jz")]
    Jz,
    #[strum(serialize = "jnz")]
    Jnz,
    #[strum(serialize = "jbe")]
    Jbe,
    #[strum(serialize = "ja")]
    Ja,
    #[strum(serialize = "js")]
    Js,
    #[strum(serialize = "jns")]
    Jns,
    #[strum(serialize = "jp")]
    Jp,
    #[strum(serialize = "jnp")]
    Jnp,
    #[strum(serialize = "jl")]
    Jl,
    #[strum(serialize = "jge")]
    Jge,
    #[strum(serialize = "jle")]
    Jle,
    #[strum(serialize = "jg")]
    Jg,
}

pub const JUMP_CONDITIONS: [JumpCondition; 16] = [
    JumpCondition::Jo,
    JumpCondition::Jno,
    JumpCondition::Jb,
    JumpCondition::Jnb,
    JumpCondition::Jz,
    JumpCondition::Jnz,
    JumpCondition::Jbe,
    JumpCondition::Ja,
    JumpCondition::Js,
    JumpCondition::Jns,
    JumpCondition::Jp,
    JumpCondition::Jnp,
    JumpCondition::Jl,
    JumpCondition::Jge,
    JumpCondition::Jle,
    JumpCondition::Jg,
];

#[derive(Debug, PartialEq, Eq, Copy, Clone, Display)]
pub enum LoopCondition {
    #[strum(serialize = "loopnz")]
    Loopnz,
    #[strum(serialize = "loopz")]
    Loopz,
    #[strum(serialize = "loop")]
    Loop,
    #[strum(serialize = "jcxz")]
    Jcxz,
}

pub const LOOP_CONDITIONS: [LoopCondition; 4] = [
    LoopCondition::Loopnz,
    LoopCondition::Loopz,
    LoopCondition::Loop,
    LoopCondition::Jcxz,
];

#[derive(Debug, PartialEq, Eq, Copy, Clone, Display)]
pub enum BinaryOp {
    #[strum(serialize = "mov")]
    Mov,
    #[strum(serialize = "add")]
    Add,
    #[strum(serialize = "sub")]
    Sub,
    #[strum(serialize = "cmp")]
    Cmp,
}

impl BinaryOp {
    pub fn arithmetic_op_from_encoding(op: u8) -> Self {
        match op {
            0b000 => BinaryOp::Add,
            0b101 => BinaryOp::Sub,
            0b111 => BinaryOp::Cmp,
            _ => panic!("Invalid arithmetic op encoding: {:b}", op),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Instruction {
    Bo(BinaryOp, Operand, Operand),
    Jmp(JumpCondition, i8),
    Loop(LoopCondition, i8),
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bo(op, dest, src) => write!(f, "{} {}, {}", op, dest, src),
            Self::Jmp(cond, disp) => {
                let target_offset = *disp as i16 + 2;
                write!(f, "{} short ${:+}", cond, target_offset)
            }
            Self::Loop(cond, disp) => {
                let target_offset = *disp as i16 + 2;
                write!(f, "{} ${:+}", cond, target_offset)
            }
        }
    }
}

impl Instruction {
    /// Returns a tuple (opcode_clocks, effective_address_clocks) for the instruction.
    /// Clock cycles are based on the Intel 8086 User's Manual.
    /// Only supports MOV, ADD, SUB instructions.
    pub fn to_clocks(self) -> (u32, u32) {
        match self {
            Instruction::Bo(op, dest, src) => {
                let opcode_clocks;
                let mut ea_clocks = 0;

                match op {
                    BinaryOp::Mov => {
                        match (dest, src) {
                            (Operand::R(_), Operand::R(_)) => {
                                // MOV R, R
                                opcode_clocks = 2;
                            }
                            (Operand::R(d_reg), Operand::Ma(s_mem)) => {
                                // MOV R, M
                                // Special case: MOV AL/AX, [disp16] (direct address)
                                if (d_reg == Register::Al || d_reg == Register::Ax)
                                    && matches!(s_mem, MemoryAddress::Da(_))
                                {
                                    opcode_clocks = 10;
                                } else {
                                    opcode_clocks = 8;
                                    ea_clocks = s_mem.calculate_ea_clocks();
                                }
                            }
                            (Operand::Ma(d_mem), Operand::R(s_reg)) => {
                                // MOV M, R
                                // Special case: MOV [disp16], AL/AX (direct address)
                                if (s_reg == Register::Al || s_reg == Register::Ax)
                                    && matches!(d_mem, MemoryAddress::Da(_))
                                {
                                    opcode_clocks = 10;
                                } else {
                                    opcode_clocks = 9;
                                    ea_clocks = d_mem.calculate_ea_clocks();
                                }
                            }
                            (Operand::R(_d_reg), Operand::Imm8(_))
                            | (Operand::R(_d_reg), Operand::Imm16(_)) => {
                                // MOV R, I
                                opcode_clocks = 4;
                            }
                            (Operand::Ma(d_mem), Operand::Imm8(_))
                            | (Operand::Ma(d_mem), Operand::Imm16(_)) => {
                                // MOV M, I
                                opcode_clocks = 10;
                                ea_clocks = d_mem.calculate_ea_clocks();
                            }
                            _ => {
                                eprintln!(
                                    "Warning: Unhandled MOV instruction operand combination: dest={:?}, src={:?}",
                                    dest, src
                                );
                                // Default to 0,0 for unhandled combinations
                                return (0, 0);
                            }
                        }
                    }
                    BinaryOp::Add | BinaryOp::Sub => {
                        match (dest, src) {
                            (Operand::R(_), Operand::R(_)) => {
                                // ADD/SUB R, R
                                opcode_clocks = 3;
                            }
                            (Operand::R(_), Operand::Ma(s_mem)) => {
                                // ADD/SUB R, M
                                opcode_clocks = 9;
                                ea_clocks = s_mem.calculate_ea_clocks();
                            }
                            (Operand::Ma(d_mem), Operand::R(_)) => {
                                // ADD/SUB M, R
                                opcode_clocks = 16;
                                ea_clocks = d_mem.calculate_ea_clocks();
                            }
                            (Operand::R(_), Operand::Imm8(_))
                            | (Operand::R(_), Operand::Imm16(_)) => {
                                // ADD/SUB R, I
                                // The manual lists 4 clocks for both accumulator and general register immediate.
                                opcode_clocks = 4;
                            }
                            (Operand::Ma(d_mem), Operand::Imm8(_))
                            | (Operand::Ma(d_mem), Operand::Imm16(_)) => {
                                // ADD/SUB M, I
                                opcode_clocks = 17;
                                ea_clocks = d_mem.calculate_ea_clocks();
                            }
                            _ => {
                                eprintln!(
                                    "Warning: Unhandled ADD/SUB instruction operand combination: dest={:?}, src={:?}",
                                    dest, src
                                );
                                return (0, 0);
                            }
                        }
                    }
                    BinaryOp::Cmp => {
                        eprintln!("Warning: CMP instruction clock cycles not implemented.");
                        return (0, 0);
                    }
                }
                (opcode_clocks, ea_clocks)
            }
            Instruction::Jmp(_, _) | Instruction::Loop(_, _) => {
                eprintln!("Warning: JMP/LOOP instruction clock cycles not implemented.");
                (0, 0)
            }
        }
    }
}

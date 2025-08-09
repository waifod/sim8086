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


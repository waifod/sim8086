use crate::instruction::{
    BinaryOp, Instruction, MemoryAddress, Operand, JUMP_CONDITIONS, LOOP_CONDITIONS, REGISTERS,
};
use std::collections::HashMap;

pub struct Decoder<'a> {
    bytes: &'a [u8],
    pub pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub fn decode_all(&mut self) -> HashMap<usize, Instruction> {
        let mut instructions = HashMap::new();
        while !self.is_eof() {
            let start_pos = self.pos;
            let instruction = self.decode_next_instruction();
            instructions.insert(start_pos, instruction);
        }
        instructions
    }

    pub fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn read_u8(&mut self) -> u8 {
        if self.pos >= self.bytes.len() {
            panic!("Fatal error: Unexpected end of byte stream while trying to read a byte at position {}", self.pos);
        }
        let byte = self.bytes[self.pos];
        self.pos += 1;
        byte
    }

    fn read_u16_le(&mut self) -> u16 {
        let low = self.read_u8() as u16;
        let high = self.read_u8() as u16;
        low | (high << 8)
    }

    pub fn decode_next_instruction(&mut self) -> Instruction {
        let start_pos = self.pos;
        let opcode = self.read_u8();
        match opcode {
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
        }
    }

    fn decode_mov_imm_to_rm(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1) != 0;
        let mod_rm_byte = self.read_u8();

        let dest = self.decode_rm_operand(mod_rm_byte, w);

        let src = if w {
            Operand::Imm16(self.read_u16_le())
        } else {
            Operand::Imm8(self.read_u8())
        };

        Instruction::Bo(BinaryOp::Mov, dest, src)
    }

    fn decode_loop(&mut self, opcode: u8) -> Instruction {
        let cond_index = (opcode - 0xE0) as usize;
        let cond = *LOOP_CONDITIONS
            .get(cond_index)
            .expect("Fatal error: Invalid loop condition opcode during decoding.");
        let disp = self.read_u8() as i8;
        Instruction::Loop(cond, disp)
    }

    fn decode_jump(&mut self, opcode: u8) -> Instruction {
        let cond_index = (opcode & 0x0F) as usize;
        let cond = *JUMP_CONDITIONS
            .get(cond_index)
            .expect("Fatal error: Invalid jump condition opcode during decoding.");
        let disp = self.read_u8() as i8;
        Instruction::Jmp(cond, disp)
    }

    fn decode_arithmetic_reg_mem(&mut self, opcode: u8) -> Instruction {
        let op = BinaryOp::arithmetic_op_from_encoding((opcode >> 3) & 0b111);
        let d = (opcode & 0b10) != 0;
        let w = (opcode & 0b1) != 0;
        let (dest, src) = self.decode_reg_and_rm_operands(d, w);
        Instruction::Bo(op, dest, src)
    }

    fn decode_mov_reg_mem(&mut self, opcode: u8) -> Instruction {
        let d = (opcode & 0b10) != 0;
        let w = (opcode & 0b1) != 0;
        let (dest, src) = self.decode_reg_and_rm_operands(d, w);
        Instruction::Bo(BinaryOp::Mov, dest, src)
    }

    fn decode_arithmetic_imm_to_rm(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1) != 0;
        let mod_rm_byte = self.read_u8();
        let op_code = (mod_rm_byte >> 3) & 0b111;
        let op = BinaryOp::arithmetic_op_from_encoding(op_code);
        let dest = self.decode_rm_operand(mod_rm_byte, w);
        let src = if w {
            if opcode == 0x83 {
                Operand::Imm16(self.read_u8() as i8 as u16)
            } else {
                Operand::Imm16(self.read_u16_le())
            }
        } else {
            Operand::Imm8(self.read_u8())
        };
        Instruction::Bo(op, dest, src)
    }

    fn decode_arithmetic_imm_to_acc(&mut self, opcode: u8) -> Instruction {
        let op = BinaryOp::arithmetic_op_from_encoding((opcode >> 3) & 0b111);
        let w = (opcode & 0b1) != 0;
        let dest_reg = REGISTERS
            .get(w as usize)
            .expect("Fatal error: Invalid W bit for accumulator register during decoding.")
            .first()
            .expect("Fatal error: Missing accumulator register in lookup table.");
        let dest = Operand::R(*dest_reg);
        let src = if w {
            Operand::Imm16(self.read_u16_le())
        } else {
            Operand::Imm8(self.read_u8())
        };
        Instruction::Bo(op, dest, src)
    }

    fn decode_mov_imm_to_reg(&mut self, opcode: u8) -> Instruction {
        let w = (opcode & 0b1000) != 0;
        let reg_encoding = (opcode & 0b111) as usize;
        let dest_reg = REGISTERS
            .get(w as usize)
            .expect("Fatal error: Invalid W bit for MOV immediate to register.")
            .get(reg_encoding)
            .expect("Fatal error: Invalid REG encoding for MOV immediate to register.");
        let dest = Operand::R(*dest_reg);
        let src = if w {
            Operand::Imm16(self.read_u16_le())
        } else {
            Operand::Imm8(self.read_u8())
        };
        Instruction::Bo(BinaryOp::Mov, dest, src)
    }

    fn decode_mov_mem_accumulator(&mut self, opcode: u8) -> Instruction {
        let d = (opcode & 0b10) == 0;
        let w = (opcode & 0b1) != 0;
        let acc_reg = REGISTERS
            .get(w as usize)
            .expect("Fatal error: Invalid W bit for MOV memory to accumulator.")
            .first()
            .expect("Fatal error: Missing accumulator register in lookup table.");
        let acc_op = Operand::R(*acc_reg);
        let mem_op = Operand::Ma(MemoryAddress::Da(self.read_u16_le()));
        let (dest, src) = if d {
            (acc_op, mem_op)
        } else {
            (mem_op, acc_op)
        };
        Instruction::Bo(BinaryOp::Mov, dest, src)
    }

    fn decode_reg_and_rm_operands(&mut self, d: bool, w: bool) -> (Operand, Operand) {
        let mod_rm_byte = self.read_u8();
        let reg_encoding = (mod_rm_byte >> 3) & 0b111;
        let reg_op = Operand::R(
            *REGISTERS
                .get(w as usize)
                .expect("Fatal error: Invalid W bit during ModR/M decoding.")
                .get(reg_encoding as usize)
                .expect("Fatal error: Invalid REG encoding during ModR/M decoding."),
        );
        let rm_op = self.decode_rm_operand(mod_rm_byte, w);

        if d {
            (reg_op, rm_op)
        } else {
            (rm_op, reg_op)
        }
    }

    fn decode_rm_operand(&mut self, mod_rm_byte: u8, w: bool) -> Operand {
        let mod_field = mod_rm_byte >> 6;
        let rm_field = (mod_rm_byte & 0b111) as usize;
        if mod_field == 0b11 {
            let reg = REGISTERS
                .get(w as usize)
                .expect("Fatal error: Invalid W bit for R/M register decoding.")
                .get(rm_field)
                .expect("Fatal error: Invalid R/M encoding for register.");
            return Operand::R(*reg);
        }
        let addr = match mod_field {
            0b00 => {
                if rm_field == 0b110 {
                    MemoryAddress::Da(self.read_u16_le())
                } else {
                    Self::decode_effective_address_no_disp(rm_field as u8)
                }
            }
            0b01 => Self::decode_effective_address_disp8(rm_field as u8, self.read_u8()),
            0b10 => Self::decode_effective_address_disp16(rm_field as u8, self.read_u16_le()),
            _ => panic!("Invalid MOD field during ModR/M decoding: {:b}", mod_field),
        };
        Operand::Ma(addr)
    }

    fn decode_effective_address_no_disp(rm_field: u8) -> MemoryAddress {
        match rm_field {
            0b000 => MemoryAddress::Rr(REGISTERS[1][3], REGISTERS[1][6]),
            0b001 => MemoryAddress::Rr(REGISTERS[1][3], REGISTERS[1][7]),
            0b010 => MemoryAddress::Rr(REGISTERS[1][5], REGISTERS[1][6]),
            0b011 => MemoryAddress::Rr(REGISTERS[1][5], REGISTERS[1][7]),
            0b100 => MemoryAddress::R(REGISTERS[1][6]),
            0b101 => MemoryAddress::R(REGISTERS[1][7]),
            0b111 => MemoryAddress::R(REGISTERS[1][3]),
            _ => panic!(
                "Fatal error: Invalid R/M encoding for MOD=00 with no displacement: {}",
                rm_field
            ),
        }
    }

    fn decode_effective_address_disp8(rm_field: u8, disp: u8) -> MemoryAddress {
        match rm_field {
            0b000 => MemoryAddress::Rrd8(REGISTERS[1][3], REGISTERS[1][6], disp),
            0b001 => MemoryAddress::Rrd8(REGISTERS[1][3], REGISTERS[1][7], disp),
            0b010 => MemoryAddress::Rrd8(REGISTERS[1][5], REGISTERS[1][6], disp),
            0b011 => MemoryAddress::Rrd8(REGISTERS[1][5], REGISTERS[1][7], disp),
            0b100 => MemoryAddress::Rd8(REGISTERS[1][6], disp),
            0b101 => MemoryAddress::Rd8(REGISTERS[1][7], disp),
            0b110 => MemoryAddress::Rd8(REGISTERS[1][5], disp),
            0b111 => MemoryAddress::Rd8(REGISTERS[1][3], disp),
            _ => panic!("Invalid R/M encoding for MOD=01 with 8-bit displacement."),
        }
    }

    fn decode_effective_address_disp16(rm_field: u8, disp: u16) -> MemoryAddress {
        match rm_field {
            0b000 => MemoryAddress::Rrd16(REGISTERS[1][3], REGISTERS[1][6], disp),
            0b001 => MemoryAddress::Rrd16(REGISTERS[1][3], REGISTERS[1][7], disp),
            0b010 => MemoryAddress::Rrd16(REGISTERS[1][5], REGISTERS[1][6], disp),
            0b011 => MemoryAddress::Rrd16(REGISTERS[1][5], REGISTERS[1][7], disp),
            0b100 => MemoryAddress::Rd16(REGISTERS[1][6], disp),
            0b101 => MemoryAddress::Rd16(REGISTERS[1][7], disp),
            0b110 => MemoryAddress::Rd16(REGISTERS[1][5], disp),
            0b111 => MemoryAddress::Rd16(REGISTERS[1][3], disp),
            _ => panic!("Invalid R/M encoding for MOD=10 with 16-bit displacement."),
        }
    }
}

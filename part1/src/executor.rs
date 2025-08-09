use crate::decoder::Decoder;
use crate::instruction::{
    BinaryOp, Instruction, JumpCondition, LoopCondition, MemoryAddress, Operand, Register,
};
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct RegisterRow {
    low: u8,
    high: u8,
}

impl Display for RegisterRow {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{:02x} {:02x}", self.high, self.low)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Flags {
    parity: bool,
    zero: bool,
    sign: bool,
    carry: bool,
    auxiliary_carry: bool,
}

impl fmt::Display for Flags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.parity {
            write!(f, "P")?;
        }
        if self.zero {
            write!(f, "Z")?;
        }
        if self.sign {
            write!(f, "S")?;
        }
        if self.carry {
            write!(f, "C")?;
        }
        if self.auxiliary_carry {
            write!(f, "A")?;
        }
        Ok(())
    }
}

pub struct Executor<'a> {
    instructions: HashMap<usize, Instruction>,
    bytes: &'a [u8],
    memory: [u8; 1024 * 1024],
    register_rows: [RegisterRow; 8],
    flags: Flags,
    ip: usize,
}

impl<'a> Executor<'a> {
    pub fn new(
        decoded_instructions: HashMap<usize, Instruction>,
        original_bytes: &'a [u8],
    ) -> Self {
        Self {
            instructions: decoded_instructions,
            bytes: original_bytes,
            memory: [0; 1024 * 1024],
            register_rows: [RegisterRow { low: 0, high: 0 }; 8],
            flags: Flags {
                parity: false,
                zero: false,
                sign: false,
                carry: false,
                auxiliary_carry: false,
            },
            ip: 0,
        }
    }

    pub fn run(&mut self) {
        println!("\nStart of execution log:");
        println!("------------------------");
        while self.ip < self.bytes.len() {
            let instruction = *self.instructions
                .get(&self.ip)
                .expect("Fatal error: No instruction found at current IP. This may indicate an invalid jump target.");

            let start_ip = self.ip;
            let start_flags = self.flags;

            let instruction_length = self.get_instruction_length(start_ip);

            let initial_state = self.get_operand_values_for_log(&instruction);
            self.execute_instruction(&instruction);
            let final_state = self.get_operand_values_for_log(&instruction);

            if self.ip == start_ip {
                self.ip += instruction_length;
            }

            let end_ip = self.ip;
            let end_flags = self.flags;

            self.log_instruction_execution(
                &instruction,
                start_ip,
                end_ip,
                initial_state,
                final_state,
                start_flags,
                end_flags,
            );
        }
        println!("------------------------");

        println!("\nFinal registers:");
        let f = |r: Register| {
            let val = self.get_16bit_register_value(r);
            if val != 0 {
                println!("     {}: 0x{:04x} ({})", r, val, val);
            }
        };
        f(Register::Ax);
        f(Register::Cx);
        f(Register::Dx);
        f(Register::Bx);
        f(Register::Sp);
        f(Register::Bp);
        f(Register::Si);
        f(Register::Di);
        println!("     ip: 0x{:04x} ({})", self.ip, self.ip);
        println!("  flags: {}\n", self.flags);
    }

    fn get_instruction_length(&self, byte_offset: usize) -> usize {
        let mut temp_decoder = Decoder::new(&self.bytes[byte_offset..]);
        temp_decoder.decode_next_instruction();
        temp_decoder.pos
    }

    fn log_instruction_execution(
        &self,
        instruction: &Instruction,
        start_ip: usize,
        end_ip: usize,
        initial_state: Vec<(Operand, u16)>,
        final_state: Vec<(Operand, u16)>,
        start_flags: Flags,
        end_flags: Flags,
    ) {
        let mut log_line = format!("{}", instruction);
        let mut state_changes = Vec::new();

        for i in 0..initial_state.len() {
            let (op, initial_val) = &initial_state[i];
            let (_, final_val) = &final_state[i];
            if initial_val != final_val {
                if let Operand::R(reg) = op {
                    state_changes.push(format!("{}:0x{:x}->0x{:x}", reg, initial_val, final_val));
                }
            }
        }

        if start_ip != end_ip {
            state_changes.push(format!("ip:0x{:x}->0x{:x}", start_ip, end_ip));
        }

        if start_flags != end_flags {
            let start_flags_str = format!("{}", start_flags);
            let end_flags_str = format!("{}", end_flags);
            state_changes.push(format!("flags:{}->{}", start_flags_str, end_flags_str));
        }

        if !state_changes.is_empty() {
            log_line.push_str(&format!(" ; {}", state_changes.join(" ")));
        }

        println!("{}", log_line);
    }

    fn get_operand_values_for_log(&self, instruction: &Instruction) -> Vec<(Operand, u16)> {
        let mut values = Vec::new();
        // This function logs the state of the destination operand.
        // It's assumed to be a binary op with a destination.
        if let Instruction::Bo(_, dest, _) = instruction {
            if let Operand::R(reg) = dest {
                let (w, _) = reg.get_index();
                values.push((*dest, self.get_operand_value(*dest, w)));
            }
        }
        values
    }

    fn get_register_row(&mut self, reg: Register) -> &mut RegisterRow {
        let (_w, i) = reg.get_index();
        let idx = i as usize;
        self.register_rows
            .get_mut(idx)
            .expect("Fatal error: Attempted to get register row with invalid index.")
    }

    fn get_16bit_register_value(&self, reg: Register) -> u16 {
        let (_w, i) = reg.get_index();
        let idx = i as usize;
        let row = self
            .register_rows
            .get(idx)
            .expect("Fatal error: Invalid register index for 16-bit register access.");
        (row.high as u16) << 8 | (row.low as u16)
    }

    fn set_16bit_register_value(&mut self, reg: Register, val: u16) {
        let row = self.get_register_row(reg);
        row.low = val as u8;
        row.high = (val >> 8) as u8;
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
        let row = self
            .register_rows
            .get(idx as usize)
            .expect("Fatal error: Invalid register index for 8-bit register access.");
        if (i >> 2) != 0 {
            row.high
        } else {
            row.low
        }
    }

    fn set_8bit_register_low_value(&mut self, reg: Register, val: u8) {
        let row = self.get_register_row(reg);
        row.low = val;
    }

    fn set_8bit_register_high_value(&mut self, reg: Register, val: u8) {
        let row = self.get_register_row(reg);
        row.high = val;
    }

    fn execute_instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Bo(op, arg1, arg2) => self.execute_bo_instruction(*op, *arg1, *arg2),
            Instruction::Jmp(op, arg) => self.execute_jump_instruction(*op, *arg),
            Instruction::Loop(op, arg) => self.execute_loop_instruction(*op, *arg),
        }
    }

    fn execute_loop_instruction(&mut self, op: LoopCondition, arg: i8) {
        let cx_val = self.get_16bit_register_value(Register::Cx);
        let new_cx_val = cx_val - 1;
        self.set_16bit_register_value(Register::Cx, new_cx_val);

        let should_loop = match op {
            LoopCondition::Loop => new_cx_val != 0,
            _ => panic!("Loop condition {:?} not supported", op),
        };

        if should_loop {
            let displacement = arg as isize;
            self.ip = (self.ip as isize + 2 + displacement) as usize;
        }
    }

    fn execute_jump_instruction(&mut self, op: JumpCondition, arg: i8) {
        let should_jump = match op {
            JumpCondition::Jz => self.flags.zero,
            JumpCondition::Jnz => !self.flags.zero,
            _ => panic!("Jump condition {:?} not supported", op),
        };

        if should_jump {
            let displacement = arg as isize;
            self.ip = (self.ip as isize + 2 + displacement) as usize;
        }
    }

    fn execute_bo_instruction(&mut self, op: BinaryOp, arg1: Operand, arg2: Operand) {
        // We determine the width of the operation from the destination operand (arg1).
        let width = match arg1 {
            Operand::R(reg) => reg.get_index().0,
            Operand::Ma(_) => {
                // If the destination is memory, we assume the width is determined by the source operand (arg2)
                match arg2 {
                    Operand::R(reg) => reg.get_index().0,
                    Operand::Imm8(_) => 0,
                    Operand::Imm16(_) => 1,
                    _ => panic!("Fatal error: Cannot infer operand width from {:?}", arg2),
                }
            }
            _ => panic!(
                "Fatal error: Invalid destination operand for binary op {:?}",
                arg1
            ),
        };

        match op {
            BinaryOp::Mov => self.execute_mov_instruction(arg1, arg2, width),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Cmp => {
                self.execute_arithmetic_instruction(op, arg1, arg2, width)
            }
        }
    }

    fn execute_mov_instruction(&mut self, dest: Operand, src: Operand, width: u8) {
        let val = self.get_operand_value(src, width);
        self.set_operand_value(dest, val, width);
    }

    fn execute_arithmetic_instruction(
        &mut self,
        op: BinaryOp,
        arg1: Operand,
        arg2: Operand,
        width: u8,
    ) {
        let val1 = self.get_operand_value(arg1, width);
        let val2 = self.get_operand_value(arg2, width);

        let result: i16;
        let carry_flag_val: bool;
        let auxiliary_carry_flag_val: bool;

        match op {
            BinaryOp::Add => {
                let (res, overflow) = (val1 as i16).overflowing_add(val2 as i16);
                carry_flag_val = overflow;
                result = res;

                let res_nibble = (val1 & 0xF) + (val2 & 0xF);
                auxiliary_carry_flag_val = res_nibble > 0xF;
            }
            BinaryOp::Sub | BinaryOp::Cmp => {
                let (res, overflow) = (val1 as i16).overflowing_sub(val2 as i16);
                carry_flag_val = overflow;
                result = res;

                let res_nibble = (val1 & 0xF) as i16 - (val2 & 0xF) as i16;
                auxiliary_carry_flag_val = res_nibble < 0;
            }
            _ => unreachable!(),
        };

        self.flags.carry = carry_flag_val;
        self.flags.auxiliary_carry = auxiliary_carry_flag_val;
        self.flags.zero = result == 0;
        self.flags.sign = result < 0;
        self.update_parity_flag(result);

        if op == BinaryOp::Add || op == BinaryOp::Sub {
            self.set_operand_value(arg1, result as u16, width);
        }
    }

    fn update_parity_flag(&mut self, val: i16) {
        let low_byte = (val as u8).count_ones();
        self.flags.parity = low_byte % 2 == 0;
    }

    // Retrieves the value from an operand, respecting the specified width (0 for 8-bit, 1 for 16-bit).
    fn get_operand_value(&self, arg: Operand, width: u8) -> u16 {
        match arg {
            Operand::Imm8(val) => val as u16,
            Operand::Imm16(val) => val,
            Operand::R(reg) => {
                // The get_register_value function already handles the width
                let (w, _) = reg.get_index();
                if w == 1 {
                    self.get_16bit_register_value(reg)
                } else {
                    self.get_8bit_register_value(reg) as u16
                }
            }
            Operand::Ma(addr) => {
                let index = self.get_memory_index(addr);
                let mem_slice = self
                    .memory
                    .get(index..)
                    .expect("Memory access out of bounds");
                if width == 0 {
                    mem_slice[0] as u16
                } else {
                    (mem_slice[1] as u16) << 8 | (mem_slice[0] as u16)
                }
            }
        }
    }

    fn get_memory_index(&self, addr: MemoryAddress) -> usize {
        (match addr {
            MemoryAddress::Rd8(reg, disp) => self.get_16bit_register_value(reg) + (disp as u16),
            MemoryAddress::Rd16(reg, disp) => self.get_16bit_register_value(reg) + disp,
            MemoryAddress::Rr(reg1, reg2) => {
                self.get_16bit_register_value(reg1) + self.get_16bit_register_value(reg2)
            }
            MemoryAddress::Rrd8(reg1, reg2, disp) => {
                self.get_16bit_register_value(reg1)
                    + self.get_16bit_register_value(reg2)
                    + (disp as u16)
            }
            MemoryAddress::Rrd16(reg1, reg2, disp) => {
                self.get_16bit_register_value(reg1) + self.get_16bit_register_value(reg2) + disp
            }
            MemoryAddress::Da(addr) => addr,
            MemoryAddress::R(reg) => self.get_16bit_register_value(reg),
        }) as usize
    }

    // A helper function that writes a value to an operand, respecting the width (0 for 8-bit, 1 for 16-bit).
    fn set_operand_value(&mut self, arg: Operand, val: u16, width: u8) {
        match arg {
            Operand::R(reg) => {
                let (w, i) = reg.get_index();
                if w == 1 {
                    self.set_16bit_register_value(reg, val);
                } else if i < 4 {
                    self.set_8bit_register_low_value(reg, val as u8);
                } else {
                    self.set_8bit_register_high_value(reg, val as u8);
                }
            }
            Operand::Ma(addr) => {
                let index = self.get_memory_index(addr);
                if width == 0 {
                    let mem_byte = self
                        .memory
                        .get_mut(index)
                        .expect("Memory access out of bounds");
                    *mem_byte = val as u8;
                } else {
                    let mem_slice = self
                        .memory
                        .get_mut(index..index + 2)
                        .expect("Memory access out of bounds");
                    mem_slice[0] = val as u8;
                    mem_slice[1] = (val >> 8) as u8;
                }
            }
            _ => panic!("Fatal error: Invalid destination operand for set_operand_value."),
        }
    }
}

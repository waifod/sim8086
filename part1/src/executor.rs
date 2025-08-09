use crate::decoder::Decoder;
use crate::instruction::{
    BinaryOp, Instruction, JumpCondition, LoopCondition, MemoryAddress, Operand, Register,
};
use std::fmt::{self, Display, Formatter};

// --- Helper Structs for State Management ---

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct RegisterRow {
    low: u8,
    high: u8,
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Flags {
    carry: bool,
    parity: bool,
    auxiliary_carry: bool,
    zero: bool,
    sign: bool,
}

// --- Executor ---

pub struct Executor {
    /// The 1MB memory space of the 8086. Program code is loaded here. We use a Box to avoid the
    /// program crashing due to a stack overflow.
    memory: Box<[u8; 1024 * 1024]>,
    /// The 8 general-purpose 16-bit registers, modeled as pairs of 8-bit values.
    register_rows: [RegisterRow; 8],
    /// The CPU flags.
    flags: Flags,
    /// Instruction Pointer: offset into the `memory` array for the next instruction.
    ip: usize,
    /// The offset of the end of the loaded program in memory.
    end: usize,
}

impl Executor {
    /// Creates a new Executor and loads the provided program bytes into its memory.
    pub fn new(program_bytes: &[u8]) -> Self {
        let mut memory = Box::new([0; 1024 * 1024]);
        memory[..program_bytes.len()].copy_from_slice(program_bytes);

        Self {
            memory,
            register_rows: [RegisterRow { low: 0, high: 0 }; 8],
            flags: Flags {
                carry: false,
                parity: false,
                auxiliary_carry: false,
                zero: false,
                sign: false,
            },
            ip: 0,
            end: program_bytes.len(),
        }
    }

    /// Runs the simulation loop, decoding and executing instructions one by one
    /// until the end of the program is reached.
    pub fn run(&mut self) {
        println!("\nStart of execution log:");
        println!("--------------------------");

        while self.ip < self.end {
            let start_ip = self.ip;
            let start_flags = self.flags;

            // Create a decoder for the current slice of memory
            let mut decoder = Decoder::new(&self.memory[self.ip..self.end]);

            // Decode one instruction and get its length
            let (instruction, instruction_length) = decoder.decode_next_instruction();

            let initial_state = self.get_operand_values_for_log(&instruction);
            self.execute_instruction(&instruction);
            let final_state = self.get_operand_values_for_log(&instruction);

            // If a jump did not occur, advance IP by the instruction's length
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
        println!("--------------------------");
        self.log_final_state();
    }

    // --- Execution Logic ---

    fn execute_instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Bo(op, arg1, arg2) => self.execute_bo_instruction(*op, *arg1, *arg2),
            Instruction::Jmp(op, arg) => self.execute_jump_instruction(*op, *arg),
            Instruction::Loop(op, arg) => self.execute_loop_instruction(*op, *arg),
        }
    }

    fn execute_loop_instruction(&mut self, op: LoopCondition, arg: i8) {
        let cx_val = self.get_16bit_register_value(Register::Cx);
        let new_cx_val = cx_val.wrapping_sub(1);
        self.set_16bit_register_value(Register::Cx, new_cx_val);

        let should_loop = match op {
            LoopCondition::Loop => new_cx_val != 0,
            LoopCondition::Loopz => new_cx_val != 0 && self.flags.zero,
            LoopCondition::Loopnz => new_cx_val != 0 && !self.flags.zero,
            LoopCondition::Jcxz => cx_val == 0,
        };

        if should_loop {
            // Jumps are relative to the *next* instruction's address.
            // For a 2-byte loop instruction, this is ip + 2.
            let displacement = arg as isize;
            self.ip = (self.ip as isize + 2 + displacement) as usize;
        }
    }

    fn execute_jump_instruction(&mut self, op: JumpCondition, arg: i8) {
        let should_jump = match op {
            JumpCondition::Jz => self.flags.zero,
            JumpCondition::Jnz => !self.flags.zero,
            // Add other jump conditions here
            _ => panic!("Jump condition {:?} not supported", op),
        };

        if should_jump {
            // Jumps are relative to the *next* instruction's address.
            // For a 2-byte jump instruction, this is ip + 2.
            let displacement = arg as isize;
            self.ip = (self.ip as isize + 2 + displacement) as usize;
        }
    }

    fn execute_bo_instruction(&mut self, op: BinaryOp, dest: Operand, src: Operand) {
        // Determine operation width (0 for 8-bit, 1 for 16-bit)
        let width = match dest {
            Operand::R(reg) => reg.get_index().0,
            Operand::Ma(_) => match src {
                // If dest is memory, width is inferred from source
                Operand::R(reg) => reg.get_index().0,
                Operand::Imm8(_) => 0,
                Operand::Imm16(_) => 1,
                _ => panic!("Fatal error: Cannot infer operand width from memory-to-memory mov"),
            },
            _ => panic!(
                "Fatal error: Invalid destination operand for binary op {:?}",
                dest
            ),
        };

        match op {
            BinaryOp::Mov => {
                let val = self.get_operand_value(src, width);
                self.set_operand_value(dest, val, width);
            }
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Cmp => {
                self.execute_arithmetic_instruction(op, dest, src, width)
            }
        }
    }

    fn execute_arithmetic_instruction(
        &mut self,
        op: BinaryOp,
        dest: Operand,
        src: Operand,
        width: u8,
    ) {
        let val1 = self.get_operand_value(dest, width);
        let val2 = self.get_operand_value(src, width);

        let (result, carry, aux_carry) = if width == 0 {
            // 8-bit
            let v1 = val1 as u8;
            let v2 = val2 as u8;
            let (res, c) = match op {
                BinaryOp::Add => v1.overflowing_add(v2),
                BinaryOp::Sub | BinaryOp::Cmp => v1.overflowing_sub(v2),
                _ => unreachable!(),
            };
            let ac = match op {
                BinaryOp::Add => (v1 & 0xF) + (v2 & 0xF) > 0xF,
                BinaryOp::Sub | BinaryOp::Cmp => (v1 & 0xF) < (v2 & 0xF),
                _ => unreachable!(),
            };
            (res as u16, c, ac)
        } else {
            // 16-bit
            let (res, c) = match op {
                BinaryOp::Add => val1.overflowing_add(val2),
                BinaryOp::Sub | BinaryOp::Cmp => val1.overflowing_sub(val2),
                _ => unreachable!(),
            };
            let ac = match op {
                BinaryOp::Add => (val1 & 0xF) + (val2 & 0xF) > 0xF,
                BinaryOp::Sub | BinaryOp::Cmp => (val1 & 0xF) < (val2 & 0xF),
                _ => unreachable!(),
            };
            (res, c, ac)
        };

        self.flags.carry = carry;
        self.flags.auxiliary_carry = aux_carry;
        self.flags.zero = result == 0;
        self.flags.sign = if width == 0 {
            (result as u8 & 0x80) != 0
        } else {
            (result & 0x8000) != 0
        };
        self.update_parity_flag(result as u8);

        if op != BinaryOp::Cmp {
            self.set_operand_value(dest, result, width);
        }
    }

    fn update_parity_flag(&mut self, val: u8) {
        self.flags.parity = val.count_ones() % 2 == 0;
    }

    // --- Getters and Setters for Operands ---

    fn get_operand_value(&self, arg: Operand, width: u8) -> u16 {
        match arg {
            Operand::Imm8(val) => val as u16,
            Operand::Imm16(val) => val,
            Operand::R(reg) => {
                let (w, _) = reg.get_index();
                if w == 1 {
                    self.get_16bit_register_value(reg)
                } else {
                    self.get_8bit_register_value(reg) as u16
                }
            }
            Operand::Ma(addr) => {
                let index = self.get_memory_index(addr);
                if width == 0 {
                    self.memory[index] as u16
                } else {
                    u16::from_le_bytes([self.memory[index], self.memory[index + 1]])
                }
            }
        }
    }

    fn set_operand_value(&mut self, arg: Operand, val: u16, width: u8) {
        match arg {
            Operand::R(reg) => {
                let (w, i) = reg.get_index();
                if w == 1 {
                    self.set_16bit_register_value(reg, val);
                } else if i < 4 {
                    // Low byte registers (al, cl, dl, bl)
                    self.set_8bit_register_low_value(reg, val as u8);
                } else {
                    // High byte registers (ah, ch, dh, bh)
                    self.set_8bit_register_high_value(reg, val as u8);
                }
            }
            Operand::Ma(addr) => {
                let index = self.get_memory_index(addr);
                if width == 0 {
                    self.memory[index] = val as u8;
                } else {
                    let bytes = val.to_le_bytes();
                    self.memory[index] = bytes[0];
                    self.memory[index + 1] = bytes[1];
                }
            }
            _ => panic!("Fatal error: Invalid destination operand for set_operand_value."),
        }
    }

    fn get_memory_index(&self, addr: MemoryAddress) -> usize {
        let offset = match addr {
            MemoryAddress::R(reg) => self.get_16bit_register_value(reg),
            MemoryAddress::Rd8(reg, disp) => self
                .get_16bit_register_value(reg)
                .wrapping_add(disp as i8 as u16),
            MemoryAddress::Rd16(reg, disp) => self.get_16bit_register_value(reg).wrapping_add(disp),
            MemoryAddress::Rr(r1, r2) => self
                .get_16bit_register_value(r1)
                .wrapping_add(self.get_16bit_register_value(r2)),
            MemoryAddress::Rrd8(r1, r2, disp) => self
                .get_16bit_register_value(r1)
                .wrapping_add(self.get_16bit_register_value(r2))
                .wrapping_add(disp as i8 as u16),
            MemoryAddress::Rrd16(r1, r2, disp) => self
                .get_16bit_register_value(r1)
                .wrapping_add(self.get_16bit_register_value(r2))
                .wrapping_add(disp),
            MemoryAddress::Da(addr) => addr,
        };
        offset as usize
    }

    // --- Register Helpers ---

    fn get_register_row(&mut self, reg: Register) -> &mut RegisterRow {
        &mut self.register_rows[reg.get_index().1 as usize]
    }

    fn get_16bit_register_value(&self, reg: Register) -> u16 {
        let row = self.register_rows[reg.get_index().1 as usize];
        u16::from_le_bytes([row.low, row.high])
    }

    fn set_16bit_register_value(&mut self, reg: Register, val: u16) {
        let row = self.get_register_row(reg);
        let [low, high] = val.to_le_bytes();
        row.low = low;
        row.high = high;
    }

    fn get_8bit_register_value(&self, reg: Register) -> u8 {
        let (_, i) = reg.get_index();
        let row = self.register_rows[(i & 0b11) as usize];
        if (i >> 2) != 0 {
            row.high
        } else {
            row.low
        }
    }

    fn set_8bit_register_low_value(&mut self, reg: Register, val: u8) {
        self.get_register_row(reg).low = val;
    }

    fn set_8bit_register_high_value(&mut self, reg: Register, val: u8) {
        self.get_register_row(reg).high = val;
    }

    // --- Logging ---

    fn get_operand_values_for_log(&self, instruction: &Instruction) -> Vec<(Operand, u16)> {
        let mut values = Vec::new();
        if let Instruction::Bo(_, dest, src) = instruction {
            if let Operand::R(reg) = dest {
                values.push((*dest, self.get_operand_value(*dest, reg.get_index().0)));
            }
            if let Operand::R(reg) = src {
                values.push((*src, self.get_operand_value(*src, reg.get_index().0)));
            }
        }
        values
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
        let mut log_line = format!("{: <20}", instruction.to_string());
        let mut state_changes = Vec::new();

        for (op, initial_val) in &initial_state {
            if let Some((_, final_val)) = final_state.iter().find(|(f_op, _)| f_op == op) {
                if initial_val != final_val {
                    if let Operand::R(reg) = op {
                        state_changes
                            .push(format!("{}:0x{:x}->0x{:x}", reg, initial_val, final_val));
                    }
                }
            }
        }

        if start_flags != end_flags {
            state_changes.push(format!("flags:{}->{}", start_flags, end_flags));
        }

        if start_ip != end_ip {
            state_changes.push(format!("ip:0x{:x}->0x{:x}", start_ip, end_ip));
        }

        if !state_changes.is_empty() {
            log_line.push_str(&format!("; {}", state_changes.join(" ")));
        }

        println!("{}", log_line);
    }

    fn log_final_state(&self) {
        println!("\nFinal registers:");
        let registers_to_log = [
            Register::Ax,
            Register::Cx,
            Register::Dx,
            Register::Bx,
            Register::Sp,
            Register::Bp,
            Register::Si,
            Register::Di,
        ];
        for reg in registers_to_log {
            let val = self.get_16bit_register_value(reg);
            if val != 0 {
                println!("     {}: 0x{:04x} ({})", reg, val, val);
            }
        }
        println!("     ip: 0x{:04x} ({})", self.ip, self.ip);
        println!("  flags: {}\n", self.flags);
    }
}

// --- Display Implementations for Logging ---

impl Display for Flags {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut s = String::new();
        if self.carry {
            s.push('C');
        }
        if self.parity {
            s.push('P');
        }
        if self.auxiliary_carry {
            s.push('A');
        }
        if self.zero {
            s.push('Z');
        }
        if self.sign {
            s.push('S');
        }
        write!(f, "{}", s)
    }
}

use std::fmt::Write;

pub struct PrettyPrinter<W> {
    wrt: W,
    indent_stack: Vec<u16>,
    cur_text: u16,
    cur_indent: u16,
}

pub trait PP: Write {
    fn open_box(&mut self);
    fn close_box(&mut self);
}

impl<W: Write> PrettyPrinter<W> {
    pub fn start(wrt: W) -> Self {
        PrettyPrinter {
            wrt,
            indent_stack: Vec::new(),
            cur_text: 0,
            cur_indent: 0,
        }
    }
}

impl<W: Write> PP for PrettyPrinter<W> {
    fn open_box(&mut self) {
        self.indent_stack.push(self.cur_indent);
        self.cur_indent += self.cur_text;
        self.cur_text = 0;
    }

    fn close_box(&mut self) {
        let prev_indent = self.indent_stack.pop().unwrap_or(0);
        self.cur_text += self.cur_indent - prev_indent;
        self.cur_indent = prev_indent;
    }
}

impl<W: Write> Write for PrettyPrinter<W> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let mut cur_text = self.cur_text as usize;
        for (ndx, line) in s.split('\n').enumerate() {
            if ndx == 0 {
                cur_text += line.len();
            } else {
                self.wrt.write_char('\n')?;
                // better way?
                for _ in 0..self.cur_indent {
                    self.wrt.write_char(' ')?;
                }
                cur_text = line.len();
            }
            self.wrt.write_str(line)?;
        }

        self.cur_text = cur_text.try_into().unwrap();
        Ok(())
    }
}

pub struct IoAsFmt<W>(pub W);

impl<W: std::io::Write> std::fmt::Write for IoAsFmt<W> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write_all(s.as_bytes()).unwrap();
        Ok(())
    }
}

use std::io::Write;

use smallvec::SmallVec;

pub struct PrettyPrinter<W> {
    wrt: W,
    indent_stack: Vec<u16>,
    cur_text: u16,
    cur_indent: u16,
    indent_next_chunk: bool,
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
            indent_next_chunk: false,
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
        let prev_indent = self.indent_stack.pop().unwrap();
        if !self.indent_next_chunk {
            self.cur_text += self.cur_indent - prev_indent;
        }
        self.cur_indent = prev_indent;
    }
}

impl<W: Write> Write for PrettyPrinter<W> {
    fn write(&mut self, s: &[u8]) -> std::io::Result<usize> {
        let mut cur_text = self.cur_text as usize;

        for line in s.split_inclusive(|ch| *ch == b'\n') {
            if self.indent_next_chunk {
                for _ in 0..self.cur_indent {
                    self.wrt.write_all(b" ")?;
                }
            }

            self.wrt.write_all(line)?;
            if line.ends_with(b"\n") {
                cur_text = 0;
                self.indent_next_chunk = true;
            } else {
                cur_text += line.len();
                self.indent_next_chunk = false;
            }
        }

        self.cur_text = cur_text.try_into().unwrap();

        Ok(s.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.wrt.flush()
    }
}

pub struct IoAsFmt<W>(pub W);

impl<W: std::io::Write> std::fmt::Write for IoAsFmt<W> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write_all(s.as_bytes()).unwrap();
        Ok(())
    }
}

pub struct FmtAsIoUTF8<W>(pub W);

impl<W: std::fmt::Write> std::io::Write for FmtAsIoUTF8<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let s = std::str::from_utf8(buf).unwrap();
        self.0.write_str(s).unwrap();
        Ok(s.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn pp_to_string(action: impl FnOnce(&mut PrettyPrinter<&mut Vec<u8>>)) -> String {
    let mut bytes = Vec::new();
    let mut pp = PrettyPrinter::start(&mut bytes);
    action(&mut pp);

    String::from_utf8_lossy(&bytes).into_owned()
}

pub struct MultiWriter<W>(SmallVec<[W; 4]>);

impl<W> MultiWriter<W> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        MultiWriter(SmallVec::new())
    }

    pub fn add_writer(&mut self, wrt: W) {
        self.0.push(wrt);
    }
}
impl<W: std::io::Write> std::io::Write for MultiWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for wrt in &mut self.0 {
            let _ = wrt.write(buf);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        for wrt in &mut self.0 {
            let _ = wrt.flush();
        }
        Ok(())
    }
}

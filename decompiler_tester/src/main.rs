use std::{
    borrow::Cow,
    fmt::Write,
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result};
use decompiler::test_tool::Tester;
use egui::TextBuffer;
use ouroboros::self_referencing;

fn main() {
    let mut args = std::env::args();
    let self_name = args.next().unwrap();
    let Some(exe_filename) = args.next() else {
        eprintln!("Usage: {} EXE_FILENAME", self_name);
        std::process::exit(1)
    };

    let exe_filename: PathBuf = exe_filename.into();

    let res = eframe::run_native(
        "decompiler test app",
        eframe::NativeOptions::default(),
        Box::new(|_cctx| {
            let mut app = Box::new(App::new());
            app.open_executable(exe_filename);
            Ok(app)
        }),
    );

    if let Err(err) = res {
        eprintln!("eframe error: {:?}", err);
        std::process::exit(1);
    }
}

#[self_referencing]
struct Exe {
    exe_bytes: Vec<u8>,

    #[borrows(exe_bytes)]
    #[covariant]
    tester: Tester<'this>,
}

struct App {
    restore_file: RestoreFile,
    exe: Option<Result<Exe>>,
    ui_function_name: String,
    process_log: String,
    file_view: FileView,
    status: StatusView,
}

#[derive(serde::Serialize, Default, Clone)]
struct RestoreFile {
    exe_filename: Option<PathBuf>,
    function_name: Option<String>,
}

impl App {
    fn new() -> Self {
        App {
            restore_file: RestoreFile::default(),
            exe: None,
            ui_function_name: String::new(),
            process_log: String::new(),
            file_view: FileView::default(),
            status: StatusView::default(),
        }
    }

    fn open_executable(&mut self, path: PathBuf) {
        self.restore(RestoreFile {
            exe_filename: Some(path),
            function_name: None,
        });
    }

    fn restore(&mut self, restore_file: RestoreFile) {
        // move off thread?

        let prev_restore_file = std::mem::replace(&mut self.restore_file, restore_file);

        if self.restore_file.exe_filename != prev_restore_file.exe_filename {
            self.exe = self
                .restore_file
                .exe_filename
                .as_ref()
                .map(|p| load_executable(&p));
        }

        if let Some(Ok(exe)) = &mut self.exe {
            if self.restore_file.function_name != prev_restore_file.function_name {
                match &self.restore_file.function_name {
                    Some(function_name) => {
                        exe.with_tester_mut(|tester| {
                            let mut log = Vec::new();
                            let mut pp = decompiler::pp::PrettyPrinter::start(&mut log);
                            let res = tester.process_function(function_name, &mut pp);

                            let mut log = String::from_utf8(log).unwrap();
                            if let Err(err) = res {
                                writeln!(log, "\nfinished with error: {:?}", err).unwrap();
                            }

                            self.process_log = log;
                            self.ui_function_name = function_name.clone();
                        });
                    }
                    None => {
                        self.process_log = format!("No function loaded.");
                    }
                }
            }
        }
    }
}

fn load_executable(path: &Path) -> Result<Exe> {
    use std::io::Read as _;

    let mut exe_bytes = Vec::new();
    let mut elf = File::open(&path).context("opening file")?;
    elf.read_to_end(&mut exe_bytes).context("reading file")?;

    ExeTryBuilder {
        exe_bytes,
        tester_builder: |exe_bytes| Tester::start(&exe_bytes).context("parsing executable"),
    }
    .try_build()
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Quit").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                self.status.show(ui);
            });

            if ui
                .text_edit_singleline(&mut self.ui_function_name)
                .lost_focus()
            {
                self.restore(RestoreFile {
                    function_name: Some(self.ui_function_name.clone()),
                    ..self.restore_file.clone()
                });
                return;
            }

            self.file_view.show(ui, &self.process_log);
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        match ron::to_string(&self.restore_file) {
            Ok(payload) => {
                storage.set_string("state", payload);
                self.status.push(StatusMessage {
                    text: "State saved successfully.".into(),
                    ..Default::default()
                });
            }
            Err(err) => {
                self.status.push(StatusMessage {
                    text: format!("while saving application state: {}", err).into(),
                    category: StatusCategory::Error,
                    ..Default::default()
                });
            }
        }
    }
}

struct StatusView {
    cur_msg: Option<StatusMessage>,
}

impl StatusView {
    fn push(&mut self, msg: StatusMessage) {
        eprintln!("status: {:?}", msg);
        self.cur_msg = Some(msg);
    }

    fn show(&mut self, ui: &mut egui::Ui) {
        if let Some(msg) = &self.cur_msg {
            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.label(msg.text.as_str());
            });
        }
    }
}

impl Default for StatusView {
    fn default() -> Self {
        StatusView { cur_msg: None }
    }
}

#[derive(Debug)]
struct StatusMessage {
    text: Cow<'static, str>,
    category: StatusCategory,
    timeout: Duration,
}
#[derive(Debug)]
enum StatusCategory {
    Info,
    Error,
}
impl Default for StatusMessage {
    fn default() -> Self {
        StatusMessage {
            text: "".into(),
            category: StatusCategory::Info,
            timeout: Duration::from_secs(5),
        }
    }
}

struct FileView {}

impl Default for FileView {
    fn default() -> Self {
        FileView {}
    }
}

impl FileView {
    fn show(&mut self, ui: &mut egui::Ui, log: &str) {
        ui.label(log);
    }
}

use std::{
    borrow::Cow,
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result};
use decompiler::test_tool::Tester;

fn main() -> eframe::Result<()> {
    let mut args = std::env::args();
    let self_name = args.next().unwrap();
    let Some(exe_filename) = args.next() else {
        eprintln!("Usage: {} EXE_FILENAME", self_name);
        std::process::exit(1);
    };

    let exe_filename: PathBuf = exe_filename.into();

    eframe::run_native(
        "decompiler test app",
        eframe::NativeOptions::default(),
        Box::new(|_cctx| {
            let mut app = Box::new(App::new());
            app.open_executable(exe_filename);
            Ok(app)
        }),
    )
}

struct App {
    state: State,
    file_view: FileView,
    status: StatusView,
}

#[derive(serde::Serialize, Default)]
struct State {
    exe_filename: Option<PathBuf>,
    function_name: Option<String>,
}

impl App {
    fn new() -> Self {
        App {
            state: State::default(),
            status: StatusView::default(),
            file_view: FileView::default(),
        }
    }

    fn open_executable(&mut self, path: PathBuf) {
        // move off thread?

        self.state = State {
            exe_filename: Some(path),
            function_name: None,
        };

        let raw_binary = load_executable(path);
    }
}

fn load_executable(path: &Path) -> Result<Tester> {
    use std::io::Read as _;

    let mut raw_binary = Vec::new();
    let mut elf = File::open(&path).context("opening file")?;
    elf.read_to_end(&mut raw_binary).context("reading file")?;

    Ok(decompiler::test_tool::Tester::start(&raw_binary))
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Hello, world!");
            if ui.button("Quit").clicked() {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        match ron::to_string(&self.state) {
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
    queue: Vec<StatusMessage>,
}

impl StatusView {
    fn push(&mut self, msg: StatusMessage) {
        self.queue.push(msg);
    }
}

impl Default for StatusView {
    fn default() -> Self {
        StatusView { queue: Vec::new() }
    }
}

struct StatusMessage {
    text: Cow<'static, str>,
    category: StatusCategory,
    timeout: Duration,
}
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
    fn show(&mut self, ctx: egui::Context) {
        // --
    }
}

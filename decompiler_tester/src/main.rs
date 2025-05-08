use std::{
    borrow::Cow,
    collections::BTreeMap,
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result};
use decompiler::Executable;
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
        Box::new(|cctx| {
            let mut app = Box::new(App::new());

            // TODO: remove, take this from the app state, allow picking exe from gui
            app.open_executable(&exe_filename);

            if let Some(storage) = cctx.storage {
                app.load(storage);
            }
            cctx.egui_ctx.set_theme(app.theme_preference);

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
    exe: Executable<'this>,
}

struct App {
    theme_preference: egui::ThemePreference,
    status: StatusView,
    stage_exe: Option<StageExe>,
}

struct StageExe {
    exe: Result<Exe>,
    path: PathBuf,
    function_selector: Option<FunctionSelector>,
    stage_func: Option<Result<StageFunc, decompiler::Error>>,
    stage_func_tree: egui_tiles::Tree<Pane>,
}
struct StageFunc {
    df: decompiler::DecompiledFunction,
    problems_is_visible: bool,
    problems_title: String,
    problems_error: Option<String>,

    assembly: Assembly,
    mil_lines: Vec<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, Copy)]
enum Pane {
    Assembly,
    Mil,
    Ssa,
    SsaPreXform,
    Ast,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
struct RestoreFile {
    exe_filename: Option<PathBuf>,
    function_name: Option<String>,
    tree: Option<egui_tiles::Tree<Pane>>,
}

impl App {
    fn new() -> Self {
        App {
            theme_preference: egui::ThemePreference::Light,
            status: StatusView::default(),
            stage_exe: None,
        }
    }

    const SK_STATE: &'static str = "state";

    fn load(&mut self, storage: &dyn eframe::Storage) {
        let Some(serial) = storage.get_string(Self::SK_STATE) else {
            return;
        };

        let restore_file = match ron::from_str::<RestoreFile>(&serial) {
            Ok(rf) => rf,
            Err(err) => {
                let text = format!("unable to parse application state file: {:?}", err).into();
                self.status.push(StatusMessage {
                    text,
                    category: StatusCategory::Error,
                    ..Default::default()
                });
                return;
            }
        };

        let RestoreFile {
            exe_filename,
            function_name,
            tree,
        } = restore_file;

        // TODO: enable after exe filename is no longer taken from CLI
        if false {
            if let Some(exe_filename) = exe_filename {
                self.open_executable(&exe_filename);
            }
        }

        if let Some(function_name) = function_name {
            if let Some(stage_exe) = self.stage_exe.as_mut() {
                stage_exe.load_function(&function_name);
            }
        }

        if let Some(tree) = tree {
            if let Some(stage_exe) = &mut self.stage_exe {
                stage_exe.stage_func_tree = tree;
            }
        }
    }

    const TREE_ID: &'static str = "stage_func_tree";

    fn open_executable(&mut self, path: &Path) {
        let exe = load_executable(path);
        self.stage_exe = Some(StageExe {
            exe,
            path: path.to_path_buf(),
            function_selector: None,
            stage_func: None,
            stage_func_tree: egui_tiles::Tree::new_horizontal(
                Self::TREE_ID,
                vec![Pane::Mil, Pane::Ast],
            ),
        });
    }

    fn function_name(&self) -> Option<&str> {
        let stage_exe = self.stage_exe.as_ref()?;
        let stage_func_res = stage_exe.stage_func.as_ref()?;
        let stage_func = stage_func_res.as_ref().ok()?;
        Some(stage_func.df.name())
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let stage_exe = match &mut self.stage_exe {
            Some(stage_exe) => stage_exe,
            None => {
                egui::CentralPanel::default()
                    .show(ctx, |ui| ui.label("No executable loaded. (To do!)"));
                return;
            }
        };

        egui::TopBottomPanel::top("top_bar")
            .resizable(false)
            .show_separator_line(false)
            .exact_height(25.)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    stage_exe.show_topbar(ui);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }

                        let (label, value) = match self.theme_preference {
                            // not super correct, but whatever
                            egui::ThemePreference::System | egui::ThemePreference::Dark => {
                                ("Light mode", egui::ThemePreference::Light)
                            }
                            egui::ThemePreference::Light => {
                                ("Dark mode", egui::ThemePreference::Dark)
                            }
                        };

                        if ui.button(label).clicked() {
                            self.theme_preference = value;
                            ctx.set_theme(value);
                        }
                    });
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            stage_exe.show_central(ctx, ui);
        });

        egui::TopBottomPanel::bottom("statusbar")
            .exact_height(20.)
            .resizable(false)
            .show_separator_line(false)
            .show(ctx, |ui| {
                if let Some(stage_exe) = &mut self.stage_exe {
                    stage_exe.show_status(ui);
                }
                self.status.show(ui);
            });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let stage_exe = self.stage_exe.as_ref();
        let restore_file = RestoreFile {
            exe_filename: stage_exe.map(|st| st.path.clone()),
            function_name: self.function_name().map(ToOwned::to_owned),
            tree: stage_exe.map(|st| st.stage_func_tree.clone()),
        };

        match ron::to_string(&restore_file) {
            Ok(payload) => {
                storage.set_string(Self::SK_STATE, payload);
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

impl StageExe {
    fn show_topbar(&mut self, ui: &mut egui::Ui) {
        let Ok(exe) = &mut self.exe else { return };

        if ui.button("Load function…").clicked() {
            let mut all_names: Vec<_> = exe
                .borrow_exe()
                .function_names()
                .map(|s| s.to_owned())
                .collect();
            all_names.sort();
            self.function_selector = Some(FunctionSelector::new("modal load function", all_names));
        }

        match &mut self.stage_func {
            Some(Ok(stage_func)) => {
                stage_func.show_topbar(ui, &mut self.stage_func_tree);
            }
            Some(Err(_)) => {
                // error shown in central area
            }
            None => {
                ui.label("No function loaded.");
            }
        }
    }

    fn show_central(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        match &mut self.stage_func {
            Some(Ok(stage_func)) => {
                stage_func.show_panels(ui);
                self.stage_func_tree.ui(stage_func, ui);
            }
            Some(Err(err)) => {
                egui::Frame::new().show(ui, |ui| {
                    ui.label("Error while loading executable");
                    // TODO cache instead of alloc'ing and deallocing every frame
                    ui.label(err.to_string());
                });
            }
            None => {
                // message shown in topbar
            }
        }

        if let Some(function_selector) = &mut self.function_selector {
            let res = function_selector.show(ctx);
            if let Some(function_name) = res.inner.cloned() {
                self.load_function(&function_name);
                self.function_selector = None;
            } else if res.should_close() {
                self.function_selector = None;
            }
        }
    }

    fn load_function(&mut self, function_name: &str) {
        let Ok(exe) = &mut self.exe else {
            // TODO: show this in statusbar (with a cleaner message?)
            eprintln!("unable to load function: no exe loaded");
            return;
        };

        let mut stage_func = exe.with_exe_mut(|exe| {
            let df = exe.decompile_function(function_name)?;
            Ok(StageFunc::new(df, exe))
        });
        if let Ok(stage_func) = &mut stage_func {
            stage_func.problems_is_visible =
                stage_func.df.error().is_some() || !stage_func.df.warnings().is_empty();
        }

        self.stage_func = Some(stage_func);
    }

    fn show_status(&mut self, ui: &mut egui::Ui) {
        if let Some(Ok(stage_func)) = &mut self.stage_func {
            stage_func.show_status(ui);
        }
    }
}

impl StageFunc {
    fn new(df: decompiler::DecompiledFunction, exe: &Executable) -> Self {
        let title = format!(
            "{}{} warnings",
            if df.error().is_some() { "ERROR!, " } else { "" },
            df.warnings().len(),
        );
        let error_label = df.error().map(|err| err.to_string());

        let decoder = df.disassemble(exe);
        let assembly = Assembly::from_decoder(decoder);

        let mil_lines = {
            let mut lines = Vec::new();

            match df.mil() {
                None => {
                    lines.push("No MIL generated!".to_string());
                }
                Some(mil) => {
                    let buf = format!("{:?}", mil);
                    lines.extend(buf.lines().map(|line| line.to_string()))
                }
            }

            lines
        };

        StageFunc {
            df,
            problems_title: title,
            problems_error: error_label,
            problems_is_visible: false,
            assembly,
            mil_lines,
        }
    }

    fn show_topbar(&mut self, ui: &mut egui::Ui, tree: &mut egui_tiles::Tree<Pane>) {
        ui.label(self.df.name());

        ui.menu_button("Add view", |ui| {
            for (pane, label) in [
                (Pane::Assembly, "Assembly"),
                (Pane::Mil, "MIL"),
                (Pane::Ssa, "SSA"),
                (Pane::SsaPreXform, "SSA pre-xform"),
                (Pane::Ast, "AST"),
            ] {
                if ui.button(label).clicked() {
                    let root_id = *tree
                        .root
                        .get_or_insert_with(|| tree.tiles.insert_vertical_tile(vec![]));
                    let child = tree.tiles.insert_pane(pane);
                    let egui_tiles::Tile::Container(root) = tree.tiles.get_mut(root_id).unwrap()
                    else {
                        panic!()
                    };
                    root.add_child(child);
                    // do not close menu, in case the user wants another tile
                }
            }
        });
    }

    fn show_panels(&mut self, ui: &mut egui::Ui) {
        if self.problems_is_visible {
            egui::TopBottomPanel::bottom("func_errors")
                .resizable(true)
                .default_height(ui.text_style_height(&egui::TextStyle::Body) * 10.0)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            // TODO cache
                            ui.heading(&self.problems_title);

                            if let Some(error_label) = &self.problems_error {
                                ui.label(
                                    egui::RichText::new(error_label).color(egui::Color32::DARK_RED),
                                );
                            }

                            for warn in self.df.warnings() {
                                // TODO cache
                                ui.label(warn.to_string());
                            }

                            ui.add_space(50.0);
                        });
                });
        }
    }

    fn show_status(&mut self, ui: &mut egui::Ui) {
        ui.toggle_value(&mut self.problems_is_visible, &self.problems_title);
    }

    fn ui_tab_assembly(&mut self, ui: &mut egui::Ui) {
        let height = ui.text_style_height(&egui::TextStyle::Monospace);
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show_rows(ui, height, self.assembly.lines.len(), |ui, ndxs| {
                for ndx in ndxs {
                    let asm_line = &self.assembly.lines[ndx];
                    ui.horizontal(|ui| {
                        ui.allocate_ui(egui::Vec2::new(100.0, 18.0), |ui| {
                            ui.monospace(format!("0x{:x}", asm_line.addr));
                        });
                        ui.monospace(&asm_line.text);
                    });
                }
            });
    }

    fn ui_tab_mil(&mut self, ui: &mut egui::Ui) {
        let height = ui.text_style_height(&egui::TextStyle::Monospace);
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show_rows(ui, height, self.mil_lines.len(), |ui, ndxs| {
                for ndx in ndxs {
                    ui.monospace(&self.mil_lines[ndx]);
                }
            });
    }
}

impl egui_tiles::Behavior<Pane> for StageFunc {
    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        match pane {
            Pane::Assembly => {
                self.ui_tab_assembly(ui);
            }
            Pane::Mil => {
                self.ui_tab_mil(ui);
            }
            _ => {
                ui.label(format!("{:?}", pane));
            }
        }

        egui_tiles::UiResponse::None
    }

    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        match pane {
            Pane::Mil => "MIL",
            Pane::Ssa => "SSA",
            Pane::SsaPreXform => "SSA pre-transforms",
            Pane::Ast => "AST",
            Pane::Assembly => "Assembly",
        }
        .into()
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            all_panes_must_have_tabs: true,
            ..Default::default()
        }
    }

    fn is_tab_closable(
        &self,
        _tiles: &egui_tiles::Tiles<Pane>,
        _tile_id: egui_tiles::TileId,
    ) -> bool {
        true
    }
}

fn load_executable(path: &Path) -> Result<Exe> {
    use std::io::Read as _;

    let mut exe_bytes = Vec::new();
    let mut elf = File::open(&path).context("opening file")?;
    elf.read_to_end(&mut exe_bytes).context("reading file")?;

    ExeTryBuilder {
        exe_bytes,
        exe_builder: |exe_bytes| Executable::parse(&exe_bytes).context("parsing executable"),
    }
    .try_build()
}

struct FunctionSelector {
    id: &'static str,
    line: String,
    line_lower: String,
    all_names: Vec<String>,
    all_names_lower: Vec<String>,
    filtered_ndxs: Vec<usize>,
}

impl FunctionSelector {
    fn new(id: &'static str, all_names: Vec<String>) -> Self {
        let all_names_lower = all_names.iter().map(|s| s.to_lowercase()).collect();
        FunctionSelector {
            id,
            line: String::new(),
            line_lower: String::new(),
            all_names,
            all_names_lower,
            filtered_ndxs: Vec::new(),
        }
    }

    fn show(&mut self, ctx: &egui::Context) -> egui::ModalResponse<Option<&String>> {
        egui::Modal::new(self.id.into()).show(ctx, |ui| {
            let screen_rect = ctx.screen_rect();
            ui.set_min_size(egui::Vec2::new(
                screen_rect.width() * 0.6,
                screen_rect.height() * 0.6,
            ));

            egui::TextEdit::singleline(&mut self.line)
                .font(egui::TextStyle::Monospace)
                .hint_text("Function name...")
                .desired_width(f32::INFINITY)
                .show(ui);
            self.line_lower = self.line.to_lowercase();

            ui.add_space(5.0);

            self.filtered_ndxs.clear();
            self.filtered_ndxs.extend(
                self.all_names_lower
                    .iter()
                    .enumerate()
                    .filter(|(_, name_lower)| {
                        self.line_lower
                            .split_whitespace()
                            .map(|word| name_lower.contains(word))
                            .all(|x| x)
                    })
                    .map(|(ndx, _)| ndx),
            );

            use egui::scroll_area::ScrollBarVisibility;
            let mut response_inner = None;
            egui::ScrollArea::vertical()
                .scroll_bar_visibility(ScrollBarVisibility::AlwaysVisible)
                .show_rows(ui, 18.0, self.filtered_ndxs.len(), |ui, ndxs| {
                    ui.set_min_width(ui.available_width());
                    for ndx in ndxs {
                        let name = &self.all_names[self.filtered_ndxs[ndx]];
                        if ui.selectable_label(false, name).clicked() {
                            response_inner = Some(name);
                        }
                    }
                });

            response_inner
        })
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

struct Assembly {
    lines: Vec<AssemblyLine>,
    ndx_of_addr: BTreeMap<u64, usize>,
}
struct AssemblyLine {
    addr: u64,
    code_size: u8,
    text: String,
}

impl Assembly {
    fn from_decoder(decoder: iced_x86::Decoder) -> Self {
        let mut lines = Vec::new();
        let mut ndx_of_addr = BTreeMap::new();

        let mut formatter = iced_x86::IntelFormatter::new();
        for instr in decoder {
            use iced_x86::Formatter as _;

            let addr = instr.ip();
            let code_size: u8 = instr.len().try_into().unwrap();
            let mut text = String::new();
            formatter.format(&instr, &mut text);

            let ndx = lines.len();
            ndx_of_addr.insert(addr, ndx);
            lines.push(AssemblyLine {
                addr,
                code_size,
                text,
            });
        }

        Assembly { lines, ndx_of_addr }
    }
}

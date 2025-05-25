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
    ssa_vcache: SSAViewCache,
    ssa_px_vcache: SSAViewCache,
    ast: ast_view::Ast,

    hl: Highlight,
}

struct SSAViewCache {
    expanded: Vec<decompiler::ExpandedInsn>,
    height_of_block: decompiler::BlockMap<Option<f32>>,
}
impl SSAViewCache {
    fn new(ssa: &decompiler::SSAProgram) -> Self {
        let expanded = ssa_to_expanded(ssa);
        let height_of_block = decompiler::BlockMap::new(ssa.cfg(), None);
        SSAViewCache {
            expanded,
            height_of_block,
        }
    }
    fn new_from_option(ssa: Option<&decompiler::SSAProgram>) -> Self {
        match ssa {
            Some(ssa) => Self::new(ssa),
            None => SSAViewCache {
                expanded: Vec::new(),
                height_of_block: decompiler::BlockMap::empty(),
            },
        }
    }
}

#[derive(Default)]
struct Highlight {
    // NOTE changing the set of HighlightItem<_> members requires also changing
    // Highlight::clear_dirty_bit
    reg: HighlightItem<decompiler::Reg>,
    block: HighlightItem<decompiler::BlockID>,
    // same length as Assembly::lines
    asm_lines: Vec<AsmLineHighlight>,
    dirty_linked: bool,
}
#[derive(PartialEq, Eq)]
struct HighlightItem<T> {
    pinned: Option<T>,
    hovered: Option<T>,
}
impl<T> Default for HighlightItem<T> {
    fn default() -> Self {
        HighlightItem {
            pinned: None,
            hovered: None,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum AsmLineHighlight {
    /// The asm line is not highlighted.
    None,
    /// The asm instruction in this line corresponds directly with the highlighted SSA
    /// instruction. (Multiple SSA instructions may correspond to the same asm line.)
    Insn,
    /// The asm instruction in this line is in the same control-flow graph block as the
    /// selected block.
    Block,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, Copy)]
enum Pane {
    Assembly,
    Mil,
    Ssa,
    SsaPreXform,
    Ast,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct RestoreFile {
    theme_preference: egui::ThemePreference,
    exe_filename: Option<PathBuf>,
    function_name: Option<String>,
    tree: Option<egui_tiles::Tree<Pane>>,
}

impl Default for RestoreFile {
    fn default() -> Self {
        RestoreFile {
            theme_preference: egui::ThemePreference::System,
            exe_filename: None,
            function_name: None,
            tree: None,
        }
    }
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
            theme_preference,
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

        self.theme_preference = theme_preference;
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
            theme_preference: self.theme_preference,
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

        let ssa_vcache = SSAViewCache::new_from_option(df.ssa());
        let ssa_px_vcache = SSAViewCache::new_from_option(df.ssa_pre_xform());

        let ast = match df.ssa() {
            Some(ssa) => ast_view::Ast::from_ssa(ssa),
            None => ast_view::Ast::empty(),
        };

        StageFunc {
            df,
            problems_is_visible: false,
            problems_title: title,
            problems_error: error_label,
            assembly,
            mil_lines,
            ssa_vcache,
            ssa_px_vcache,
            ast,
            hl: Highlight::default(),
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
        self.refresh_hl_links();

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

    fn refresh_hl_links(&mut self) {
        if !self.hl.dirty_linked {
            return;
        }

        self.hl.dirty_linked = false;

        self.hl
            .asm_lines
            .resize(self.assembly.lines.len(), AsmLineHighlight::None);
        self.hl.asm_lines.fill(AsmLineHighlight::None);

        let Some(ssa) = self.df.ssa() else {
            return;
        };

        // TODO anything better than this cascade of if-let's?

        if let Some(bid) = self.hl.block.pinned {
            for reg in ssa.block_regs(bid) {
                if let Some(iv) = ssa.get(reg) {
                    if let Some(&ndx) = self.assembly.ndx_of_addr.get(&iv.addr) {
                        self.hl.asm_lines[ndx] = AsmLineHighlight::Block;
                    }
                }
            }
        }

        if let Some(reg) = self.hl.reg.pinned {
            if let Some(iv) = ssa.get(reg) {
                if let Some(&ndx) = self.assembly.ndx_of_addr.get(&iv.addr) {
                    self.hl.asm_lines[ndx] = AsmLineHighlight::Insn;
                }
            }
        }
    }

    fn show_status(&mut self, ui: &mut egui::Ui) {
        ui.toggle_value(&mut self.problems_is_visible, &self.problems_title);
    }

    fn ui_tab_assembly(&mut self, ui: &mut egui::Ui) {
        let hl = &self.hl.asm_lines;
        let height = ui.text_style_height(&egui::TextStyle::Monospace);

        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show_rows(ui, height, self.assembly.lines.len(), |ui, ndxs| {
                let is_hl_valid = hl.len() == self.assembly.lines.len();

                for ndx in ndxs {
                    let asm_line = &self.assembly.lines[ndx];
                    ui.horizontal_top(|ui| {
                        ui.allocate_ui(egui::Vec2::new(100.0, 18.0), |ui| {
                            let text = format!("0x{:x}", asm_line.addr);

                            let line_hl = if is_hl_valid {
                                hl[ndx]
                            } else {
                                AsmLineHighlight::None
                            };
                            let (bg, fg) = match line_hl {
                                AsmLineHighlight::None => {
                                    (egui::Color32::TRANSPARENT, ui.visuals().text_color())
                                }
                                AsmLineHighlight::Insn => (COLOR_RED_LIGHT, egui::Color32::BLACK),
                                AsmLineHighlight::Block => {
                                    (COLOR_GREEN_LIGHT, egui::Color32::BLACK)
                                }
                            };

                            ui.label(
                                egui::RichText::new(text)
                                    .monospace()
                                    .background_color(bg)
                                    .color(fg),
                            );
                        });
                        ui.add(
                            egui::Label::new(egui::RichText::new(&asm_line.text).monospace())
                                .extend(),
                        );
                    });
                }
            });
    }

    fn ui_tab_mil(&mut self, ui: &mut egui::Ui) {
        let height = ui.text_style_height(&egui::TextStyle::Monospace);
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show_rows(ui, height, self.mil_lines.len(), |ui, ndxs| {
                for ndx in ndxs {
                    ui.add(
                        egui::Label::new(egui::RichText::new(&self.mil_lines[ndx]).monospace())
                            .extend(),
                    );
                }
            });
    }

    fn ui_tab_ssa(&mut self, ui: &mut egui::Ui) {
        match self.df.ssa() {
            Some(ssa) => {
                show_ssa(ui, ssa, &mut self.ssa_vcache, &mut self.hl);
            }
            None => {
                ui.label("No SSA generated");
            }
        };
    }

    fn ui_tab_ssa_px(&mut self, ui: &mut egui::Ui) {
        match self.df.ssa_pre_xform() {
            Some(ssa) => {
                show_ssa(ui, ssa, &mut self.ssa_px_vcache, &mut self.hl);
            }
            None => {
                ui.label("No SSA generated");
            }
        };
    }

    fn ui_tab_ast(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.ast.show(ui, &mut self.hl);
            });
    }
}

fn ssa_to_expanded(ssa: &decompiler::SSAProgram) -> Vec<decompiler::ExpandedInsn> {
    (0..ssa.reg_count())
        .map(|ndx| {
            let insn = ssa[decompiler::Reg(ndx)].get();
            decompiler::to_expanded(&insn)
        })
        .collect()
}

fn show_ssa(
    ui: &mut egui::Ui,
    ssa: &decompiler::SSAProgram,
    vcache: &mut SSAViewCache,
    hl: &mut Highlight,
) {
    use decompiler::{BlockCont, Dest};
    fn show_dest(ui: &mut egui::Ui, dest: &Dest, hl: &mut Highlight) {
        match dest {
            Dest::Block(bid) => {
                label_block_ref(ui, *bid, hl);
            }
            Dest::Ext(addr) => {
                ui.label(format!("ext @ 0x{:x}", addr));
            }
            Dest::Indirect => {
                ui.label("<at runtime>");
            }
            Dest::Return => {
                ui.label("<return>");
            }
            Dest::Undefined => {
                ui.label("<undefined>");
            }
        }
    }
    fn show_continuation(ui: &mut egui::Ui, cont: &BlockCont, hl: &mut Highlight) {
        ui.horizontal(|ui| {
            ui.label("⮩");
            match cont {
                BlockCont::Always(dest) => show_dest(ui, dest, hl),
                BlockCont::Conditional { pos, neg } => {
                    ui.label("if ... then");
                    show_dest(ui, pos, hl);
                    ui.label("else");
                    show_dest(ui, neg, hl);
                }
            }
        });
    }

    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show_viewport(ui, |ui, viewport_rect| {
            let mut cur_y = 0.0;
            for bid in ssa.cfg().block_ids_rpo() {
                if let &Some(height) = &vcache.height_of_block[bid] {
                    if viewport_rect.min.y > cur_y + height || viewport_rect.max.y < cur_y {
                        let size = egui::Vec2 {
                            x: viewport_rect.width(),
                            y: height,
                        };
                        ui.allocate_exact_size(size, egui::Sense::empty());
                        cur_y += height;
                        continue;
                    }
                }

                const HL_BORDER_SIZE: f32 = 3.0;

                let block_res = ui.horizontal(|ui| {
                    ui.add_space(HL_BORDER_SIZE + 2.0);
                    ui.vertical(|ui| {
                        ui.separator();

                        label_block_def(ui, bid, hl);
                        ui.horizontal(|ui| {
                            ui.label("from:");
                            for &pred in ssa.cfg().block_preds(bid) {
                                label_block_ref(ui, pred, hl);
                            }
                        });

                        for reg in ssa.block_regs(bid) {
                            if ssa[reg].get() == decompiler::Insn::Void {
                                continue;
                            }

                            ui.horizontal(|ui| {
                                label_reg_def(ui, reg, hl);

                                // TODO show type information
                                // TODO use label_reg for parts of the instruction as well
                                ui.label(" <- ");

                                // NOTE: ssa_vcache is produced by mapping
                                // ssa.insns_rpo(), so its order matches this for loop
                                let ndx = reg.reg_index() as usize;
                                let insnx = &vcache.expanded[ndx];
                                ui.label(insnx.variant_name);
                                ui.label("(");
                                for (name, value) in insnx.fields.iter() {
                                    ui.label(*name);
                                    ui.label(":");
                                    match value {
                                        decompiler::ExpandedValue::Reg(reg) => {
                                            label_reg_ref(ui, *reg, hl);
                                        }
                                        decompiler::ExpandedValue::Generic(debug_str) => {
                                            ui.label(debug_str);
                                        }
                                    }
                                }
                                ui.label(")");
                            });
                        }

                        show_continuation(ui, &ssa.cfg().block_cont(bid), hl);
                    })
                });

                let block_rect = block_res.response.rect;
                let mut hl_rect = block_rect.clone();
                hl_rect.set_width(HL_BORDER_SIZE);

                if hl.block.pinned == Some(bid) {
                    ui.painter().rect_filled(hl_rect, 0.0, COLOR_GREEN_DARK);
                } else if hl.block.hovered == Some(bid) {
                    ui.painter().rect_stroke(
                        hl_rect,
                        0.0,
                        egui::Stroke {
                            color: COLOR_GREEN_DARK,
                            width: 1.0,
                        },
                        egui::StrokeKind::Inside,
                    );
                }

                let height = block_rect.height();
                vcache.height_of_block[bid] = Some(height);
                cur_y += height;
            }

            ui.separator();
            ui.label(format!("{} instructions/registers", ssa.reg_count()));
        });
}

const COLOR_BLUE_LIGHT: egui::Color32 = egui::Color32::from_rgb(166, 206, 227);
const COLOR_BLUE_DARK: egui::Color32 = egui::Color32::from_rgb(31, 120, 180);
const COLOR_GREEN_LIGHT: egui::Color32 = egui::Color32::from_rgb(178, 223, 138);
const COLOR_GREEN_DARK: egui::Color32 = egui::Color32::from_rgb(51, 160, 44);
const COLOR_RED_LIGHT: egui::Color32 = egui::Color32::from_rgb(251, 154, 153);
const COLOR_RED_DARK: egui::Color32 = egui::Color32::from_rgb(227, 26, 28);
const COLOR_ORANGE_LIGHT: egui::Color32 = egui::Color32::from_rgb(253, 191, 111);
const COLOR_ORANGE_DARK: egui::Color32 = egui::Color32::from_rgb(255, 127, 0);
const COLOR_PURPLE_LIGHT: egui::Color32 = egui::Color32::from_rgb(202, 178, 214);
const COLOR_PURPLE_DARK: egui::Color32 = egui::Color32::from_rgb(106, 61, 154);
const COLOR_BROWN_LIGHT: egui::Color32 = egui::Color32::from_rgb(255, 255, 153);
const COLOR_BROWN_DARK: egui::Color32 = egui::Color32::from_rgb(177, 89, 40);

fn label_reg_def(ui: &mut egui::Ui, reg: decompiler::Reg, hl: &mut Highlight) {
    let fg = egui::Color32::WHITE;
    let bg = COLOR_BLUE_DARK;
    // TODO avoid alloc?
    let text = format!("{:?}", reg);
    hl.dirty_linked |= hl_label(ui, text, &reg, &mut hl.reg, bg, fg);
}

fn label_reg_ref(ui: &mut egui::Ui, reg: decompiler::Reg, hl: &mut Highlight) {
    let fg = egui::Color32::BLACK;
    let bg = COLOR_BLUE_LIGHT;
    // TODO avoid alloc?
    let text = format!("{:?}", reg);
    hl.dirty_linked |= hl_label(ui, text, &reg, &mut hl.reg, bg, fg);
}

fn label_block_def(ui: &mut egui::Ui, bid: decompiler::BlockID, hl: &mut Highlight) {
    let fg = egui::Color32::WHITE;
    let bg = COLOR_GREEN_DARK;
    // TODO avoid alloc?
    let text = format!("{:?}", bid);
    hl.dirty_linked |= hl_label(ui, text, &bid, &mut hl.block, bg, fg);
}

fn label_block_ref(ui: &mut egui::Ui, bid: decompiler::BlockID, hl: &mut Highlight) {
    let fg = egui::Color32::BLACK;
    let bg = COLOR_GREEN_LIGHT;
    // TODO avoid alloc?
    let text = format!("{:?}", bid);
    hl.dirty_linked |= hl_label(ui, text, &bid, &mut hl.block, bg, fg);
}

fn hl_label<T: PartialEq + Eq + Clone>(
    ui: &mut egui::Ui,
    text: String,
    item: &T,
    hli: &mut HighlightItem<T>,
    background_color: egui::Color32,
    color: egui::Color32,
) -> bool {
    let mut rt = egui::RichText::new(text);
    let mut stroke = egui::Stroke {
        width: 1.0,
        color: egui::Color32::TRANSPARENT,
    };

    let is_pinned = hli.pinned.as_ref() == Some(item);
    let is_hovered = hli.hovered.as_ref() == Some(item);
    if is_pinned {
        rt = rt.background_color(background_color).color(color);
    } else if is_hovered {
        stroke.color = background_color;
    };

    let res = egui::Frame::new()
        .stroke(stroke)
        .show(ui, |ui| ui.label(rt))
        .inner;

    if res.clicked() {
        // toggle selection
        hli.pinned = if is_pinned { None } else { Some(item.clone()) };
        return true;
    }
    if res.hovered() {
        // toggle selection
        hli.hovered = Some(item.clone());
        ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
        return true;
    }

    return false;
}

impl egui_tiles::Behavior<Pane> for StageFunc {
    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        match pane {
            Pane::Assembly => self.ui_tab_assembly(ui),
            Pane::Mil => self.ui_tab_mil(ui),
            Pane::Ssa => self.ui_tab_ssa(ui),
            Pane::SsaPreXform => self.ui_tab_ssa_px(ui),
            Pane::Ast => self.ui_tab_ast(ui),
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

mod ast_view {
    // target features for the ast view:
    //
    // - structured ("indented")
    // - highlight def/use
    // - folding (hide/show subtrees)
    // - highlight indirect def/use chains
    // - fast rendering
    //
    // I'm skeptic that we can just walk through the SSAProgram's def/use chains
    // and get everything we want without a slow (allocation heavy) algorithm.
    // better to just do a single transformation at the beginning and pick a
    // representation that suits the rendering.

    use core::str;
    use std::{cell::RefCell, fmt::Debug, ops::Range};

    use decompiler::Insn;

    use super::Highlight;
    use crate::{label_block_def, label_block_ref, label_reg_def, label_reg_ref};

    pub struct Ast {
        // the tree is represented as a flat Vec of Nodes.
        // the last element is the root node
        nodes: Vec<Node>,
        is_node_shown: RefCell<Vec<bool>>,
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum Node {
        Open { kind: SeqKind, count: usize },
        Error(String),
        Ref(decompiler::Reg),
        LitNum(i64),
        Generic(String),
        Kw(&'static str),
        BlockHeader(decompiler::BlockID),
        BlockRef(decompiler::BlockID),
        RegDef(decompiler::Reg),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SeqKind {
        Vertical,
        Flow,
    }

    #[derive(Clone, Copy)]
    pub struct NodeID(usize);

    impl Ast {
        pub fn empty() -> Self {
            Ast {
                nodes: Vec::new(),
                is_node_shown: RefCell::new(Vec::new()),
            }
        }

        pub fn from_ssa(ssa: &decompiler::SSAProgram) -> Self {
            let ast = Builder::new(ssa).build();

            for (ndx, node) in ast.nodes.iter().enumerate() {
                eprintln!("{:4} - {:?}", ndx, node);
            }

            ast
        }

        pub fn show(&self, ui: &mut egui::Ui, hl: &mut Highlight) {
            {
                let mut mask = self.is_node_shown.borrow_mut();
                mask.fill(false);
            }

            self.show_block(ui, 0..self.nodes.len(), SeqKind::Vertical, hl);
        }

        fn show_block(
            &self,
            ui: &mut egui::Ui,
            ndx_range: Range<usize>,
            kind: SeqKind,
            hl: &mut Highlight,
        ) -> usize {
            match kind {
                SeqKind::Vertical => {
                    let mut child_rect = ui.available_rect_before_wrap();

                    let line_x = child_rect.min.x;
                    child_rect.min.x += 30.0;

                    let res = ui.scope_builder(egui::UiBuilder::new().max_rect(child_rect), |ui| {
                        self.show_block_content(ui, ndx_range, hl)
                    });

                    let y_min = res.response.rect.min.y;
                    let y_max = res.response.rect.max.y;
                    ui.painter().line_segment(
                        [
                            egui::Pos2::new(line_x, y_min),
                            egui::Pos2::new(line_x, y_max),
                        ],
                        ui.visuals().window_stroke(),
                    );

                    res.inner
                }
                SeqKind::Flow => {
                    ui.horizontal(|ui| self.show_block_content(ui, ndx_range, hl))
                        .inner
                }
            }
        }

        fn show_block_content(
            &self,
            ui: &mut egui::Ui,
            ndx_range: Range<usize>,
            hl: &mut Highlight,
        ) -> usize {
            let mut ndx = ndx_range.start;
            while ndx < ndx_range.end {
                let new_ndx = self.show_node(ui, ndx, hl);
                assert!(new_ndx > ndx);
                ndx = new_ndx;
            }
            assert_eq!(ndx, ndx_range.end);
            ndx
        }

        fn show_node(&self, ui: &mut egui::Ui, ndx: usize, hl: &mut Highlight) -> usize {
            let already_visited = {
                let mut mask = self.is_node_shown.borrow_mut();
                std::mem::replace(&mut mask[ndx], true)
            };
            assert!(!already_visited);

            match &self.nodes[ndx] {
                Node::Open { kind, count } => {
                    // ndx      Open { count: 3 }
                    // ndx + 1  A
                    // ndx + 2  B
                    // ndx + 3  C
                    // ndx + 4  TheNextThing
                    let start_ndx = ndx + 1; // skip the Open node
                    let end_ndx = start_ndx + count;
                    let check_end_ndx = self.show_block(ui, start_ndx..end_ndx, *kind, hl);
                    if check_end_ndx != end_ndx {
                        eprintln!(
                            "warning: @ {}: skipped {} nodes",
                            ndx,
                            end_ndx as isize - check_end_ndx as isize
                        )
                    }
                    return end_ndx;
                }
                Node::Error(err_msg) => {
                    egui::Frame::new()
                        .stroke(egui::Stroke::new(1.0, egui::Color32::DARK_RED))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Internal error:");
                                ui.label(err_msg);
                            });
                        });
                }

                // TODO define specific styles
                Node::Ref(reg) => {
                    label_reg_ref(ui, *reg, hl);
                }
                Node::RegDef(reg) => {
                    label_reg_def(ui, *reg, hl);
                }
                Node::LitNum(num) => {
                    // TODO avoid alloc
                    ui.label(format!("{:?}", num));
                }
                Node::Generic(text) => {
                    ui.label(text);
                }
                Node::Kw(kw) => {
                    ui.label(*kw);
                }
                Node::BlockHeader(bid) => {
                    // TODO avoid alloc
                    ui.add_space(12.0);
                    label_block_def(ui, *bid, hl);
                }
                Node::BlockRef(bid) => {
                    label_block_ref(ui, *bid, hl);
                }
            }

            ndx + 1
        }
    }

    struct Builder<'a> {
        nodes: Vec<Node>,
        ssa: &'a decompiler::SSAProgram,
        rdr_count: decompiler::RegMap<usize>,

        // just to check that the algo is correct:
        block_status: decompiler::BlockMap<BlockStatus>,
        open_stack: Vec<decompiler::BlockID>,
        let_was_printed: decompiler::RegMap<bool>,
    }
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum BlockStatus {
        Pending,
        Started,
        Finished,
    }
    impl<'a> Builder<'a> {
        fn new(ssa: &'a decompiler::SSAProgram) -> Self {
            let rdr_count = decompiler::count_readers(ssa);
            let block_status = decompiler::BlockMap::new(ssa.cfg(), BlockStatus::Pending);
            let let_was_printed = decompiler::RegMap::for_program(ssa, false);
            Builder {
                nodes: Vec::new(),
                ssa,
                rdr_count,
                block_status,
                open_stack: Vec::new(),
                let_was_printed,
            }
        }

        fn build(mut self) -> Ast {
            self.transform_block_labeled(self.ssa.cfg().entry_block_id());
            let is_node_shown = vec![false; self.nodes.len()];
            Ast {
                nodes: self.nodes,
                is_node_shown: RefCell::new(is_node_shown),
            }
        }

        fn seq<R>(&mut self, kind: SeqKind, add_contents: impl FnOnce(&mut Self) -> R) -> R {
            const REF_NODE: Node = Node::Kw("<bug:seq!>");

            let ndx = self.nodes.len();
            self.nodes.push(REF_NODE);

            let ret = add_contents(self);

            assert_eq!(self.nodes[ndx], REF_NODE);
            let count = self.nodes.len() - ndx - 1;
            self.nodes[ndx] = Node::Open { kind, count };

            ret
        }
        fn transform_block_labeled(&mut self, bid: decompiler::BlockID) {
            self.emit(Node::BlockHeader(bid));
            self.transform_block_unlabeled(bid);
        }

        fn transform_block_unlabeled(&mut self, bid: decompiler::BlockID) {
            {
                self.open_stack.push(bid);
                assert_eq!(self.block_status[bid], BlockStatus::Pending);
                self.block_status[bid] = BlockStatus::Started;
            }

            // TODO fix: remove .to_vec(). split builder core from visitors (then we can
            // borrow &self.scheduler and &mut self.nodes)
            let block_sched: Vec<_> = self.ssa.block_regs(bid).collect();
            for reg in block_sched {
                if self.rdr_count[reg] > 1 {
                    self.emit_let_def(reg);
                } else if self.ssa[reg].get().has_side_effects()
                    && self.ssa.reg_type(reg) != decompiler::RegType::Control
                {
                    self.transform_def(reg);
                }
            }

            let cont = self.ssa.cfg().block_cont(bid);
            match cont {
                decompiler::BlockCont::Always(dest) => {
                    self.transform_dest(bid, &dest);
                }
                decompiler::BlockCont::Conditional { pos, neg } => {
                    let cond = self.ssa.find_last_matching(bid, |insn| {
                        decompiler::match_get!(insn, decompiler::Insn::SetJumpCondition(cond), cond)
                    });

                    if let Some(cond) = cond {
                        self.seq(SeqKind::Flow, |s| {
                            s.emit(Node::Kw("if"));
                            s.transform_value(cond);
                        });
                        self.seq(SeqKind::Vertical, |s| {
                            s.transform_dest(bid, &pos);
                        });
                        self.emit(Node::Kw("else"));
                        self.transform_dest(bid, &neg);
                    } else {
                        self.emit(Node::Error(format!("bug: no condition!")));
                    }
                }
            }

            // process other blocks dominated by this one,
            let dom_tree = self.ssa.cfg().dom_tree();
            for &child_bid in dom_tree.children_of(bid) {
                if self.block_status[child_bid] == BlockStatus::Pending {
                    self.transform_block_labeled(child_bid);
                }
            }

            {
                let check_bid = self.open_stack.pop().unwrap();
                assert_eq!(bid, check_bid);
                assert_eq!(self.block_status[bid], BlockStatus::Started);
                self.block_status[bid] = BlockStatus::Finished;
            }
        }

        fn transform_value(&mut self, reg: decompiler::Reg) {
            // TODO! specific representation of operands
            if self.rdr_count[reg] > 1 {
                if self.let_was_printed[reg] {
                    self.emit(Node::Ref(reg));
                } else {
                    self.seq(SeqKind::Flow, |s| {
                        s.emit(Node::Kw("<bug:let!>"));
                        s.emit(Node::Ref(reg));
                    });
                }
            } else {
                self.transform_def(reg);
            }
        }

        fn transform_def(&mut self, reg: decompiler::Reg) {
            let mut insn = self.ssa[reg].get();
            match insn {
                Insn::Void => {
                    self.emit(Node::Kw("void"));
                }
                Insn::True => {
                    self.emit(Node::Kw("true"));
                }
                Insn::False => {
                    self.emit(Node::Kw("false"));
                }
                Insn::Undefined => {
                    self.emit(Node::Kw("undefined"));
                }

                Insn::Phi => self.transform_regular_insn("Phi", std::iter::empty()),
                Insn::Const { value, size: _ } => {
                    self.emit(Node::LitNum(value));
                }

                Insn::Ancestral(aname) => {
                    self.emit(Node::Kw("ancestral"));
                    self.emit(Node::Kw(aname.name()));
                }

                Insn::StoreMem {
                    mem: _,
                    addr,
                    value,
                } => {
                    self.seq(SeqKind::Flow, |s| {
                        s.transform_value(addr);
                        s.emit(Node::Kw(".@"));
                        s.emit(Node::Kw(":="));
                        s.transform_value(value);
                    });
                }
                Insn::LoadMem {
                    mem: _,
                    addr,
                    size: _,
                } => {
                    self.seq(SeqKind::Flow, |s| {
                        s.transform_value(addr);
                        s.emit(Node::Kw(".@"));
                    });
                }

                Insn::Part { src, offset, size } => {
                    self.seq(SeqKind::Flow, |s| {
                        s.transform_value(src);
                        s.emit(Node::Kw("["));
                        s.emit(Node::LitNum(offset as i64));
                        s.emit(Node::Kw(".."));
                        s.emit(Node::LitNum((offset + size) as i64));
                        s.emit(Node::Kw("]"));
                    });
                }
                Insn::Concat { lo, hi } => {
                    self.seq(SeqKind::Flow, |s| {
                        s.transform_value(hi);
                        s.emit(Node::Kw("++"));
                        s.transform_value(lo);
                    });
                }

                Insn::StructGetMember {
                    struct_value,
                    name,
                    size: _,
                } => {
                    self.seq(SeqKind::Flow, |s| {
                        s.transform_value(struct_value);
                        s.emit(Node::Kw("."));
                        s.emit(Node::Kw(name));
                    });
                }

                Insn::Widen {
                    reg,
                    target_size,
                    sign: _,
                } => {
                    self.seq(SeqKind::Flow, |s| {
                        s.transform_value(reg);
                        s.emit(Node::Kw("as"));
                        s.emit(Node::Generic(format!("i{}", target_size * 8)));
                    });
                }

                Insn::Arith(op, a, b) => {
                    self.emit_binop(
                        op.symbol(),
                        |s| s.transform_value(a),
                        |s| s.transform_value(b),
                    );
                }
                Insn::ArithK(op, a, bk) => {
                    self.emit_binop(
                        op.symbol(),
                        |s| s.transform_value(a),
                        |s| {
                            s.emit(Node::LitNum(bk));
                        },
                    );
                }
                Insn::Cmp(op, a, b) => {
                    self.emit_binop(
                        op.symbol(),
                        |s| s.transform_value(a),
                        |s| s.transform_value(b),
                    );
                }
                Insn::Bool(op, a, b) => {
                    self.emit_binop(
                        op.symbol(),
                        |s| s.transform_value(a),
                        |s| s.transform_value(b),
                    );
                }

                Insn::Not(arg) => {
                    self.seq(SeqKind::Flow, |s| {
                        s.emit(Node::Kw("!"));
                        s.transform_value(arg);
                    });
                }

                Insn::NotYetImplemented(msg) => {
                    self.seq(SeqKind::Flow, |s| {
                        s.emit(Node::Kw("NYI:"));
                        s.emit(Node::Generic(msg.to_string()));
                    });
                }

                Insn::SetReturnValue(_) | Insn::SetJumpCondition(_) | Insn::SetJumpTarget(_) => {
                    // handled through control flow / transform_dest
                }

                Insn::Call { callee, first_arg } => {
                    self.seq(SeqKind::Flow, |s| {
                        s.transform_value(callee);
                        s.emit(Node::Kw("("));

                        for (ndx, arg) in s.ssa.get_call_args(first_arg).enumerate() {
                            if ndx > 0 {
                                s.emit(Node::Kw(","));
                            }
                            s.transform_value(arg);
                        }

                        s.emit(Node::Kw(")"));
                    });
                }

                Insn::CArg { .. } => {
                    self.emit(Node::Kw("<bug:CArg>"));
                }
                Insn::Control(_) => {
                    self.emit(Node::Kw("<bug:Control>"));
                }

                Insn::Upsilon { value, phi_ref } => {
                    self.seq(SeqKind::Flow, |s| {
                        s.emit(Node::Ref(phi_ref));
                        s.emit(Node::Kw(":="));
                        s.transform_value(value);
                    });
                }

                Insn::Get(_)
                | Insn::OverflowOf(_)
                | Insn::CarryOf(_)
                | Insn::SignOf(_)
                | Insn::IsZero(_)
                | Insn::Parity(_) => {
                    self.transform_regular_insn(
                        Self::opcode_name(&insn),
                        insn.input_regs_iter().map(|x| *x),
                    );
                }
            }
        }

        fn opcode_name(insn: &Insn) -> &'static str {
            match insn {
                Insn::Void => "Void",
                Insn::True => "True",
                Insn::False => "False",
                Insn::Const { .. } => "Const",
                Insn::Get(_) => "Get",
                Insn::Part { .. } => "Part",
                Insn::Concat { .. } => "Concat",
                Insn::StructGetMember { .. } => "StructGetMember",
                Insn::Widen { .. } => "Widen",
                Insn::Arith(_, _, _) => "Arith",
                Insn::ArithK(_, _, _) => "ArithK",
                Insn::Cmp(_, _, _) => "Cmp",
                Insn::Bool(_, _, _) => "Bool",
                Insn::Not(_) => "Not",
                Insn::Call { .. } => "Call",
                Insn::CArg { .. } => "CArg",
                Insn::SetReturnValue(_) => "SetReturnValue",
                Insn::SetJumpCondition(_) => "SetJumpCondition",
                Insn::SetJumpTarget(_) => "SetJumpTarget",
                Insn::Control(_) => "Control",
                Insn::NotYetImplemented(_) => "NotYetImplemented",
                Insn::LoadMem { .. } => "LoadMem",
                Insn::StoreMem { .. } => "StoreMem",
                Insn::OverflowOf(_) => "OverflowOf",
                Insn::CarryOf(_) => "CarryOf",
                Insn::SignOf(_) => "SignOf",
                Insn::IsZero(_) => "IsZero",
                Insn::Parity(_) => "Parity",
                Insn::Undefined => "Undefined",
                Insn::Ancestral(_) => "Ancestral",
                Insn::Phi => "Phi",
                Insn::Upsilon { .. } => "Upsilon",
            }
        }

        fn transform_regular_insn(
            &mut self,
            opcode: &'static str,
            inputs: impl IntoIterator<Item = decompiler::Reg>,
        ) {
            self.seq(SeqKind::Flow, |s| {
                s.emit(Node::Generic(opcode.to_string()));
                s.emit(Node::Kw("("));
                for (ndx, input) in inputs.into_iter().enumerate() {
                    if ndx > 0 {
                        s.emit(Node::Kw(","));
                    }
                    s.transform_value(input);
                }
                s.emit(Node::Kw(")"));
            });
        }

        fn emit(&mut self, init: Node) -> NodeID {
            let nid = NodeID(self.nodes.len());
            self.nodes.push(init);
            nid
        }

        fn transform_dest(&mut self, src_bid: decompiler::BlockID, dest: &decompiler::Dest) {
            match dest {
                decompiler::Dest::Ext(addr) => {
                    self.seq(SeqKind::Flow, |s| {
                        s.emit(Node::Kw("goto"));
                        s.emit(Node::LitNum(*addr as i64));
                    });
                }
                decompiler::Dest::Block(bid) => {
                    let block_preds = self.ssa.cfg().block_preds(*bid);
                    if block_preds.len() == 1 {
                        if block_preds[0] != src_bid {
                            // TODO emit a warning (not our fault, but a bug in the cfg)
                        }

                        // just print the block inline, no "goto"s
                        self.transform_block_unlabeled(*bid);
                    } else {
                        self.seq(SeqKind::Flow, |s| {
                            s.emit(Node::Kw("goto"));
                            // TODO specific node type
                            s.emit(Node::BlockRef(*bid));
                        });
                    }
                }
                decompiler::Dest::Indirect => {
                    let tgt = self.ssa.find_last_matching(src_bid, |insn| {
                        decompiler::match_get!(insn, decompiler::Insn::SetJumpTarget(tgt), tgt)
                    });

                    if let Some(tgt) = tgt {
                        self.seq(SeqKind::Flow, |s| {
                            s.emit(Node::Kw("goto"));
                            s.seq(SeqKind::Flow, |s| {
                                s.emit(Node::Kw("*"));
                                s.transform_value(tgt);
                            });
                        });
                    } else {
                        self.emit(Node::Error(format!("bug: no jump target!")));
                    }
                }
                decompiler::Dest::Return => {
                    let ret = self.ssa.find_last_matching(src_bid, |insn| {
                        decompiler::match_get!(insn, decompiler::Insn::SetReturnValue(val), val)
                    });

                    if let Some(ret) = ret {
                        self.seq(SeqKind::Flow, |s| {
                            s.emit(Node::Kw("return"));
                            s.transform_value(ret);
                        });
                    } else {
                        self.emit(Node::Error(format!("bug: no return value!")));
                    }
                }
                decompiler::Dest::Undefined => {
                    self.emit(Node::Kw("goto undefined"));
                }
            }
        }

        fn emit_let_def(&mut self, reg: decompiler::Reg) {
            self.seq(SeqKind::Flow, |s| {
                if s.let_was_printed[reg] {
                    s.emit(Node::Kw("<bug:dupe let>"));
                }

                s.emit(Node::Kw("let"));
                // TODO make the name editable
                s.emit(Node::RegDef(reg));
                s.emit(Node::Kw("="));
                s.transform_def(reg);
            });

            self.let_was_printed[reg] = true;
        }

        fn emit_binop<F, G>(&mut self, op_s: &'static str, a: F, b: G)
        where
            F: FnOnce(&mut Self),
            G: FnOnce(&mut Self),
        {
            self.seq(SeqKind::Flow, |s| {
                a(s);
                s.emit(Node::Kw(op_s));
                b(s);
            });
        }
    }
}

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

            if let Some(storage) = cctx.storage {
                app.load(storage);
            }

            if !app.is_exe_loaded(&exe_filename) {
                // TODO: remove, take this from the app state, allow picking exe from gui
                app.open_executable(&exe_filename);
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

const TREE_ID_STAGES_EXE: &'static str = "tree_id_stages_exe";
const TREE_ID_STAGE_FUNC: &'static str = "tree_id_stage_func";

struct App {
    theme_preference: egui::ThemePreference,
    status: StatusView,
    stages_exe: slotmap::SlotMap<ExeID, StageExeFallible>,
    stages_exe_tree: egui_tiles::Tree<ExeID>,
}

slotmap::new_key_type! {
    struct ExeID;
}

struct StageExeFallible {
    path: PathBuf,
    stage_exe: Result<StageExe>,
}
struct StageExe {
    exe: Exe,
    function_selector: Option<FunctionSelector>,
    stages_func: Vec<Result<StageFunc, decompiler::Error>>,
}
struct StageFunc {
    df: DecompiledFunction,
    tree: egui_tiles::Tree<Pane>,
}
struct DecompiledFunction {
    df: decompiler::DecompiledFunction,
    problems_is_visible: bool,
    problems_title: String,
    problems_error: Option<String>,

    assembly: Assembly,
    mil_lines: Vec<String>,
    ssa_vcache: SSAViewCache,
    ssa_px_vcache: SSAViewCache,
    ast: ast_view::Ast,

    hl: hl::Highlight,
}

struct SSAViewCache {
    height_of_block: decompiler::BlockMap<Option<f32>>,
}
impl SSAViewCache {
    fn new(ssa: &decompiler::SSAProgram) -> Self {
        let height_of_block = decompiler::BlockMap::new(ssa.cfg(), None);
        SSAViewCache { height_of_block }
    }
    fn new_from_option(ssa: Option<&decompiler::SSAProgram>) -> Self {
        match ssa {
            Some(ssa) => Self::new(ssa),
            None => SSAViewCache {
                height_of_block: decompiler::BlockMap::empty(),
            },
        }
    }
}

mod hl {
    #[derive(Default)]
    pub(crate) struct Highlight {
        // directly tracking user input (mouse hover, clicks)
        pub(super) reg: Item<decompiler::Reg>,
        pub(super) block: Item<decompiler::BlockID>,
        pub(super) asm_line_ndx: Item<usize>,

        // derived from the above:
        // same length as Assembly::lines
        related_asm: Vec<AsmLineRelation>,
        related_ssa: Option<decompiler::RegMap<bool>>,
    }

    impl Highlight {
        pub(super) fn set_asm_line_count(&mut self, count: usize) {
            self.related_asm.resize(count, AsmLineRelation::default());
        }

        pub(super) fn asm_line_rel(&self, ndx: usize) -> Option<&AsmLineRelation> {
            self.related_asm.get(ndx)
        }

        pub(super) fn is_ssa_asm_related(&self, reg: decompiler::Reg) -> bool {
            self.related_ssa.as_ref().map(|m| m[reg]).unwrap_or(false)
        }

        pub(super) fn update(&mut self, ssa: &decompiler::SSAProgram, asm: &super::Assembly) {
            // TODO update dependent values from user-tracking stuff
            // NOTE not short-circuiting!
            let is_dirty =
                self.reg.tick_frame() | self.block.tick_frame() | self.asm_line_ndx.tick_frame();
            if !is_dirty {
                return;
            }

            self.related_asm.fill(AsmLineRelation::default());

            // TODO anything better than this cascade of if-let's?

            if let Some(bid) = self.block.pinned {
                for reg in ssa.block_regs(bid) {
                    if let Some(iv) = ssa.get(reg) {
                        if let Some(&ndx) = asm.ndx_of_addr.get(&iv.addr) {
                            self.related_asm[ndx].block = true;
                        }
                    }
                }
            }
            if let Some(reg) = self.reg.pinned {
                if let Some(iv) = ssa.get(reg) {
                    if let Some(&ndx) = asm.ndx_of_addr.get(&iv.addr) {
                        self.related_asm[ndx].ssa = true;
                    }
                }
            }

            if let Some(rel_ssa) = &mut self.related_ssa {
                if rel_ssa.reg_count() != ssa.reg_count() {
                    *rel_ssa = decompiler::RegMap::for_program(ssa, false);
                }
            }

            let related_ssa = self
                .related_ssa
                .get_or_insert_with(|| decompiler::RegMap::for_program(ssa, false));
            assert_eq!(related_ssa.reg_count(), ssa.reg_count());

            related_ssa.fill(false);
            if let Some(asm_line_ndx) = self.asm_line_ndx.pinned {
                if let Some(asm_line) = asm.lines.get(asm_line_ndx) {
                    for reg in ssa.registers() {
                        let iv = ssa.get(reg).unwrap();
                        if iv.addr == asm_line.addr {
                            related_ssa[reg] = true;
                        }
                    }
                }
            }
        }
    }

    #[derive(PartialEq, Eq)]
    pub(crate) struct Item<T> {
        pinned: Option<T>,
        is_pinned_dirty: bool,
        // used for "scroll to"
        was_pinned_just_cleared: bool,
        // `get` reads `hovered`, while `set` writes to `hovered_next_frame`
        // This allows to easily detect:
        //
        // 1. whether the hovered selection has changed (and update dependent
        // data as a consequence);
        //
        // 2. whether the mouse is currently not hovering anything (because
        // that's on the *absence* of mousehover events, but we still need to
        // know what happened on the last frame!)
        hovered: Option<T>,
        hovered_next_frame: Option<T>,
    }
    impl<T> Default for Item<T> {
        fn default() -> Self {
            Item {
                pinned: None,
                is_pinned_dirty: false,
                was_pinned_just_cleared: false,
                hovered: None,
                hovered_next_frame: None,
            }
        }
    }
    impl<T: PartialEq + Eq> Item<T> {
        pub(super) fn pinned(&self) -> Option<&T> {
            self.pinned.as_ref()
        }
        pub(super) fn did_pinned_just_change(&self) -> bool {
            self.was_pinned_just_cleared
        }
        pub(super) fn set_pinned(&mut self, value: Option<T>) {
            self.pinned = value;
            self.is_pinned_dirty = true;
        }
        pub(super) fn hovered(&self) -> Option<&T> {
            self.hovered.as_ref()
        }
        pub(super) fn set_hovered(&mut self, value: Option<T>) {
            self.hovered_next_frame = value;
        }
        fn tick_frame(&mut self) -> bool {
            let is_pinned_dirty = self.is_pinned_dirty;
            self.is_pinned_dirty = false;
            self.was_pinned_just_cleared = is_pinned_dirty;

            let next_value = self.hovered_next_frame.take();
            let is_hovered_dirty = self.hovered != next_value;
            self.hovered = next_value;

            is_pinned_dirty || is_hovered_dirty
        }
    }

    #[derive(Default, Clone, Copy, PartialEq, Eq)]
    pub struct AsmLineRelation {
        /// True iff the assembly insn in this line is related to the selected
        /// (pinned) SSA instruction.
        pub ssa: bool,
        /// True iff the assembly insn in this line is in the same control-flow
        /// graph block as the selected (pinned) block.
        pub block: bool,
    }

    pub struct Colors {
        pub background: egui::Color32,
        pub background_pinned: egui::Color32,
        /// Text color. Set to `None` to keep the default text color.
        pub text: Option<egui::Color32>,
        pub text_pinned: egui::Color32,
        pub border_hovered: egui::Color32,
        pub border_pinned: egui::Color32,
    }

    impl Default for Colors {
        fn default() -> Self {
            Colors {
                background: egui::Color32::TRANSPARENT,
                background_pinned: egui::Color32::BLACK,
                text: None,
                text_pinned: egui::Color32::WHITE,
                border_hovered: egui::Color32::TRANSPARENT,
                border_pinned: egui::Color32::TRANSPARENT,
            }
        }
    }

    pub const COLOR_BLUE_LIGHT: egui::Color32 = egui::Color32::from_rgb(166, 206, 227);
    pub const COLOR_BLUE_DARK: egui::Color32 = egui::Color32::from_rgb(31, 120, 180);
    pub const COLOR_GREEN_LIGHT: egui::Color32 = egui::Color32::from_rgb(178, 223, 138);
    pub const COLOR_GREEN_DARK: egui::Color32 = egui::Color32::from_rgb(51, 160, 44);
    pub const COLOR_RED_LIGHT: egui::Color32 = egui::Color32::from_rgb(251, 154, 153);
    pub const COLOR_RED_DARK: egui::Color32 = egui::Color32::from_rgb(227, 26, 28);
    pub const COLOR_ORANGE_LIGHT: egui::Color32 = egui::Color32::from_rgb(253, 191, 111);
    pub const COLOR_ORANGE_DARK: egui::Color32 = egui::Color32::from_rgb(255, 127, 0);
    pub const COLOR_PURPLE_LIGHT: egui::Color32 = egui::Color32::from_rgb(202, 178, 214);
    pub const COLOR_PURPLE_DARK: egui::Color32 = egui::Color32::from_rgb(106, 61, 154);
    pub const COLOR_BROWN_LIGHT: egui::Color32 = egui::Color32::from_rgb(255, 255, 153);
    pub const COLOR_BROWN_DARK: egui::Color32 = egui::Color32::from_rgb(177, 89, 40);
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, Copy)]
enum Pane {
    Assembly,
    Mil,
    Ssa,
    SsaPreXform,
    Ast,
}

mod persistence {
    use super::Pane;

    use std::path::PathBuf;

    #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
    pub struct File {
        pub theme_preference: egui::ThemePreference,
        pub exes: Vec<Exe>,
    }
    #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
    pub struct Exe {
        pub filename: PathBuf,
        pub funcs: Vec<Func>,
    }
    #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
    pub struct Func {
        pub name: String,
        pub tree: egui_tiles::Tree<Pane>,
    }

    impl Default for File {
        fn default() -> Self {
            File {
                theme_preference: egui::ThemePreference::System,
                exes: Vec::new(),
            }
        }
    }
}

impl App {
    fn new() -> Self {
        App {
            theme_preference: egui::ThemePreference::Light,
            status: StatusView::default(),
            stages_exe: slotmap::SlotMap::with_key(),
            stages_exe_tree: egui_tiles::Tree::empty(TREE_ID_STAGES_EXE),
        }
    }

    const SK_STATE: &'static str = "state";

    fn load(&mut self, storage: &dyn eframe::Storage) {
        let Some(serial) = storage.get_string(Self::SK_STATE) else {
            return;
        };

        let restore_file: persistence::File = match ron::from_str(&serial) {
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

        let mut panes = Vec::with_capacity(restore_file.exes.len());

        for restore_exe in restore_file.exes {
            let exe_res = load_executable(&restore_exe.filename);
            let stage_exe = exe_res.map(|mut exe| {
                let stages_func = restore_exe
                    .funcs
                    .into_iter()
                    .map(|restore_func| {
                        let mut stage_func_or_err = load_function(&mut exe, &restore_func.name);
                        if let Ok(stage_func) = &mut stage_func_or_err {
                            stage_func.tree = restore_func.tree;
                        }
                        stage_func_or_err
                    })
                    .collect();

                StageExe {
                    exe,
                    function_selector: None,
                    stages_func,
                }
            });
            // We "hope" that the keys here resemble the
            let stage_exe_fallible = StageExeFallible {
                path: (&restore_exe.filename).to_path_buf(),
                stage_exe,
            };

            let key = self.stages_exe.insert(stage_exe_fallible);
            panes.push(key);
        }

        self.stages_exe_tree = egui_tiles::Tree::new_tabs(TREE_ID_STAGES_EXE, panes);
        self.theme_preference = restore_file.theme_preference;
    }

    fn is_exe_loaded(&self, path: &Path) -> bool {
        self.stages_exe
            .values()
            .any(|stage_exe_flbl| stage_exe_flbl.path == path)
    }

    fn open_executable(&mut self, path: &Path) {
        let exe_res = load_executable(path);
        let stage_exe = exe_res.map(|exe| StageExe {
            exe,
            function_selector: None,
            stages_func: Vec::new(),
        });
        let stage_exe_fallible = StageExeFallible {
            path: path.to_path_buf(),
            stage_exe,
        };
        let key = self.stages_exe.insert(stage_exe_fallible);
        self.stages_exe_tree = egui_tiles::Tree::new_tabs(TREE_ID_STAGES_EXE, vec![key]);
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::bottom("statusbar")
            .exact_height(20.)
            .resizable(false)
            .show_separator_line(false)
            .show(ctx, |ui| {
                let stage_exe_flbl = self
                    .stages_exe_tree
                    .active_tiles()
                    .first()
                    .and_then(|tile_id| self.stages_exe_tree.tiles.get_pane(tile_id))
                    .and_then(|exe_id| self.stages_exe.get_mut(*exe_id));
                if let Some(StageExeFallible {
                    stage_exe: Ok(stage_exe),
                    ..
                }) = stage_exe_flbl
                {
                    stage_exe.show_status(ui);
                }

                self.status.show(ui);
            });

        if self.stages_exe_tree.is_empty() {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.label("No executable loaded. Load one using File > Open.")
            });
        } else {
            egui::CentralPanel::default()
                .frame(egui::Frame::central_panel(&ctx.style()).inner_margin(0))
                .show(ctx, |ui| {
                    let stages_exe = &mut self.stages_exe;
                    let mut beh = ExeTabsBehavior {
                        exes: stages_exe,
                        theme_preference: &mut self.theme_preference,
                    };
                    self.stages_exe_tree.ui(&mut beh, ui);
                });
        }
    }
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let restore_file = persistence::File {
            theme_preference: self.theme_preference,
            exes: self
                .stages_exe
                .iter()
                .map(|(_, stage_exe_fallible)| {
                    let funcs = match &stage_exe_fallible.stage_exe {
                        // If the executable loaded successfully, collect its functions
                        Ok(stage_exe) => {
                            stage_exe
                                .stages_func
                                .iter()
                                // Only save successfully loaded functions
                                .filter_map(|stage_func_res| stage_func_res.as_ref().ok())
                                .map(|stage_func| persistence::Func {
                                    name: stage_func.df.df.name().to_string(),
                                    tree: stage_func.tree.clone(),
                                })
                                .collect()
                        }
                        // If the executable failed to load, save an empty list of functions
                        Err(_) => Vec::new(),
                    };

                    persistence::Exe {
                        filename: stage_exe_fallible.path.clone(),
                        funcs,
                    }
                })
                .collect(),
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

fn load_function(exe: &mut Exe, function_name: &str) -> Result<StageFunc, decompiler::Error> {
    let mut stage_func = exe.with_exe_mut(|exe| {
        let df = exe.decompile_function(function_name)?;
        Ok(StageFunc::new(df, exe))
    });

    if let Ok(stage_func) = &mut stage_func {
        stage_func.df.problems_is_visible =
            stage_func.df.df.error().is_some() || !stage_func.df.df.warnings().is_empty();
    }

    stage_func
}

trait ExeGetter {
    fn exe_mut(&mut self, exe_id: ExeID) -> Option<&mut StageExeFallible>;
}
impl ExeGetter for slotmap::SlotMap<ExeID, StageExeFallible> {
    fn exe_mut(&mut self, exe_id: ExeID) -> Option<&mut StageExeFallible> {
        self.get_mut(exe_id)
    }
}

struct ExeTabsBehavior<'a, G: ExeGetter> {
    exes: &'a mut G,
    theme_preference: &'a mut egui::ThemePreference,
}

impl<G: ExeGetter> egui_tiles::Behavior<ExeID> for ExeTabsBehavior<'_, G> {
    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        key: &mut ExeID,
    ) -> egui_tiles::UiResponse {
        let Some(stage_exe_fallible) = self.exes.exe_mut(*key) else {
            ui.label("BUG: invalid ExeID");
            return egui_tiles::UiResponse::None;
        };

        match &mut stage_exe_fallible.stage_exe {
            Ok(stage_exe) => {
                ui.horizontal(|ui| {
                    stage_exe.show_topbar(ui);
                });
                stage_exe.show_central(ui);
            }
            Err(err) => {
                // TODO avoid alloc / cache?
                ui.label(format!("Error loading this exe: {:?}", err));
            }
        }

        egui_tiles::UiResponse::None
    }

    fn tab_title_for_pane(&mut self, key: &ExeID) -> egui::WidgetText {
        // TODO shorten tab filenames
        match self.exes.exe_mut(*key) {
            Some(StageExeFallible { path, stage_exe: _ }) => path.to_string_lossy().into(),
            None => "???".into(),
        }
    }

    fn top_bar_right_ui(
        &mut self,
        _tiles: &egui_tiles::Tiles<ExeID>,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        _tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            if ui.button("Quit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }

            let (label, value) = match self.theme_preference {
                // not super correct, but whatever
                egui::ThemePreference::System | egui::ThemePreference::Dark => {
                    ("Light mode", egui::ThemePreference::Light)
                }
                egui::ThemePreference::Light => ("Dark mode", egui::ThemePreference::Dark),
            };

            if ui.button(label).clicked() {
                *self.theme_preference = value;
                ui.ctx().set_theme(value);
            }
        });
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            all_panes_must_have_tabs: true,
            ..Default::default()
        }
    }
}

impl StageExe {
    fn show_topbar(&mut self, ui: &mut egui::Ui) {
        if ui.button("Load function…").clicked() {
            let mut all_names: Vec<_> = self
                .exe
                .borrow_exe()
                .function_names()
                .map(|s| s.to_owned())
                .collect();
            all_names.sort();
            self.function_selector = Some(FunctionSelector::new("modal load function", all_names));
        }

        match self.stages_func.last_mut() {
            Some(Ok(stage_func)) => {
                stage_func.show_topbar(ui);
            }
            Some(Err(_)) => {
                // error shown in central area
            }
            None => {
                ui.label("No function loaded.");
            }
        }
    }

    fn show_central(&mut self, ui: &mut egui::Ui) {
        match self.stages_func.last_mut() {
            Some(Ok(stage_func)) => {
                stage_func.show_panels(ui);
                stage_func.tree.ui(&mut stage_func.df, ui);
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
            let res = function_selector.show(ui.ctx());
            if let Some(function_name) = res.inner.cloned() {
                let stage_func_or_err = load_function(&mut self.exe, &function_name);
                self.add_function(stage_func_or_err);
                self.function_selector = None;
            } else if res.should_close() {
                self.function_selector = None;
            }
        }
    }

    fn add_function(&mut self, stage_func_or_err: Result<StageFunc, decompiler::Error>) {
        // TODO make this plural
        self.stages_func.push(stage_func_or_err);
    }

    fn show_status(&mut self, ui: &mut egui::Ui) {
        for stage_func in &mut self.stages_func {
            if let Ok(stage_func) = stage_func {
                stage_func.show_status(ui);
            }
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
            df: DecompiledFunction {
                df,
                problems_is_visible: false,
                problems_title: title,
                problems_error: error_label,
                assembly,
                mil_lines,
                ssa_vcache,
                ssa_px_vcache,
                ast,
                hl: hl::Highlight::default(),
            },
            tree: egui_tiles::Tree::new_horizontal(
                TREE_ID_STAGE_FUNC,
                vec![Pane::Assembly, Pane::Ast],
            ),
        }
    }

    fn show_topbar(&mut self, ui: &mut egui::Ui) {
        ui.label(self.df.df.name());

        ui.menu_button("Add view", |ui| {
            for (pane, label) in [
                (Pane::Assembly, "Assembly"),
                (Pane::Mil, "MIL"),
                (Pane::Ssa, "SSA"),
                (Pane::SsaPreXform, "SSA pre-xform"),
                (Pane::Ast, "AST"),
            ] {
                if ui.button(label).clicked() {
                    let root_id = *self
                        .tree
                        .root
                        .get_or_insert_with(|| self.tree.tiles.insert_vertical_tile(vec![]));
                    let child = self.tree.tiles.insert_pane(pane);
                    let egui_tiles::Tile::Container(root) =
                        self.tree.tiles.get_mut(root_id).unwrap()
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
        if let Some(ssa) = self.df.df.ssa() {
            self.df.hl.update(ssa, &self.df.assembly);
        }

        if self.df.problems_is_visible {
            egui::TopBottomPanel::bottom("func_errors")
                .resizable(true)
                .default_height(ui.text_style_height(&egui::TextStyle::Body) * 10.0)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            // TODO cache
                            ui.heading(&self.df.problems_title);

                            if let Some(error_label) = &self.df.problems_error {
                                ui.label(
                                    egui::RichText::new(error_label).color(egui::Color32::DARK_RED),
                                );
                            }

                            for warn in self.df.df.warnings() {
                                // TODO cache
                                ui.label(warn.to_string());
                            }

                            ui.add_space(50.0);
                        });
                });
        }
    }

    fn show_status(&mut self, ui: &mut egui::Ui) {
        ui.toggle_value(&mut self.df.problems_is_visible, &self.df.problems_title);
    }
}

impl DecompiledFunction {
    fn ui_tab_assembly(&mut self, ui: &mut egui::Ui) {
        self.hl.set_asm_line_count(self.assembly.lines.len());

        let height = ui.text_style_height(&egui::TextStyle::Monospace);
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show_rows(ui, height, self.assembly.lines.len(), |ui, ndxs| {
                for ndx in ndxs {
                    let asm_line = &self.assembly.lines[ndx];
                    ui.horizontal_top(|ui| {
                        ui.allocate_ui(egui::Vec2::new(100.0, 18.0), |ui| {
                            let text = format!("0x{:x}", asm_line.addr);

                            let line_hl = self.hl.asm_line_rel(ndx).unwrap();
                            let is_pinned = self.hl.asm_line_ndx.pinned() == Some(&ndx);
                            let (bg, fg) = if is_pinned {
                                (hl::COLOR_RED_DARK, egui::Color32::WHITE)
                            } else if line_hl.block {
                                (hl::COLOR_GREEN_LIGHT, egui::Color32::BLACK)
                            } else if line_hl.ssa {
                                (hl::COLOR_RED_LIGHT, egui::Color32::BLACK)
                            } else {
                                (egui::Color32::TRANSPARENT, ui.visuals().text_color())
                            };

                            let stroke = if self.hl.asm_line_ndx.hovered() == Some(&ndx) {
                                hl::COLOR_RED_LIGHT
                            } else {
                                egui::Color32::TRANSPARENT
                            };

                            let res = egui::Frame::new()
                                .stroke(egui::Stroke {
                                    width: 1.0,
                                    color: stroke,
                                })
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(text)
                                            .monospace()
                                            .background_color(bg)
                                            .color(fg),
                                    )
                                })
                                .inner;

                            if res.hovered() {
                                self.hl.asm_line_ndx.set_hovered(Some(ndx));
                            }
                            if res.clicked() {
                                // TODO refactor this into a "toggle" method
                                self.hl.asm_line_ndx.set_pinned(
                                    match self.hl.asm_line_ndx.pinned() {
                                        Some(&pinned_ndx) if pinned_ndx == ndx => None,
                                        _ => Some(ndx),
                                    },
                                );
                            }
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

fn show_ssa(
    ui: &mut egui::Ui,
    ssa: &decompiler::SSAProgram,
    vcache: &mut SSAViewCache,
    hl: &mut hl::Highlight,
) {
    use decompiler::{BlockCont, Dest};
    fn show_dest(ui: &mut egui::Ui, dest: &Dest, hl: &mut hl::Highlight) {
        match dest {
            Dest::Block(bid) => {
                label_block_ref(ui, *bid, hl, format!("{:?}", *bid).into());
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
    fn show_continuation(ui: &mut egui::Ui, cont: &BlockCont, hl: &mut hl::Highlight) {
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

                        label_block_def(
                            ui,
                            bid,
                            hl,
                            format!(" -- block B{}", bid.as_number()).into(),
                        );
                        ui.horizontal(|ui| {
                            ui.label("from:");
                            for &pred in ssa.cfg().block_preds(bid) {
                                label_block_ref(
                                    ui,
                                    pred,
                                    hl,
                                    format!("B{}", pred.as_number()).into(),
                                );
                            }
                        });

                        for reg in ssa.block_regs(bid) {
                            let insn = ssa[reg].get();
                            if insn == decompiler::Insn::Void {
                                continue;
                            }

                            let insnx = decompiler::to_expanded(&insn);
                            ui.horizontal(|ui| {
                                let label_res =
                                    label_reg_def(ui, reg, hl, format!("{:?}", reg).into());
                                if hl.reg.pinned() == Some(&reg) && hl.reg.did_pinned_just_change()
                                {
                                    label_res.scroll_to_me(Some(egui::Align::Center));
                                }

                                // TODO show type information
                                // TODO use label_reg for parts of the instruction as well
                                ui.label(" <- ");

                                ui.label(insnx.variant_name);
                                ui.label("(");
                                for (name, value) in insnx.fields.iter() {
                                    ui.label(*name);
                                    ui.label(":");
                                    match value {
                                        decompiler::ExpandedValue::Reg(reg) => {
                                            label_reg_ref(
                                                ui,
                                                *reg,
                                                hl,
                                                format!("{:?}", *reg).into(),
                                            );
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
                let mut hl_rect = block_rect;
                hl_rect.set_width(HL_BORDER_SIZE);

                if hl.block.pinned() == Some(&bid) {
                    ui.painter().rect_filled(hl_rect, 0.0, hl::COLOR_GREEN_DARK);
                } else if hl.block.hovered() == Some(&bid) {
                    ui.painter().rect_stroke(
                        hl_rect,
                        0.0,
                        egui::Stroke {
                            color: hl::COLOR_GREEN_DARK,
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

// TODO Move these to an `ssa` module (when the refactoring happens)

fn label_reg_def(
    ui: &mut egui::Ui,
    reg: decompiler::Reg,
    hl: &mut hl::Highlight,
    text: egui::WidgetText,
) -> egui::Response {
    let is_asm_related = hl.is_ssa_asm_related(reg);
    let mut colors = TextRole::RegDef.colors();
    if is_asm_related {
        colors.background = hl::COLOR_RED_LIGHT;
        colors.text = Some(egui::Color32::BLACK);
    }
    hl_label(ui, &reg, &mut hl.reg, &colors, text)
}

fn label_reg_ref(
    ui: &mut egui::Ui,
    reg: decompiler::Reg,
    hl: &mut hl::Highlight,
    text: egui::WidgetText,
) -> egui::Response {
    hl_label(ui, &reg, &mut hl.reg, &TextRole::RegRef.colors(), text)
}

fn label_block_def(
    ui: &mut egui::Ui,
    bid: decompiler::BlockID,
    hl: &mut hl::Highlight,
    text: egui::WidgetText,
) -> egui::Response {
    hl_label(ui, &bid, &mut hl.block, &TextRole::BlockDef.colors(), text)
}

fn label_block_ref(
    ui: &mut egui::Ui,
    bid: decompiler::BlockID,
    hl: &mut hl::Highlight,
    text: egui::WidgetText,
) -> egui::Response {
    hl_label(ui, &bid, &mut hl.block, &TextRole::BlockRef.colors(), text)
}

fn hl_label<T: PartialEq + Eq + Clone>(
    ui: &mut egui::Ui,
    value: &T,
    hli: &mut hl::Item<T>,
    colors: &hl::Colors,
    text: egui::WidgetText,
) -> egui::Response {
    let add_contents = |ui: &mut egui::Ui| ui.label(text);

    let res = highlight(ui, value, hli, colors, add_contents);

    res
}

fn highlight<T: PartialEq + Eq + Clone>(
    ui: &mut egui::Ui,
    value: &T,
    hli: &mut hl::Item<T>,
    colors: &hl::Colors,
    add_contents: impl FnOnce(&mut egui::Ui) -> egui::Response,
) -> egui::Response {
    let is_pinned = hli.pinned() == Some(value);
    let is_hovered = hli.hovered() == Some(value);

    let bg = if is_pinned {
        colors.background_pinned
    } else {
        colors.background
    };
    let fg = if is_pinned {
        Some(colors.text_pinned)
    } else {
        colors.text
    };
    let stroke = if is_hovered {
        colors.border_hovered
    } else if is_pinned {
        colors.border_pinned
    } else {
        egui::Color32::TRANSPARENT
    };

    let res = egui::Frame::new()
        .fill(bg)
        .stroke(egui::Stroke {
            width: 1.0,
            color: stroke,
        })
        .show(ui, |ui| {
            ui.visuals_mut().override_text_color = fg;
            add_contents(ui)
        })
        .inner;

    if res.clicked() {
        // toggle selection (TODO refactor into 'toggle' method)
        hli.set_pinned(if is_pinned { None } else { Some(value.clone()) });
    }
    if res.hovered() {
        // toggle selection
        hli.set_hovered(Some(value.clone()));
        ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
    }
    res
}

impl egui_tiles::Behavior<Pane> for DecompiledFunction {
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
    let mut elf = File::open(path).context("opening file")?;
    elf.read_to_end(&mut exe_bytes).context("reading file")?;

    ExeTryBuilder {
        exe_bytes,
        exe_builder: |exe_bytes| Executable::parse(exe_bytes).context("parsing executable"),
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
                            .all(|word| name_lower.contains(word))
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

#[derive(Default)]
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

/// Preprocessed version of the original assembly program, geared towards being
/// showed on screen.
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

#[derive(Debug, PartialEq, Eq, Clone)]
enum TextRole {
    Generic,
    RegRef, // TODO Rename to 'RegRef'
    RegDef,
    BlockDef,
    BlockRef,
    Literal, // TODO Rename to 'Literal'
    Kw,      // TODO Rename to 'keyword'
    Error,
    Ident,
}
impl TextRole {
    fn colors(&self) -> hl::Colors {
        match self {
            TextRole::Generic => hl::Colors::default(),
            TextRole::Ident => hl::Colors {
                text: Some(hl::COLOR_ORANGE_DARK),
                ..Default::default()
            },
            TextRole::RegRef => hl::Colors {
                background_pinned: hl::COLOR_BLUE_LIGHT,
                text_pinned: egui::Color32::BLACK,
                border_hovered: hl::COLOR_BLUE_LIGHT,
                ..Default::default()
            },
            TextRole::RegDef => hl::Colors {
                background_pinned: hl::COLOR_BLUE_DARK,
                border_hovered: hl::COLOR_BLUE_DARK,
                ..Default::default()
            },
            TextRole::BlockRef => hl::Colors {
                background_pinned: hl::COLOR_GREEN_LIGHT,
                text_pinned: egui::Color32::BLACK,
                border_hovered: hl::COLOR_GREEN_LIGHT,
                ..Default::default()
            },
            TextRole::BlockDef => hl::Colors {
                background_pinned: hl::COLOR_GREEN_DARK,
                border_hovered: hl::COLOR_GREEN_DARK,
                ..Default::default()
            },
            TextRole::Literal => hl::Colors {
                text: Some(hl::COLOR_GREEN_DARK),
                ..Default::default()
            },
            TextRole::Kw => hl::Colors {
                background_pinned: egui::Color32::WHITE,
                text_pinned: egui::Color32::BLACK,
                border_hovered: egui::Color32::BLACK,
                border_pinned: egui::Color32::BLACK,
                ..Default::default()
            },
            TextRole::Error => hl::Colors {
                background: hl::COLOR_RED_DARK,
                text: Some(egui::Color32::WHITE),
                ..Default::default()
            },
        }
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

    use super::hl::Highlight;
    use crate::hl_label;

    pub struct Ast {
        nodes: Vec<Node>,
        /// only for assert
        was_node_shown: RefCell<Vec<bool>>,
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum Node {
        Seq(Seq),
        Element(Element),
        Space(u16),
    }
    #[derive(Debug, PartialEq, Eq, Clone)]
    struct Seq {
        kind: SeqKind,
        count: usize,
        anchor: Option<Anchor>,
    }
    #[derive(Debug, PartialEq, Eq, Clone)]
    struct Element {
        // TODO this could use some string interning
        text: String,
        anchor: Option<Anchor>,
        role: TextRole,
    }

    use super::TextRole;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SeqKind {
        Vertical,
        Flow,
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum Anchor {
        Reg(decompiler::Reg),
        Block(decompiler::BlockID),
    }

    impl Ast {
        pub fn empty() -> Self {
            Ast {
                nodes: Vec::new(),
                was_node_shown: RefCell::new(Vec::new()),
            }
        }

        pub fn from_ssa(ssa: &decompiler::SSAProgram) -> Self {
            Builder::new(ssa).build()
        }

        pub fn show(&self, ui: &mut egui::Ui, hl: &mut Highlight) {
            {
                let mut mask = self.was_node_shown.borrow_mut();
                mask.fill(false);
            }

            self.show_block(ui, 0..self.nodes.len(), SeqKind::Vertical, hl, None);
        }

        fn show_block(
            &self,
            ui: &mut egui::Ui,
            ndx_range: Range<usize>,
            kind: SeqKind,
            hl: &mut Highlight,
            anchor: Option<&Anchor>,
        ) -> usize {
            match kind {
                SeqKind::Vertical => {
                    let indent_width = 30.0;
                    let mut child_rect = ui.available_rect_before_wrap();

                    let line_x = child_rect.min.x;
                    child_rect.min.x += indent_width;

                    let res = ui.scope_builder(egui::UiBuilder::new().max_rect(child_rect), |ui| {
                        self.show_block_content(ui, ndx_range, hl)
                    });

                    let y_min = res.response.rect.min.y;
                    let y_max = res.response.rect.max.y;

                    let indent_rect = egui::Rect {
                        min: egui::Pos2::new(line_x, y_min),
                        max: egui::Pos2::new(line_x + indent_width, y_max),
                    };

                    let sense = if anchor.is_some() {
                        egui::Sense::HOVER | egui::Sense::CLICK
                    } else {
                        egui::Sense::empty()
                    };
                    let indent_response = ui.allocate_rect(indent_rect, sense);

                    let painter = ui.painter();
                    if indent_response.hovered() {
                        let stroke = ui.visuals().widgets.active.bg_stroke;
                        painter.line_segment(
                            [indent_rect.left_top(), indent_rect.right_top()],
                            stroke,
                        );
                        painter.line_segment(
                            [indent_rect.left_top(), indent_rect.left_bottom()],
                            stroke,
                        );
                        painter.line_segment(
                            [indent_rect.left_bottom(), indent_rect.right_bottom()],
                            stroke,
                        );
                    } else {
                        let stroke = ui.visuals().widgets.inactive.bg_stroke;
                        painter.line_segment(
                            [indent_rect.left_top(), indent_rect.left_bottom()],
                            stroke,
                        );
                    }

                    res.inner
                }
                SeqKind::Flow => {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 5.0;

                        let handle_size = 5.0;
                        let handle_response_painter = if anchor.is_some() {
                            let handle_space = egui::Vec2 {
                                x: handle_size,
                                y: ui.text_style_height(&egui::TextStyle::Body),
                            };
                            Some(ui.allocate_painter(
                                handle_space,
                                egui::Sense::CLICK | egui::Sense::HOVER,
                            ))
                        } else {
                            None
                        };

                        let ret = self.show_block_content(ui, ndx_range, hl);

                        if let Some((response, painter)) = handle_response_painter {
                            let stroke = egui::Stroke {
                                width: 1.0,
                                color: ui.visuals().text_color(),
                            };

                            if response.hovered() {
                                ui.painter().rect(
                                    ui.min_rect(),
                                    0.,
                                    egui::Color32::TRANSPARENT,
                                    stroke,
                                    egui::StrokeKind::Outside,
                                );
                            } else {
                                let points = [
                                    egui::vec2(1., 1.),
                                    egui::vec2(handle_size, 1.),
                                    egui::vec2(1., handle_size),
                                ];
                                for i in 0..points.len() {
                                    let from = response.rect.min + points[i];
                                    let to = response.rect.min + points[(i + 1) % points.len()];
                                    painter.line_segment([from, to], stroke);
                                }
                            }
                        }

                        ret
                    })
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
                let mut mask = self.was_node_shown.borrow_mut();
                std::mem::replace(&mut mask[ndx], true)
            };
            assert!(!already_visited);

            match &self.nodes[ndx] {
                Node::Seq(Seq {
                    kind,
                    count,
                    anchor,
                }) => {
                    // ndx      Open { count: 3 }
                    // ndx + 1  A
                    // ndx + 2  B
                    // ndx + 3  C
                    // ndx + 4  TheNextThing
                    let start_ndx = ndx + 1; // skip the Open node
                    let end_ndx = start_ndx + count;
                    let check_end_ndx =
                        self.show_block(ui, start_ndx..end_ndx, *kind, hl, anchor.as_ref());
                    if check_end_ndx != end_ndx {
                        eprintln!(
                            "warning: @ {}: skipped {} nodes",
                            ndx,
                            end_ndx as isize - check_end_ndx as isize
                        )
                    }
                    return end_ndx;
                }
                Node::Element(Element { text, anchor, role }) => {
                    match anchor {
                        Some(Anchor::Reg(reg)) => {
                            hl_label(ui, reg, &mut hl.reg, &role.colors(), text.into());
                        }
                        Some(Anchor::Block(block_id)) => {
                            hl_label(ui, block_id, &mut hl.block, &role.colors(), text.into());
                        }
                        None => {
                            ui.label(text);
                        }
                    };
                }
                &Node::Space(amount) => {
                    ui.add_space(amount as f32);
                }
            }

            ndx + 1
        }
    }

    struct Builder<'a> {
        nodes: Vec<Node>,
        ssa: &'a decompiler::SSAProgram,
        value_mode: decompiler::RegMap<ValueMode>,

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
    #[derive(Clone, Copy)]
    enum ValueMode {
        /// Value's definition is printed inline on every use, as part of each
        /// reader expression.
        Inline,
        /// The value is "defined" as a let statement (e.g., `let r123 = ...`)
        /// before the first use. Every use then refers to it by name (e.g.,
        /// `r123`).
        NamedStmt,
        /// The value is not named, but its definition is still printed as a
        /// naked statement (e.g., a call whose return value is unused)
        UnnamedStmt,
    }
    impl<'a> Builder<'a> {
        fn new(ssa: &'a decompiler::SSAProgram) -> Self {
            let rdr_count = decompiler::count_readers(ssa);
            let value_mode = rdr_count.map(|reg, rdr_count| {
                let insn = ssa[reg].get();
                if matches!(insn, Insn::Ancestral(_) | Insn::Const { .. }) {
                    ValueMode::Inline
                } else if matches!(insn, Insn::Phi) || *rdr_count > 1 {
                    ValueMode::NamedStmt
                } else if *rdr_count == 0 && insn.has_side_effects() {
                    // even without readers/users, effectful instructions need
                    // to be printed at their scheduled slot
                    ValueMode::UnnamedStmt
                } else {
                    ValueMode::Inline
                }
            });
            let block_status = decompiler::BlockMap::new(ssa.cfg(), BlockStatus::Pending);
            let let_was_printed = decompiler::RegMap::for_program(ssa, false);
            Builder {
                nodes: Vec::new(),
                ssa,
                value_mode,
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
                was_node_shown: RefCell::new(is_node_shown),
            }
        }

        fn seq<R>(
            &mut self,
            kind: SeqKind,
            anchor: Option<Anchor>,
            add_contents: impl FnOnce(&mut Self) -> R,
        ) -> R {
            self.emit(Node::Seq(Seq {
                kind,
                count: 0,
                anchor,
            }));
            let len_pre = self.nodes.len();
            let ret = add_contents(self);
            let count = self.nodes.len() - len_pre;

            let Node::Seq(Seq {
                count: seq_head_count,
                ..
            }) = &mut self.nodes[len_pre - 1]
            else {
                unreachable!()
            };
            *seq_head_count = count;

            ret
        }
        fn transform_block_labeled(&mut self, bid: decompiler::BlockID) {
            self.emit(Node::Space(20));
            self.emit(Node::Element(Element {
                text: format!(" -- block B{}", bid.as_number()),
                anchor: Some(Anchor::Block(bid)),
                role: TextRole::BlockDef,
            }));
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
                match self.value_mode[reg] {
                    ValueMode::Inline => {
                        // just skip; reader expressions will pick this up
                    }
                    ValueMode::NamedStmt => {
                        self.emit_let_def(reg);
                    }
                    ValueMode::UnnamedStmt => {
                        self.transform_def(reg, 0);
                    }
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
                        self.seq(SeqKind::Flow, None, |s| {
                            s.emit(Node::Element(Element {
                                text: "if".to_string(),
                                anchor: None,
                                role: TextRole::Kw,
                            }));
                            s.transform_value(cond, 0);
                        });
                        self.seq(SeqKind::Vertical, None, |s| {
                            s.transform_dest(bid, &pos);
                        });
                        self.emit(Node::Element(Element {
                            text: "else".to_string(),
                            anchor: None,
                            role: TextRole::Kw,
                        }));
                        self.transform_dest(bid, &neg);
                    } else {
                        self.emit(Node::Element(Element {
                            text: "bug: no condition!".to_string(),
                            anchor: None,
                            role: TextRole::Error,
                        }));
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

        fn transform_value(
            &mut self,
            reg: decompiler::Reg,
            parent_prec: decompiler::PrecedenceLevel,
        ) {
            // TODO! specific representation of operands
            match self.value_mode[reg] {
                ValueMode::Inline => {
                    self.transform_def(reg, parent_prec);
                }
                ValueMode::NamedStmt => {
                    let was_let_printed = self.let_was_printed[reg];
                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        if !was_let_printed {
                            s.emit_bug_tag("let!");
                        }
                        s.emit_reg_ref(reg);
                    });
                }
                ValueMode::UnnamedStmt => {
                    // This should never happen!
                    self.emit_bug_tag("unnamed");
                    self.transform_def(reg, parent_prec);
                }
            }
        }

        fn emit_reg_ref(&mut self, reg: decompiler::Reg) {
            self.emit(Node::Element(Element {
                text: format!("{:?}", reg),
                anchor: Some(Anchor::Reg(reg)),
                role: TextRole::RegRef,
            }))
        }

        fn transform_def(
            &mut self,
            reg: decompiler::Reg,
            parent_prec: decompiler::PrecedenceLevel,
        ) {
            let mut insn = self.ssa[reg].get();
            let prec = decompiler::precedence(&insn);

            if prec < parent_prec {
                self.emit_simple(TextRole::Kw, "(".to_string());
            }

            match insn {
                Insn::Void => {
                    self.emit(Node::Element(Element {
                        text: "void".to_string(),
                        anchor: None,
                        role: TextRole::Kw,
                    }));
                }
                Insn::True => {
                    self.emit(Node::Element(Element {
                        text: "true".to_string(),
                        anchor: None,
                        role: TextRole::Kw,
                    }));
                }
                Insn::False => {
                    self.emit(Node::Element(Element {
                        text: "false".to_string(),
                        anchor: None,
                        role: TextRole::Kw,
                    }));
                }
                Insn::Undefined => {
                    self.emit(Node::Element(Element {
                        text: "undefined".to_string(),
                        anchor: None,
                        role: TextRole::Kw,
                    }));
                }

                Insn::Phi => self.transform_regular_insn(reg, "Phi", std::iter::empty()),
                Insn::Const { value, size: _ } => {
                    self.emit_simple(TextRole::Literal, format!("{}", value));
                }

                Insn::Ancestral(aname) => {
                    self.emit(Node::Element(Element {
                        text: aname.name().to_string(),
                        anchor: Some(Anchor::Reg(reg)),
                        role: TextRole::RegRef,
                    }));
                }

                Insn::StoreMem { addr, value } => {
                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        s.transform_value(addr, 255);
                        s.emit(Node::Element(Element {
                            text: ".*".to_string(),
                            anchor: Some(Anchor::Reg(addr)),
                            role: TextRole::Kw,
                        }));
                        s.emit(Node::Element(Element {
                            text: ":=".to_string(),
                            anchor: Some(Anchor::Reg(reg)),
                            role: TextRole::RegDef,
                        }));
                        s.transform_value(value, prec);
                    });
                }
                Insn::LoadMem { addr, size: _ } => {
                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        s.transform_value(addr, prec);

                        s.emit_simple(TextRole::Kw, ".*".to_string());
                    });
                }

                Insn::Part { src, offset, size } => {
                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        s.transform_value(src, prec);
                        s.emit(Node::Element(Element {
                            text: format!("[{} .. {}]", offset, offset + size),
                            anchor: Some(Anchor::Reg(reg)),
                            role: TextRole::Kw,
                        }));
                    });
                }
                Insn::Concat { lo, hi } => {
                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        s.transform_value(hi, prec);
                        s.emit(Node::Element(Element {
                            text: "++".to_string(),
                            anchor: Some(Anchor::Reg(reg)),
                            role: TextRole::Kw,
                        }));
                        s.transform_value(lo, prec);
                    });
                }

                Insn::StructGetMember {
                    struct_value,
                    name,
                    size: _,
                } => {
                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        s.transform_value(struct_value, prec);
                        s.emit(Node::Element(Element {
                            text: format!(".{}", name),
                            anchor: Some(Anchor::Reg(reg)),
                            role: TextRole::Kw,
                        }));
                    });
                }

                Insn::Widen {
                    reg,
                    target_size,
                    sign: _,
                } => {
                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        s.transform_value(reg, prec);
                        s.emit(Node::Element(Element {
                            text: format!("as i{}", target_size * 8),
                            anchor: Some(Anchor::Reg(reg)),
                            role: TextRole::Kw,
                        }));
                    });
                }

                Insn::Arith(op, a, b) => {
                    self.emit_binop(
                        reg,
                        op.symbol(),
                        |s| s.transform_value(a, prec),
                        |s| s.transform_value(b, prec),
                    );
                }
                Insn::ArithK(op, a, bk) => {
                    self.emit_binop(
                        reg,
                        op.symbol(),
                        |s| s.transform_value(a, prec),
                        |s| {
                            s.emit_simple(TextRole::Literal, format!("{}", bk));
                        },
                    );
                }
                Insn::Cmp(op, a, b) => {
                    self.emit_binop(
                        reg,
                        op.symbol(),
                        |s| s.transform_value(a, prec),
                        |s| s.transform_value(b, prec),
                    );
                }
                Insn::Bool(op, a, b) => {
                    self.emit_binop(
                        reg,
                        op.symbol(),
                        |s| s.transform_value(a, prec),
                        |s| s.transform_value(b, prec),
                    );
                }

                Insn::Not(arg) => {
                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        s.emit_simple(TextRole::Kw, "!".to_string());
                        s.transform_value(arg, prec);
                    });
                }

                Insn::NotYetImplemented(msg) => {
                    self.emit_simple(TextRole::Kw, format!("NYI:{}", msg));
                }

                Insn::SetReturnValue(_) | Insn::SetJumpCondition(_) | Insn::SetJumpTarget(_) => {
                    // handled through control flow / transform_dest
                }

                Insn::Call { callee, first_arg } => {
                    // Not quite correct (why would we print the type name?) but
                    // happens to be always correct for well formed programs
                    let callee_type_name = self
                        .ssa
                        .get(callee)
                        .map(|iv| {
                            // TODO This name should not be copied; rather, it
                            // should be shared (in order to be editable)
                            self.ssa
                                .types()
                                .get_through_alias(iv.tyid.get())
                                .unwrap()
                                .name
                                .to_string()
                        })
                        .filter(|name| !name.is_empty());

                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        if let Some(name) = callee_type_name {
                            let role = TextRole::Ident;
                            s.emit(Node::Element(Element {
                                text: name,
                                anchor: Some(Anchor::Reg(callee)),
                                role,
                            }));
                        } else {
                            s.transform_value(callee, prec);
                        }

                        s.emit_simple(TextRole::Kw, "(".to_string());

                        for (ndx, arg) in s.ssa.get_call_args(first_arg).enumerate() {
                            if ndx > 0 {
                                s.emit_simple(TextRole::Kw, ",".to_string());
                            }
                            s.transform_value(arg, prec);
                        }

                        s.emit_simple(TextRole::Kw, ")".to_string());
                    });
                }

                Insn::CArg { .. } => {
                    self.emit_bug_tag("CArg");
                }
                Insn::Control(_) => {
                    self.emit_bug_tag("Control");
                }

                Insn::Upsilon { value, phi_ref } => {
                    self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                        s.emit_reg_ref(phi_ref);
                        s.emit(Node::Element(Element {
                            text: ":=".to_string(),
                            anchor: Some(Anchor::Reg(reg)),
                            role: TextRole::Kw,
                        }));
                        s.transform_value(value, prec);
                    });
                }

                Insn::Get(_)
                | Insn::OverflowOf(_)
                | Insn::CarryOf(_)
                | Insn::SignOf(_)
                | Insn::IsZero(_)
                | Insn::Parity(_) => {
                    self.transform_regular_insn(
                        reg,
                        Self::opcode_name(&insn),
                        insn.input_regs_iter().map(|x| *x),
                    );
                }
            }

            if prec < parent_prec {
                self.emit_simple(TextRole::Kw, ")".to_string());
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
            result: decompiler::Reg,
            opcode: &'static str,
            inputs: impl IntoIterator<Item = decompiler::Reg>,
        ) {
            self.seq(SeqKind::Flow, Some(Anchor::Reg(result)), |s| {
                s.emit_simple(TextRole::Generic, opcode.to_string());
                s.emit_simple(TextRole::Kw, "(".to_string());
                for (ndx, input) in inputs.into_iter().enumerate() {
                    if ndx > 0 {
                        s.emit_simple(TextRole::Kw, ",".to_string());
                    }
                    s.transform_value(input, 0);
                }
                s.emit_simple(TextRole::Kw, ")".to_string());
            });
        }

        fn emit(&mut self, init: Node) {
            self.nodes.push(init);
        }

        fn emit_simple(&mut self, role: TextRole, text: String) {
            self.emit(Node::Element(Element {
                text,
                anchor: None,
                role,
            }));
        }

        fn transform_dest(&mut self, src_bid: decompiler::BlockID, dest: &decompiler::Dest) {
            match dest {
                decompiler::Dest::Ext(addr) => {
                    self.seq(SeqKind::Flow, None, |s| {
                        s.emit_simple(TextRole::Kw, "goto".to_string());
                        s.emit_simple(TextRole::Literal, format!("{}", *addr));
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
                        self.seq(SeqKind::Flow, None, |s| {
                            s.emit_simple(TextRole::Kw, "goto".to_string());
                            s.emit(Node::Element(Element {
                                text: format!("B{}", bid.as_number()),
                                anchor: Some(Anchor::Block(*bid)),
                                role: TextRole::BlockRef,
                            }));
                        });
                    }
                }
                decompiler::Dest::Indirect => {
                    let tgt = self.ssa.find_last_matching(src_bid, |insn| {
                        decompiler::match_get!(insn, decompiler::Insn::SetJumpTarget(tgt), tgt)
                    });

                    if let Some(tgt) = tgt {
                        self.seq(SeqKind::Flow, None, |s| {
                            s.emit_simple(TextRole::Kw, "goto".to_string());
                            s.seq(SeqKind::Flow, None, |s| {
                                s.emit_simple(TextRole::Kw, "*".to_string());
                                s.transform_value(tgt, 0);
                            });
                        });
                    } else {
                        self.emit_simple(TextRole::Error, "bug: no jump target!".to_string());
                    }
                }
                decompiler::Dest::Return => {
                    let ret = self.ssa.find_last_matching(src_bid, |insn| {
                        decompiler::match_get!(insn, decompiler::Insn::SetReturnValue(val), val)
                    });

                    if let Some(ret) = ret {
                        self.seq(SeqKind::Flow, None, |s| {
                            s.emit_simple(TextRole::Kw, "return".to_string());
                            s.transform_value(ret, 0);
                        });
                    } else {
                        self.emit_simple(TextRole::Error, "bug: no return value!".to_string());
                    }
                }
                decompiler::Dest::Undefined => {
                    self.emit_simple(TextRole::Kw, "goto undefined".to_string());
                }
            }
        }

        fn emit_let_def(&mut self, reg: decompiler::Reg) {
            self.seq(SeqKind::Flow, Some(Anchor::Reg(reg)), |s| {
                if s.let_was_printed[reg] {
                    s.emit_bug_tag("dupe let");
                }

                s.emit_simple(TextRole::Kw, "let".to_string());
                // TODO make the name editable

                s.emit(Node::Element(Element {
                    text: format!("{:?}", reg),
                    anchor: Some(Anchor::Reg(reg)),
                    role: TextRole::RegDef,
                }));
                s.emit_simple(TextRole::Kw, "=".to_string());
                s.transform_def(reg, 0);
            });

            self.let_was_printed[reg] = true;
        }

        fn emit_binop<F, G>(&mut self, result: decompiler::Reg, op_s: &'static str, a: F, b: G)
        where
            F: FnOnce(&mut Self),
            G: FnOnce(&mut Self),
        {
            self.seq(SeqKind::Flow, Some(Anchor::Reg(result)), |s| {
                a(s);
                s.emit(Node::Element(Element {
                    text: op_s.to_string(),
                    anchor: Some(Anchor::Reg(result)),
                    role: TextRole::Kw,
                }));
                b(s);
            });
        }

        fn emit_bug_tag(&mut self, tag: &str) {
            self.emit(Node::Element(Element {
                text: format!("<bug:{}>", tag),
                anchor: None,
                role: TextRole::Error,
            }));
        }
    }
}

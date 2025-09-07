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
    is_warnings_visible: bool,
    is_ssa_visible: bool,
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
    }

    impl Highlight {
        pub(super) fn update(&mut self, ssa: &decompiler::SSAProgram, asm: &super::Assembly) {
            // TODO update dependent values from user-tracking stuff
            // NOTE not short-circuiting!
            let is_dirty =
                self.reg.tick_frame() | self.block.tick_frame() | self.asm_line_ndx.tick_frame();
            if !is_dirty {
                return;
            }

            // TODO eliminate this function
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
        pub fn colors_for(&self, value: &T, colors: &Colors) -> FrameColors {
            let is_pinned = self.pinned() == Some(value);
            let is_hovered = self.hovered() == Some(value);

            let background = if is_pinned {
                colors.background_pinned
            } else {
                colors.background
            };

            let text = if is_pinned {
                Some(colors.text_pinned)
            } else {
                colors.text
            };

            let border = if is_hovered {
                colors.border_hovered
            } else if is_pinned {
                colors.border_pinned
            } else {
                egui::Color32::TRANSPARENT
            };

            FrameColors {
                background,
                text,
                border,
            }
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

    /// Colors of a specific link in a specific frame.
    ///
    /// Given that a single value is pinned/hovered at any given time, a link
    /// has a specific background, text, and border color.
    pub struct FrameColors {
        pub background: egui::Color32,
        /// Optional text override color.
        ///
        /// When None, use egui's default.
        pub text: Option<egui::Color32>,
        pub border: egui::Color32,
    }

    pub fn highlight<T: PartialEq + Eq + Clone>(
        ui: &mut egui::Ui,
        value: &T,
        hli: &mut Item<T>,
        colors: &Colors,
        add_contents: impl FnOnce(&mut egui::Ui) -> egui::Response,
    ) -> egui::Response {
        let colors = hli.colors_for(value, colors);
        let res = egui::Frame::new()
            .fill(colors.background)
            .stroke(egui::Stroke {
                width: 1.0,
                color: colors.border,
            })
            .show(ui, |ui| {
                ui.visuals_mut().override_text_color = colors.text;
                add_contents(ui)
            })
            .inner;

        anchor_interaction(ui, value, hli, &res);

        res
    }

    pub fn anchor_interaction<T: PartialEq + Eq + Clone>(
        ui: &mut egui::Ui,
        value: &T,
        hli: &mut Item<T>,
        res: &egui::Response,
    ) {
        let is_pinned = hli.pinned() == Some(value);
        if res.clicked() {
            // toggle selection (TODO refactor into 'toggle' method)
            hli.set_pinned(if is_pinned { None } else { Some(value.clone()) });
        }
        if res.hovered() {
            // toggle selection
            hli.set_hovered(Some(value.clone()));
            ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
        }
    }
}

mod persistence {
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
                    .map(|restore_func| load_function(&mut exe, &restore_func.name))
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

/// Ephemeral struct, wrapping a pointer to an ExeGetter.
///
/// Just enough data and impl to render the UI for an Exe.
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
                stage_func.show(ui);
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
            is_warnings_visible: false,
            is_ssa_visible: true,
        }
    }

    /// Show widgets specific for this function to be laid out on the top bar
    /// (which is visually located at the executable level).
    fn show_topbar(&mut self, ui: &mut egui::Ui) {
        ui.allocate_ui(
            egui::vec2(300.0, ui.text_style_height(&egui::TextStyle::Body)),
            |ui| {
                ui.label(self.df.df.name());
                ui.allocate_space(ui.available_size());
            },
        );
        ui.checkbox(&mut self.is_ssa_visible, "SSA");
    }

    fn show(&mut self, ui: &mut egui::Ui) {
        self.show_panels(ui);
        self.show_central(ui);
    }

    /// Show the side panels (top, bottom, right, left) to be laid out around
    /// the central are, with content specific to this function.
    ///
    /// To be called before `show_central`.
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

    /// Show the widgets to be laid out in the central/main area assigned to this function.
    fn show_central(&mut self, ui: &mut egui::Ui) {
        {
            let warnings = self.df.ast.warnings();
            if warnings.len() > 0 {
                let text = format!("AST generation: {} warnings", warnings.len());
                ui.toggle_value(&mut self.is_warnings_visible, text);
                if self.is_warnings_visible {
                    egui::Window::new("AST warnings")
                        .open(&mut self.is_warnings_visible)
                        .show(ui.ctx(), |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                for warning in warnings {
                                    ui.horizontal(|ui| {
                                        ui.label(" - ");
                                        ui.add(egui::Label::new(warning.to_string()).wrap());
                                    });
                                }
                            });
                        });
                }
            }
        }

        self.show_integrated_ast(ui);
    }

    fn show_integrated_ast(&mut self, ui: &mut egui::Ui) {
        let ast = &self.df.ast;
        let mut col_ssa = self.df.df.ssa().map(|ssa| SSAColumn::new(ssa));
        let mut col_blk = BlockIDColumn;
        let mut col_ast = ast_view::AstColumn::new(ast);

        let widths = &[300.0, 80.0, columns::EXPANDING_WIDTH];
        let visible = &[self.is_ssa_visible, true, true];

        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ast_view::show(
                    ui,
                    ast,
                    visible,
                    widths,
                    &mut [&mut col_ssa.as_mut(), &mut col_blk, &mut col_ast],
                );
            });
    }
}

pub struct SSAColumn<'a> {
    ssa: &'a decompiler::SSAProgram,
}
impl<'a> SSAColumn<'a> {
    pub fn new(ssa: &'a decompiler::SSAProgram) -> Self {
        SSAColumn { ssa }
    }
}
impl ast_view::Column for SSAColumn<'_> {
    fn push_stmt(&mut self, ui: &mut egui::Ui, stmt: &ast_view::Stmt) {
        if let &ast_view::Stmt::BlockLabel(block_id) = stmt {
            for reg in self.ssa.block_regs(block_id) {
                let insn = self.ssa[reg].get();
                ui.label(format!("{:?} <- {:?}", reg, insn));
            }
            ui.label(format!("{:?}", self.ssa.cfg().block_cont(block_id)));
        }
    }
}

pub struct BlockIDColumn;
impl ast_view::Column for BlockIDColumn {
    fn push_stmt(&mut self, ui: &mut egui::Ui, stmt: &ast_view::Stmt) {
        if let &ast_view::Stmt::BlockLabel(block_id) = stmt {
            ui.label(format!("B{}", block_id.as_number()));
        }
    }
}

impl DecompiledFunction {
    fn ui_tab_assembly(&mut self, ui: &mut egui::Ui) {
        let height = ui.text_style_height(&egui::TextStyle::Monospace);
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show_rows(ui, height, self.assembly.lines.len(), |ui, ndxs| {
                for ndx in ndxs {
                    let asm_line = &self.assembly.lines[ndx];
                    ui.horizontal_top(|ui| {
                        ui.allocate_ui(egui::Vec2::new(100.0, 18.0), |ui| {
                            let text = format!("0x{:x}", asm_line.addr);

                            let is_pinned = self.hl.asm_line_ndx.pinned() == Some(&ndx);
                            let (bg, fg) = if is_pinned {
                                (hl::COLOR_RED_DARK, egui::Color32::WHITE)
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
    hl::highlight(ui, &reg, &mut hl.reg, &TextRole::RegDef.colors(), |ui| {
        ui.label(text)
    })
}

fn label_reg_ref(
    ui: &mut egui::Ui,
    reg: decompiler::Reg,
    hl: &mut hl::Highlight,
    text: egui::WidgetText,
) -> egui::Response {
    hl::highlight(ui, &reg, &mut hl.reg, &TextRole::RegRef.colors(), |ui| {
        ui.label(text)
    })
}

fn label_block_def(
    ui: &mut egui::Ui,
    bid: decompiler::BlockID,
    hl: &mut hl::Highlight,
    text: egui::WidgetText,
) -> egui::Response {
    hl::highlight(
        ui,
        &bid,
        &mut hl.block,
        &TextRole::BlockDef.colors(),
        |ui| ui.label(text),
    )
}

fn label_block_ref(
    ui: &mut egui::Ui,
    bid: decompiler::BlockID,
    hl: &mut hl::Highlight,
    text: egui::WidgetText,
) -> egui::Response {
    hl::highlight(
        ui,
        &bid,
        &mut hl.block,
        &TextRole::BlockRef.colors(),
        |ui| ui.label(text),
    )
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
    // This module is responsible for generating and displaying the Abstract Syntax Tree (AST)
    // from the SSA (Static Single Assignment) form of the decompiled code.
    // It focuses on creating a structured, navigable, and highlightable representation
    // of the code for the user interface.

    use core::str;
    use std::borrow::Cow;
    use std::fmt::Debug;
    use std::iter::Peekable;
    use std::sync::Arc;

    use anyhow::anyhow;
    use arrayvec::ArrayVec;
    use decompiler::{BlockID, Insn};

    use crate::columns;

    use super::TextRole;

    /// Represents the Abstract Syntax Tree (AST) itself. It holds a flat list of nodes
    /// that are rendered hierarchically based on `Seq` nodes.
    pub struct Ast {
        plan: Vec<Stmt>,
        warnings: Vec<anyhow::Error>,
    }

    pub enum Stmt {
        /// A label marking the start of a block.
        ///
        /// The stmts included in block follow this BlockLabel.
        BlockLabel(BlockID),
        ExprStmt(ExprTree),
        NamedStmt {
            name: Arc<String>,
            value: ExprTree,
        },
        Dedent,
        Indent,
        Comment(String),
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    pub enum ExprTree {
        Seq(Seq),
        Term(Term),
        Null,
    }
    impl From<Seq> for ExprTree {
        fn from(value: Seq) -> Self {
            ExprTree::Seq(value)
        }
    }
    impl From<Term> for ExprTree {
        fn from(value: Term) -> Self {
            ExprTree::Term(value)
        }
    }

    /// A sequence of AST expression nodes.
    #[derive(Debug, PartialEq, Eq, Clone)]
    struct Seq {
        /// Used for linking and highlighting, often to a specific register or block.
        anchor: Option<Anchor>,
        /// Children nodes, intended to be laid out sequentially
        children: Vec<ExprTree>,

        /// Whether this sequence is wrapped by parentheses (in order to
        /// maintain correct evaluation order while still respecting operator
        /// precedence)
        parentheses: bool,
    }

    /// Represents a single textual element in the AST.
    #[derive(Debug, PartialEq, Eq, Clone)]
    struct Term {
        /// Optional link to a register or block for highlighting and navigation.
        anchor: Option<Anchor>,
        /// The actual string content to be displayed.
        // TODO this could use some string interning
        text: String,
        /// Defines the semantic role of the text (e.g., keyword, register reference, literal)
        /// which is used to determine its display style (colors).
        role: TextRole,
    }

    /// Defines the type of entity an AST element or sequence can be anchored to.
    /// - `Reg`: Anchored to a specific SSA register.
    /// - `Block`: Anchored to a specific control flow graph block.
    #[derive(Debug, PartialEq, Eq, Clone)]
    enum Anchor {
        Reg(decompiler::Reg),
        Block(decompiler::BlockID),
    }

    impl Seq {
        fn new() -> Self {
            Seq {
                anchor: None,
                children: Vec::new(),
                parentheses: false,
            }
        }

        fn with_anchor(mut self, anchor: Anchor) -> Self {
            self.anchor = Some(anchor);
            self
        }

        fn with_child(mut self, child: ExprTree) -> Self {
            self.add_child(child);
            self
        }
        fn add_child(&mut self, child: ExprTree) {
            self.children.push(child);
        }
    }

    /// Implementation block for the `Ast` struct, handling AST construction and UI rendering.
    impl Ast {
        /// Creates an empty AST.
        pub fn empty() -> Self {
            Ast {
                plan: Vec::new(),
                warnings: Vec::new(),
            }
        }

        /// Constructs an `Ast` from an SSA program using the `Builder`. This is the primary
        /// entry point for AST generation.
        pub fn from_ssa(ssa: &decompiler::SSAProgram) -> Self {
            Builder::new(ssa).build()
        }

        fn show_expr(&self, ui: &mut egui::Ui, value: &ExprTree) {
            match value {
                ExprTree::Null => {}
                ExprTree::Seq(Seq {
                    anchor: _,
                    children,
                    parentheses,
                }) => {
                    if *parentheses {
                        ui.label("(");
                    }
                    for child in children {
                        self.show_expr(ui, child);
                    }
                    if *parentheses {
                        ui.label(")");
                    }
                }
                ExprTree::Term(Term { anchor, text, role }) => {
                    if text.trim().is_empty() {
                        ui.label(format!("{:?}", value));
                    } else {
                        ui.label(text);
                    }
                }
            }
        }

        pub(crate) fn warnings(&self) -> impl ExactSizeIterator<Item = &dyn std::error::Error> {
            self.warnings.iter().map(|err| err.as_ref())
        }
    }

    pub trait Column {
        fn push_stmt(&mut self, ui: &mut egui::Ui, stmt: &Stmt);
    }
    impl<C: Column> Column for Option<&mut C> {
        fn push_stmt(&mut self, ui: &mut egui::Ui, stmt: &Stmt) {
            if let Some(c) = self {
                c.push_stmt(ui, stmt);
            }
        }
    }

    pub struct AstColumn<'a> {
        indent_level: usize,
        ast: &'a Ast,
    }
    impl<'a> AstColumn<'a> {
        pub fn new(ast: &'a Ast) -> Self {
            AstColumn {
                indent_level: 0,
                ast,
            }
        }
    }
    impl Column for AstColumn<'_> {
        fn push_stmt(&mut self, ui: &mut egui::Ui, stmt: &Stmt) {
            match stmt {
                &Stmt::BlockLabel(_) => {}
                Stmt::ExprStmt(value_xp) => {
                    ui.horizontal_top(|ui| {
                        ui.add_space(self.indent_level as f32 * 20.0);
                        self.ast.show_expr(ui, value_xp);
                    });
                }
                Stmt::NamedStmt {
                    name,
                    value: value_xp,
                } => {
                    ui.horizontal_top(|ui| {
                        ui.add_space(self.indent_level as f32 * 20.0);
                        ui.label(format!("let {} = ", name));
                        self.ast.show_expr(ui, value_xp);
                    });
                }
                Stmt::Dedent => {
                    self.indent_level -= 1;
                }
                Stmt::Indent => {
                    self.indent_level += 1;
                }
                Stmt::Comment(comment) => {
                    ui.horizontal(|ui| {
                        ui.visuals_mut().override_text_color = Some(egui::Color32::RED);
                        ui.label("//");
                        ui.label(comment);
                    });
                }
            }
        }
    }

    /// Renders the AST within the provided egui UI.
    pub fn show(
        ui: &mut egui::Ui,
        ast: &Ast,
        visible: &[bool],
        widths: &[f32],
        columns: &mut [&mut dyn Column],
    ) {
        assert_eq!(visible.len(), widths.len());
        assert_eq!(visible.len(), columns.len());

        let widths_masked: ArrayVec<f32, 16> = widths
            .into_iter()
            .zip(visible.into_iter())
            .map(|(&w, &is_visible)| if is_visible { w } else { 0.0 })
            .collect();

        columns::show(ui, &widths_masked, |col_uis| {
            for stmt in &ast.plan {
                if let Stmt::BlockLabel(_) = stmt {
                    col_uis.clear();
                    for ndx in 0..col_uis.count() {
                        col_uis.ui(ndx).separator();
                    }
                }

                for (ndx, col) in columns.iter_mut().enumerate() {
                    if visible[ndx] {
                        col.push_stmt(col_uis.ui(ndx), stmt);
                    }
                }
            }
        });
    }

    /// The `Builder` is responsible for transforming the SSAProgram into the `Ast`'s flat `Vec<Node>` representation.
    /// It traverses the SSA graph, making decisions about how instructions and control flow
    /// should be represented in the AST (e.g., inline values vs. `let` statements).
    struct Builder<'a> {
        block_order: Vec<BlockID>,
        plan: Vec<Stmt>,
        ssa: &'a decompiler::SSAProgram,
        value_mode: decompiler::RegMap<ValueMode>,

        // just to check that the algo is correct:
        block_status: decompiler::BlockMap<BlockStatus>,
        open_stack: Vec<decompiler::BlockID>,
        let_was_printed: decompiler::RegMap<bool>,
    }
    /// Tracks the processing status of a control flow graph block during AST generation.
    /// - `Pending`: Block has not yet been processed.
    /// - `Started`: Block processing has begun.
    /// - `Finished`: Block has been fully processed.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum BlockStatus {
        Pending,
        Started,
        Finished,
    }
    /// Determines how a register's value definition is handled in the AST.
    /// - `Inline`: The value is printed directly where it's used.
    /// - `NamedStmt`: The value is defined using a `let` statement and then referred to by name.
    /// - `UnnamedStmt`: The value's definition is printed as a statement, but it's not named (e.g., a call with unused return).
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
    /// Implementation block for the `Builder` struct, containing the logic for
    /// converting SSA form into the AST nodes.
    impl<'a> Builder<'a> {
        /// Creates a new `Builder` instance. It initializes the `value_mode` for each register
        /// based on usage count and instruction type, determining how each value will be represented.
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
                block_order: ssa.cfg().block_ids_rpo().collect(),
                plan: Vec::new(),
                ssa,
                value_mode,
                block_status,
                open_stack: Vec::new(),
                let_was_printed,
            }
        }

        /// Builds the `Ast` by starting the transformation process from the entry block
        /// of the SSA program.
        fn build(mut self) -> Ast {
            let block_order = std::mem::replace(&mut self.block_order, Vec::new());

            let mut iter = block_order.iter().copied().peekable();
            while self.transform_block(&mut iter) {}

            let mut warnings = Vec::new();
            {
                let block_order_check: Vec<_> = self
                    .plan
                    .iter()
                    .filter_map(|stmt| match stmt {
                        &Stmt::BlockLabel(bid) => Some(bid),
                        _ => None,
                    })
                    .collect();
                if block_order_check != block_order {
                    warnings.push(anyhow!(
                        "AST does not cover requested block order:\nrequested={:?};\nvisited={:?}",
                        block_order,
                        block_order_check
                    ));
                }
            }

            Ast {
                plan: self.plan,
                warnings,
            }
        }

        fn if_of_cond(&mut self, bid: BlockID) -> Option<Stmt> {
            let cond = self.ssa.find_last_matching(bid, |insn| {
                decompiler::match_get!(insn, decompiler::Insn::SetJumpCondition(cond), cond)
            })?;
            let value_xpr = self.transform_value_use(cond, 0);
            Some(Stmt::ExprStmt(
                Seq::new()
                    .with_child(mk_lit("if".into()).into())
                    .with_child(value_xpr)
                    .into(),
            ))
        }

        /// Transforms a basic block. This is the core of the AST generation for blocks.
        /// It iterates through scheduled registers, deciding whether to emit `let` statements,
        /// inline definitions, or unnamed statements. It then handles control flow (jumps, conditionals, returns).
        /// Finally, it recursively processes dominated blocks that haven't been visited yet.
        fn transform_block<I>(&mut self, iter: &mut Peekable<I>) -> bool
        where
            I: Iterator<Item = decompiler::BlockID>,
        {
            let Some(bid) = iter.next() else { return false };

            self.plan.push(Stmt::BlockLabel(bid));

            for reg in self.ssa.block_regs(bid) {
                match self.value_mode[reg] {
                    ValueMode::Inline => {
                        // just skip; reader expressions will pick this up
                    }
                    ValueMode::NamedStmt => {
                        self.let_was_printed[reg] = true;
                        self.plan.push(Stmt::NamedStmt {
                            name: Arc::new(format!("{:?}", reg)),
                            value: self.transform_expr(reg, 0),
                        })
                    }
                    ValueMode::UnnamedStmt => {
                        self.plan.push(Stmt::ExprStmt(self.transform_expr(reg, 0)))
                    }
                }
            }

            match self.ssa.cfg().block_cont(bid) {
                decompiler::BlockCont::Always(decompiler::Dest::Block(dest_bid))
                    if iter.peek() == Some(&dest_bid) =>
                {
                    // don't print the 'goto', just 'print' the next block inline
                    self.transform_block(iter);
                }
                decompiler::BlockCont::Always(dest) => {
                    let dest = self.transform_jump(bid, &dest);
                    self.plan.push(Stmt::ExprStmt(dest));
                }

                decompiler::BlockCont::Conditional { pos, neg } => {
                    let if_header = self.if_of_cond(bid).unwrap_or_else(|| {
                        Stmt::ExprStmt(mk_error_term("bug: no condition!".to_string()))
                    });
                    self.plan.push(if_header);

                    self.plan.push(Stmt::Indent);
                    match pos {
                        decompiler::Dest::Block(pos_bid) if iter.peek() == Some(&pos_bid) => {
                            self.transform_block(iter);
                        }
                        _ => {
                            let jump = self.transform_jump(bid, &pos);
                            self.plan.push(Stmt::ExprStmt(jump));
                        }
                    }
                    self.plan.push(Stmt::Dedent);

                    self.plan.push(Stmt::ExprStmt(mk_kw("else".into()).into()));
                    self.plan.push(Stmt::Indent);
                    match neg {
                        decompiler::Dest::Block(neg_bid) if iter.peek() == Some(&neg_bid) => {
                            self.transform_block(iter);
                        }
                        _ => {
                            let jump = self.transform_jump(bid, &neg);
                            self.plan.push(Stmt::ExprStmt(jump));
                        }
                    }
                    self.plan.push(Stmt::Dedent);

                    self.plan.push(Stmt::ExprStmt(mk_kw("endif".into()).into()));
                }
            }

            true
        }

        /// Transforms a value (SSA register) into its AST representation.
        /// The representation depends on its `ValueMode` (inline, named statement, or unnamed statement).
        fn transform_value_use(
            &self,
            reg: decompiler::Reg,
            parent_prec: decompiler::PrecedenceLevel,
        ) -> ExprTree {
            // TODO! specific representation of operands
            match self.value_mode[reg] {
                ValueMode::Inline => self.transform_expr(reg, parent_prec),
                ValueMode::NamedStmt => {
                    let reg_ref = mk_reg_ref(reg);
                    if self.let_was_printed[reg] {
                        reg_ref
                    } else {
                        Seq::new()
                            .with_child(mk_error_term("bug:let!".to_string()))
                            .with_child(reg_ref)
                            .into()
                    }
                }
                ValueMode::UnnamedStmt => Seq::new()
                    .with_child(mk_error_term("bug:unnamed-ref!".to_string()))
                    .with_child(mk_reg_ref(reg))
                    .into(),
            }
        }

        /// Transforms an SSA definition (instruction) into its AST representation.
        /// This function handles various instruction types and generates corresponding
        /// AST nodes, including arithmetic operations, memory access, calls, etc.
        /// It also handles precedence for operators by adding parentheses when necessary.
        ///
        /// This function only generates expressions (or fragments of them). It
        /// never generates a `let _ = ` form.
        fn transform_expr(
            &self,
            reg: decompiler::Reg,
            parent_prec: decompiler::PrecedenceLevel,
        ) -> ExprTree {
            let mut insn = self.ssa[reg].get();
            let prec = decompiler::precedence(&insn);

            let anchor = Anchor::Reg(reg);
            let mut expr: ExprTree = match insn {
                Insn::Void => mk_kw("void".into()),
                Insn::True => mk_kw("true".into()),
                Insn::False => mk_kw("false".into()),
                Insn::UndefinedBool => ExprTree::Seq(
                    Seq::new()
                        .with_anchor(anchor)
                        .with_child(mk_kw("undefined".into()).into())
                        .with_child(mk_kw("bool".into()).into()),
                ),
                Insn::UndefinedBytes { size } => ExprTree::Seq(
                    Seq::new()
                        .with_anchor(anchor)
                        .with_child(mk_kw("undefined".into()).into())
                        .with_child(mk_kw("bytes".into()).into())
                        .with_child(mk_kw(format!("{}", size).into()).into()),
                ),

                Insn::Phi => self.transform_regular_insn(reg, "Phi", std::iter::empty()),
                Insn::Const { value, size: _ } => mk_lit(format!("{}", value).into()),

                Insn::Ancestral(aname) => ExprTree::Term(Term {
                    text: aname.name().to_string(),
                    anchor: Some(Anchor::Reg(reg)),
                    role: TextRole::RegRef,
                }),

                Insn::StoreMem { addr, value } => Seq::new()
                    .with_anchor(anchor)
                    .with_child(self.transform_value_use(addr, 255))
                    .with_child(mk_kw(".*".into()))
                    .with_child(ExprTree::Term(Term {
                        text: ":=".to_string(),
                        anchor: Some(Anchor::Reg(reg)),
                        role: TextRole::RegDef,
                    }))
                    .with_child(self.transform_value_use(value, prec))
                    .into(),

                Insn::LoadMem { addr, size: _ } => Seq::new()
                    .with_anchor(anchor)
                    .with_child(self.transform_value_use(addr, 0))
                    .with_child(mk_kw(".*".into()))
                    .into(),

                Insn::Part { src, offset, size } => Seq::new()
                    .with_anchor(anchor)
                    .with_child(self.transform_value_use(src, prec))
                    .with_child(mk_kw(format!("[{} .. {}]", offset, offset + size).into()))
                    .into(),

                Insn::Concat { lo, hi } => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(self.transform_value_use(hi, prec))
                    .with_child(mk_kw("++".into()))
                    .with_child(self.transform_value_use(lo, prec))
                    .into(),

                Insn::StructGetMember {
                    struct_value,
                    name,
                    size: _,
                } => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(self.transform_value_use(struct_value, prec))
                    .with_child(mk_kw(format!(".{}", name).into()))
                    .into(),
                Insn::ArrayGetElement { array, index, size } => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(self.transform_value_use(array, prec))
                    .with_child(mk_kw(format!("[{}]", index).into()))
                    .into(),

                Insn::Widen {
                    reg,
                    target_size,
                    sign: _,
                } => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(self.transform_value_use(reg, prec))
                    .with_child(mk_kw(format!("as i{}", target_size * 8).into()))
                    .into(),

                Insn::Arith(op, a, b) => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(self.transform_value_use(a, prec))
                    .with_child(ExprTree::Term(Term {
                        text: op.symbol().to_string(),
                        anchor: Some(Anchor::Reg(reg)),
                        role: TextRole::Kw,
                    }))
                    .with_child(self.transform_value_use(b, prec))
                    .into(),

                Insn::ArithK(op, a, bk) => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(self.transform_value_use(a, prec))
                    .with_child(ExprTree::Term(Term {
                        text: op.symbol().to_string(),
                        anchor: Some(Anchor::Reg(reg)),
                        role: TextRole::Kw,
                    }))
                    .with_child(ExprTree::Term(Term {
                        text: format!("{}", bk),
                        anchor: None,
                        role: TextRole::Literal,
                    }))
                    .into(),

                Insn::Cmp(op, a, b) => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(self.transform_value_use(a, prec))
                    .with_child(ExprTree::Term(Term {
                        text: op.symbol().to_string(),
                        anchor: Some(Anchor::Reg(reg)),
                        role: TextRole::Kw,
                    }))
                    .with_child(self.transform_value_use(b, prec))
                    .into(),

                Insn::Bool(op, a, b) => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(self.transform_value_use(a, prec))
                    .with_child(ExprTree::Term(Term {
                        text: op.symbol().to_string(),
                        anchor: Some(Anchor::Reg(reg)),
                        role: TextRole::Kw,
                    }))
                    .with_child(self.transform_value_use(b, prec))
                    .into(),

                Insn::Not(arg) => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(ExprTree::Term(Term {
                        text: "!".to_string(),
                        anchor: None,
                        role: TextRole::Kw,
                    }))
                    .with_child(self.transform_value_use(arg, prec))
                    .into(),

                Insn::NotYetImplemented(msg) => ExprTree::Term(Term {
                    text: format!("NYI:{}", msg),
                    anchor: None,
                    role: TextRole::Kw,
                }),

                Insn::SetReturnValue(_) | Insn::SetJumpCondition(_) | Insn::SetJumpTarget(_) => {
                    ExprTree::Null
                }

                Insn::Call { callee, first_arg } => {
                    let callee_type_name = self
                        .ssa
                        .value_type(callee)
                        .and_then(|tyid| self.ssa.types().name(tyid))
                        .map(|s| s.to_string());
                    let mut seq = Seq::new().with_anchor(Anchor::Reg(reg));

                    if let Some(name) = callee_type_name {
                        seq.add_child(ExprTree::Term(Term {
                            text: name,
                            anchor: Some(Anchor::Reg(callee)),
                            role: TextRole::Ident,
                        }));
                    } else {
                        seq.add_child(self.transform_value_use(callee, prec));
                    }

                    seq.add_child(mk_kw("(".into()));

                    for (_ndx, arg) in self.ssa.get_call_args(first_arg).enumerate() {
                        // if ndx > 0 {
                        //     seq.add_child(mk_kw(",".into()));
                        // }
                        seq.add_child(self.transform_value_use(arg, prec));
                    }

                    seq.with_child(mk_kw(")".into())).into()
                }

                Insn::CArg { .. } => mk_error_term("<bug:CArg>".into()),
                Insn::Control(_) => mk_error_term("<bug:Control>".into()),

                Insn::Upsilon { value, phi_ref } => Seq::new()
                    .with_anchor(Anchor::Reg(reg))
                    .with_child(ExprTree::Term(Term {
                        text: format!("{:?}", phi_ref),
                        anchor: Some(Anchor::Reg(phi_ref)),
                        role: TextRole::RegRef,
                    }))
                    .with_child(mk_kw(":=".into()))
                    .with_child(self.transform_value_use(value, prec))
                    .into(),

                Insn::Get(_)
                | Insn::OverflowOf(_)
                | Insn::CarryOf(_)
                | Insn::SignOf(_)
                | Insn::IsZero(_)
                | Insn::Parity(_) => self.transform_regular_insn(
                    reg,
                    Self::opcode_name(&insn),
                    insn.input_regs_iter().map(|x| *x),
                ),
            };

            if let ExprTree::Seq(seq) = &mut expr {
                seq.parentheses = prec < parent_prec;
            }

            expr
        }

        /// Returns a static string name for a given SSA instruction opcode.
        fn opcode_name(insn: &Insn) -> &'static str {
            // TODO use facet for this!
            match insn {
                Insn::Void => "Void",
                Insn::True => "True",
                Insn::False => "False",
                Insn::Const { .. } => "Const",
                Insn::Get(_) => "Get",
                Insn::Part { .. } => "Part",
                Insn::Concat { .. } => "Concat",
                Insn::StructGetMember { .. } => "StructGetMember",
                Insn::ArrayGetElement { .. } => "ArrayGetElement",
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
                Insn::UndefinedBool => "UndefinedBool",
                Insn::UndefinedBytes { .. } => "UndefinedBytes",
                Insn::Ancestral(_) => "Ancestral",
                Insn::Phi => "Phi",
                Insn::Upsilon { .. } => "Upsilon",
            }
        }

        /// Transforms a generic SSA instruction (opcode and inputs) into a flow-style AST sequence.
        /// It formats the instruction as `opcode(input1, input2, ...)`
        fn transform_regular_insn(
            &self,
            result: decompiler::Reg,
            opcode: &'static str,
            inputs: impl IntoIterator<Item = decompiler::Reg>,
        ) -> ExprTree {
            let mut seq = Seq::new()
                .with_anchor(Anchor::Reg(result))
                .with_child(ExprTree::Term(Term {
                    anchor: None,
                    text: opcode.to_string(),
                    role: TextRole::Generic,
                }))
                .with_child(mk_kw("(".into()));

            for (_ndx, input) in inputs.into_iter().enumerate() {
                // if ndx > 0 {
                //     seq.add_child(mk_kw(",".into()));
                // }
                seq.add_child(self.transform_value_use(input, 0));
            }

            seq.add_child(mk_kw(")".into()));
            seq.into()
        }

        /// Transforms a control flow destination into its AST representation.
        /// Handles external jumps, jumps to other blocks (which may be inlined or
        /// represented as `goto` statements), indirect jumps, and function returns.
        ///
        /// `next_bid` is used, when available, as the BlockID that is laid out
        /// right after `src_bid` in order to lay out certain blocks inline or
        /// avoid/insert `goto` expression.
        fn transform_jump(
            &mut self,
            src_bid: decompiler::BlockID,
            dest: &decompiler::Dest,
        ) -> ExprTree {
            match dest {
                decompiler::Dest::Ext(addr) => Seq::new()
                    .with_child(mk_kw("goto".into()))
                    .with_child(mk_lit(format!("{}", *addr).into()))
                    .into(),
                decompiler::Dest::Block(bid) => Seq::new()
                    .with_child(mk_kw("goto".into()))
                    .with_child(
                        Term {
                            text: format!("B{}", bid.as_number()),
                            anchor: Some(Anchor::Block(*bid)),
                            role: TextRole::BlockRef,
                        }
                        .into(),
                    )
                    .into(),
                decompiler::Dest::Indirect => {
                    let tgt = self.ssa.find_last_matching(src_bid, |insn| {
                        decompiler::match_get!(insn, decompiler::Insn::SetJumpTarget(tgt), tgt)
                    });

                    if let Some(tgt) = tgt {
                        Seq::new()
                            .with_child(mk_kw("goto".into()))
                            .with_child(mk_kw("*".into()))
                            .with_child(self.transform_value_use(tgt, 0))
                            .into()
                    } else {
                        mk_error_term("<bug:no jump target>".into()).into()
                    }
                }
                decompiler::Dest::Return => {
                    let ret = self.ssa.find_last_matching(src_bid, |insn| {
                        decompiler::match_get!(insn, decompiler::Insn::SetReturnValue(val), val)
                    });

                    if let Some(ret) = ret {
                        Seq::new()
                            .with_child(mk_kw("return".into()))
                            .with_child(self.transform_value_use(ret, 0))
                            .into()
                    } else {
                        mk_error_term("<bug:no return value>".into()).into()
                    }
                }
                decompiler::Dest::Undefined => mk_kw("goto undefined".into()).into(),
            }
        }
    }

    fn mk_reg_ref(reg: decompiler::Reg) -> ExprTree {
        ExprTree::Term(Term {
            anchor: Some(Anchor::Reg(reg)),
            text: format!("{:?}", reg),
            role: TextRole::RegRef,
        })
    }

    fn mk_kw(text: Cow<'static, str>) -> ExprTree {
        ExprTree::Term(Term {
            anchor: None,
            text: text.to_string(),
            role: TextRole::Kw,
        })
    }

    fn mk_lit(text: Cow<'static, str>) -> ExprTree {
        ExprTree::Term(Term {
            anchor: None,
            text: text.to_string(),
            role: TextRole::Literal,
        })
    }

    fn mk_error_term(text: String) -> ExprTree {
        ExprTree::Term(Term {
            anchor: None,
            text,
            role: TextRole::Error,
        })
    }
}

mod columns {
    use arrayvec::ArrayVec;

    pub const EXPANDING_WIDTH: f32 = f32::INFINITY;

    /// Sets up a multi-column layout where each column can be filled
    /// independently by the given closure.
    ///
    /// The desired width for each column is set via the `width` array (also
    /// determines the number of columns). Use `EXPANDING_WIDTH` for columns
    /// that should take up the remaining available space equally.
    ///
    /// The `add_contents` closure is given an array of [`Column`], allowing
    /// access to a separate `egui::Ui` for each column.
    pub fn show(ui: &mut egui::Ui, widths: &[f32], add_contents: impl FnOnce(&mut Columns)) {
        ui.horizontal(move |ui| {
            let width_fixed: f32 = widths.into_iter().filter(|w| w.is_finite()).sum();
            let width_expanding_count = widths.into_iter().filter(|w| w.is_infinite()).count();
            let width_available = ui.available_width();
            let width_expanding_each: f32 =
                (width_available - width_fixed) / width_expanding_count as f32;

            let uis: ArrayVec<_, MAX_COUNT> = widths
                .into_iter()
                .map(|width| {
                    let width = match *width {
                        w if w.is_infinite() => width_expanding_each,
                        other => other,
                    };

                    let (_, col_rect) = ui.allocate_space(egui::vec2(width, 0.0));
                    let col_ui = ui.new_child(
                        egui::UiBuilder::new()
                            .max_rect(col_rect)
                            .layout(egui::Layout::top_down(egui::Align::TOP)),
                    );
                    ui.advance_cursor_after_rect(col_rect);

                    col_ui
                })
                .collect();

            let mut columns = Columns { uis };

            add_contents(&mut columns);

            for col in columns.uis {
                // we're doing a custom layout, so we have to do this where egui
                // would have done this automatically
                ui.expand_to_include_rect(col.min_rect());
            }
        });
    }

    const MAX_COUNT: usize = 16;

    pub struct Columns {
        uis: ArrayVec<egui::Ui, MAX_COUNT>,
    }
    impl Columns {
        pub fn ui(&mut self, ndx: usize) -> &mut egui::Ui {
            &mut self.uis[ndx]
        }

        pub fn count(&self) -> usize {
            self.uis.len()
        }

        pub fn clear(&mut self) {
            let y_clear = self
                .uis
                .iter()
                .map(|ui| ui.min_rect().max.y)
                .reduce(|ay, by| ay.max(by))
                .unwrap();

            for col in &mut self.uis {
                col.advance_cursor_after_rect(col.cursor().with_max_y(y_clear));
            }
        }
    }
}

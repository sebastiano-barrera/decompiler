use std::{
    collections::BTreeMap,
    f32,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use decompiler::{BlockID, Executable};
use ouroboros::self_referencing;
use tracing_subscriber::EnvFilter;

mod search;

fn main() {
    tracing_subscriber::fmt::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let mut args = std::env::args();
    let self_name = args.next().unwrap();
    let Some(exe_filename) = args.next() else {
        eprintln!(
            "usage: {} EXE_NAME\n\nOpens the given executable in the decompiler GUI",
            self_name
        );
        std::process::exit(1);
    };
    let exe_filename: PathBuf = exe_filename.into();

    let app = match App::new(&exe_filename) {
        Ok(app) => Box::new(app),
        Err(err) => {
            eprintln!("could not load executable: {:?}", err);
            std::process::exit(1);
        }
    };
    let res = eframe::run_native(
        "decompiler test app",
        eframe::NativeOptions::default(),
        Box::new(move |cctx| {
            cctx.egui_ctx.set_theme(egui::ThemePreference::System);
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
    exe: Exe,
    function_selector: Option<FunctionSelector>,
    type_search: Option<search::TypeSearchEngine>,
    func_type_force_dialog: Option<FuncTypeForceDialog>,
    func_view: Option<Result<FunctionView, decompiler::Error>>,
    type_selector: Option<TypeSelectorDialog>,
}
struct FunctionView {
    df: decompiler::DecompiledFunction,
    /// Cached from df's function type
    param_names: Vec<String>,
    name_of_reg: decompiler::RegMap<String>,
    ast: Option<decompiler::Ast>,
    cfg_layout: cfg::Layout,
    asm: Assembly,
    type_windows: Vec<TypeDetailsWindow>,

    problems_is_visible: bool,
    #[allow(dead_code)]
    problems_title: String,
    #[allow(dead_code)]
    problems_error: Option<String>,

    hl: hl::State,
    asm_hl_mask: Cache<BlockID, Vec<bool>>,

    is_asm_visible: bool,
    is_cfg_visible: bool,
    is_block_order_visible: bool,
    block_order: Vec<BlockID>,
    block_order_error: Option<String>,
}

struct TypeDetailsWindow {
    tyid: decompiler::ty::TypeID,
    // window title, cached
    title: String,
    is_open: bool,
}
impl TypeDetailsWindow {
    fn new(tyid: decompiler::ty::TypeID, rtx: &decompiler::ty::ReadTxRef<'_>) -> Self {
        let ty_dump = decompiler::pp::pp_to_string(|pp| {
            rtx.dump_type_ref(pp, tyid).unwrap();
        });
        TypeDetailsWindow {
            tyid,
            title: format!("{} ({})", ty_dump, tyid.0),
            is_open: true,
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top_panel").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                self.show_topbar(ui);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                    egui::widgets::global_theme_preference_switch(ui);

                    if let Some(Ok(func_view)) = &mut self.func_view {
                        ui.toggle_value(
                            &mut func_view.is_asm_visible,
                            egui::RichText::new("ASM").monospace(),
                        );
                        ui.toggle_value(
                            &mut func_view.is_block_order_visible,
                            egui::RichText::new("BlkOrd").monospace(),
                        );
                        ui.toggle_value(
                            &mut func_view.is_cfg_visible,
                            egui::RichText::new("CFG").monospace(),
                        );
                    }
                });
            });

            if let Some(Ok(stage_func)) = self.func_view.as_mut() {
                stage_func.show_topbar_2(ui, self.exe.borrow_exe());
            }
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.show_central(ui);
        });
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // NOTE: a bright gray makes the shadows of the windows look weird.
        // We use a bit of transparency so that if the user switches on the
        // `transparent()` option they get immediate results.
        egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).to_normalized_gamma_f32()

        // _visuals.window_fill() would also be a natural choice
    }

    fn persist_egui_memory(&self) -> bool {
        true
    }
}

impl App {
    fn new(path: &Path) -> Result<Self> {
        let time_load_start = Instant::now();

        let exe = load_executable(path)?;

        let load_time = Instant::now().duration_since(time_load_start);
        eprintln!("exe loaded in {:?}", load_time);

        let mut app = App {
            exe,
            function_selector: None,
            func_view: None,
            type_search: None,
            type_selector: None,
            func_type_force_dialog: None,
        };
        app.start_function_selector();
        Ok(app)
    }

    fn show_topbar(&mut self, ui: &mut egui::Ui) {
        if ui.button("Lookup types…").clicked() {
            self.start_type_lookup();
        }
        if ui.button("Load function…").clicked() {
            self.start_function_selector();
        }

        self.exe.borrow_exe().types();

        match self.func_view.as_mut() {
            Some(Ok(stage_func)) => {
                ui.label(stage_func.df.name());
                if ui.button("Force type...").clicked() {
                    self.func_type_force_dialog =
                        Some(FuncTypeForceDialog::new("func type force dialog"));
                }
            }
            Some(Err(_)) => {
                // error shown in central area
            }
            None => {
                ui.label("No function loaded.");
            }
        }
    }

    fn ensure_type_search_engine(&mut self) -> &mut search::TypeSearchEngine {
        self.type_search.get_or_insert_with(|| {
            let mut engine = search::TypeSearchEngine::new();
            let tys = Arc::clone(self.exe.borrow_exe().types());
            engine.load_types(tys);
            engine
        })
    }

    fn start_type_lookup(&mut self) {
        self.ensure_type_search_engine();
        self.type_selector = Some(TypeSelectorDialog::new("modal type selector"));
    }

    fn start_function_selector(&mut self) {
        let mut all_names: Vec<_> = self
            .exe
            .borrow_exe()
            .function_names()
            .map(|s| s.to_owned())
            .collect();
        all_names.sort();
        self.function_selector = Some(FunctionSelector::new("modal load function", all_names));
    }

    fn show_central(&mut self, ui: &mut egui::Ui) {
        match self.func_view.as_mut() {
            Some(Ok(stage_func)) => {
                stage_func.show(ui, self.exe.borrow_exe());

                if let (Some(type_selector), Some(engine)) =
                    (&mut self.type_selector, &mut self.type_search)
                {
                    let res = type_selector.show(ui, engine);

                    if let Some(tyid) = res.inner {
                        if let Ok(rtx) = self.exe.borrow_exe().types().read_tx() {
                            let rtx = rtx.read();
                            stage_func.add_type_window(tyid, &rtx);
                        }
                        self.type_selector = None;
                    } else if res.should_close() {
                        self.type_selector = None;
                    }
                }
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
            if let Some(function_name) = res.inner {
                let function_name = function_name.clone();
                self.reload_function(&decompiler::DecompileParams {
                    function_name: &function_name,
                    force_function_type: None,
                });
                self.function_selector = None;
            } else if res.should_close() {
                self.function_selector = None;
            }
        }

        if self.func_type_force_dialog.is_some() {
            self.ensure_type_search_engine();
            let engine = self.type_search.as_mut().unwrap();
            let dialog = self.func_type_force_dialog.as_mut().unwrap();
            let mut res = dialog.show(ui, engine);
            if let Some(type_builder) = res.inner.take() {
                let types = self.exe.borrow_exe().types();
                match type_builder.build_types(types) {
                    Ok(new_tyid) => {
                        // changing the function type potentially changes the
                        // entire SSA and AST, so a full reload is necessary
                        if let Some(Ok(func_view)) = &mut self.func_view {
                            let function_name = func_view.df.name().to_string();
                            self.reload_function(&decompiler::DecompileParams {
                                function_name: &function_name,
                                force_function_type: Some(new_tyid),
                            });
                        }
                    }
                    Err(err) => {
                        // TODO make a dialog or toast or something
                        eprintln!("error building the requested type: {err:?}");
                    }
                }
            }
            if res.should_close() {
                self.func_type_force_dialog = None;
            }
        }
    }

    fn reload_function(&mut self, params: &decompiler::DecompileParams) {
        let mut stage_func_or_err = self.exe.with_exe_mut(|exe| {
            eprintln!("reload function: {:?}", params);
            let df = exe.decompile_function_ex(params)?;
            Ok(FunctionView::new(df, exe))
        });

        if let Ok(func_view) = &mut stage_func_or_err {
            func_view.problems_is_visible =
                func_view.df.error().is_some() || !func_view.df.warnings().is_empty();
        }

        self.func_view = Some(stage_func_or_err);
    }
}

impl FunctionView {
    fn new(df: decompiler::DecompiledFunction, exe: &Executable) -> Self {
        let title = format!(
            "{}{} warnings",
            if df.error().is_some() { "ERROR!, " } else { "" },
            df.warnings().len(),
        );
        let error_label = df.error().map(|err| err.to_string());

        let machine_code = df.machine_code(exe);
        let base_ip = df.base_ip();
        let asm = Assembly::disassemble_x86(machine_code, base_ip);

        let name_of_reg = if let Some(prog) = df.ssa() {
            let mut names = decompiler::RegMap::for_program(prog, String::new());
            for reg in prog.registers() {
                names[reg] = format!("r{}", reg.reg_index());
            }
            names
        } else {
            decompiler::RegMap::empty()
        };

        let param_names = df
            .ssa()
            .and_then(|ssa| get_param_names(exe, ssa))
            .unwrap_or_default();

        let mut fv = FunctionView {
            df,
            param_names,
            name_of_reg,
            problems_is_visible: false,
            problems_title: title,
            problems_error: error_label,
            cfg_layout: cfg::Layout::default(),
            ast: None,
            asm,
            type_windows: Vec::new(),
            hl: hl::State::empty(),
            asm_hl_mask: Cache::default(),
            is_asm_visible: false,
            is_block_order_visible: false,
            block_order: Vec::new(),
            block_order_error: None,
            is_cfg_visible: false,
        };
        fv.rebuild_ast();
        fv
    }

    // Rebuild the AST from the SSA, using the current block order if provided.
    //
    // If `block_order` is empty, the default block order from the AST will be
    // used (or an error will be shown if the AST cannot be built with any block
    // order).
    fn rebuild_ast(&mut self) {
        self.ast = None;
        self.block_order_error = None;
        self.cfg_layout = cfg::Layout::default();

        let Some(ssa) = self.df.ssa() else {
            return;
        };

        self.cfg_layout = cfg::Layout::from_cfg(ssa.cfg());

        let mut builder = decompiler::AstBuilder::new(ssa);
        builder.set_all_blocks_named(true);
        self.block_order_error = builder
            .set_block_order(&self.block_order)
            .err()
            .map(|err| err.to_string());
        let mut ast = builder.build();

        decompiler::ast::edit::cleanup_with_ssa(&mut ast, self.df.ssa_mut().unwrap());

        if self.block_order.is_empty() {
            self.block_order = Vec::from(ast.block_order());
        }
        self.ast = Some(ast);
    }

    /// Show the "second" top bar, i.e. the are right under the top bar
    fn show_topbar_2(&mut self, ui: &mut egui::Ui, exe: &decompiler::Executable) {
        let Some(tyid) = self.df.ssa().and_then(|ssa| ssa.function_type_id()) else {
            ui.label("unknown function type");
            return;
        };

        let Some(rtx) = exe.types().read_tx().ok() else {
            ui.label("could not start read tx");
            return;
        };
        let ty = match rtx.read().get_through_alias(tyid) {
            Ok(Some(ty)) => ty,
            Ok(None) => {
                ui.label(format!("no such type: {:?}", tyid));
                return;
            }
            Err(err) => {
                ui.label(format!("could not fetch type: {:?}: {:?}", tyid, err));
                return;
            }
        };
        let func_ty = match ty.into_owned() {
            decompiler::ty::Ty::Subroutine(func_ty) => func_ty,
            other => {
                ui.label(format!("not a subroutine type: {:?}", other));
                return;
            }
        };

        ui.horizontal_wrapped(move |ui| {
            ui.label(egui::RichText::new("RETURN TYPE").small().strong());
            self.ui_type_ref(ui, func_ty.return_tyid, &rtx.read());

            ui.add_space(10.0);

            let count = func_ty.param_names.len();
            let title_label = format!("ARGUMENTS ({})", count);
            ui.label(egui::RichText::new(title_label).small().strong());

            for (arg_ndx, (param_name, param_tyid)) in func_ty
                .param_names
                .iter()
                .zip(&func_ty.param_tyids)
                .enumerate()
            {
                let arg_ndx = arg_ndx.try_into().unwrap();
                let param_name = param_name
                    .as_ref()
                    .map(|arc| arc.as_str())
                    .unwrap_or("<unnamed>");
                ui.horizontal(|ui| {
                    ui.label(" • ");
                    ast::print_ident_def(ui, &mut self.hl, param_name, hl::Focus::Arg(arg_ndx));
                    ui.label(": ");
                    self.ui_type_ref(ui, *param_tyid, &rtx.read());
                });
            }
        });
    }

    fn ui_type_ref(
        &mut self,
        ui: &mut egui::Ui,
        tyid: decompiler::ty::TypeID,
        rtx: &decompiler::ty::ReadTxRef,
    ) {
        let text = {
            let mut bytes = Vec::new();
            let mut pp = decompiler::pp::PrettyPrinter::start(&mut bytes);
            rtx.dump_type_ref(&mut pp, tyid)
                .map(|_| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_else(|_| "<error in printing>".to_string())
        };

        let res = ui.button(text);
        if res.clicked() {
            self.add_type_window(tyid, rtx);
        } else {
            res.on_hover_text(format!("{:?}", tyid));
        }
    }

    fn add_type_window(
        &mut self,
        tyid: decompiler::ty::TypeID,
        rtx: &decompiler::ty::ReadTxRef<'_>,
    ) {
        if self
            .type_windows
            .iter()
            .find(|win| win.tyid == tyid)
            .is_none()
        {
            self.type_windows.push(TypeDetailsWindow::new(tyid, rtx));
        }
    }

    fn show(&mut self, ui: &mut egui::Ui, exe: &Executable) {
        self.hl.frame_started();
        self.show_panels(ui, exe);
        self.show_central(ui);

        if !self.hl.hovered.was_set_this_frame() {
            self.hl.hovered.set_focus(None);
        }

        self.show_type_details_windows(ui, exe.types());
    }

    fn show_type_details_windows(&mut self, ui: &mut egui::Ui, types: &decompiler::ty::TypeSet) {
        let Ok(rtx) = types.read_tx() else {
            return;
        };

        let mut type_windows = std::mem::take(&mut self.type_windows);
        for win in type_windows.iter_mut() {
            egui::Window::new(&win.title)
                .id(egui::Id::new(win.tyid))
                .open(&mut win.is_open)
                // this window is NOT resizable; rather, the content occasionally is
                .resizable(false)
                .show(ui.ctx(), |ui| {
                    self.show_type_window_content(ui, rtx.read(), win.tyid);
                });
        }

        type_windows.retain(|win| win.is_open);
        // some windows may have been added in the meantime (during show_type_details_content)
        type_windows.append(&mut self.type_windows);
        std::mem::swap(&mut self.type_windows, &mut type_windows);
    }

    fn show_type_window_content(
        &mut self,
        ui: &mut egui::Ui,
        rtx: decompiler::ty::ReadTxRef<'_>,
        tyid: decompiler::ty::TypeID,
    ) {
        // TODO handle recursive types
        use decompiler::ty::{Int, Ty};

        let ty = match rtx.get_through_alias(tyid) {
            Ok(Some(ty)) => ty,
            Ok(None) => {
                ui.label("Type not found");
                return;
            }
            Err(err) => {
                ui.label(format!("Error reading type: {:?}", err));
                return;
            }
        };

        match ty.as_ref() {
            Ty::Flag => {
                ui.heading("Flag type");
                ui.small(
                    "(like a bool, but does not occupy physical memory/registers; e.g. CPU flags)",
                );
            }
            Ty::Int(Int { size, signed }) => {
                ui.heading("Integer");
                show_details_ty_int(ui, ("integer_type", tyid), *size, signed);
            }
            Ty::Float(decompiler::ty::Float { size }) => {
                ui.heading("Floating-point");
                ui.label(format!("Size: {} bits", size * 8));
            }
            Ty::Bool(decompiler::ty::Bool { size }) => {
                ui.heading("Boolean");
                ui.label(format!("Size: {} bits", size * 8));
            }
            Ty::Enum(decompiler::ty::Enum {
                variants,
                base_type,
            }) => {
                ui.heading("Enumeration");
                ui.add_space(5.0);

                ui.horizontal(|ui| {
                    ui.label("Base type:");
                    show_details_ty_int(
                        ui,
                        ("enum_base_type", tyid),
                        base_type.size,
                        &base_type.signed,
                    );
                });
                ui.add_space(5.0);

                ui.label("Variants:");
                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        egui::Grid::new(("enum_variants", tyid)).show(ui, |ui| {
                            for variant in variants {
                                ui.monospace(format!("{}", variant.value));
                                ui.monospace(variant.name.as_str());
                                ui.end_row();
                            }
                        });
                    });
            }
            Ty::Ptr(pointee_tyid) => {
                ui.horizontal(|ui| {
                    ui.label("Pointer ☞");
                    ui.add_space(5.0);
                    ui.vertical(|ui| {
                        self.show_type_window_content(ui, rtx, *pointee_tyid);
                    });
                });
            }
            Ty::Struct(decompiler::ty::Struct { members, size }) => {
                ui.heading("Struct");
                ui.label(format!("Size: {} bytes", size));
                ui.add_space(5.0);
                ui.label("Members:");
                egui::Resize::default().show(ui, |ui| {
                    egui::ScrollArea::both()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            egui::Grid::new(("struct_members", tyid)).show(ui, |ui| {
                                ui.strong("Name");
                                ui.strong("Offset");
                                ui.strong("Type");
                                ui.end_row();

                                for member in members {
                                    ui.monospace(member.name.as_str());
                                    ui.label(format!("{}", member.offset));
                                    self.ui_type_ref(ui, member.tyid, &rtx);
                                    ui.end_row();
                                }
                            });
                        });
                });
            }
            Ty::Array(decompiler::ty::Array {
                element_tyid,
                index_subrange,
            }) => {
                ui.heading("Array");

                egui::Grid::new(("array_details", tyid)).show(ui, |ui| {
                    ui.label("Indices:");
                    let decompiler::ty::Subrange { lo, hi } = index_subrange;
                    if let Some(hi) = hi {
                        ui.label(format!("{} to {}", lo, hi));
                    } else {
                        ui.label(format!("{} onwards (to indefinite length)", lo));
                    }
                    ui.end_row();

                    ui.label("Element type:");
                    self.ui_type_ref(ui, *element_tyid, &rtx);
                    ui.end_row();
                });
            }
            Ty::Subroutine(decompiler::ty::Subroutine {
                return_tyid,
                param_names,
                param_tyids,
            }) => {
                ui.heading("Subroutine");
                ui.add_space(5.0);

                ui.label("Return type:");
                self.ui_type_ref(ui, *return_tyid, &rtx);
                ui.add_space(5.0);

                let count = param_tyids.len();
                ui.label(format!("Parameters ({}):", count));
                egui::Grid::new(("subroutine_params", tyid)).show(ui, |ui| {
                    for (param_name, param_tyid) in param_names.iter().zip(param_tyids) {
                        let param_name = param_name
                            .as_ref()
                            .map(|arc| arc.as_str())
                            .unwrap_or("<unnamed>");
                        ui.monospace(param_name);
                        self.ui_type_ref(ui, *param_tyid, &rtx);
                        ui.end_row();
                    }
                });
            }
            Ty::Unknown => {
                ui.heading("Unknown type");
                ui.label("No further information is available for this type.");
            }
            Ty::Void => {
                ui.heading("Void type");
                ui.label("Represents the absence of a value, or a value of an unknown type.");
            }
            Ty::Alias(aliasee_tyid) => {
                ui.heading(format!("Alias to type ID: {:?}", aliasee_tyid));
                self.show_type_window_content(ui, rtx, *aliasee_tyid);
            }
        }
    }

    /// Show the side panels (top, bottom, right, left) to be laid out around
    /// the central are, with content specific to this function.
    ///
    /// To be called before `show_central`.
    fn show_panels(&mut self, ui: &mut egui::Ui, exe: &Executable) {
        if self.is_asm_visible {
            egui::Panel::left("asm_panel").show_inside(ui, |ui| {
                self.show_assembly(ui);
            });
        }

        if self.is_block_order_visible {
            let is_block_order_changed = egui::Panel::left("block_order_panel")
                .show_inside(ui, |ui| {
                    ui.heading("Block order");

                    let res = egui_dnd::dnd(ui, "block_order_dnd").show_vec(
                        &mut self.block_order,
                        |ui, bid, _handle, _item_state| {
                            _handle.ui(ui, |ui| {
                                ast::print_block_ref(ui, &mut self.hl, *bid);
                            });
                        },
                    );

                    res.update.is_some()
                })
                .inner;

            if is_block_order_changed {
                self.block_order = std::mem::take(&mut self.block_order);
                self.rebuild_ast();
            }
        }

        self.show_hl_details_panel(ui, exe);
    }

    fn show_hl_details_panel(&mut self, ui: &mut egui::Ui, exe: &Executable<'_>) {
        egui::Panel::bottom("details_panel")
            .resizable(true)
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    let Some(hl::Focus::Reg(reg)) = self.hl.pinned.focus() else {
                        return;
                    };

                    let name = &mut self.name_of_reg[reg];
                    let res = egui::TextEdit::singleline(name)
                        .font(egui::TextStyle::Heading)
                        .show(ui);
                    if res.response.changed() {
                        self.rebuild_ast();
                    }

                    let Some(ssa) = self.df.ssa() else {
                        ui.label("(Error: No SSA!)");
                        return;
                    };
                    let hl_tyid = ssa.value_type(reg);

                    let rtx = match exe.types().read_tx() {
                        Ok(rtx) => rtx,
                        Err(err) => {
                            ui.label(format!(
                                "<error while starting read transaction: {:?}>",
                                err
                            ));
                            return;
                        }
                    };
                    let ll_ty_str = format!("{:?}", ssa.ll_type(reg));

                    egui::Grid::new("reg_details_grid").show(ui, |ui| {
                        ui.strong("Low-level type:");
                        ui.label(ll_ty_str);
                        ui.end_row();

                        ui.label(egui::RichText::new("High-level type:").strong());
                        ui.horizontal(|ui| {
                            match hl_tyid {
                                Some(hl_tyid) => {
                                    self.ui_type_ref(ui, hl_tyid, &rtx.read());
                                }
                                None => {
                                    ui.label("No Type ID.");
                                }
                            }

                            ui.add_space(5.0);

                            egui::containers::menu::MenuButton::new("Change...").ui(ui, |ui| {
                                for type_window in &self.type_windows {
                                    if ui.button(&type_window.title).clicked() {
                                        let tyid = type_window.tyid;

                                        let program = self.df.ssa_mut().unwrap();
                                        program.mutate(|mut prog| {
                                            prog.set_value_type(reg, Some(tyid));
                                        });
                                        self.df.reoptimize(exe.types());
                                    }
                                }
                            });
                        });
                        ui.end_row();
                    });
                });
            });
    }

    /// Show the widgets to be laid out in the central/main area assigned to this function.
    fn show_central(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| match (self.df.ssa_mut(), self.ast.as_mut()) {
                (Some(ssa), Some(ast)) => {
                    if self.is_cfg_visible {
                        self.cfg_layout.render(ui);
                    } else {
                        let mut params = ast::Params {
                            ast,
                            name_of_reg: &self.name_of_reg,
                            ssa,
                            hl: &mut self.hl,
                            param_names: &self.param_names,
                        };
                        ast::render(ui, &mut params);
                    }
                }
                _ => {
                    ui.centered_and_justified(|ui| {
                        ui.label("- No AST -");
                    });
                }
            });
    }

    fn show_assembly(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("ASSEMBLY").small().strong());
        ui.add_space(2.0);

        let height = ui.text_style_height(&egui::TextStyle::Monospace);
        let mask = if let Some(hl::Focus::Block(bid)) = self.hl.pinned.focus() {
            let mask = self.asm_hl_mask.get_or_insert(&bid, |&bid| {
                self.asm.compute_highlight_mask(bid, self.df.ssa())
            });
            mask.as_slice()
        } else {
            self.asm_hl_mask.reset();
            &[]
        };

        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show_rows(ui, height, self.asm.lines.len(), |ui, ndxs| {
                for ndx in ndxs {
                    let asm_line = &self.asm.lines[ndx];
                    ui.horizontal_top(|ui| {
                        ui.allocate_ui(egui::Vec2::new(100.0, 18.0), |ui| {
                            let text = format!("0x{:x}", asm_line.addr);
                            ui.label(egui::RichText::new(text).monospace());
                        });

                        let is_highlighted = mask.get(ndx).copied().unwrap_or(false);
                        if is_highlighted {
                            let rect = egui::Rect::from_min_size(
                                ui.cursor().left_top(),
                                egui::vec2(4.0, height),
                            );
                            ui.painter().rect_filled(rect, 0.0, theme::COLOR_BLUE_DARK);
                            ui.add_space(8.0);
                        } else {
                            ui.add_space(8.0);
                        }

                        ui.add(
                            egui::Label::new(egui::RichText::new(&asm_line.text).monospace())
                                .extend(),
                        );
                    });
                }
            });
    }
}

fn get_param_names(exe: &Executable<'_>, ssa: &decompiler::SSAProgram) -> Option<Vec<String>> {
    let rtx = exe.types().read_tx().ok()?;
    let tyid = ssa.function_type_id()?;
    let ty = rtx.read().get_through_alias(tyid).ok().flatten()?;
    let decompiler::ty::Ty::Subroutine(func_ty) = ty.into_owned() else {
        return None;
    };
    let param_names = func_ty
        .param_names
        .iter()
        .enumerate()
        .map(|(ndx, s)| match s {
            Some(s) => s.to_string(),
            None => format!("arg{}", ndx),
        })
        .collect();
    Some(param_names)
}

fn show_details_ty_int(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash,
    size: u8,
    signedness: &decompiler::ty::Signedness,
) {
    egui::Grid::new(id).show(ui, |ui| {
        ui.label("Size:");
        ui.label(format!("{} bits", size * 8));
        ui.end_row();

        ui.label("Signedness:");
        ui.label(match *signedness {
            decompiler::ty::Signedness::Signed => "Signed",
            decompiler::ty::Signedness::Unsigned => "Unsigned",
        });
        ui.end_row();
    });
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
            let screen_rect = ctx.viewport_rect();
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

struct TypeSelectorDialog {
    id: &'static str,
    query: String,
    selector: TypeSelector,
}
impl TypeSelectorDialog {
    fn new(id: &'static str) -> Self {
        TypeSelectorDialog {
            id,
            query: String::new(),
            selector: TypeSelector::new(),
        }
    }

    fn show(
        &mut self,
        ui: &mut egui::Ui,
        engine: &mut search::TypeSearchEngine,
    ) -> egui::ModalResponse<Option<decompiler::ty::TypeID>> {
        egui::Modal::new(self.id.into()).show(ui.ctx(), |ui| {
            let screen_rect = ui.ctx().viewport_rect();
            ui.set_min_size(egui::Vec2::new(
                screen_rect.width() * 0.6,
                screen_rect.height() * 0.6,
            ));

            egui::TextEdit::singleline(&mut self.query)
                .font(egui::TextStyle::Monospace)
                .hint_text("Type query...")
                .desired_width(f32::INFINITY)
                .show(ui);

            ui.add_space(5.0);

            let res = self.selector.show(ui, engine, &self.query);
            res.selected_tyid
        })
    }
}

struct TypeSelector {
    query_prev_frame: String,
    results: Vec<search::TypeRecord>,
}
impl TypeSelector {
    fn new() -> Self {
        TypeSelector {
            query_prev_frame: String::new(),
            results: Vec::new(),
        }
    }

    fn show(
        &mut self,
        ui: &mut egui::Ui,
        engine: &mut search::TypeSearchEngine,
        query: &str,
    ) -> TypeSelectorResponse {
        if query != self.query_prev_frame {
            let is_append = query.starts_with(&self.query_prev_frame);
            engine.set_query(query, is_append);
            self.query_prev_frame = query.to_string();
        }

        if engine.tick() {
            engine.fetch_current_results(&mut self.results);
        }

        let mut selected_tyid = None;

        egui::ScrollArea::vertical()
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
            .show_rows(ui, 18.0, self.results.len(), |ui, ndxs| {
                ui.set_min_width(ui.available_width());
                for ndx in ndxs {
                    let record = &self.results[ndx];
                    let label = format!(
                        "{:15} {} (ID: {:?})",
                        record.category.as_str(),
                        record.name,
                        record.tyid
                    );
                    let label = egui::RichText::new(label).monospace();
                    if ui.selectable_label(false, label).clicked() {
                        selected_tyid = Some(record.tyid);
                    }
                }
            });

        TypeSelectorResponse { selected_tyid }
    }
}
struct TypeSelectorResponse {
    selected_tyid: Option<decompiler::ty::TypeID>,
}

struct FuncTypeForceDialog {
    id: &'static str,
    text_buffer: String,
    is_recently_changed: bool,
    change_time: Instant,
    status: decompiler::ty::notation::ParseResult<decompiler::ty::notation::TypeBuilder>,
    type_selector: Option<TypeSelector>,
}

impl FuncTypeForceDialog {
    const CHANGE_DEBOUNCE_TIME: Duration = Duration::from_secs(1);

    fn new(id: &'static str) -> Self {
        FuncTypeForceDialog {
            id,
            text_buffer: String::new(),
            is_recently_changed: false,
            change_time: Instant::now(),
            status: Ok(decompiler::ty::notation::TypeBuilder::empty()),
            type_selector: None,
        }
    }

    fn show(
        &mut self,
        ctx: &egui::Context,
        engine: &mut search::TypeSearchEngine,
    ) -> egui::ModalResponse<Option<decompiler::ty::notation::TypeBuilder>> {
        egui::Modal::new(self.id.into()).show(ctx, |ui| {
            let mut ret = None;

            ui.label(egui::RichText::new(
                "Type the definition of the new type for the function.\n\
                Syntax:\n\
                 - array: [12]T\n\
                 - struct: struct { T; U; V }\n\
                 - primitives: u64, u8, f32\n\
                 - functions: func(T, U) R\n\
                 - search for type: ?name of type ...\n",
            ));

            let res = egui::TextEdit::multiline(&mut self.text_buffer)
                .min_size(egui::vec2(ui.available_width(), 50.0))
                .show(ui);

            if let Some(qmark_pos) = self.text_buffer.find('?') {
                let ts = self.type_selector.get_or_insert_with(TypeSelector::new);
                // .show(ui, ctx, &self.text_buffer[qmark_pos + 1..]);
                let name_query = &self.text_buffer[qmark_pos + 1..];
                let ts_res = ui
                    .allocate_ui(egui::Vec2::new(ui.available_width(), 200.0), |ui| {
                        ui.label("Searching type by name:");
                        ts.show(ui, engine, name_query)
                    })
                    .inner;

                if let Some(tyid) = ts_res.selected_tyid {
                    self.text_buffer.truncate(qmark_pos);
                    self.text_buffer.push_str(&format!("#{}", tyid.0));
                }
            } else {
                self.type_selector = None;

                if res.response.changed() {
                    self.change_time = Instant::now();
                    self.is_recently_changed = true;
                }

                if self.is_recently_changed
                    && Instant::now().duration_since(self.change_time) >= Self::CHANGE_DEBOUNCE_TIME
                {
                    self.is_recently_changed = false;
                    self.status = decompiler::ty::notation::parse(&self.text_buffer);
                }

                ui.horizontal(|ui| {
                    ui.label("Status: ");
                    match &mut self.status {
                        Ok(tb) => {
                            ui.horizontal_centered(|ui| {
                                if ui.button("  Apply  ").clicked() {
                                    ret = Some(std::mem::take(tb));
                                    ui.close();
                                }
                            });
                        }
                        Err(err) => {
                            ui.label(format!("error: {:?}", err));
                        }
                    };
                });
            }

            ret
        })
    }
}

/// Preprocessed version of the original assembly program, geared towards being
/// showed on screen.
struct Assembly {
    #[allow(dead_code)]
    machine_code: Vec<u8>,
    lines: Vec<AssemblyLine>,
    ndx_of_addr: BTreeMap<u64, usize>,
}
struct AssemblyLine {
    addr: u64,
    #[allow(dead_code)]
    machine_code_range: std::ops::Range<usize>,
    text: String,
}

impl Assembly {
    fn disassemble_x86(machine_code: &[u8], base_ip: u64) -> Self {
        let decoder =
            iced_x86::Decoder::with_ip(64, machine_code, base_ip, iced_x86::DecoderOptions::NONE);

        let machine_code = machine_code.to_vec();
        let mut lines = Vec::new();
        let mut ndx_of_addr = BTreeMap::new();

        let mut formatter = iced_x86::IntelFormatter::new();
        let mut offset = 0;
        for (ndx, instr) in decoder.into_iter().enumerate() {
            use iced_x86::Formatter as _;

            let addr = instr.ip();
            let code_size = instr.len();

            ndx_of_addr.insert(addr, ndx);

            let mut text = String::new();
            formatter.format(&instr, &mut text);
            lines.push(AssemblyLine {
                addr,
                machine_code_range: offset..offset + code_size,
                text,
            });

            offset += code_size;
        }

        Assembly {
            machine_code,
            lines,
            ndx_of_addr,
        }
    }

    fn compute_highlight_mask(
        &self,
        bid: BlockID,
        ssa: Option<&decompiler::SSAProgram>,
    ) -> Vec<bool> {
        let mut mask = vec![false; self.lines.len()];
        if let Some(ssa) = ssa {
            for reg in ssa.block_regs(bid) {
                if let Some(addr) = ssa.machine_addr(reg)
                    && let Some(&ndx) = self.ndx_of_addr.get(&addr) {
                        mask[ndx] = true;
                    }
            }
        }
        mask
    }
}

mod ast {
    use std::borrow::Cow;

    use decompiler::{
        BlockID, Insn,
        ast::{Stmt, StmtID},
    };

    use super::hl;
    use crate::{columns, theme};

    pub struct Params<'a> {
        pub ast: &'a mut decompiler::Ast,
        pub name_of_reg: &'a decompiler::RegMap<String>,
        pub ssa: &'a mut decompiler::SSAProgram,
        pub hl: &'a mut hl::State,
        pub param_names: &'a [String],
    }

    pub fn render(ui: &mut egui::Ui, params: &mut Params) {
        let cmd = &mut Cmd::None;

        columns::show(ui, [40.0, columns::EXPANDING_WIDTH], |cols| {
            let root_sid = params.ast.root();
            let [block_def_col, main_col] = cols.uis();
            render_stmt(block_def_col, main_col, params, root_sid, cmd);
        });

        cmd.execute(params.ast, params.ssa);
    }

    fn render_stmt(
        block_def_col: &mut egui::Ui,
        ui: &mut egui::Ui,
        s: &mut Params<'_>,
        sid: StmtID,
        command: &mut Cmd,
    ) {
        // TODO replace tail-calls with a loop continue (or similar)
        ui.vertical(|ui| match s.ast.get(sid) {
            &Stmt::NamedBlock { bid, body } => {
                block_def_col.advance_cursor_after_rect(
                    block_def_col.cursor().with_max_y(ui.cursor().min.y - 3.0),
                );
                print_block_def(block_def_col, s.hl, bid);

                ui.horizontal(|ui| {
                    render_stmt(block_def_col, ui, s, body, command);
                });
            }
            &Stmt::Let { name, value, body } => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "let");
                    print_reg_def(ui, s, name, hl::Focus::Reg(value));
                    print_kw(ui, s.hl, "=");
                    render_expr_def(ui, s, value, 0);
                    print_kw(ui, s.hl, "in");
                });
                render_stmt(block_def_col, ui, s, body, command);
            }
            &Stmt::LetPhi { name, body } => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "letphi");
                    print_reg_def(ui, s, name, hl::Focus::Reg(name));
                });
                render_stmt(block_def_col, ui, s, body, command);
            }
            &Stmt::Seq { first, then } => {
                render_stmt(block_def_col, ui, s, first, command);
                render_stmt(block_def_col, ui, s, then, command);
            }
            &Stmt::Eval(reg) => {
                ui.horizontal(|ui| {
                    render_expr_def(ui, s, reg, 0);
                });
            }
            &Stmt::If { cond, cons, alt } => {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        if print_kw(ui, s.hl, "if")
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .double_clicked()
                        {
                            *command = Cmd::InvertIf(sid);
                        }

                        match cond {
                            Some(cond) => {
                                render_expr_def(ui, s, cond, 0);
                            }
                            None => {
                                print_error_tag(ui, s, "undefined condition");
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.add_space(20.0);
                        render_stmt(block_def_col, ui, s, cons, command);
                    });

                    if s.ast.get(alt) != &Stmt::Pass {
                        print_kw(ui, s.hl, "else");
                        ui.horizontal(|ui| {
                            ui.add_space(20.0);
                            render_stmt(block_def_col, ui, s, alt, command);
                        });
                    }

                    print_kw(ui, s.hl, "end");
                });
            }
            &Stmt::Return(reg) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "return");
                    render_expr(ui, s, reg, 0);
                });
            }
            Stmt::JumpUndefined => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "goto");
                    print_kw(ui, s.hl, "undefined");
                });
            }
            &Stmt::JumpExternal(addr) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "goto");
                    ui.label(format!("0x{:x}", addr));
                });
            }
            &Stmt::JumpIndirect(reg) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "goto");
                    ui.label("(");
                    render_expr_def(ui, s, reg, 0);
                    ui.label(").*");
                });
            }
            Stmt::Loop(block_id) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "loop");
                    print_block_ref(ui, s.hl, *block_id);
                });
            }
            Stmt::Jump(block_id) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "goto");
                    print_block_ref(ui, s.hl, *block_id);
                });
            }
            Stmt::Pass => {
                print_kw(ui, s.hl, "pass");
            }
        });
    }

    fn print_error_tag(ui: &mut egui::Ui, _s: &mut Params<'_>, text: &str) {
        ui.label(egui::RichText::new(text).color(egui::Color32::DARK_RED));
    }

    fn render_expr(
        ui: &mut egui::Ui,
        s: &mut Params<'_>,
        reg: decompiler::Reg,
        parent_prec: decompiler::PrecedenceLevel,
    ) {
        if s.ast.is_value_named(reg) {
            print_reg_ref(ui, s, reg, hl::Focus::Reg(reg));
        } else {
            render_expr_def(ui, s, reg, parent_prec);
        }
    }

    fn render_expr_def(
        ui: &mut egui::Ui,
        s: &mut Params<'_>,
        reg: decompiler::Reg,
        parent_prec: decompiler::PrecedenceLevel,
    ) {
        let Some(insn) = s.ssa.get(reg) else {
            print_error_tag(ui, s, "invalid reg");
            return;
        };

        let my_prec = decompiler::precedence(insn);
        if my_prec < parent_prec {
            print_kw(ui, s.hl, "(");
        }
        match insn {
            Insn::Void => {
                print_kw(ui, s.hl, "void");
            }
            Insn::True => {
                print_kw(ui, s.hl, "true");
            }
            Insn::False => {
                print_kw(ui, s.hl, "false");
            }
            Insn::Bytes(bytes) => {
                ui.label(format!("{:?}", bytes.as_slice()));
            }
            Insn::Global(identifier) => {
                ui.label(*identifier);
            }
            Insn::Int { value, size: _ } => {
                ui.label(format!("{}", value));
            }
            Insn::Get(r) => {
                render_expr(ui, s, *r, my_prec);
            }
            &Insn::Part { src, offset, size } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, src, my_prec);
                    ui.label(format!("[{} .. {}]", offset, offset + size));
                });
            }
            &Insn::Concat { lo, hi } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, hi, my_prec);
                    print_kw(ui, s.hl, "++");
                    render_expr(ui, s, lo, my_prec);
                });
            }
            &Insn::StructGetMember {
                struct_value,
                name,
                size: _,
            } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, struct_value, my_prec);
                    print_kw(ui, s.hl, ".");
                    print_ident(ui, s.hl, name);
                });
            }
            Insn::Struct {
                type_name,
                members,
                size: _,
            } => {
                let type_name = *type_name;
                // sad: I don't know how to get around cloning this potentially large struct.
                // temporary allocator?
                let members = members.clone();
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        print_kw(ui, s.hl, "struct");
                        print_ident(ui, s.hl, type_name);
                        print_kw(ui, s.hl, "{");
                    });
                    ui.horizontal(|ui| {
                        ui.add_space(5.0);
                        ui.vertical(|ui| {
                            if members.is_empty() {
                                ui.label("(* no members *)");
                            } else {
                                for member in members {
                                    ui.horizontal(|ui| {
                                        print_ident(ui, s.hl, member.name);
                                        print_kw(ui, s.hl, ":");
                                        render_expr(ui, s, member.value, my_prec);
                                        print_kw(ui, s.hl, ";");
                                    });
                                }
                            }
                        });
                    });
                    print_kw(ui, s.hl, "}");
                });
            }
            &Insn::ArrayGetElement {
                array,
                index,
                size: _,
            } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, array, my_prec);
                    ui.label(format!("[{}]", index));
                });
            }
            &Insn::Widen {
                reg: inner,
                target_size,
                sign: _,
            } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, inner, my_prec);
                    print_kw(ui, s.hl, "as");
                    ui.label(format!("i{}", target_size * 8));
                });
            }
            &Insn::Arith(arith_op, a, b) => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, a, my_prec);
                    print_kw(ui, s.hl, arith_op.symbol());
                    render_expr(ui, s, b, my_prec);
                });
            }
            &Insn::ArithK(arith_op, a, k) => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, a, my_prec);
                    print_kw(ui, s.hl, arith_op.symbol());
                    ui.label(format!("{}", k));
                });
            }
            &Insn::Cmp(cmp_op, a, b) => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, a, my_prec);
                    print_kw(ui, s.hl, cmp_op.symbol());
                    render_expr(ui, s, b, my_prec);
                });
            }
            &Insn::Bool(bool_op, a, b) => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, a, my_prec);
                    print_kw(ui, s.hl, bool_op.symbol());
                    render_expr(ui, s, b, my_prec);
                });
            }
            &Insn::Not(a) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "!");
                    render_expr(ui, s, a, my_prec);
                });
            }
            Insn::Call {
                callee,
                args,
                ret_ll_type: _,
            } => {
                let callee = *callee;
                let args = args.clone();
                ui.horizontal(|ui| {
                    render_expr(ui, s, callee, my_prec);
                    print_kw(ui, s.hl, "(");
                    for (ndx, a) in args.into_iter().enumerate() {
                        if ndx > 0 {
                            print_kw(ui, s.hl, ",");
                        }
                        render_expr(ui, s, a, my_prec);
                    }
                    print_kw(ui, s.hl, ")");
                });
            }
            &Insn::LoadMem { addr, size } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, addr, my_prec);
                    print_kw(ui, s.hl, &format!(".* [..{}]", size));
                });
            }
            &Insn::StoreMem { addr, value } => {
                ui.horizontal(|ui| {
                    ui.label("(");
                    render_expr(ui, s, addr, my_prec);
                    ui.label(")");
                    print_kw(ui, s.hl, ".*");
                    print_kw(ui, s.hl, ":=");
                    render_expr(ui, s, value, my_prec);
                });
            }
            &Insn::OverflowOf(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "overflow_of");
                    render_expr(ui, s, r, my_prec);
                });
            }
            &Insn::CarryOf(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "carry_of");
                    render_expr(ui, s, r, my_prec);
                });
            }
            &Insn::SignOf(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "sign_of");
                    render_expr(ui, s, r, my_prec);
                });
            }
            &Insn::IsZero(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "is_zero");
                    render_expr(ui, s, r, my_prec);
                });
            }
            &Insn::Parity(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "parity");
                    render_expr(ui, s, r, my_prec);
                });
            }
            Insn::UndefinedBool => {
                print_kw(ui, s.hl, "undefined");
                print_kw(ui, s.hl, "bool");
            }
            Insn::UndefinedBytes { size } => {
                ui.horizontal(|ui| {
                    print_kw(ui, s.hl, "undefined");
                    ui.label(format!("bytes({})", size));
                });
            }
            &Insn::FuncArgument { index, ll_type: _ } => {
                let name = s
                    .param_names
                    .get(index as usize)
                    .map(Cow::Borrowed)
                    .unwrap_or_else(|| Cow::Owned(format!("arg{}", index)));
                print_ident_ref(ui, s.hl, name.as_str(), hl::Focus::Arg(index));
            }
            Insn::Ancestral {
                anc_name,
                ll_type: _,
            } => {
                print_kw(ui, s.hl, anc_name.name());
            }
            Insn::Phi => {
                print_kw(ui, s.hl, "phi");
            }
            &Insn::Upsilon { value, phi_ref } => {
                ui.horizontal(|ui| {
                    print_reg_ref(ui, s, phi_ref, hl::Focus::Reg(phi_ref));
                    print_kw(ui, s.hl, ":=");
                    render_expr(ui, s, value, my_prec);
                });
            }

            Insn::Control(_)
            | Insn::SetReturnValue(_)
            | Insn::SetJumpCondition(_)
            | Insn::SetJumpTarget(_) => {
                print_error_tag(ui, s, &format!("unexpected as expr: {:?}", insn));
            }

            Insn::NotYetImplemented(msg) => {
                print_error_tag(
                    ui,
                    s,
                    &format!("/* not yet implemented: {:?}: {} */", insn, msg),
                );
            }

            &Insn::StructSetMember {
                struct_value,
                name,
                value,
            } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, struct_value, my_prec);
                    print_kw(ui, s.hl, ".");
                    print_ident(ui, s.hl, name);
                    print_kw(ui, s.hl, ":=");
                    render_expr(ui, s, value, my_prec);
                });
            }

            &Insn::ArraySetElement {
                array,
                index,
                value,
            } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, array, my_prec);
                    ui.label(format!("[{}]", index));
                    print_kw(ui, s.hl, ":=");
                    render_expr(ui, s, value, my_prec);
                });
            }
        }

        if my_prec < parent_prec {
            print_kw(ui, s.hl, ")");
        }
    }

    fn print_reg_def(
        ui: &mut egui::Ui,
        s: &mut Params<'_>,
        reg: decompiler::Reg,
        focus: hl::Focus,
    ) {
        let ident = &s.name_of_reg[reg];
        print_ident_def(ui, s.hl, ident, focus);
    }
    fn print_reg_ref(
        ui: &mut egui::Ui,
        s: &mut Params<'_>,
        reg: decompiler::Reg,
        focus: hl::Focus,
    ) {
        let ident = &s.name_of_reg[reg];
        print_ident_ref(ui, s.hl, ident, focus);
    }

    fn print_ident(ui: &mut egui::Ui, _s: &mut hl::State, ident: &str) {
        ui.label(ident);
    }
    pub fn print_ident_def(ui: &mut egui::Ui, s: &mut hl::State, ident: &str, focus: hl::Focus) {
        let colors = theme::colors(focus, theme::Role::Definition);
        active_label(ui, s, focus, colors, ident);
    }
    pub fn print_ident_ref(ui: &mut egui::Ui, s: &mut hl::State, ident: &str, focus: hl::Focus) {
        let colors = theme::colors(focus, theme::Role::Reference);
        active_label(ui, s, focus, colors, ident);
    }
    fn print_kw(ui: &mut egui::Ui, _s: &mut hl::State, kw: &str) -> egui::Response {
        ui.label(egui::RichText::new(kw).strong())
    }

    fn print_block_def(ui: &mut egui::Ui, s: &mut hl::State, bid: BlockID) {
        let text = format!("♦{}", bid.as_number());
        let colors = theme::colors(hl::Focus::Block(bid), theme::Role::Definition);
        active_label(ui, s, hl::Focus::Block(bid), colors, &text);
    }
    pub fn print_block_ref(ui: &mut egui::Ui, s: &mut hl::State, bid: BlockID) {
        let text = format!("♦{}", bid.as_number());
        let colors = theme::colors(hl::Focus::Block(bid), theme::Role::Reference);
        active_label(ui, s, hl::Focus::Block(bid), colors, &text);
    }

    fn active_label(
        ui: &mut egui::Ui,
        hl: &mut hl::State,
        focus: hl::Focus,
        colors_active: &theme::Colors,
        text: &str,
    ) {
        const TRANSPARENT: egui::Color32 = egui::Color32::TRANSPARENT;
        let col_text_normal = ui.visuals().text_color();

        let (col_bg, col_border, col_text) = if hl.pinned.focus() == Some(focus) {
            (colors_active.background, TRANSPARENT, colors_active.text)
        } else if hl.hovered.focus() == Some(focus) {
            (TRANSPARENT, colors_active.background, col_text_normal)
        } else {
            (TRANSPARENT, TRANSPARENT, col_text_normal)
        };

        let res = egui::Frame::new()
            .stroke(egui::Stroke {
                width: 1.0,
                color: col_border,
            })
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(text)
                        .monospace()
                        .background_color(col_bg)
                        .color(col_text),
                )
            })
            .inner;

        if res.clicked() {
            // toggle pinned state when clicking on something that's already pinned
            if Some(focus) == hl.pinned.focus() {
                hl.pinned.set_focus(None);
            } else {
                hl.pinned.set_focus(Some(focus));
            }
        } else if res.hovered() {
            hl.hovered.set_focus(Some(focus));
        }

        res.on_hover_cursor(egui::CursorIcon::PointingHand);
    }

    enum Cmd {
        None,
        InvertIf(StmtID),
    }
    impl Cmd {
        fn execute(&self, ast: &mut decompiler::Ast, ssa: &mut decompiler::SSAProgram) {
            match *self {
                Cmd::None => {}
                Cmd::InvertIf(sid) => {
                    if let Err(err) = decompiler::ast::edit::invert_if(ast, ssa, sid) {
                        eprintln!("Error inverting if: {:?}", err);
                    }
                }
            }
        }
    }
}

mod cfg {
    use decompiler::{BlockID, BlockMap};

    use std::collections::HashMap;

    #[derive(Default)]
    pub struct Layout {
        nodes: Vec<Node>,
        edges: HashMap<(NodeID, NodeID), EdgeData>,
        level_count: u32,
    }
    type NodeID = usize;

    #[derive(Debug)]
    struct Node {
        content: NodeContent,
        succs: Vec<NodeID>,
        level: u32,
        horiz_ord_key: u32,
    }

    #[derive(Default)]
    struct EdgeData {
        is_inverted: bool,
    }

    #[derive(Debug)]
    enum NodeContent {
        Block(BlockID),
        Dummy,
    }

    

    impl Layout {
        fn add_node(&mut self, content: NodeContent) -> NodeID {
            let ndx = self.nodes.len();
            self.nodes.push(Node {
                content,
                succs: Vec::new(),
                level: 0,
                horiz_ord_key: 0,
            });
            ndx
        }

        fn get(&self, nid: NodeID) -> &Node {
            self.nodes.get(nid).unwrap()
        }
        fn get_mut(&mut self, nid: NodeID) -> &mut Node {
            self.nodes.get_mut(nid).unwrap()
        }

        fn add_edge(&mut self, from: NodeID, to: NodeID) -> &mut EdgeData {
            let node = self.nodes.get_mut(from).unwrap();
            if !node.succs.contains(&to) {
                node.succs.push(to);
            }
            self.edges
                .entry((from, to))
                .or_insert_with(EdgeData::default)
        }

        pub(super) fn from_cfg(cfg: &decompiler::Graph) -> Self {
            // Modified Sugiyama algorithm for CFG layout
            // Heavily inspired by this blog post: https://spidermonkey.dev/blog/2025/10/28/iongraph-web.html

            // General outline:
            let mut layout = Layout::default();

            // 1. Cycle breaking, where the direction of some edges are flipped in order to produce a DAG.

            // 2. Leveling, where vertices are assigned into horizontal layers according to their depth in the graph, and dummy vertices are added to any edge that crosses multiple layers.
            let nid_of_bid =
                BlockMap::new_with(cfg, |bid| layout.add_node(NodeContent::Block(bid)));

            for bid in cfg.block_ids_rpo() {
                let nid = nid_of_bid[bid];
                layout.get_mut(nid).level = 1 + cfg
                    .block_preds(bid)
                    .iter()
                    .map(|&pred| layout.get(nid_of_bid[pred]).level)
                    .max()
                    .unwrap_or(0u32);
            }
            layout.level_count = 1 + layout.nodes.iter().map(|n| n.level).max().unwrap_or(0u32);

            // 3. Crossing minimization, where vertices on a layer are reordered
            // in order to minimize the number of edge crossings.

            // 3.1 Create edges and dummy nodes
            for bid in cfg.block_ids() {
                let start_nid = nid_of_bid[bid];
                for succ in cfg.direct().successors(bid) {
                    let nid_end = nid_of_bid[succ];

                    // If the edge is inverted (i.e. upwards), then we still
                    // create a downward edge, but we mark it "inverted" so
                    // that the arrow is rendered the other way. The graph as
                    // represented for layout is a guaranteed DAG.
                    let lvl_start = layout.get(start_nid).level;
                    let lvl_end = layout.get(nid_end).level;
                    let (lvl_start, nid_start, lvl_end, nid_end, is_inverted) =
                        if lvl_start <= lvl_end {
                            (lvl_start, start_nid, lvl_end, nid_end, false)
                        } else {
                            (lvl_end, nid_end, lvl_start, start_nid, true)
                        };

                    assert!(lvl_start <= lvl_end);

                    let mut src = nid_start;
                    for inter_level in lvl_start + 1..lvl_end {
                        let dummy = layout.add_node(NodeContent::Dummy);
                        layout.get_mut(dummy).level = inter_level;

                        let edge = layout.add_edge(src, dummy);
                        edge.is_inverted = is_inverted;

                        src = dummy;
                    }
                    let edge = layout.add_edge(src, nid_end);
                    edge.is_inverted = is_inverted;
                }
            }

            // 4. Vertex positioning, where vertices are horizontally positioned
            // in order to make the edges as straight as possible.

            // my heuristic here is to just do a DFS preorder numbering of the DAG,
            // using the ordering as a sort key
            {
                let entry_nid = nid_of_bid[cfg.entry_block_id()];
                let mut work = vec![entry_nid];

                #[cfg(test)]
                let mut is_visited = {
                    use std::collections::HashSet;
                    HashSet::new()
                };

                let mut number = 0;
                while let Some(nid) = work.pop() {
                    layout.get_mut(nid).horiz_ord_key = number;
                    number += 1;

                    #[cfg(test)]
                    {
                        let is_newly_visited = is_visited.insert(nid);
                        assert!(is_newly_visited);
                    }

                    for &succ in &layout.get(nid).succs {
                        work.push(succ);
                    }
                }
            }

            layout
        }
    }

    impl Layout {
        pub fn render(&self, ui: &mut egui::Ui) {
            ui.vertical(|ui| {
                ui.label(format!("{} levels", self.level_count));
                let mut ordered = Vec::new();
                for level in 0..self.level_count {
                    ordered.clear();
                    ordered.extend(
                        self.nodes
                            .iter()
                            .enumerate()
                            .filter(|&(_, node)| node.level == level)
                            .map(|(nid, _)| nid),
                    );
                    ordered.sort_by_key(|&nid| self.nodes[nid].horiz_ord_key);

                    ui.horizontal_top(|ui| {
                        ui.label(format!("[{}]", ordered.len()));
                        ui.vertical(|ui| {
                            for &nid in &ordered {
                                let node = self.get(nid);
                                ui.label(format!("{:?}:{:?}", nid, node.content));
                                for &succ in &node.succs {
                                    ui.label(format!("-> {:?}", succ));
                                }
                            }
                        });
                    });
                }
            });
        }
    }
}

mod hl {
    pub struct State {
        pub pinned: Set,
        pub hovered: Set,
    }

    impl State {
        pub fn empty() -> Self {
            Self {
                pinned: Set::empty(),
                hovered: Set::empty(),
            }
        }

        pub fn frame_started(&mut self) {
            self.pinned.frame_started();
            self.hovered.frame_started();
        }
    }

    pub struct Set {
        focus: Option<Focus>,
        was_changed_this_frame: bool,
        was_set_this_frame: bool,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Focus {
        Reg(decompiler::Reg),
        Block(decompiler::BlockID),
        Arg(u16),
    }

    impl Set {
        pub fn empty() -> Self {
            Self {
                focus: None,
                was_changed_this_frame: false,
                was_set_this_frame: false,
            }
        }

        pub fn frame_started(&mut self) {
            self.was_changed_this_frame = false;
            self.was_set_this_frame = false;
        }
        pub fn was_set_this_frame(&self) -> bool {
            self.was_set_this_frame
        }

        pub fn focus(&self) -> Option<Focus> {
            self.focus
        }

        pub fn set_focus(&mut self, focus: Option<Focus>) {
            self.was_set_this_frame = true;

            if focus != self.focus {
                self.focus = focus;
                self.was_changed_this_frame = true;
            }
        }
    }
}

#[allow(dead_code)]
mod theme {
    use crate::hl;

    pub struct Colors {
        pub background: egui::Color32,
        pub text: egui::Color32,
    }

    pub enum Role {
        Definition,
        Reference,
    }

    pub fn colors(focus: hl::Focus, role: Role) -> &'static Colors {
        match (focus, role) {
            (hl::Focus::Reg(_) | hl::Focus::Arg(_), Role::Definition) => {
                &(Colors {
                    background: COLOR_GREEN_DARK,
                    text: egui::Color32::WHITE,
                })
            }
            (hl::Focus::Reg(_) | hl::Focus::Arg(_), Role::Reference) => {
                &(Colors {
                    background: COLOR_GREEN_LIGHT,
                    text: egui::Color32::BLACK,
                })
            }
            (hl::Focus::Block(_), Role::Definition) => {
                &(Colors {
                    background: COLOR_BLUE_DARK,
                    text: egui::Color32::WHITE,
                })
            }
            (hl::Focus::Block(_), Role::Reference) => {
                &(Colors {
                    background: COLOR_BLUE_LIGHT,
                    text: COLOR_BLUE_DARK,
                })
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

struct Cache<K, V>(Option<(K, V)>);

impl<K, V> Default for Cache<K, V> {
    fn default() -> Self {
        Cache(None)
    }
}

impl<K, V> Cache<K, V>
where
    K: Clone + PartialEq + std::fmt::Debug,
{
    fn get_or_insert(&mut self, key: &K, recompute: impl FnOnce(&K) -> V) -> &V {
        let needs_recompute = match self.0.as_ref() {
            Some((cur_key, _)) => cur_key != key,
            None => true,
        };

        if needs_recompute {
            let new_value = recompute(key);
            self.0 = Some((key.clone(), new_value));
        }

        let (_, value_ref) = self.0.as_ref().unwrap();
        value_ref
    }

    fn reset(&mut self) {
        self.0 = None;
    }
}

mod columns {
    pub const EXPANDING_WIDTH: f32 = f32::INFINITY;

    /// Sets up a multi-column layout where each column can be filled independently by the given closure.
    ///
    /// The desired width for each column is set via the `width` array (also
    /// determines the number of columns). Use `EXPANDING_WIDTH` for columns
    /// that should take up the remaining available space equally.
    ///
    /// The `add_contents` closure is given an array of [`Column`], allowing
    /// access to a separate `egui::Ui` for each column.
    pub fn show<const N: usize>(
        ui: &mut egui::Ui,
        widths: [f32; N],
        add_contents: impl FnOnce(&mut Columns<N>),
    ) {
        ui.horizontal(move |ui| {
            let width_fixed: f32 = widths.into_iter().filter(|w| w.is_finite()).sum();
            let width_expanding_count = widths.into_iter().filter(|w| w.is_infinite()).count();
            let width_available = ui.available_width();
            let width_expanding_each: f32 =
                (width_available - width_fixed) / width_expanding_count as f32;

            let uis = std::array::from_fn(|ndx| {
                let width = match widths[ndx] {
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
            });

            let mut columns = Columns { uis };

            add_contents(&mut columns);

            for col in columns.uis {
                // we're doing a custom layout, so we have to do this where egui
                // would have done this automatically
                ui.expand_to_include_rect(col.min_rect());
            }
        });
    }

    pub struct Columns<const N: usize> {
        uis: [egui::Ui; N],
    }
    impl<const N: usize> Columns<N> {
        pub fn uis(&mut self) -> [&mut egui::Ui; N] {
            self.uis.each_mut()
        }

        pub fn ui(&mut self, ndx: usize) -> &mut egui::Ui {
            &mut self.uis[ndx]
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

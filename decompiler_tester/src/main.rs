use std::{
    collections::BTreeMap,
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use decompiler::{BlockID, Executable};
use ouroboros::self_referencing;

fn main() {
    tracing_subscriber::fmt::fmt()
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
    stage_func: Option<Result<FunctionView, decompiler::Error>>,
}
struct FunctionView {
    df: decompiler::DecompiledFunction,
    problems_is_visible: bool,
    problems_title: String,
    problems_error: Option<String>,

    hl: hl::State,
    ast: Option<decompiler::Ast>,

    asm: Assembly,
    asm_hl_mask: Cache<BlockID, Vec<bool>>,

    pinned_ty: Cache<decompiler::Reg, Option<TypeInfo>>,

    is_asm_visible: bool,
    is_cfg_visible: bool,
}

struct TypeInfo {
    tyid: decompiler::ty::TypeID,
    hl_ty_str: String,
    ll_ty_str: String,
}
impl TypeInfo {
    fn read(exe: &Executable, ssa: &decompiler::SSAProgram, reg: decompiler::Reg) -> Option<Self> {
        let ll_ty_str = format!("{:?}", ssa.reg_type(reg));

        let tyid = ssa.value_type(reg)?;
        let hl_ty_str = decompiler::pp::pp_to_string(|pp| {
            let rtx = exe.types().read_tx().unwrap();
            rtx.read().dump_type_ref(pp, tyid).unwrap();
        });

        Some(TypeInfo {
            tyid,
            ll_ty_str,
            hl_ty_str,
        })
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                self.show_topbar(ui);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                    egui::widgets::global_theme_preference_switch(ui);

                    if let Some(Ok(func_view)) = &mut self.stage_func {
                        ui.toggle_value(
                            &mut func_view.is_cfg_visible,
                            egui::RichText::new("CFG").monospace(),
                        );
                        ui.toggle_value(
                            &mut func_view.is_asm_visible,
                            egui::RichText::new("ASM").monospace(),
                        );
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
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
            stage_func: None,
        };
        app.start_function_selector();
        Ok(app)
    }

    fn show_topbar(&mut self, ui: &mut egui::Ui) {
        if ui.button("Load function…").clicked() {
            self.start_function_selector();
        }

        match self.stage_func.as_mut() {
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
        match self.stage_func.as_mut() {
            Some(Ok(stage_func)) => {
                stage_func.show(ui, self.exe.borrow_exe());
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
                let stage_func_or_err = load_function(&mut self.exe, function_name);
                self.stage_func = Some(stage_func_or_err);
                self.function_selector = None;
            } else if res.should_close() {
                self.function_selector = None;
            }
        }
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

        let ast = df.ssa().map(|ssa| decompiler::AstBuilder::new(ssa).build());

        FunctionView {
            df,
            problems_is_visible: false,
            problems_title: title,
            problems_error: error_label,
            asm,
            asm_hl_mask: Cache::default(),
            ast,
            hl: hl::State::empty(),
            pinned_ty: Cache::default(),
            is_asm_visible: false,
            is_cfg_visible: false,
        }
    }

    /// Show widgets specific for this function to be laid outj on the top bar
    /// (which is visually located at the executable level).
    fn show_topbar(&mut self, ui: &mut egui::Ui) {
        ui.label(self.df.name());
    }

    fn show(&mut self, ui: &mut egui::Ui, exe: &Executable) {
        self.hl.frame_started();
        self.show_panels(ui, exe);
        self.show_central(ui);

        if !self.hl.hovered.was_set_this_frame() {
            self.hl.hovered.set_focus(None);
        }
    }

    /// Show the side panels (top, bottom, right, left) to be laid out around
    /// the central are, with content specific to this function.
    ///
    /// To be called before `show_central`.
    fn show_panels(&mut self, ui: &mut egui::Ui, exe: &Executable) {
        if self.is_asm_visible {
            egui::SidePanel::left("asm_panel").show_inside(ui, |ui| {
                self.show_assembly(ui);
            });
        }

        if self.is_cfg_visible {
            egui::SidePanel::right("cfg_panel").show_inside(ui, |ui| {
                ui.heading("Control-flow graph");
            });
        }

        self.show_hl_details_panel(ui, exe);
    }

    fn show_hl_details_panel(&mut self, ui: &mut egui::Ui, exe: &Executable<'_>) {
        match self.hl.pinned.focus() {
            Some(hl::Focus::Reg(reg)) => {
                let type_info = self
                    .pinned_ty
                    .get_or_insert(&reg, |_| {
                        let ssa = self.df.ssa()?;
                        TypeInfo::read(exe, ssa, reg)
                    })
                    .as_ref();
                egui::TopBottomPanel::bottom("details_panel")
                    .resizable(true)
                    .show_inside(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.heading(format!("{:?}", reg));

                            egui::Grid::new("reg_details_grid").show(ui, |ui| {
                                ui.strong("Low-level type:");
                                ui.label(
                                    type_info.map(|ti| ti.ll_ty_str.as_str()).unwrap_or("???"),
                                );
                                ui.end_row();

                                ui.label(egui::RichText::new("High-level type:").strong());
                                if let Some(type_info) = type_info {
                                    if ui.button(type_info.hl_ty_str.as_str()).clicked() {
                                        println!("open window for type: {:?}", type_info.tyid);
                                    }
                                } else {
                                    ui.label("???");
                                }
                                ui.end_row();
                            });
                        });
                    });
            }
            _ => {
                self.pinned_ty.reset();
            }
        }
    }

    /// Show the widgets to be laid out in the central/main area assigned to this function.
    fn show_central(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .show(ui, |ui| match (self.df.ssa(), self.ast.as_ref()) {
                (Some(ssa), Some(ast)) => {
                    ast::render(ui, ast, ssa, &mut self.hl);
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

fn load_function(exe: &mut Exe, function_name: &str) -> Result<FunctionView, decompiler::Error> {
    let mut stage_func = exe.with_exe_mut(|exe| {
        let df = exe.decompile_function(function_name)?;
        Ok(FunctionView::new(df, exe))
    });

    if let Ok(func_view) = &mut stage_func {
        func_view.problems_is_visible =
            func_view.df.error().is_some() || !func_view.df.warnings().is_empty();
    }

    stage_func
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

/// Preprocessed version of the original assembly program, geared towards being
/// showed on screen.
struct Assembly {
    machine_code: Vec<u8>,
    lines: Vec<AssemblyLine>,
    ndx_of_addr: BTreeMap<u64, usize>,
}
struct AssemblyLine {
    addr: u64,
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
                if let Some(addr) = ssa.machine_addr(reg) {
                    if let Some(&ndx) = self.ndx_of_addr.get(&addr) {
                        mask[ndx] = true;
                    }
                }
            }
        }
        mask
    }
}

mod ast {
    use decompiler::{
        BlockID, Insn,
        ast::{Stmt, StmtID},
    };

    use super::hl;
    use crate::theme;

    pub fn render(
        ui: &mut egui::Ui,
        ast: &decompiler::Ast,
        ssa: &decompiler::SSAProgram,
        hl: &mut hl::State,
    ) {
        let mut s = State { ast, ssa, hl };
        render_stmt(ui, &mut s, ast.root())
    }

    struct State<'a> {
        ast: &'a decompiler::Ast,
        ssa: &'a decompiler::SSAProgram,
        hl: &'a mut hl::State,
    }

    fn render_stmt(ui: &mut egui::Ui, s: &mut State<'_>, sid: StmtID) {
        // TODO replace tail-calls with a loop continue (or similar)
        ui.vertical(|ui| {
            match s.ast.get(sid) {
                Stmt::NamedBlock { bid, body } => {
                    ui.horizontal(|ui| {
                        print_block_def(ui, s, *bid);
                        render_stmt(ui, s, *body);
                    });
                }
                Stmt::Let { name, value, body } => {
                    ui.horizontal(|ui| {
                        print_kw(ui, s, "let");
                        print_ident_def(ui, s, name, hl::Focus::Reg(*value));
                        print_kw(ui, s, "=");
                        render_expr_def(ui, s, *value, 0);
                        print_kw(ui, s, "in");
                    });
                    render_stmt(ui, s, *body);
                }
                Stmt::LetPhi { name, body } => {
                    ui.horizontal(|ui| {
                        print_kw(ui, s, "letphi");
                        // no reg available here in the AST node; use a placeholder reg index 0
                        print_ident_def(ui, s, name, hl::Focus::Reg(decompiler::Reg(0)));
                    });
                    render_stmt(ui, s, *body);
                }
                Stmt::Seq { first, then } => {
                    render_stmt(ui, s, *first);
                    render_stmt(ui, s, *then);
                }
                Stmt::Eval(reg) => {
                    ui.horizontal(|ui| {
                        render_expr_def(ui, s, *reg, 0);
                    });
                }
                Stmt::If { cond, cons, alt } => {
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            print_kw(ui, s, "if");
                            match *cond {
                                Some(cond) => {
                                    render_expr_def(ui, s, cond, 0);
                                }
                                None => {
                                    print_error_tag(ui, s, "undefined condition");
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            print_kw(ui, s, "then");
                            render_stmt(ui, s, *cons);
                        });
                        ui.horizontal(|ui| {
                            print_kw(ui, s, "else");
                            render_stmt(ui, s, *alt);
                        });
                    });
                }
                Stmt::Return(reg) => {
                    ui.horizontal(|ui| {
                        print_kw(ui, s, "return");
                        render_expr_def(ui, s, *reg, 0);
                    });
                }
                Stmt::JumpUndefined => {
                    ui.horizontal(|ui| {
                        print_kw(ui, s, "goto");
                        print_kw(ui, s, "undefined");
                    });
                }
                Stmt::JumpExternal(addr) => {
                    ui.horizontal(|ui| {
                        print_kw(ui, s, "goto");
                        ui.label(format!("0x{:x}", *addr));
                    });
                }
                Stmt::JumpIndirect(reg) => {
                    ui.horizontal(|ui| {
                        print_kw(ui, s, "goto");
                        ui.label("(");
                        render_expr_def(ui, s, *reg, 0);
                        ui.label(").*");
                    });
                }
                Stmt::Loop(block_id) => {
                    ui.horizontal(|ui| {
                        print_kw(ui, s, "loop");
                        print_block_ref(ui, s, *block_id);
                    });
                }
                Stmt::Jump(block_id) => {
                    ui.horizontal(|ui| {
                        print_kw(ui, s, "goto");
                        print_block_ref(ui, s, *block_id);
                    });
                }
            }
        });
    }

    fn print_error_tag(ui: &mut egui::Ui, _s: &mut State<'_>, text: &str) {
        ui.label(egui::RichText::new(text).color(egui::Color32::DARK_RED));
    }

    fn render_expr(
        ui: &mut egui::Ui,
        s: &mut State<'_>,
        reg: decompiler::Reg,
        parent_prec: decompiler::PrecedenceLevel,
    ) {
        if s.ast.is_value_named(reg) {
            let text = format!("r{}", reg.reg_index());
            print_ident_ref(ui, s, &text, hl::Focus::Reg(reg));
        } else {
            render_expr_def(ui, s, reg, parent_prec);
        }
    }

    fn render_expr_def(
        ui: &mut egui::Ui,
        s: &mut State<'_>,
        reg: decompiler::Reg,
        parent_prec: decompiler::PrecedenceLevel,
    ) {
        let Some(insn) = s.ssa.get(reg) else {
            print_error_tag(ui, s, "invalid reg");
            return;
        };

        let my_prec = decompiler::precedence(&insn);
        if my_prec < parent_prec {
            print_kw(ui, s, "(");
        }
        match insn {
            Insn::Void => {
                print_kw(ui, s, "void");
            }
            Insn::True => {
                print_kw(ui, s, "true");
            }
            Insn::False => {
                print_kw(ui, s, "false");
            }
            Insn::Bytes(bytes) => {
                ui.label(format!("{:?}", bytes.as_slice()));
            }
            Insn::Int { value, size: _ } => {
                ui.label(format!("{}", value));
            }
            Insn::Get(r) => {
                render_expr(ui, s, r, my_prec);
            }
            Insn::Part { src, offset, size } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, src, my_prec);
                    ui.label(format!("[{} .. {}]", offset, offset + size));
                });
            }
            Insn::Concat { lo, hi } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, hi, my_prec);
                    print_kw(ui, s, "++");
                    render_expr(ui, s, lo, my_prec);
                });
            }
            Insn::StructGetMember {
                struct_value,
                name,
                size: _,
            } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, struct_value, my_prec);
                    print_kw(ui, s, ".");
                    print_ident(ui, s, name);
                });
            }
            Insn::Struct {
                type_name,
                first_member,
                size: _,
            } => {
                ui.horizontal(|ui| {
                    print_kw(ui, s, "struct");
                    print_ident(ui, s, type_name);
                    if let Some(first) = first_member {
                        render_expr(ui, s, first, my_prec);
                    } else {
                        ui.label("(no members)");
                    }
                });
            }
            Insn::StructMember { name, value, next } => {
                ui.horizontal(|ui| {
                    print_ident(ui, s, name);
                    print_kw(ui, s, ":");
                    render_expr(ui, s, value, my_prec);
                    print_kw(ui, s, ";");
                });
                if let Some(next_val) = next {
                    render_expr(ui, s, next_val, my_prec);
                }
            }
            Insn::ArrayGetElement {
                array,
                index,
                size: _,
            } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, array, my_prec);
                    ui.label(format!("[{}]", index));
                });
            }
            Insn::Widen {
                reg: inner,
                target_size,
                sign: _,
            } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, inner, my_prec);
                    print_kw(ui, s, "as");
                    ui.label(format!("i{}", target_size * 8));
                });
            }
            Insn::Arith(arith_op, a, b) => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, a, my_prec);
                    print_kw(ui, s, arith_op.symbol());
                    render_expr(ui, s, b, my_prec);
                });
            }
            Insn::ArithK(arith_op, a, k) => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, a, my_prec);
                    print_kw(ui, s, arith_op.symbol());
                    ui.label(format!("{}", k));
                });
            }
            Insn::Cmp(cmp_op, a, b) => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, a, my_prec);
                    print_kw(ui, s, cmp_op.symbol());
                    render_expr(ui, s, b, my_prec);
                });
            }
            Insn::Bool(bool_op, a, b) => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, a, my_prec);
                    print_kw(ui, s, bool_op.symbol());
                    render_expr(ui, s, b, my_prec);
                });
            }
            Insn::Not(a) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s, "!");
                    render_expr(ui, s, a, my_prec);
                });
            }
            Insn::Call { callee, first_arg } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, callee, my_prec);
                    print_kw(ui, s, "(");
                    if let Some(arg) = first_arg {
                        // show call args rendered as expressions
                        for (ndx, a) in s.ssa.get_call_args(arg).enumerate() {
                            if ndx > 0 {
                                print_kw(ui, s, ",");
                            }
                            render_expr(ui, s, a, my_prec);
                        }
                    }
                    print_kw(ui, s, ")");
                });
            }
            Insn::LoadMem { addr, size } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, addr, my_prec);
                    print_kw(ui, s, ".*");
                    print_kw(ui, s, &format!(".i{}", size));
                });
            }
            Insn::StoreMem { addr, value } => {
                ui.horizontal(|ui| {
                    ui.label("(");
                    render_expr(ui, s, addr, my_prec);
                    ui.label(")");
                    print_kw(ui, s, ".*");
                    print_kw(ui, s, ":=");
                    render_expr(ui, s, value, my_prec);
                });
            }
            Insn::OverflowOf(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s, "overflow_of");
                    render_expr(ui, s, r, my_prec);
                });
            }
            Insn::CarryOf(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s, "carry_of");
                    render_expr(ui, s, r, my_prec);
                });
            }
            Insn::SignOf(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s, "sign_of");
                    render_expr(ui, s, r, my_prec);
                });
            }
            Insn::IsZero(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s, "is_zero");
                    render_expr(ui, s, r, my_prec);
                });
            }
            Insn::Parity(r) => {
                ui.horizontal(|ui| {
                    print_kw(ui, s, "parity");
                    render_expr(ui, s, r, my_prec);
                });
            }
            Insn::UndefinedBool => {
                print_kw(ui, s, "undefined");
                print_kw(ui, s, "bool");
            }
            Insn::UndefinedBytes { size } => {
                ui.horizontal(|ui| {
                    print_kw(ui, s, "undefined");
                    ui.label(format!("bytes({})", size));
                });
            }
            Insn::FuncArgument { index, reg_type: _ } => {
                print_ident(ui, s, &format!("$arg{}", index));
            }
            Insn::Ancestral {
                anc_name,
                reg_type: _,
            } => {
                print_kw(ui, s, anc_name.name());
            }
            Insn::Phi => {
                print_kw(ui, s, "phi");
            }
            Insn::Upsilon { value, phi_ref } => {
                ui.horizontal(|ui| {
                    render_expr(ui, s, phi_ref, my_prec);
                    print_kw(ui, s, ":=");
                    render_expr(ui, s, value, my_prec);
                });
            }

            Insn::CArg { .. }
            | Insn::Control(_)
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
        }

        if my_prec < parent_prec {
            print_kw(ui, s, ")");
        }
    }

    fn print_ident(ui: &mut egui::Ui, _s: &mut State<'_>, ident: &str) {
        ui.label(ident);
    }
    fn print_ident_def(ui: &mut egui::Ui, s: &mut State<'_>, ident: &str, focus: hl::Focus) {
        let colors = theme::colors(focus, theme::Role::Definition);
        active_label(ui, s, focus, colors, ident);
    }
    fn print_ident_ref(ui: &mut egui::Ui, s: &mut State, ident: &str, focus: hl::Focus) {
        let colors = theme::colors(focus, theme::Role::Reference);
        active_label(ui, s, focus, colors, ident);
    }

    fn print_kw(ui: &mut egui::Ui, _s: &mut State<'_>, kw: &str) {
        ui.label(egui::RichText::new(kw).strong());
    }

    fn print_block_def(ui: &mut egui::Ui, s: &mut State<'_>, bid: BlockID) {
        let text = format!("♦{}", bid.as_number());
        let colors = theme::colors(hl::Focus::Block(bid), theme::Role::Definition);
        active_label(ui, s, hl::Focus::Block(bid), colors, &text);
    }
    fn print_block_ref(ui: &mut egui::Ui, s: &mut State<'_>, bid: BlockID) {
        let text = format!("♦{}", bid.as_number());
        let colors = theme::colors(hl::Focus::Block(bid), theme::Role::Reference);
        active_label(ui, s, hl::Focus::Block(bid), colors, &text);
    }

    fn active_label(
        ui: &mut egui::Ui,
        s: &mut State<'_>,
        focus: hl::Focus,
        colors_active: &theme::Colors,
        text: &str,
    ) {
        const TRANSPARENT: egui::Color32 = egui::Color32::TRANSPARENT;
        let col_text_normal = ui.visuals().text_color();

        let (col_bg, col_border, col_text) = if s.hl.pinned.focus() == Some(focus) {
            (colors_active.background, TRANSPARENT, colors_active.text)
        } else if s.hl.hovered.focus() == Some(focus) {
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
            if Some(focus) == s.hl.pinned.focus() {
                s.hl.pinned.set_focus(None);
            } else {
                s.hl.pinned.set_focus(Some(focus));
            }
        } else if res.hovered() {
            s.hl.hovered.set_focus(Some(focus));
        }

        res.on_hover_cursor(egui::CursorIcon::PointingHand);
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
        pub fn was_changed_this_frame(&self) -> bool {
            self.was_changed_this_frame
        }
        pub fn was_set_this_frame(&self) -> bool {
            self.was_set_this_frame
        }

        pub fn focus(&self) -> Option<Focus> {
            self.focus.clone()
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
            (hl::Focus::Reg(_), Role::Definition) => {
                &(Colors {
                    background: COLOR_GREEN_DARK,
                    text: egui::Color32::WHITE,
                })
            }
            (hl::Focus::Reg(_), Role::Reference) => {
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
            let new_value = recompute(&key);
            self.0 = Some((key.clone(), new_value));
        }

        let (_, value_ref) = self.0.as_ref().unwrap();
        value_ref
    }

    fn reset(&mut self) {
        self.0 = None;
    }
}

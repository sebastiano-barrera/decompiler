use std::sync::{
    Arc,
    atomic::{self, AtomicBool},
};

use decompiler::ty::Ty;

#[derive(Clone)]
pub(super) struct TypeRecord {
    pub tyid: decompiler::ty::TypeID,
    pub category: TypeCategory,
    pub name: String,
}

pub(super) struct TypeSearchEngine {
    nuc: nucleo::Nucleo<TypeRecord>,
    injector_thread: Option<std::thread::JoinHandle<()>>,
    injector_running: Arc<AtomicBool>,
}
impl TypeSearchEngine {
    pub fn new() -> Self {
        TypeSearchEngine {
            nuc: nucleo::Nucleo::new(
                nucleo::Config::DEFAULT,
                // notify
                Arc::new(|| {}),
                None, // num_threads
                2,
            ),
            injector_thread: None,
            injector_running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn load_types(&mut self, tys: Arc<decompiler::ty::TypeSet>) {
        self.stop_loading();

        assert!(self.injector_running.load(atomic::Ordering::SeqCst) == false);
        assert!(self.injector_thread.is_none());

        self.injector_running.store(true, atomic::Ordering::SeqCst);
        let injector = self.nuc.injector();
        let running_flag = Arc::clone(&self.injector_running);
        let jh = std::thread::spawn(move || inject_types(injector, &*tys, &*running_flag));
        self.injector_thread = Some(jh);
    }

    pub fn stop_loading(&mut self) {
        self.injector_running.store(false, atomic::Ordering::SeqCst);
        if let Some(jh) = self.injector_thread.take() {
            if let Err(err) = jh.join() {
                eprintln!("Injector thread panicked: {err:?}");
            }
        }
    }

    /// By specifying `is_append` the caller promises that text passed to the
    /// previous `set_query` invocation is a prefix of the text passed here now.
    /// This enables additional optimizations but can lead to missing matches if
    /// an incorrect value is passed.
    pub fn set_query(&mut self, query: &str, is_append: bool) {
        self.nuc.pattern.reparse(
            1,
            query,
            nucleo::pattern::CaseMatching::Smart,
            nucleo::pattern::Normalization::Never,
            is_append,
        );
    }

    pub fn tick(&mut self) -> bool {
        self.nuc.tick(10).changed
    }

    pub fn fetch_current_results(&self, results: &mut Vec<TypeRecord>) {
        let snapshot = self.nuc.snapshot();
        results.clear();
        let count = snapshot.matched_item_count();
        for item in snapshot.matched_items(0..count) {
            results.push(item.data.clone());
        }
    }
}
impl Drop for TypeSearchEngine {
    fn drop(&mut self) {
        self.stop_loading();
    }
}

fn inject_types(
    injector: nucleo::Injector<TypeRecord>,
    tys: &decompiler::ty::TypeSet,
    running_flag: &AtomicBool,
) {
    if let Err(err) = inject_types_fallible(injector, tys, running_flag) {
        // TODO report the error outside somehow
        eprintln!("Failed to acquire read transaction: {err}");
    }
}
fn inject_types_fallible(
    injector: nucleo::Injector<TypeRecord>,
    tys: &decompiler::ty::TypeSet,
    running_flag: &AtomicBool,
) -> decompiler::ty::Result<()> {
    let rtx = tys.read_tx()?;
    let rtx = rtx.read();

    const CHECK_COUNTDOWN_MAX: u8 = 50;
    let mut check_countdown = CHECK_COUNTDOWN_MAX;

    for (ndx, (tyid, ty)) in rtx.scan_types()?.enumerate() {
        if let Ty::Alias(_) = ty {
            continue;
        }

        check_countdown -= 1;
        if check_countdown == 0 {
            if !running_flag.load(atomic::Ordering::SeqCst) {
                break;
            }
            check_countdown = CHECK_COUNTDOWN_MAX;
            eprintln!("scanning types ({})...", ndx);
        }

        let name = rtx.name(tyid)?.unwrap_or_else(|| String::new());
        let record = TypeRecord {
            tyid,
            category: TypeCategory::of_type(&ty),
            name,
        };
        injector.push(record, |record, columns| {
            // TODO lots of copying! can this be avoided?
            columns[0] = record.category.to_column_string();
            columns[1] = record.name.to_string().into();
        });
    }

    Ok(())
}

#[derive(Clone)]
pub enum TypeCategory {
    Flag,
    Int,
    Bool,
    Enum,
    Struct,
    Array,
    Ptr,
    Float,
    Function,
    Unknown,
    Void,
    Alias,
}
impl TypeCategory {
    fn of_type(ty: &Ty) -> TypeCategory {
        match ty {
            Ty::Flag => TypeCategory::Flag,
            Ty::Int(_) => TypeCategory::Int,
            Ty::Bool(_) => TypeCategory::Bool,
            Ty::Enum(_) => TypeCategory::Enum,
            Ty::Struct(_) => TypeCategory::Struct,
            Ty::Array(_) => TypeCategory::Array,
            Ty::Ptr(_) => TypeCategory::Ptr,
            Ty::Float(_) => TypeCategory::Float,
            Ty::Subroutine(_) => TypeCategory::Function,
            Ty::Unknown => TypeCategory::Unknown,
            Ty::Void => TypeCategory::Void,
            Ty::Alias(_) => TypeCategory::Alias,
        }
    }

    fn to_column_string(&self) -> nucleo::Utf32String {
        self.as_str()
            // TODO can we avoid going through UTF-8?
            .to_owned()
            .into()
    }

    pub fn as_str(&self) -> &str {
        match self {
            TypeCategory::Flag => "Flag",
            TypeCategory::Int => "Int",
            TypeCategory::Bool => "Bool",
            TypeCategory::Enum => "Enum",
            TypeCategory::Struct => "Struct",
            TypeCategory::Array => "Array",
            TypeCategory::Ptr => "Ptr",
            TypeCategory::Float => "Float",
            TypeCategory::Function => "Function",
            TypeCategory::Unknown => "Unknown",
            TypeCategory::Void => "Void",
            TypeCategory::Alias => "Alias",
        }
    }
}

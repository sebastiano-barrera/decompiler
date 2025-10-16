use std::{
    collections::HashMap,
    io::Read,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use actix_web::{App, HttpResponse, HttpServer, Responder, web};
use anyhow::Context;
use decompiler::proto;

#[actix_web::main]
async fn main() {
    anyhow_main().await.unwrap()
}

async fn anyhow_main() -> anyhow::Result<()> {
    env_logger::init();

    let cli_opts = CliOptions::parse()?;

    let exe_name: String = cli_opts
        .exe_filename
        .file_name()
        .map(|oss| oss.to_string_lossy().into_owned())
        .unwrap_or_else(|| "(exe name?)".to_owned());

    let raw_binary: &'static [u8] = {
        let mut f = std::fs::File::open(cli_opts.exe_filename).context("opening executable")?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).context("reading executable")?;
        buf.leak()
    };

    let exe = decompiler::Executable::parse(raw_binary).context("parsing binary file as ELF")?;
    let shared = Arc::new(Mutex::new(SharedState::new(exe_name, exe)));

    // TODO pick random port to bind to
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(Arc::clone(&shared)))
            .service(get_exe)
            .service(get_function)
            .service(web::redirect("/", "/p/"))
            .service(pages)
            .service(actix_files::Files::new("/assets/", "./assets/").index_file("index.html"))
    })
    .bind(("127.0.0.1", 1993))?
    .run()
    .await?;

    Ok(())
}

#[actix_web::get("/exe")]
async fn get_exe(shared: web::Data<Arc<Mutex<SharedState>>>) -> impl Responder {
    let shared = shared.lock().unwrap();
    HttpResponse::Ok().json(&shared.exe_data)
}

#[actix_web::get("/p/{tail:.*}")]
async fn pages() -> impl Responder {
    // serve index.html for every page. the frontend router will do the rest
    std::fs::File::open("./assets/index.html")
        .and_then(|mut f| {
            let mut buf = String::new();
            f.read_to_string(&mut buf)?;
            Ok(buf)
        })
        .map(|content| HttpResponse::Ok().body(content))
        .unwrap_or_else(HttpResponse::from_error)
}

#[actix_web::get("/functions/{name}")]
async fn get_function(
    path: web::Path<String>,
    shared: web::Data<Arc<Mutex<SharedState>>>,
) -> impl Responder {
    let function_name = path.into_inner();

    let mut shared = shared.lock().unwrap();
    let df = match shared.get_or_create(&function_name) {
        Ok(df) => df,
        Err(err) => {
            return HttpResponse::InternalServerError().body(err.to_string());
        }
    };
    let df = df.lock().unwrap();

    HttpResponse::Ok().json(&*df)
}

mod proto_web {
    #[derive(serde::Serialize)]
    pub struct Exe {
        pub name: String,
        pub functions: Vec<String>,
    }
}

struct SharedState {
    exe: decompiler::Executable<'static>,
    exe_data: proto_web::Exe,
    df_by_name: HashMap<String, Arc<Mutex<proto::Function>>>,
    autoreloader: minijinja_autoreload::AutoReloader,
}

impl SharedState {
    fn new(exe_name: String, exe: decompiler::Executable<'static>) -> Self {
        let autoreloader = minijinja_autoreload::AutoReloader::new(|notifier| {
            let mut tmpl_env = minijinja::Environment::new();
            tmpl_env.set_loader(minijinja::path_loader("templates/"));
            notifier.watch_path("templates/", false);
            Ok(tmpl_env)
        });

        let mut functions: Vec<_> = exe.function_names().map(|s| s.to_string()).collect();
        functions.sort();

        SharedState {
            autoreloader,
            exe,
            exe_data: proto_web::Exe {
                name: exe_name,
                functions,
            },
            df_by_name: HashMap::new(),
        }
    }

    fn tmpl_env(&self) -> impl std::ops::Deref<Target = minijinja::Environment<'static>> {
        self.autoreloader.acquire_env().unwrap()
    }

    fn get_or_create(
        &mut self,
        function_name: &str,
    ) -> anyhow::Result<Arc<Mutex<proto::Function>>> {
        if let Some(df_data) = self.df_by_name.get(function_name) {
            return Ok(Arc::clone(df_data));
        }

        let df = self.exe.decompile_function(&function_name)?;
        let df = proto::Function::from(&df);
        let df = Arc::new(Mutex::new(df));

        self.df_by_name
            .insert(function_name.to_string(), Arc::clone(&df));
        Ok(df)
    }
}

struct CliOptions {
    exe_filename: PathBuf,
}

impl CliOptions {
    fn parse() -> anyhow::Result<Self> {
        let mut args = std::env::args().skip(1);

        let exe_filename = args.next();
        let Some(exe_filename) = exe_filename else {
            anyhow::bail!("missing required argument: executable filename");
        };
        let exe_filename = PathBuf::from(exe_filename);

        Ok(CliOptions { exe_filename })
    }
}

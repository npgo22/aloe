//! Aloe - Hobby-rocket flight simulator

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "aloe")]
#[command(about = "Hobby-rocket flight simulator with sensor fusion")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run CLI batch/single simulation mode
    Cli {
        /// Pass remaining arguments to aloe-cli
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Launch web GUI
    Gui {
        /// Port to bind to (defaults to $PORT, else 8080)
        #[arg(short, long)]
        port: Option<u16>,

        /// Host to bind to
        #[arg(short = 'H', long, default_value = "0.0.0.0")]
        host: String,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize comprehensive tracing with OTEL support
    aloe_gui::tracing_init::init_tracing().expect("Failed to initialize tracing");

    let cli = Cli::parse();

    let result = match cli.command {
        Some(Commands::Cli { args }) => {
            // Pass through to aloe-cli
            let cli_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            run_cli(&cli_args)
        }
        Some(Commands::Gui { port, host }) => run_gui(resolve_port(port), &host),
        None => {
            // Default to GUI mode
            run_gui(resolve_port(None), "0.0.0.0")
        }
    };

    // Ensure spans are flushed before exit
    aloe_gui::tracing_init::shutdown_tracing();

    result
}

fn run_cli(args: &[&str]) -> anyhow::Result<()> {
    // Convert args back to a format that can be parsed by the CLI
    let mut full_args = vec!["aloe-cli"];
    full_args.extend(args);

    // Parse and run the CLI
    match aloe_cli::run_cli_main(&full_args) {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("CLI error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_gui(port: u16, host: &str) -> anyhow::Result<()> {
    use std::net::SocketAddr;
    use tokio::net::TcpListener;
    use tokio::runtime::Runtime;
    let rt = Runtime::new()?;
    rt.block_on(async {
        let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
        let app = aloe_gui::create_router();
        println!("listening on http://{}", addr);
        tracing::info!("GUI server started on {}", addr);
        let listener = TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    })
}

fn resolve_port(cli_port: Option<u16>) -> u16 {
    if let Some(port) = cli_port {
        return port;
    }

    std::env::var("PORT")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(8080)
}

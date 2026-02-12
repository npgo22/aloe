//! Aloe - Hobby-rocket flight simulator

use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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
        /// Port to bind to
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Host to bind to
        #[arg(short = 'H', long, default_value = "0.0.0.0")]
        host: String,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Cli { args }) => {
            // Pass through to aloe-cli
            let cli_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            run_cli(&cli_args)
        }
        Some(Commands::Gui { port, host }) => run_gui(port, &host),
        None => {
            // Default to GUI mode
            run_gui(8080, "0.0.0.0")
        }
    }
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

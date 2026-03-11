//! Tracing and OpenTelemetry initialization for aloe-gui
//!
//! Provides flexible logging configuration supporting:
//! - Structured JSON logging
//! - Pretty-printed console logging
//! - OpenTelemetry span export
//! - Environment-based configuration

use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Logging format configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Pretty-printed human-readable logs
    Pretty,
    /// Structured JSON logs for machine parsing
    Json,
    /// Compact format for production
    Compact,
}

impl LogFormat {
    /// Parse log format from environment variable
    pub fn from_env() -> Self {
        match std::env::var("LOG_FORMAT")
            .unwrap_or_else(|_| "pretty".to_string())
            .to_lowercase()
            .as_str()
        {
            "json" => LogFormat::Json,
            "compact" => LogFormat::Compact,
            _ => LogFormat::Pretty,
        }
    }
}

/// Initialize tracing with optional OpenTelemetry support
///
/// # Environment Variables
///
/// - `RUST_LOG`: Log level filter (e.g., "info", "aloe=debug,tower_http=trace")
/// - `LOG_FORMAT`: Output format - "pretty" (default), "json", or "compact"
/// - `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP endpoint (e.g., "http://localhost:4317")
/// - `OTEL_SERVICE_NAME`: Service name for traces (default: "aloe-gui")
///
/// # Examples
///
/// ```bash
/// # Pretty console logs at info level
/// RUST_LOG=info cargo run
///
/// # JSON logs for production
/// LOG_FORMAT=json RUST_LOG=info cargo run
///
/// # With OpenTelemetry export to Jaeger
/// OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 cargo run
/// ```
pub fn init_tracing() -> Result<(), Box<dyn std::error::Error>> {
    let format = LogFormat::from_env();

    // Set up environment filter
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    // Check if OpenTelemetry is enabled
    let otel_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();
    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "aloe-gui".to_string());

    let subscriber = tracing_subscriber::registry().with(env_filter);

    if let Some(_endpoint) = otel_endpoint {
        // OpenTelemetry support - Currently requires additional configuration
        // TODO: Implement OTLP exporter with version 0.31.x API
        tracing::warn!(
            "OpenTelemetry endpoint specified but OTLP exporter not yet configured for this version"
        );
        tracing::warn!(
            "Falling back to local logging only (service: {}, format: {:?})",
            service_name,
            format
        );

        // Fall through to standard logging
    }

    {
        // No OpenTelemetry - just format layer
        match format {
            LogFormat::Json => {
                let json_layer = tracing_subscriber::fmt::layer()
                    .json()
                    .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE);
                subscriber.with(json_layer).init();
            }
            LogFormat::Compact => {
                let compact_layer = tracing_subscriber::fmt::layer().compact();
                subscriber.with(compact_layer).init();
            }
            LogFormat::Pretty => {
                let pretty_layer = tracing_subscriber::fmt::layer().pretty();
                subscriber.with(pretty_layer).init();
            }
        }

        tracing::info!("Tracing initialized (format: {:?})", format);
    }

    Ok(())
}

/// Shutdown OpenTelemetry provider gracefully
///
/// Call this before application exit to ensure all spans are flushed
pub fn shutdown_tracing() {
    // TODO: Implement proper OTLP shutdown when OTLP exporter is configured
    tracing::debug!("Shutting down tracing");
}

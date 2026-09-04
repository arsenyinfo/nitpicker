//! Log subscriber and optional OpenTelemetry trace export.
//!
//! The `tracing` subscriber has two layers: the stderr formatter every build has, and — behind
//! the `otel` cargo feature — an OTLP span exporter. Activation is driven by the standard
//! `OTEL_*` environment variables only, never by `nitpicker.toml`: in `pr` mode that file comes
//! from the target repository, and an endpoint there would redirect the run's telemetry.
//!
//! The formatter deliberately never sees spans: with a per-layer filter that rejects them, the
//! log lines keep their pre-span shape (no `chat{...}:` scope prefix) whether or not the exporter
//! is attached. The exporter sees spans only — no log events — so the `info!`/`warn!` lines that
//! carry tool arguments or provider error bodies never leave the process.

use std::future::Future;

use eyre::Result;
use tracing::{Instrument, Span};
use tracing_subscriber::filter::{FilterExt, filter_fn};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use crate::output::UsageReport;
use crate::progress;

const SUPPORTED_PROTOCOL: &str = "http/protobuf";

/// The subset of the OTLP environment that decides whether nitpicker exports at all.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OtelEnv {
    pub sdk_disabled: Option<String>,
    pub endpoint: Option<String>,
    pub traces_endpoint: Option<String>,
    pub protocol: Option<String>,
    pub traces_protocol: Option<String>,
}

impl OtelEnv {
    pub fn from_process() -> Self {
        let var = |name: &str| std::env::var(name).ok();
        Self {
            sdk_disabled: var("OTEL_SDK_DISABLED"),
            endpoint: var("OTEL_EXPORTER_OTLP_ENDPOINT"),
            traces_endpoint: var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),
            protocol: var("OTEL_EXPORTER_OTLP_PROTOCOL"),
            traces_protocol: var("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"),
        }
    }

    /// Signal-specific variables win over the generic ones; blank values count as unset.
    pub fn activation(&self) -> Activation {
        let set = |value: &Option<String>| {
            value
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_string)
        };
        if set(&self.sdk_disabled).is_some_and(|v| v.eq_ignore_ascii_case("true")) {
            return Activation::Off;
        }
        if set(&self.traces_endpoint)
            .or_else(|| set(&self.endpoint))
            .is_none()
        {
            return Activation::Off;
        }
        let protocol = set(&self.traces_protocol)
            .or_else(|| set(&self.protocol))
            .unwrap_or_else(|| SUPPORTED_PROTOCOL.to_string());
        match protocol == SUPPORTED_PROTOCOL {
            true => Activation::On,
            false => Activation::Unsupported { protocol },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activation {
    /// No endpoint, or `OTEL_SDK_DISABLED=true`.
    Off,
    On,
    /// An endpoint is set but the requested wire protocol is not built in. Decided before any
    /// exporter is constructed: the HTTP exporter would otherwise fall back silently.
    Unsupported {
        protocol: String,
    },
}

/// Only nitpicker's own spans are exported: never log events, never third-party crates' spans.
#[cfg(any(feature = "otel", test))]
pub fn exported_span(meta: &tracing::Metadata<'_>) -> bool {
    meta.is_span() && meta.target().starts_with("nitpicker")
}

/// Run a review or debate body under its root span and close the span's outcome fields:
/// `nitpicker.degraded` and the usage totals on success, `otel.status_code = ERROR` on failure.
pub async fn record_run<T>(
    span: Span,
    body: impl Future<Output = Result<T>>,
    summary: impl FnOnce(&T) -> (bool, &UsageReport),
) -> Result<T> {
    let result = body.instrument(span.clone()).await;
    match &result {
        Ok(outcome) => {
            let (degraded, usage) = summary(outcome);
            span.record("nitpicker.degraded", degraded);
            span.record("gen_ai.usage.input_tokens", usage.input_tokens);
            span.record("gen_ai.usage.output_tokens", usage.output_tokens);
        }
        Err(_) => {
            span.record("otel.status_code", "ERROR");
        }
    }
    result
}

/// Handle to the exporter, if one was attached. Dropping it without `shutdown` loses the tail
/// of the trace, so `main` calls `shutdown` after the runtime is gone.
pub struct Telemetry {
    #[cfg(feature = "otel")]
    provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

impl Telemetry {
    /// Flush and stop the exporter. A failure here is reported on stderr and never changes the
    /// exit code: telemetry must not alter the run's outcome.
    pub fn shutdown(self) {
        #[cfg(feature = "otel")]
        if let Some(provider) = self.provider {
            if let Err(err) = provider.shutdown() {
                eprintln!("warning: OpenTelemetry export did not shut down cleanly: {err}");
            }
        }
    }
}

/// Install the global subscriber. `default_level` applies when `RUST_LOG` is unset. Logs go to
/// stderr — stdout is reserved for the deliverable so `pr --json` stays a single object.
pub fn init(default_level: &str) -> Telemetry {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(progress::stderr_log_writer)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .with_level(true)
        .with_ansi(progress::stderr_supports_color())
        .compact()
        .with_filter(env_filter.and(filter_fn(|meta| !meta.is_span())));
    let registry = tracing_subscriber::registry().with(fmt_layer);

    let env = OtelEnv::from_process();
    #[cfg(feature = "otel")]
    let (registry, telemetry, warning) = {
        let setup = otel::setup(&env);
        (
            registry.with(setup.layer),
            Telemetry {
                provider: setup.provider,
            },
            setup.warning,
        )
    };
    #[cfg(not(feature = "otel"))]
    let (telemetry, warning) = (
        Telemetry {},
        match env.activation() {
            Activation::Off => None,
            Activation::On | Activation::Unsupported { .. } => Some(
                "OTEL_EXPORTER_OTLP_ENDPOINT is set but this build has no OpenTelemetry support; rebuild with `--features otel`"
                    .to_string(),
            ),
        },
    );
    registry.init();
    if let Some(warning) = warning {
        tracing::warn!("{warning}");
    }
    telemetry
}

#[cfg(feature = "otel")]
mod otel {
    use opentelemetry::KeyValue;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing::Subscriber;
    use tracing_subscriber::Layer;
    use tracing_subscriber::filter::{FilterFn, Filtered, filter_fn};
    use tracing_subscriber::registry::LookupSpan;

    use super::{Activation, OtelEnv, SUPPORTED_PROTOCOL, exported_span};

    type OtelLayer<S> = Filtered<
        tracing_opentelemetry::OpenTelemetryLayer<S, opentelemetry_sdk::trace::Tracer>,
        FilterFn,
        S,
    >;

    pub struct Setup<S> {
        pub layer: Option<OtelLayer<S>>,
        pub provider: Option<SdkTracerProvider>,
        /// Logged once the subscriber is installed — nothing emitted before that is visible.
        pub warning: Option<String>,
    }

    pub fn setup<S>(env: &OtelEnv) -> Setup<S>
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        let off = |warning: Option<String>| Setup {
            layer: None,
            provider: None,
            warning,
        };
        match env.activation() {
            Activation::Off => off(None),
            Activation::Unsupported { protocol } => off(Some(format!(
                "OpenTelemetry export disabled: protocol {protocol:?} is not supported by this build (only {SUPPORTED_PROTOCOL})"
            ))),
            Activation::On => match build_provider() {
                Ok(provider) => {
                    let layer = tracing_opentelemetry::layer()
                        .with_tracer(provider.tracer("nitpicker"))
                        .with_tracked_inactivity(false)
                        .with_threads(false)
                        .with_location(false)
                        .with_level(false)
                        .with_filter(filter_fn(
                            exported_span as fn(&tracing::Metadata<'_>) -> bool,
                        ));
                    Setup {
                        layer: Some(layer),
                        provider: Some(provider),
                        warning: None,
                    }
                }
                Err(err) => off(Some(format!(
                    "OpenTelemetry export disabled: exporter setup failed: {err}"
                ))),
            },
        }
    }

    fn build_provider() -> eyre::Result<SdkTracerProvider> {
        // endpoint, headers, timeout and compression come from the OTEL_EXPORTER_OTLP_* env
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .build()?;
        let service_name = std::env::var("OTEL_SERVICE_NAME")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "nitpicker".to_string());
        // Resource::builder() already applied OTEL_RESOURCE_ATTRIBUTES; the two attributes set
        // here are the identity a backend needs even with nothing else configured
        let resource = Resource::builder()
            .with_service_name(service_name)
            .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
            .build();
        // sampler defaults to the env-aware SDK config (ParentBased(AlwaysOn) unless
        // OTEL_TRACES_SAMPLER says otherwise); a CLI run is one trace either way
        Ok(SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(resource)
            .build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};
    use tracing::{info_span, warn};
    use tracing_subscriber::layer::Context;

    fn env(pairs: &[(&str, &str)]) -> OtelEnv {
        let mut env = OtelEnv::default();
        for (name, value) in pairs {
            let slot = match *name {
                "OTEL_SDK_DISABLED" => &mut env.sdk_disabled,
                "OTEL_EXPORTER_OTLP_ENDPOINT" => &mut env.endpoint,
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" => &mut env.traces_endpoint,
                "OTEL_EXPORTER_OTLP_PROTOCOL" => &mut env.protocol,
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL" => &mut env.traces_protocol,
                other => panic!("unknown var {other}"),
            };
            *slot = Some(value.to_string());
        }
        env
    }

    #[test]
    fn export_is_off_without_an_endpoint() {
        assert_eq!(env(&[]).activation(), Activation::Off);
        assert_eq!(
            env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "   ")]).activation(),
            Activation::Off
        );
        assert_eq!(
            env(&[("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")]).activation(),
            Activation::Off,
            "a protocol alone activates nothing"
        );
    }

    #[test]
    fn either_endpoint_variable_turns_export_on() {
        assert_eq!(
            env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")]).activation(),
            Activation::On
        );
        assert_eq!(
            env(&[(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                "http://localhost:4318/v1/traces"
            )])
            .activation(),
            Activation::On
        );
    }

    #[test]
    fn sdk_disabled_wins_over_an_endpoint() {
        let on = ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318");
        assert_eq!(
            env(&[on, ("OTEL_SDK_DISABLED", "true")]).activation(),
            Activation::Off
        );
        assert_eq!(
            env(&[on, ("OTEL_SDK_DISABLED", "TRUE")]).activation(),
            Activation::Off
        );
        assert_eq!(
            env(&[on, ("OTEL_SDK_DISABLED", "false")]).activation(),
            Activation::On
        );
    }

    #[test]
    fn only_http_protobuf_is_supported_and_the_traces_protocol_wins() {
        let on = ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318");
        assert_eq!(
            env(&[on, ("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")]).activation(),
            Activation::Unsupported {
                protocol: "grpc".to_string()
            }
        );
        assert_eq!(
            env(&[on, ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json")]).activation(),
            Activation::Unsupported {
                protocol: "http/json".to_string()
            }
        );
        assert_eq!(
            env(&[on, ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")]).activation(),
            Activation::On
        );
        assert_eq!(
            env(&[
                on,
                ("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"),
                ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf"),
            ])
            .activation(),
            Activation::On
        );
    }

    #[derive(Clone, Default)]
    struct SpanNames(Arc<Mutex<Vec<&'static str>>>);

    impl<S: tracing::Subscriber> Layer<S> for SpanNames {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &tracing::span::Id,
            _ctx: Context<'_, S>,
        ) {
            self.0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(attrs.metadata().name());
        }
    }

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl Write for Sink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The formatter must not gain a span-scope prefix just because a span-consuming layer is
    /// attached, and that layer must still see the span the formatter ignores.
    #[test]
    fn formatter_ignores_spans_while_the_export_layer_sees_them() {
        let sink = Sink::default();
        let names = SpanNames::default();
        let writer_sink = sink.clone();
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(move || writer_sink.clone())
            .without_time()
            .with_ansi(false)
            .with_target(false)
            .compact()
            .with_filter(EnvFilter::new("warn").and(filter_fn(|meta| !meta.is_span())));
        let subscriber = tracing_subscriber::registry()
            .with(fmt_layer)
            .with(names.clone().with_filter(filter_fn(exported_span)));
        tracing::subscriber::with_default(subscriber, || {
            let span = info_span!(target: "nitpicker::telemetry::tests", "chat", model = "m");
            let _entered = span.enter();
            let foreign = info_span!(target: "hyper_util::client", "connect");
            let _foreign = foreign.enter();
            warn!("boom");
            tracing::info!("hidden by the level filter");
        });

        let output = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
        assert_eq!(output.lines().count(), 1, "{output:?}");
        assert!(output.contains("boom"), "{output:?}");
        assert!(
            !output.contains("chat"),
            "span scope leaked into the log line: {output:?}"
        );
        assert_eq!(*names.0.lock().unwrap(), vec!["chat"]);
    }
}

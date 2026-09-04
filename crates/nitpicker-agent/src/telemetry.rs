//! Span contract shared by every instrumented seam in this crate.
//!
//! Spans are plain `tracing` spans, so any subscriber can consume them; the field names follow
//! the OpenTelemetry GenAI semantic conventions (`gen_ai.*`) plus `nitpicker.*` for
//! harness-specific attributes, and `otel.name` / `otel.kind` / `otel.status_code` are the
//! `tracing-opentelemetry` conventions for span name, kind, and status. Only identifiers,
//! counts, and timings are recorded — never prompts, tool arguments, tool output, or provider
//! error text — so exporting every span is safe by construction.

use crate::llm::FinishReason;
use crate::tools::floor_char_boundary;

/// Upper bound on any exported string attribute (model ids, agent names, tool names).
const MAX_ATTR_CHARS: usize = 128;

/// Truncate an identifier to `MAX_ATTR_CHARS` bytes on a char boundary. Every string that
/// becomes a span attribute passes through here so a pathological config value stays bounded.
pub fn bounded(s: &str) -> &str {
    &s[..floor_char_boundary(s, MAX_ATTR_CHARS)]
}

/// Constant vocabulary for `gen_ai.response.finish_reasons`; `Other` carries the provider's
/// own (bounded) label.
pub(crate) fn finish_reason_label(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::None => "none",
        FinishReason::Stop => "stop",
        FinishReason::MaxTokens => "length",
        FinishReason::ToolUse => "tool_calls",
        FinishReason::Other(label) => bounded(label),
    }
}

#[cfg(test)]
pub(crate) mod capture {
    //! A test layer that records every span's name, parent, and fields so tests can assert
    //! the exported tree without a real exporter.
    //!
    //! One global subscriber is installed once per test binary and forwards to whichever
    //! `SpanCapture` the current thread has activated. Scoped per-test dispatchers are not an
    //! option: `tracing-core` recomputes its global callsite-interest and max-level caches
    //! whenever a scoped dispatcher registers or unregisters, and a span created on another
    //! thread during that window is silently disabled.
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::fmt::Debug;
    use std::sync::{Arc, Mutex, Once};

    use tracing::Subscriber;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;

    #[derive(Clone, Debug)]
    pub struct CapturedSpan {
        pub id: u64,
        pub parent: Option<u64>,
        pub name: &'static str,
        pub fields: BTreeMap<String, String>,
    }

    impl CapturedSpan {
        pub fn field(&self, name: &str) -> Option<&str> {
            self.fields.get(name).map(String::as_str)
        }
    }

    #[derive(Clone, Default)]
    pub struct SpanCapture {
        spans: Arc<Mutex<Vec<CapturedSpan>>>,
    }

    thread_local! {
        static ACTIVE: RefCell<Option<SpanCapture>> = const { RefCell::new(None) };
    }

    /// Routes this thread's spans into `capture` until dropped.
    pub struct ActiveCapture(());

    impl Drop for ActiveCapture {
        fn drop(&mut self) {
            ACTIVE.with(|active| *active.borrow_mut() = None);
        }
    }

    impl SpanCapture {
        pub fn activate(&self) -> ActiveCapture {
            static INSTALL: Once = Once::new();
            INSTALL.call_once(|| {
                let _ = tracing::subscriber::set_global_default(
                    tracing_subscriber::registry().with(Forwarder),
                );
            });
            ACTIVE.with(|active| *active.borrow_mut() = Some(self.clone()));
            ActiveCapture(())
        }

        pub fn spans(&self) -> Vec<CapturedSpan> {
            self.spans.lock().unwrap_or_else(|e| e.into_inner()).clone()
        }

        /// Fail if `secret` appears in any captured span name or field value.
        pub fn assert_no_secret(&self, secret: &str) {
            for span in self.spans() {
                assert!(
                    !span.name.contains(secret),
                    "span name leaks: {}",
                    span.name
                );
                for (key, value) in &span.fields {
                    assert!(
                        !value.contains(secret),
                        "{}.{key} leaks: {value}",
                        span.name
                    );
                }
            }
        }

        pub fn named(&self, name: &str) -> Vec<CapturedSpan> {
            self.spans()
                .into_iter()
                .filter(|s| s.name == name)
                .collect()
        }

        fn push(&self, span: CapturedSpan) {
            self.spans
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(span);
        }

        fn record(&self, id: u64, values: &Record<'_>) {
            let mut spans = self.spans.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(span) = spans.iter_mut().find(|s| s.id == id) {
                values.record(&mut FieldVisitor(&mut span.fields));
            }
        }
    }

    struct FieldVisitor<'a>(&'a mut BTreeMap<String, String>);

    impl Visit for FieldVisitor<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_i64(&mut self, field: &Field, value: i64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_bool(&mut self, field: &Field, value: bool) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
    }

    struct Forwarder;

    impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Forwarder {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            let Some(capture) = ACTIVE.with(|active| active.borrow().clone()) else {
                return;
            };
            let parent = ctx
                .span(id)
                .and_then(|span| span.parent().map(|p| p.id().into_u64()));
            let mut fields = BTreeMap::new();
            attrs.record(&mut FieldVisitor(&mut fields));
            capture.push(CapturedSpan {
                id: id.into_u64(),
                parent,
                name: attrs.metadata().name(),
                fields,
            });
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
            if let Some(capture) = ACTIVE.with(|active| active.borrow().clone()) {
                capture.record(id.into_u64(), values);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_cuts_on_a_char_boundary() {
        let short = "gpt-5";
        assert_eq!(bounded(short), short);
        let long = "é".repeat(MAX_ATTR_CHARS);
        let cut = bounded(&long);
        assert!(cut.len() <= MAX_ATTR_CHARS);
        assert_eq!(cut, "é".repeat(MAX_ATTR_CHARS / 2));
    }

    #[test]
    fn finish_reason_labels_are_constant_except_other() {
        assert_eq!(finish_reason_label(&FinishReason::Stop), "stop");
        assert_eq!(finish_reason_label(&FinishReason::MaxTokens), "length");
        assert_eq!(finish_reason_label(&FinishReason::ToolUse), "tool_calls");
        assert_eq!(finish_reason_label(&FinishReason::None), "none");
        assert_eq!(
            finish_reason_label(&FinishReason::Other("content_filter".to_string())),
            "content_filter"
        );
    }
}

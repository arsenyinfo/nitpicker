Find accidental complexity and dead weight that make the current code materially harder to
reason about or change safely.

- Reuse: identify code reimplementing an existing repository mechanism and name that mechanism.
- Simplification: find redundant or derivable state, copy-paste variation, deep nesting, dead
  imperative bookkeeping, and abstractions whose data flow could be direct.
- Dead code: zero tolerance once non-use is established. Flag unused branches, types, functions,
  configuration, compatibility shims, abandoned scaffolding, and tests for behavior that no
  longer exists. Check public APIs, feature gates, registration, reflection, build-time use, and
  generated entry points before declaring code dead; possible future use is not current use.
  A public or serialized field may be a live API or wire-format contract even without an in-repo
  reader. Require evidence that the contract is private, unreleased, versioned away, or that
  consumers have migrated before calling it dead; visibility or serialization alone is not proof
  that it remains necessary.
- Altitude: find special cases stacked on shared infrastructure when one smaller underlying
  mechanism would remove the stack. Do not add another layer to accommodate the latest case.
- Invariants: prefer closed domain types, enums/unions, dedicated identifiers, constructors,
  and boundary validation that make invalid states unrepresentable. Flag stringly typed or
  boolean-flag APIs when they permit a concrete invalid state.
- Composition: prefer small transformations, pure functions, declarative structure, and
  composition over mutable orchestration or inheritance when that exposes the invariant and
  reduces coupling.
- Rent: interfaces with one implementation, speculative extension points, premature generics,
  manager/service/factory layers, dependencies for trivial work, and expanded public surface
  must solve a present demonstrated need.
- Tests: find tests that duplicate stronger coverage, restate implementation details or
  language/type guarantees, or require disproportionate fixtures and mocking without
  protecting meaningful behavior.

Boring and explicit beats clever. Require a concrete smaller replacement and explain which
invariant, behavior, or maintenance operation it clarifies. Reject aesthetic rewrites,
paradigm dogma, generic cleanup requests, and abstractions proposed only for possible future use.

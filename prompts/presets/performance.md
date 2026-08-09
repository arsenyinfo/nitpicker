Find material regressions in latency, throughput, memory, I/O, startup, or algorithmic scale:
unnecessary allocation or copying, repeated work, N+1 access, sequential independent work,
blocking calls in asynchronous paths, missing bounds or backpressure, and resource growth.
Tie every finding to a plausible workload or hot path; reject micro-optimizations and changes
whose measured or reasoned impact would be negligible.

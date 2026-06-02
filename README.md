# queueing-theory

Queueing theory in Rust — M/M/1, M/M/c, Erlang, Jackson networks, priority queues. Model capacity before you build.

**72 tests** · `nalgebra` + `serde` · MIT OR Apache-2.0

---

## The SRE Use Case

Your API gets **1000 req/s** with an average **50ms** response time. How many servers do you need to keep **p99 wait < 200ms**?

```rust
use queueing_theory::staffing::ServerPool;

// λ = 1000 req/s, μ = 20 req/s per server (1/0.05s)
let servers = ServerPool::find_optimal_servers(0.0002, 1000.0, 20.0);
// → you need ~55 servers

let pool = ServerPool::new(servers, 1000.0, 20.0);
println!("Servers: {}", pool.num_servers);
println!("Utilization: {:.1}%", pool.utilization() * 100.0);
println!("Mean wait: {:.3}ms", pool.mean_task_wait_time() * 1000.0);
println!("P(queue): {:.3}", pool.prob_task_waits());
```

Or use the SLA planner:

```rust
use queueing_theory::staffing::SlaConfig;

let sla = SlaConfig {
    max_wait_time: 0.0002,       // 0.2ms mean wait
    max_wait_probability: 0.01,  // <1% chance of queueing
    max_utilization: 0.7,        // keep utilization under 70%
};
let plan = sla.plan_capacity(1000.0, 20.0);
println!("Recommended: {} servers", plan.recommended_servers);
```

---

## What This Does

Closed-form and algorithmic solutions for queueing system performance metrics: expected queue length, waiting time, server utilisation, blocking probability, throughput, and more.

No simulation, no Monte Carlo — exact formulas where they exist, numerically stable recursions where they don't.

| Module | System | What you get |
|---|---|---|
| `kendall` | Kendall notation parser/builder | Structured `KendallNotation` |
| `mm1` | M/M/1 | L, L_q, W, W_q, ρ, P_n, P(N>n) |
| `mmc` | M/M/c (multi-server) | Same metrics, Erlang-C formula |
| `mg1` | M/G/1 (Pollaczek–Khinchine) | L, L_q, W, W_q from variance of service time |
| `erlang` | Erlang-B (loss), Erlang-C (delay) | Blocking probability, waiting probability |
| `birth_death` | General birth-death | Steady-state probabilities via recurrence |
| `networks` | Jackson networks | Throughput, queue lengths, visit ratios |
| `priority` | M/M/1 & M/M/c with priorities | Per-class metrics (preemptive & non-preemptive) |
| `staffing` | Server/staffing planning | Optimal staffing, cost models, SLA capacity planning |

---

## Install

```toml
[dependencies]
queueing-theory = "0.1"
```

Or:

```sh
cargo add queueing-theory
```

---

## Quick Start

### M/M/1 Queue

```rust
use queueing_theory::mm1::MM1Result;

let result = MM1Result::from_lambda_mu(5.0, 8.0);
// λ = 5 arrivals/sec, μ = 8 services/sec

println!("Utilisation ρ = {:.3}", result.rho);           // 0.625
println!("Mean queue length L_q = {:.3}", result.lq);    // ~1.042
println!("Mean waiting time W_q = {:.3} s", result.wq);  // ~0.208 s
println!("P(empty) = {:.3}", result.pn(0));               // 0.375
```

### Erlang-B: How many circuits?

```rust
use queueing_theory::erlang::erlang_b;

let traffic = 12.0;  // 12 Erlangs of offered load
let servers = 20;    // try 20 circuits
let blocking = erlang_b(traffic, servers);
println!("Blocking probability with {} servers: {:.4}", servers, blocking);
```

### Jackson Network

```rust
use queueing_theory::networks::{JacksonNetwork, Node};

let mut net = JacksonNetwork::new();
net.add_node(Node::new(0, 1.0, 2.0));  // node 0: λ_ext=1, μ=2
net.add_node(Node::new(1, 0.0, 3.0));  // node 1: no ext. arrivals, μ=3
net.set_routing(0, 1, 0.6);            // 60% from node 0 → node 1
net.set_routing(1, 0, 0.3);            // 30% from node 1 → node 0

let solution = net.solve();
for (i, node_result) in solution.iter().enumerate() {
    println!("Node {}: λ_eff={:.3}, L={:.3}, W={:.3}",
        i, node_result.lambda_eff, node_result.l, node_result.w);
}
```

---

## The Math

### Little's Law

For any queueing system in steady state: **L = λW**

### Pollaczek–Khinchine (M/G/1)

- ρ = λE[S]
- L_q = (λ² · Var[S] + ρ²) / (2(1 − ρ))

### Erlang-B (M/M/c/c Loss System)

**B(c, a) = (a^c / c!) / Σ_{k=0}^{c} a^k / k!**

### Jackson's Theorem

Each node in an open Jackson network behaves as an independent M/M/c queue with effective arrival rate given by the traffic equations.

---

## License

MIT OR Apache-2.0

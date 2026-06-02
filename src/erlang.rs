//! Erlang B and C formulas for teletraffic engineering.

use serde::{Deserialize, Serialize};

/// Erlang B formula: blocking probability in a loss system (M/M/c/c, no queue).
/// B(c, a) = (a^c / c!) / Σ_{k=0}^{c} (a^k / k!)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErlangB {
    /// Number of servers/channels.
    pub channels: usize,
    /// Offered load (a = λ/μ).
    pub offered_load: f64,
}

impl ErlangB {
    pub fn new(channels: usize, offered_load: f64) -> Self {
        Self { channels, offered_load }
    }

    fn factorial(n: usize) -> f64 {
        (1..=n).fold(1.0, |acc, i| acc * i as f64)
    }

    /// Compute Erlang B blocking probability using the iterative formula:
    /// B(0, a) = 1
    /// B(c, a) = (a · B(c-1, a)) / (c + a · B(c-1, a))
    pub fn blocking_probability(&self) -> f64 {
        let a = self.offered_load;
        let c = self.channels;

        // Iterative method (more numerically stable)
        let mut b = 1.0;
        for i in 1..=c {
            b = (a * b) / (i as f64 + a * b);
        }
        b
    }

    /// Carried load = a · (1 - B(c,a)).
    pub fn carried_load(&self) -> f64 {
        self.offered_load * (1.0 - self.blocking_probability())
    }

    /// Number of busy channels on average.
    pub fn mean_busy_channels(&self) -> f64 {
        self.carried_load()
    }

    /// Channel utilization = carried load / c.
    pub fn utilization(&self) -> f64 {
        self.carried_load() / self.channels as f64
    }
}

/// Erlang C formula: probability of waiting in an M/M/c queue.
/// C(c, a) = [a^c / (c! · (1 - a/c))] / [Σ_{k=0}^{c-1} a^k/k! + a^c / (c! · (1 - a/c))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErlangC {
    /// Number of servers.
    pub servers: usize,
    /// Offered load (a = λ/μ). Must satisfy a < c.
    pub offered_load: f64,
}

impl ErlangC {
    pub fn new(servers: usize, offered_load: f64) -> Self {
        assert!(offered_load < servers as f64, "offered load must be < servers");
        Self { servers, offered_load }
    }

    fn factorial(n: usize) -> f64 {
        (1..=n).fold(1.0, |acc, i| acc * i as f64)
    }

    /// Probability of waiting (Erlang C formula).
    pub fn waiting_probability(&self) -> f64 {
        let a = self.offered_load;
        let c = self.servers;

        // Numerator: a^c / (c! · (1 - ρ))
        let rho = a / c as f64;
        let numerator = a.powi(c as i32) / (Self::factorial(c) * (1.0 - rho));

        // Denominator: sum + numerator
        let mut sum = 0.0;
        for k in 0..c {
            sum += a.powi(k as i32) / Self::factorial(k);
        }
        sum += numerator;

        numerator / sum
    }

    /// Mean number in queue: Lq = C(c,a) · ρ / (1-ρ).
    pub fn mean_queue_length(&self) -> f64 {
        let rho = self.offered_load / self.servers as f64;
        self.waiting_probability() * rho / (1.0 - rho)
    }

    /// Given service rate μ, compute mean wait time.
    pub fn mean_wait_time(&self, mu: f64) -> f64 {
        // Wq = Lq / λ = Lq * μ / a... but we need λ
        let lambda = self.offered_load * mu;
        self.mean_queue_length() / lambda
    }
}

/// Find the minimum number of channels needed to achieve a target blocking probability.
pub fn erlang_b_find_channels(offered_load: f64, target_blocking: f64) -> usize {
    let mut c = 1;
    loop {
        let eb = ErlangB::new(c, offered_load);
        if eb.blocking_probability() <= target_blocking {
            return c;
        }
        c += 1;
        if c > 10000 {
            panic!("could not find channels within 10000");
        }
    }
}

/// Find the minimum number of servers to achieve a target waiting probability.
pub fn erlang_c_find_servers(offered_load: f64, target_wait_prob: f64) -> usize {
    let start = (offered_load.ceil() as usize).max(1) + 1;
    for c in start..10000 {
        let ec = ErlangC::new(c, offered_load);
        if ec.waiting_probability() <= target_wait_prob {
            return c;
        }
    }
    panic!("could not find servers within 10000");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erlang_b_single_channel() {
        // B(1, a) = a/(1+a)
        let eb = ErlangB::new(1, 2.0);
        let expected = 2.0 / 3.0;
        assert!((eb.blocking_probability() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_erlang_b_blocking_decreases_with_channels() {
        let b1 = ErlangB::new(1, 2.0).blocking_probability();
        let b2 = ErlangB::new(2, 2.0).blocking_probability();
        let b3 = ErlangB::new(3, 2.0).blocking_probability();
        assert!(b1 > b2);
        assert!(b2 > b3);
    }

    #[test]
    fn test_erlang_b_zero_load() {
        let eb = ErlangB::new(5, 0.0);
        assert!((eb.blocking_probability() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_erlang_b_carried_load() {
        let eb = ErlangB::new(5, 3.0);
        let carried = eb.carried_load();
        assert!(carried < 3.0); // some blocked
        assert!(carried > 0.0);
    }

    #[test]
    fn test_erlang_c_single_server() {
        // C(1, a) = a (utilization)
        let ec = ErlangC::new(1, 0.5);
        assert!((ec.waiting_probability() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_erlang_c_zero_load() {
        let ec = ErlangC::new(3, 0.0);
        assert!((ec.waiting_probability() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_erlang_c_decreases_with_servers() {
        let c1 = ErlangC::new(2, 1.5).waiting_probability();
        let c2 = ErlangC::new(3, 1.5).waiting_probability();
        let c3 = ErlangC::new(4, 1.5).waiting_probability();
        assert!(c1 > c2);
        assert!(c2 > c3);
    }

    #[test]
    fn test_find_channels() {
        let c = erlang_b_find_channels(5.0, 0.01);
        let eb = ErlangB::new(c, 5.0);
        assert!(eb.blocking_probability() <= 0.01);
        // Should be reasonably tight
        let eb_prev = ErlangB::new(c - 1, 5.0);
        assert!(eb_prev.blocking_probability() > 0.01);
    }

    #[test]
    fn test_find_servers() {
        let s = erlang_c_find_servers(5.0, 0.1);
        let ec = ErlangC::new(s, 5.0);
        assert!(ec.waiting_probability() <= 0.1);
    }

    #[test]
    fn test_erlang_b_known_values() {
        // B(5, 3) ≈ 0.1101
        let eb = ErlangB::new(5, 3.0);
        assert!((eb.blocking_probability() - 0.1101).abs() < 0.001);
    }

    #[test]
    fn test_erlang_b_utilization() {
        let eb = ErlangB::new(10, 5.0);
        let u = eb.utilization();
        assert!(u > 0.0 && u < 1.0);
    }
}

//! M/M/c queue: multi-server queue with Poisson arrivals and exponential service.

use serde::{Deserialize, Serialize};

/// M/M/c queue parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MMc {
    /// Arrival rate (λ).
    pub lambda: f64,
    /// Service rate per server (μ).
    pub mu: f64,
    /// Number of servers (c).
    pub servers: usize,
}

impl MMc {
    /// Create a new M/M/c queue. Panics if λ >= c·μ (unstable).
    pub fn new(lambda: f64, mu: f64, servers: usize) -> Self {
        assert!(servers > 0, "must have at least 1 server");
        assert!(lambda < servers as f64 * mu, "system unstable: λ must be < c·μ");
        Self { lambda, mu, servers }
    }

    /// Traffic intensity per server: ρ = λ/(c·μ).
    pub fn utilization(&self) -> f64 {
        self.lambda / (self.servers as f64 * self.mu)
    }

    /// Offered load: a = λ/μ.
    pub fn offered_load(&self) -> f64 {
        self.lambda / self.mu
    }

    /// Factorial computation.
    fn factorial(n: usize) -> f64 {
        (1..=n).fold(1.0, |acc, i| acc * i as f64)
    }

    /// Erlang C formula: probability that an arriving customer must wait.
    /// C(c, a) = (a^c / c!) · (c/(c-a)) / [Σ_{n=0}^{c-1} a^n/n! + (a^c/c!)·(c/(c-a))]
    pub fn erlang_c(&self) -> f64 {
        let a = self.offered_load();
        let c = self.servers;

        let a_over_c = a / c as f64;

        // Numerator term
        let numerator = (a.powi(c as i32) / Self::factorial(c)) * (1.0 / (1.0 - a_over_c));

        // Denominator: sum of a^n/n! for n=0..c-1, plus numerator
        let mut sum = 0.0;
        for n in 0..c {
            sum += a.powi(n as i32) / Self::factorial(n);
        }
        sum += numerator;

        numerator / sum
    }

    /// Probability that an arriving customer must wait = Erlang C.
    pub fn prob_wait(&self) -> f64 {
        self.erlang_c()
    }

    /// Probability that the system is empty.
    pub fn prob_empty(&self) -> f64 {
        let a = self.offered_load();
        let c = self.servers;
        let rho = self.utilization();

        // p0 = 1 / [Σ_{n=0}^{c-1} a^n/n! + a^c/(c!·(1-ρ))]
        let mut sum = 0.0;
        for n in 0..c {
            sum += a.powi(n as i32) / Self::factorial(n);
        }
        sum += a.powi(c as i32) / (Self::factorial(c) * (1.0 - rho));
        1.0 / sum
    }

    /// Steady-state probability of n customers in system.
    pub fn prob_n(&self, n: usize) -> f64 {
        let a = self.offered_load();
        let c = self.servers;
        let p0 = self.prob_empty();

        if n <= c {
            p0 * a.powi(n as i32) / Self::factorial(n)
        } else {
            p0 * a.powi(n as i32) / (Self::factorial(c) * (c as f64).powi((n - c) as i32))
        }
    }

    /// Mean number of customers in the queue: Lq = C(c,a) · ρ/(1-ρ).
    pub fn mean_queue_length(&self) -> f64 {
        let rho = self.utilization();
        self.erlang_c() * rho / (1.0 - rho)
    }

    /// Mean number of customers in the system: L = Lq + a.
    pub fn mean_system_size(&self) -> f64 {
        self.mean_queue_length() + self.offered_load()
    }

    /// Mean wait time in queue: Wq = Lq/λ.
    pub fn mean_wait_time(&self) -> f64 {
        self.mean_queue_length() / self.lambda
    }

    /// Mean time in system: W = Wq + 1/μ.
    pub fn mean_system_time(&self) -> f64 {
        self.mean_wait_time() + 1.0 / self.mu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmc_mm1_equivalence() {
        // M/M/1 should match M/M/1 when c=1
        let mm1 = crate::mm1::MM1::new(2.0, 5.0);
        let mmc = MMc::new(2.0, 5.0, 1);

        assert!((mmc.utilization() - mm1.utilization()).abs() < 1e-12);
        assert!((mmc.mean_queue_length() - mm1.mean_queue_length()).abs() < 1e-8);
        assert!((mmc.mean_system_size() - mm1.mean_system_size()).abs() < 1e-8);
        assert!((mmc.mean_wait_time() - mm1.mean_wait_time()).abs() < 1e-8);
    }

    #[test]
    fn test_mmc_utilization() {
        let q = MMc::new(4.0, 2.0, 3);
        // ρ = 4/(3*2) = 2/3
        assert!((q.utilization() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mmc_offered_load() {
        let q = MMc::new(4.0, 2.0, 3);
        assert!((q.offered_load() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_erlang_c_single_server() {
        let q = MMc::new(2.0, 5.0, 1);
        // For M/M/1: C(1,a) = ρ = 0.4
        assert!((q.erlang_c() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_erlang_c_two_servers() {
        let q = MMc::new(3.0, 2.0, 2);
        // a = 1.5, c = 2
        // C(2, 1.5) should be < 1 and > 0
        let c = q.erlang_c();
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_mmc_littles_law() {
        let q = MMc::new(4.0, 2.0, 3);
        let l = q.mean_system_size();
        let w = q.mean_system_time();
        assert!((l - q.lambda * w).abs() < 1e-8);
    }

    #[test]
    fn test_mmc_probabilities_sum() {
        let q = MMc::new(2.0, 3.0, 2);
        let sum: f64 = (0..50).map(|n| q.prob_n(n)).sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_mmc_more_servers_less_wait() {
        let q1 = MMc::new(4.0, 5.0, 1);
        let q2 = MMc::new(4.0, 5.0, 2);
        assert!(q2.mean_wait_time() < q1.mean_wait_time());
    }

    #[test]
    fn test_prob_wait_decreases_with_servers() {
        let pw1 = MMc::new(4.0, 5.0, 1).prob_wait();
        let pw2 = MMc::new(4.0, 5.0, 2).prob_wait();
        let pw3 = MMc::new(4.0, 5.0, 3).prob_wait();
        assert!(pw1 > pw2);
        assert!(pw2 > pw3);
    }
}

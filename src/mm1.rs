//! M/M/1 queue: single-server queue with Poisson arrivals and exponential service.

use serde::{Deserialize, Serialize};

/// M/M/1 queue parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MM1 {
    /// Arrival rate (λ).
    pub lambda: f64,
    /// Service rate (μ).
    pub mu: f64,
}

impl MM1 {
    /// Create a new M/M/1 queue. Panics if λ >= μ (unstable).
    pub fn new(lambda: f64, mu: f64) -> Self {
        assert!(lambda >= 0.0 && mu > 0.0, "rates must be non-negative, μ > 0");
        assert!(lambda < mu, "system unstable: λ must be < μ for M/M/1");
        Self { lambda, mu }
    }

    /// Create allowing unstable system (for finite-horizon analysis).
    pub fn new_unchecked(lambda: f64, mu: f64) -> Self {
        Self { lambda, mu }
    }

    /// Utilization (traffic intensity) ρ = λ/μ.
    pub fn utilization(&self) -> f64 {
        self.lambda / self.mu
    }

    /// Probability that the system is empty: P₀ = 1 - ρ.
    pub fn prob_empty(&self) -> f64 {
        1.0 - self.utilization()
    }

    /// Steady-state probability of n customers: Pₙ = (1-ρ)ρⁿ.
    pub fn prob_n(&self, n: usize) -> f64 {
        let rho = self.utilization();
        (1.0 - rho) * rho.powi(n as i32)
    }

    /// Mean number of customers in the system: L = ρ/(1-ρ).
    pub fn mean_system_size(&self) -> f64 {
        let rho = self.utilization();
        rho / (1.0 - rho)
    }

    /// Mean number of customers in the queue: Lq = ρ²/(1-ρ).
    pub fn mean_queue_length(&self) -> f64 {
        let rho = self.utilization();
        rho.powi(2) / (1.0 - rho)
    }

    /// Mean time a customer spends in the system: W = 1/(μ-λ).
    pub fn mean_system_time(&self) -> f64 {
        1.0 / (self.mu - self.lambda)
    }

    /// Mean time a customer spends waiting in queue: Wq = ρ/(μ-λ).
    pub fn mean_wait_time(&self) -> f64 {
        let rho = self.utilization();
        rho / (self.mu - self.lambda)
    }

    /// Verify Little's law: L = λW.
    pub fn verify_littles_law_system(&self) -> f64 {
        let l = self.mean_system_size();
        let w = self.mean_system_time();
        (l - self.lambda * w).abs()
    }

    /// Verify Little's law for queue: Lq = λWq.
    pub fn verify_littles_law_queue(&self) -> f64 {
        let lq = self.mean_queue_length();
        let wq = self.mean_wait_time();
        (lq - self.lambda * wq).abs()
    }

    /// Variance of number in system: ρ/(1-ρ)².
    pub fn system_size_variance(&self) -> f64 {
        let rho = self.utilization();
        rho / (1.0 - rho).powi(2)
    }

    /// Probability that wait time exceeds t: P(W > t) = ρ·e^{-μ(1-ρ)t}.
    pub fn prob_wait_exceeds(&self, t: f64) -> f64 {
        let rho = self.utilization();
        rho * (-self.mu * (1.0 - rho) * t).exp()
    }

    /// Probability that system time (sojourn time) exceeds t: P(S > t) = e^{-(μ-λ)t}.
    pub fn prob_sojourn_exceeds(&self, t: f64) -> f64 {
        (-(self.mu - self.lambda) * t).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mm1_utilization() {
        let q = MM1::new(2.0, 5.0);
        assert!((q.utilization() - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_mm1_mean_system_size() {
        let q = MM1::new(2.0, 5.0);
        // L = 0.4/0.6 = 2/3
        assert!((q.mean_system_size() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mm1_mean_queue_length() {
        let q = MM1::new(2.0, 5.0);
        // Lq = 0.16/0.6 = 4/15
        assert!((q.mean_queue_length() - 4.0 / 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_mm1_mean_system_time() {
        let q = MM1::new(2.0, 5.0);
        // W = 1/3
        assert!((q.mean_system_time() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mm1_mean_wait_time() {
        let q = MM1::new(2.0, 5.0);
        // Wq = 0.4/3 = 2/15
        assert!((q.mean_wait_time() - 2.0 / 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_littles_law_system() {
        let q = MM1::new(3.0, 5.0);
        assert!(q.verify_littles_law_system() < 1e-12);
    }

    #[test]
    fn test_littles_law_queue() {
        let q = MM1::new(3.0, 5.0);
        assert!(q.verify_littles_law_queue() < 1e-12);
    }

    #[test]
    fn test_steady_state_probabilities() {
        let q = MM1::new(1.0, 2.0);
        let rho = 0.5;
        assert!((q.prob_n(0) - 0.5).abs() < 1e-12);
        assert!((q.prob_n(1) - 0.25).abs() < 1e-12);
        assert!((q.prob_n(2) - 0.125).abs() < 1e-12);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let q = MM1::new(1.0, 3.0);
        let sum: f64 = (0..50).map(|n| q.prob_n(n)).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prob_empty() {
        let q = MM1::new(2.0, 5.0);
        assert!((q.prob_empty() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_system_size_variance() {
        let q = MM1::new(1.0, 2.0);
        // Var = 0.5/0.25 = 2
        assert!((q.system_size_variance() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sojourn_exceeds() {
        let q = MM1::new(1.0, 2.0);
        // P(S>0) = 1
        assert!((q.prob_sojourn_exceeds(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn test_unstable_panics() {
        MM1::new(5.0, 3.0);
    }

    #[test]
    fn test_high_utilization() {
        let q = MM1::new(9.9, 10.0);
        assert!(q.mean_queue_length() > 90.0);
    }
}

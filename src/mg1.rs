//! M/G/1 queue: single-server queue with Poisson arrivals and general service distribution.
//! Uses the Pollaczek-Khinchine formula.

use serde::{Deserialize, Serialize};

/// M/G/1 queue parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MG1 {
    /// Arrival rate (λ).
    pub lambda: f64,
    /// Mean service time: E[S] = 1/μ.
    pub mean_service: f64,
    /// Second moment of service time: E[S²].
    pub second_moment_service: f64,
}

impl MG1 {
    /// Create from λ, μ, and σ² (variance of service time).
    /// E[S] = 1/μ, E[S²] = Var(S) + E[S]² = σ² + 1/μ².
    pub fn from_mean_and_variance(lambda: f64, mu: f64, service_variance: f64) -> Self {
        assert!(lambda > 0.0 && mu > 0.0);
        let mean_service = 1.0 / mu;
        let second_moment = service_variance + mean_service.powi(2);
        Self { lambda, mean_service, second_moment_service: second_moment }
    }

    /// Create from moments directly.
    pub fn from_moments(lambda: f64, mean_service: f64, second_moment_service: f64) -> Self {
        Self { lambda, mean_service, second_moment_service }
    }

    /// Service rate μ.
    pub fn mu(&self) -> f64 {
        1.0 / self.mean_service
    }

    /// Utilization: ρ = λ·E[S].
    pub fn utilization(&self) -> f64 {
        self.lambda * self.mean_service
    }

    /// Variance of service time: Var(S) = E[S²] - E[S]².
    pub fn service_variance(&self) -> f64 {
        self.second_moment_service - self.mean_service.powi(2)
    }

    /// Squared coefficient of variation of service time: C²_S = Var(S)/E[S]².
    pub fn squared_cv(&self) -> f64 {
        self.service_variance() / self.mean_service.powi(2)
    }

    /// Pollaczek-Khinchine formula for mean queue length:
    /// Lq = (λ²·E[S²]) / (2·(1-ρ))
    pub fn mean_queue_length(&self) -> f64 {
        let rho = self.utilization();
        (self.lambda.powi(2) * self.second_moment_service) / (2.0 * (1.0 - rho))
    }

    /// Mean number in system: L = Lq + ρ.
    pub fn mean_system_size(&self) -> f64 {
        self.mean_queue_length() + self.utilization()
    }

    /// Mean wait time in queue (Pollaczek-Khinchine):
    /// Wq = λ·E[S²] / (2·(1-ρ))
    pub fn mean_wait_time(&self) -> f64 {
        let rho = self.utilization();
        (self.lambda * self.second_moment_service) / (2.0 * (1.0 - rho))
    }

    /// Mean system time: W = Wq + E[S].
    pub fn mean_system_time(&self) -> f64 {
        self.mean_wait_time() + self.mean_service
    }
}

/// Convenience: M/D/1 (deterministic service) with service time 1/μ.
/// E[S²] = E[S]² since Var=0.
pub fn md1_metrics(lambda: f64, mu: f64) -> MG1 {
    let mean_service = 1.0 / mu;
    MG1::from_moments(lambda, mean_service, mean_service.powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mg1_matches_mm1() {
        // M/M/1: service is exponential, so E[S²] = 2/μ²
        let lambda = 2.0;
        let mu = 5.0;
        let mg1 = MG1::from_moments(lambda, 1.0 / mu, 2.0 / mu.powi(2));
        let mm1 = crate::mm1::MM1::new(lambda, mu);

        assert!((mg1.mean_queue_length() - mm1.mean_queue_length()).abs() < 1e-10);
        assert!((mg1.mean_wait_time() - mm1.mean_wait_time()).abs() < 1e-10);
        assert!((mg1.mean_system_size() - mm1.mean_system_size()).abs() < 1e-10);
    }

    #[test]
    fn test_pollaczek_khinchine_md1() {
        // M/D/1: Lq = ρ²/(2(1-ρ))
        let lambda = 3.0;
        let mu = 5.0;
        let rho = lambda / mu;
        let mg1 = md1_metrics(lambda, mu);

        let expected_lq = rho.powi(2) / (2.0 * (1.0 - rho));
        assert!((mg1.mean_queue_length() - expected_lq).abs() < 1e-12);
    }

    #[test]
    fn test_md1_lower_queue_than_mm1() {
        let lambda = 3.0;
        let mu = 5.0;
        let md1 = md1_metrics(lambda, mu);
        let mm1 = MG1::from_moments(lambda, 1.0 / mu, 2.0 / mu.powi(2));

        // Deterministic service should have lower queue length
        assert!(md1.mean_queue_length() < mm1.mean_queue_length());
    }

    #[test]
    fn test_mg1_littles_law() {
        let mg1 = MG1::from_moments(2.0, 0.3, 0.15);
        let l = mg1.mean_system_size();
        let w = mg1.mean_system_time();
        assert!((l - mg1.lambda * w).abs() < 1e-10);
    }

    #[test]
    fn test_squared_cv() {
        // Exponential: C²=1
        let mg1 = MG1::from_moments(2.0, 0.2, 2.0 * 0.2 * 0.2);
        assert!((mg1.squared_cv() - 1.0).abs() < 1e-12);

        // Deterministic: C²=0
        let md1 = md1_metrics(2.0, 5.0);
        assert!((md1.squared_cv() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_utilization() {
        let mg1 = MG1::from_moments(3.0, 0.2, 0.1);
        assert!((mg1.utilization() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_higher_variance_worse() {
        let lambda = 2.0;
        let mean_s = 0.3;
        let low_var = MG1::from_moments(lambda, mean_s, mean_s.powi(2) + 0.01);
        let high_var = MG1::from_moments(lambda, mean_s, mean_s.powi(2) + 0.1);
        assert!(high_var.mean_queue_length() > low_var.mean_queue_length());
    }
}

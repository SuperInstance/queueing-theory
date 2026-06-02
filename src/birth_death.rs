//! Birth-death processes and steady-state probabilities.

use serde::{Deserialize, Serialize};

/// Parameters for a birth-death process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BirthDeath {
    /// Birth rates (arrival rates) λ_n for state n. λ_n = births[n].
    /// If births has length N, then λ_0..λ_{N-1} are defined.
    pub births: Vec<f64>,
    /// Death rates (service rates) μ_n for state n. μ_n = deaths[n].
    /// deaths[0] is typically 0 (no deaths from state 0).
    pub deaths: Vec<f64>,
}

impl BirthDeath {
    /// Create a new birth-death process with given birth and death rate vectors.
    pub fn new(births: Vec<f64>, deaths: Vec<f64>) -> Self {
        assert!(births.len() + 1 == deaths.len() || births.len() == deaths.len(),
            "deaths must have same length or one more element than births");
        Self { births, deaths }
    }

    /// Number of states (0 to max_state).
    pub fn num_states(&self) -> usize {
        self.deaths.len().max(self.births.len() + 1)
    }

    /// Compute steady-state probabilities π_n using the product formula:
    /// π_0 = 1 / (1 + Σ_{n=1}^{N} (λ_0·λ_1·…·λ_{n-1}) / (μ_1·μ_2·…·μ_n))
    /// π_n = π_0 · (λ_0·λ_1·…·λ_{n-1}) / (μ_1·μ_2·…·μ_n)
    pub fn steady_state_probabilities(&self) -> Vec<f64> {
        let n = self.num_states();
        let mut pi = Vec::with_capacity(n);

        // Compute unnormalized ratios r_n = π_n / π_0
        let mut ratios = vec![1.0_f64]; // r_0 = 1
        for i in 1..n {
            let lambda = if i - 1 < self.births.len() { self.births[i - 1] } else { 0.0 };
            let mu = if i < self.deaths.len() { self.deaths[i] } else { 0.0 };
            if mu == 0.0 {
                break; // absorbing state
            }
            ratios.push(ratios[i - 1] * lambda / mu);
        }

        let sum: f64 = ratios.iter().sum();
        if sum == 0.0 {
            return vec![0.0; n];
        }

        for r in &ratios {
            pi.push(r / sum);
        }
        // Pad with zeros if needed
        while pi.len() < n {
            pi.push(0.0);
        }
        pi
    }

    /// Mean number in the system from steady-state probabilities.
    pub fn mean_population(&self) -> f64 {
        let pi = self.steady_state_probabilities();
        pi.iter().enumerate().map(|(n, p)| n as f64 * p).sum()
    }

    /// Variance of the number in the system.
    pub fn population_variance(&self) -> f64 {
        let pi = self.steady_state_probabilities();
        let mean = self.mean_population();
        pi.iter()
            .enumerate()
            .map(|(n, p)| p * (n as f64 - mean).powi(2))
            .sum()
    }
}

/// Build an M/M/1 birth-death process.
pub fn mm1_birth_death(lambda: f64, mu: f64, max_states: usize) -> BirthDeath {
    let births = vec![lambda; max_states];
    let mut deaths = vec![0.0; max_states + 1];
    for i in 1..=max_states {
        deaths[i] = mu;
    }
    BirthDeath::new(births, deaths)
}

/// Build an M/M/c birth-death process.
pub fn mmc_birth_death(lambda: f64, mu: f64, c: usize, max_states: usize) -> BirthDeath {
    let births = vec![lambda; max_states];
    let mut deaths = vec![0.0; max_states + 1];
    for i in 1..=max_states {
        let servers_active = i.min(c);
        deaths[i] = servers_active as f64 * mu;
    }
    BirthDeath::new(births, deaths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mm1_steady_state() {
        // M/M/1 with λ=2, μ=3 → ρ=2/3
        let bd = mm1_birth_death(2.0, 3.0, 100);
        let pi = bd.steady_state_probabilities();
        // π_n = (1-ρ)ρ^n
        let rho = 2.0 / 3.0;
        assert!((pi[0] - (1.0 - rho)).abs() < 1e-8);
        assert!((pi[1] - (1.0 - rho) * rho).abs() < 1e-8);
        assert!((pi[2] - (1.0 - rho) * rho.powi(2)).abs() < 1e-8);
    }

    #[test]
    fn test_steady_state_sums_to_one() {
        let bd = mm1_birth_death(1.0, 2.0, 30);
        let pi = bd.steady_state_probabilities();
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_population_mm1() {
        // M/M/1 mean = ρ/(1-ρ)
        let bd = mm1_birth_death(0.5, 1.0, 80);
        let mean = bd.mean_population();
        let expected = 0.5 / 0.5;
        assert!((mean - expected).abs() < 1e-8);
    }

    #[test]
    fn test_mmc_birth_death() {
        let bd = mmc_birth_death(3.0, 2.0, 3, 20);
        assert_eq!(bd.deaths[1], 2.0); // 1 server active
        assert_eq!(bd.deaths[2], 4.0); // 2 servers active
        assert_eq!(bd.deaths[3], 6.0); // 3 servers active
        assert_eq!(bd.deaths[4], 6.0); // capped at 3 servers
    }

    #[test]
    fn test_birth_death_construction() {
        let bd = BirthDeath::new(vec![1.0, 2.0], vec![0.0, 3.0, 4.0]);
        assert_eq!(bd.num_states(), 3);
    }
}

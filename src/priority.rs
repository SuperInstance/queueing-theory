//! Priority queues: preemptive and non-preemptive priority disciplines.

use serde::{Deserialize, Serialize};

/// A priority class with arrival rate, service rate, and priority level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityClass {
    /// Priority level (lower number = higher priority).
    pub priority: usize,
    /// Arrival rate for this class (λ_k).
    pub lambda: f64,
    /// Service rate for this class (μ_k).
    pub mu: f64,
}

impl PriorityClass {
    /// Mean service time for this class.
    pub fn mean_service(&self) -> f64 {
        1.0 / self.mu
    }
}

/// M/G/1 priority queue (non-preemptive).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonPreemptivePriority {
    /// Priority classes, sorted by priority (ascending).
    pub classes: Vec<PriorityClass>,
}

impl NonPreemptivePriority {
    /// Create from a list of priority classes.
    pub fn new(classes: Vec<PriorityClass>) -> Self {
        let mut c = classes;
        c.sort_by(|a, b| a.priority.cmp(&b.priority));
        Self { classes: c }
    }

    /// Total arrival rate across all classes.
    pub fn total_lambda(&self) -> f64 {
        self.classes.iter().map(|c| c.lambda).sum()
    }

    /// Total utilization.
    pub fn total_utilization(&self) -> f64 {
        self.classes.iter().map(|c| c.lambda / c.mu).sum()
    }

    /// Sum of ρ_i for classes with priority ≤ k.
    fn cumulative_utilization(&self, k: usize) -> f64 {
        self.classes.iter()
            .filter(|c| c.priority <= k)
            .map(|c| c.lambda / c.mu)
            .sum()
    }

    /// Sum of λ_i·E[S_i²] for all classes with priority ≤ k.
    fn weighted_second_moment(&self, k: usize) -> f64 {
        // Assuming exponential service: E[S²] = 2/μ²
        self.classes.iter()
            .filter(|c| c.priority <= k)
            .map(|c| c.lambda * 2.0 / c.mu.powi(2))
            .sum()
    }

    /// Total W₀ = Σ λ_i·E[S_i²] / 2 (mean residual service time contribution).
    pub fn w0(&self) -> f64 {
        // Sum of λ_i * E[S_i^2] for all i, divided by 2
        let sum: f64 = self.classes.iter()
            .map(|c| c.lambda * 2.0 / c.mu.powi(2))
            .sum();
        sum / 2.0
    }

    /// Mean wait time for class k (non-preemptive M/G/1):
    /// W_k = W_0 / ((1 - σ_{k-1})(1 - σ_k))
    /// where σ_k = Σ_{i≤k} ρ_i
    pub fn mean_wait_time(&self, priority: usize) -> f64 {
        let w0 = self.w0();
        let sigma_prev = self.cumulative_utilization(priority - 1);
        let sigma_k = self.cumulative_utilization(priority);
        w0 / ((1.0 - sigma_prev) * (1.0 - sigma_k))
    }

    /// Mean system time for class k.
    pub fn mean_system_time(&self, priority: usize) -> f64 {
        let class = self.classes.iter().find(|c| c.priority == priority)
            .expect("priority class not found");
        self.mean_wait_time(priority) + class.mean_service()
    }

    /// Mean queue length for class k: Lq_k = λ_k · Wq_k.
    pub fn mean_queue_length(&self, priority: usize) -> f64 {
        let class = self.classes.iter().find(|c| c.priority == priority)
            .expect("priority class not found");
        class.lambda * self.mean_wait_time(priority)
    }
}

/// M/M/1 preemptive-resume priority queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptivePriority {
    /// Priority classes.
    pub classes: Vec<PriorityClass>,
}

impl PreemptivePriority {
    /// Create from a list of priority classes.
    pub fn new(classes: Vec<PriorityClass>) -> Self {
        let mut c = classes;
        c.sort_by(|a, b| a.priority.cmp(&b.priority));
        Self { classes: c }
    }

    /// Total utilization.
    pub fn total_utilization(&self) -> f64 {
        self.classes.iter().map(|c| c.lambda / c.mu).sum()
    }

    /// Cumulative utilization for priority ≤ k.
    fn cumulative_utilization(&self, k: usize) -> f64 {
        self.classes.iter()
            .filter(|c| c.priority <= k)
            .map(|c| c.lambda / c.mu)
            .sum()
    }

    /// Mean system time for class k (preemptive-resume M/M/1):
    /// W_k = (1/μ_k) / (1 - σ_k) · (1 - σ_{k-1})
    /// For highest priority (k=1): W_1 = (1/μ_1) / (1 - ρ_1)
    pub fn mean_system_time(&self, priority: usize) -> f64 {
        let class = self.classes.iter().find(|c| c.priority == priority)
            .expect("priority class not found");
        let sigma_k = self.cumulative_utilization(priority);
        let sigma_prev = self.cumulative_utilization(priority - 1);
        class.mean_service() / ((1.0 - sigma_k) * (1.0 - sigma_prev))
    }

    /// Mean wait time for class k: Wq_k = W_k - E[S_k].
    pub fn mean_wait_time(&self, priority: usize) -> f64 {
        let class = self.classes.iter().find(|c| c.priority == priority)
            .expect("priority class not found");
        self.mean_system_time(priority) - class.mean_service()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_class() -> Vec<PriorityClass> {
        vec![
            PriorityClass { priority: 1, lambda: 1.0, mu: 5.0 },
            PriorityClass { priority: 2, lambda: 2.0, mu: 5.0 },
        ]
    }

    #[test]
    fn test_nonpreemptive_higher_priority_waits_less() {
        let q = NonPreemptivePriority::new(make_two_class());
        let w1 = q.mean_wait_time(1);
        let w2 = q.mean_wait_time(2);
        assert!(w1 < w2);
    }

    #[test]
    fn test_preemptive_higher_priority_faster() {
        let q = PreemptivePriority::new(make_two_class());
        let w1 = q.mean_system_time(1);
        let w2 = q.mean_system_time(2);
        assert!(w1 < w2);
    }

    #[test]
    fn test_nonpreemptive_total_utilization() {
        let q = NonPreemptivePriority::new(make_two_class());
        // ρ = 1/5 + 2/5 = 0.6
        assert!((q.total_utilization() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_nonpreemptive_littles_law() {
        let q = NonPreemptivePriority::new(make_two_class());
        for class in &q.classes {
            let lq = q.mean_queue_length(class.priority);
            let wq = q.mean_wait_time(class.priority);
            assert!((lq - class.lambda * wq).abs() < 1e-10);
        }
    }

    #[test]
    fn test_preemptive_highest_priority_like_mm1() {
        // With only one class, preemptive = M/M/1
        let classes = vec![PriorityClass { priority: 1, lambda: 2.0, mu: 5.0 }];
        let q = PreemptivePriority::new(classes);
        let w = q.mean_system_time(1);
        let mm1 = crate::mm1::MM1::new(2.0, 5.0);
        assert!((w - mm1.mean_system_time()).abs() < 1e-12);
    }
}

//! Staffing and capacity planning: applying queueing theory to optimize server pools.

use serde::{Deserialize, Serialize};
use crate::{MMc, ErlangC};

/// Configuration for an server pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerPool {
    /// Number of servers.
    pub num_servers: usize,
    /// Task arrival rate (tasks per unit time).
    pub task_arrival_rate: f64,
    /// Task service rate (tasks per unit time per server).
    pub task_service_rate: f64,
}

impl ServerPool {
    /// Create a new server pool configuration.
    pub fn new(num_servers: usize, task_arrival_rate: f64, task_service_rate: f64) -> Self {
        Self { num_servers, task_arrival_rate, task_service_rate }
    }

    /// Utilization per server.
    pub fn utilization(&self) -> f64 {
        self.task_arrival_rate / (self.num_servers as f64 * self.task_service_rate)
    }

    /// Offered load.
    pub fn offered_load(&self) -> f64 {
        self.task_arrival_rate / self.task_service_rate
    }

    /// Is the pool stable (can handle the load)?
    pub fn is_stable(&self) -> bool {
        self.utilization() < 1.0
    }

    /// Mean wait time for tasks (M/M/c model).
    pub fn mean_task_wait_time(&self) -> f64 {
        let mmc = MMc::new(self.task_arrival_rate, self.task_service_rate, self.num_servers);
        mmc.mean_wait_time()
    }

    /// Mean time to task completion (system time).
    pub fn mean_task_completion_time(&self) -> f64 {
        let mmc = MMc::new(self.task_arrival_rate, self.task_service_rate, self.num_servers);
        mmc.mean_system_time()
    }

    /// Mean number of tasks waiting.
    pub fn mean_tasks_waiting(&self) -> f64 {
        let mmc = MMc::new(self.task_arrival_rate, self.task_service_rate, self.num_servers);
        mmc.mean_queue_length()
    }

    /// Probability a task must wait (Erlang C).
    pub fn prob_task_waits(&self) -> f64 {
        let ec = ErlangC::new(self.num_servers, self.offered_load());
        ec.waiting_probability()
    }

    /// Find optimal number of servers to achieve target wait time.
    pub fn find_optimal_servers(target_wait_time: f64, task_arrival_rate: f64, task_service_rate: f64) -> usize {
        let a = task_arrival_rate / task_service_rate;
        let min_servers = (a.ceil() as usize).max(1) + 1;
        for c in min_servers..10000 {
            let pool = ServerPool::new(c, task_arrival_rate, task_service_rate);
            if pool.mean_task_wait_time() <= target_wait_time {
                return c;
            }
        }
        panic!("could not find suitable server count");
    }

    /// Find optimal servers to achieve target wait probability.
    pub fn find_servers_for_wait_prob(target_prob: f64, task_arrival_rate: f64, task_service_rate: f64) -> usize {
        crate::erlang::erlang_c_find_servers(task_arrival_rate / task_service_rate, target_prob)
    }
}

/// Task priority configuration for server scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityTaskConfig {
    /// High-priority arrival rate.
    pub high_priority_rate: f64,
    /// Normal-priority arrival rate.
    pub normal_priority_rate: f64,
    /// Low-priority arrival rate.
    pub low_priority_rate: f64,
    /// Service rate (same for all priorities).
    pub service_rate: f64,
}

impl PriorityTaskConfig {
    /// Analyze with non-preemptive priority queueing.
    pub fn analyze_nonpreemptive(&self) -> PriorityAnalysis {
        use crate::priority::{NonPreemptivePriority, PriorityClass};

        let q = NonPreemptivePriority::new(vec![
            PriorityClass { priority: 1, lambda: self.high_priority_rate, mu: self.service_rate },
            PriorityClass { priority: 2, lambda: self.normal_priority_rate, mu: self.service_rate },
            PriorityClass { priority: 3, lambda: self.low_priority_rate, mu: self.service_rate },
        ]);

        PriorityAnalysis {
            high_wait: q.mean_wait_time(1),
            normal_wait: q.mean_wait_time(2),
            low_wait: q.mean_wait_time(3),
            total_utilization: q.total_utilization(),
        }
    }
}

/// Results from priority task analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityAnalysis {
    pub high_wait: f64,
    pub normal_wait: f64,
    pub low_wait: f64,
    pub total_utilization: f64,
}

/// Capacity planning: find how many servers needed for a given SLA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaConfig {
    /// Maximum acceptable mean wait time.
    pub max_wait_time: f64,
    /// Maximum acceptable wait probability.
    pub max_wait_probability: f64,
    /// Target utilization ceiling.
    pub max_utilization: f64,
}

impl SlaConfig {
    /// Plan capacity given arrival and service rates.
    pub fn plan_capacity(&self, arrival_rate: f64, service_rate: f64) -> CapacityPlan {
        let servers_for_wait = ServerPool::find_servers_for_wait_prob(
            self.max_wait_probability, arrival_rate, service_rate
        );
        let servers_for_util = {
            let a = arrival_rate / service_rate;
            let min_for_util = (a / self.max_utilization).ceil() as usize;
            min_for_util.max(1)
        };
        let servers_for_time = ServerPool::find_optimal_servers(
            self.max_wait_time, arrival_rate, service_rate
        );

        let recommended = servers_for_wait.max(servers_for_util).max(servers_for_time);
        let pool = ServerPool::new(recommended, arrival_rate, service_rate);

        CapacityPlan {
            recommended_servers: recommended,
            servers_for_wait_prob: servers_for_wait,
            servers_for_utilization: servers_for_util,
            servers_for_wait_time: servers_for_time,
            expected_wait_time: pool.mean_task_wait_time(),
            expected_wait_prob: pool.prob_task_waits(),
            expected_utilization: pool.utilization(),
        }
    }
}

/// Capacity planning result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityPlan {
    pub recommended_servers: usize,
    pub servers_for_wait_prob: usize,
    pub servers_for_utilization: usize,
    pub servers_for_wait_time: usize,
    pub expected_wait_time: f64,
    pub expected_wait_prob: f64,
    pub expected_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MM1;

    #[test]
    fn test_server_pool_stability() {
        let pool = ServerPool::new(5, 10.0, 3.0);
        assert!(pool.is_stable());
        assert!((pool.utilization() - 10.0 / 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_server_pool_unstable() {
        let pool = ServerPool::new(2, 10.0, 3.0);
        assert!(!pool.is_stable());
    }

    #[test]
    fn test_server_pool_wait_time() {
        let pool = ServerPool::new(3, 5.0, 2.0);
        let w = pool.mean_task_wait_time();
        assert!(w > 0.0);
        assert!(w < pool.mean_task_completion_time());
    }

    #[test]
    fn test_find_optimal_servers() {
        let n = ServerPool::find_optimal_servers(0.5, 10.0, 3.0);
        let pool = ServerPool::new(n, 10.0, 3.0);
        assert!(pool.mean_task_wait_time() <= 0.5);
    }

    #[test]
    fn test_more_servers_lower_wait() {
        let p1 = ServerPool::new(4, 10.0, 3.0);
        let p2 = ServerPool::new(6, 10.0, 3.0);
        assert!(p2.mean_task_wait_time() < p1.mean_task_wait_time());
    }

    #[test]
    fn test_priority_analysis() {
        let config = PriorityTaskConfig {
            high_priority_rate: 1.0,
            normal_priority_rate: 2.0,
            low_priority_rate: 1.0,
            service_rate: 5.0,
        };
        let analysis = config.analyze_nonpreemptive();
        assert!(analysis.high_wait < analysis.normal_wait);
        assert!(analysis.normal_wait < analysis.low_wait);
        assert!(analysis.total_utilization < 1.0);
    }

    #[test]
    fn test_capacity_planning() {
        let sla = SlaConfig {
            max_wait_time: 0.2,
            max_wait_probability: 0.3,
            max_utilization: 0.8,
        };
        let plan = sla.plan_capacity(10.0, 3.0);
        assert!(plan.expected_wait_time <= 0.2 + 1e-10);
        assert!(plan.expected_wait_prob <= 0.3 + 1e-10);
        assert!(plan.expected_utilization <= 0.8 + 1e-10);
    }

    #[test]
    fn test_littles_law_server_pool() {
        let pool = ServerPool::new(5, 10.0, 3.0);
        let mmc = MMc::new(10.0, 3.0, 5);
        // L = λ * W
        let l = mmc.mean_system_size();
        let w = mmc.mean_system_time();
        assert!((l - 10.0 * w).abs() < 1e-8);
    }

    #[test]
    fn test_single_server_pool() {
        let pool = ServerPool::new(1, 2.0, 5.0);
        let mm1 = MM1::new(2.0, 5.0);
        assert!((pool.mean_task_wait_time() - mm1.mean_wait_time()).abs() < 1e-10);
    }
}

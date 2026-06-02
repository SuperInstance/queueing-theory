//! Queue networks: Jackson networks (open and closed).

use serde::{Deserialize, Serialize};
use nalgebra::DMatrix;

/// A single node (queue) in a Jackson network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkNode {
    /// Node identifier.
    pub id: usize,
    /// Number of servers at this node.
    pub servers: usize,
    /// Service rate per server at this node.
    pub mu: f64,
}

impl NetworkNode {
    /// Create a single-server node.
    pub fn single(id: usize, mu: f64) -> Self {
        Self { id, servers: 1, mu }
    }

    /// Create a multi-server node.
    pub fn multi(id: usize, mu: f64, servers: usize) -> Self {
        Self { id, servers, mu }
    }

    /// Total service rate.
    pub fn total_mu(&self) -> f64 {
        self.servers as f64 * self.mu
    }
}

/// Open Jackson network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JacksonNetwork {
    /// Nodes in the network.
    pub nodes: Vec<NetworkNode>,
    /// Routing probability matrix P[i][j]: probability of going from i to j.
    /// External arrival rates: gamma[i].
    pub routing: DMatrix<f64>,
    /// External arrival rates for each node.
    pub external_arrivals: Vec<f64>,
}

impl JacksonNetwork {
    /// Create a new Jackson network.
    pub fn new(nodes: Vec<NetworkNode>, routing: DMatrix<f64>, external_arrivals: Vec<f64>) -> Self {
        assert_eq!(nodes.len(), external_arrivals.len());
        assert_eq!(routing.nrows(), nodes.len());
        assert_eq!(routing.ncols(), nodes.len());
        Self { nodes, routing, external_arrivals }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Solve for effective arrival rates at each node using traffic equations:
    /// λ_i = γ_i + Σ_j λ_j · P[j][i]
    /// i.e., λ = γ + λ · P^T → λ(I - P^T) = γ
    pub fn effective_arrival_rates(&self) -> Vec<f64> {
        let n = self.num_nodes();
        // λ = γ + P^T · λ → (I - P^T)λ = γ
        let pt = self.routing.transpose();
        let identity = DMatrix::identity(n, n);
        let a = &identity - &pt;
        let gamma = DMatrix::from_column_slice(n, 1, &self.external_arrivals);

        // Solve a * lambda = gamma
        let lambda = a.lu().solve(&gamma).expect("routing matrix is singular");
        (0..n).map(|i| lambda[(i, 0)]).collect()
    }

    /// Check if all nodes are stable (λ_i < c_i · μ_i for all i).
    pub fn is_stable(&self) -> bool {
        let lambdas = self.effective_arrival_rates();
        for (i, node) in self.nodes.iter().enumerate() {
            if lambdas[i] >= node.total_mu() {
                return false;
            }
        }
        true
    }

    /// Utilization at each node.
    pub fn utilizations(&self) -> Vec<f64> {
        let lambdas = self.effective_arrival_rates();
        self.nodes.iter().enumerate()
            .map(|(i, node)| lambdas[i] / node.total_mu())
            .collect()
    }

    /// Mean number of customers at each node (treating each as M/M/c).
    pub fn mean_node_populations(&self) -> Vec<f64> {
        let lambdas = self.effective_arrival_rates();
        self.nodes.iter().enumerate().map(|(i, node)| {
            let rho = lambdas[i] / node.total_mu();
            if node.servers == 1 {
                // M/M/1: L = ρ/(1-ρ)
                rho / (1.0 - rho)
            } else {
                // Approximate M/M/c using Erlang C
                let mmc = crate::mmc::MMc::new(lambdas[i], node.mu, node.servers);
                mmc.mean_system_size()
            }
        }).collect()
    }

    /// Total mean population across all nodes.
    pub fn total_population(&self) -> f64 {
        self.mean_node_populations().iter().sum()
    }

    /// Throughput at each node (= effective arrival rate).
    pub fn throughputs(&self) -> Vec<f64> {
        self.effective_arrival_rates()
    }

    /// Total throughput (sum of external arrivals).
    pub fn total_throughput(&self) -> f64 {
        self.external_arrivals.iter().sum()
    }

    /// Mean system time across the entire network (Little's law): W = L / γ_total.
    pub fn mean_network_time(&self) -> f64 {
        let total_l = self.total_population();
        let total_gamma: f64 = self.external_arrivals.iter().sum();
        total_l / total_gamma
    }
}

/// Closed Jackson network (fixed population, no external arrivals).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedJacksonNetwork {
    /// Nodes.
    pub nodes: Vec<NetworkNode>,
    /// Routing matrix (rows must sum to 1).
    pub routing: DMatrix<f64>,
    /// Total population (fixed number of jobs).
    pub population: usize,
}

impl ClosedJacksonNetwork {
    /// Create a new closed Jackson network.
    pub fn new(nodes: Vec<NetworkNode>, routing: DMatrix<f64>, population: usize) -> Self {
        Self { nodes, routing, population }
    }

    /// Compute the relative throughput vector (eigenvector of routing matrix).
    /// For a closed network, solve π = π·P (left eigenvector for eigenvalue 1).
    pub fn relative_throughputs(&self) -> Vec<f64> {
        let n = self.num_nodes();
        let p = &self.routing;
        let pt = p.transpose();

        // Solve (P^T - I)v = 0 with v_1 = 1
        // Use power iteration for the stationary distribution
        let mut v = vec![1.0 / n as f64; n];
        for _ in 0..1000 {
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += v[j] * pt[(j, i)];
                }
            }
            let norm: f64 = new_v.iter().sum();
            if norm > 0.0 {
                for x in &mut new_v {
                    *x /= norm;
                }
            }
            // Check convergence
            let max_diff = new_v.iter().zip(&v).map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            v = new_v;
            if max_diff < 1e-14 {
                break;
            }
        }
        v
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_node_open_network() {
        let nodes = vec![
            NetworkNode::single(0, 5.0),
            NetworkNode::single(1, 3.0),
        ];
        // Node 0 → node 1 with prob 0.5, exits with prob 0.5
        // Node 1 → exits with prob 1.0
        let routing = DMatrix::from_row_slice(2, 2, &[
            0.0, 0.5,
            0.0, 0.0,
        ]);
        let external = vec![2.0, 0.0];

        let net = JacksonNetwork::new(nodes, routing, external);
        let lambdas = net.effective_arrival_rates();

        // λ_0 = 2.0 (external only)
        // λ_1 = 0.5 * 2.0 = 1.0
        assert!((lambdas[0] - 2.0).abs() < 1e-10);
        assert!((lambdas[1] - 1.0).abs() < 1e-10);
        assert!(net.is_stable());
    }

    #[test]
    fn test_open_network_littles_law() {
        let nodes = vec![
            NetworkNode::single(0, 5.0),
            NetworkNode::single(1, 4.0),
        ];
        let routing = DMatrix::from_row_slice(2, 2, &[
            0.0, 0.3,
            0.0, 0.0,
        ]);
        let external = vec![2.0, 0.5];

        let net = JacksonNetwork::new(nodes, routing, external);
        let total_l = net.total_population();
        let total_gamma = net.total_throughput();
        let w = net.mean_network_time();

        assert!((total_l - total_gamma * w).abs() < 1e-8);
    }

    #[test]
    fn test_three_node_network() {
        let nodes = vec![
            NetworkNode::single(0, 10.0),
            NetworkNode::single(1, 5.0),
            NetworkNode::single(2, 8.0),
        ];
        let routing = DMatrix::from_row_slice(3, 3, &[
            0.0, 0.6, 0.0,
            0.0, 0.0, 0.8,
            0.0, 0.0, 0.0,
        ]);
        let external = vec![3.0, 0.0, 0.0];

        let net = JacksonNetwork::new(nodes, routing, external);
        let lambdas = net.effective_arrival_rates();

        assert!((lambdas[0] - 3.0).abs() < 1e-10);
        assert!((lambdas[1] - 1.8).abs() < 1e-10);
        assert!((lambdas[2] - 1.44).abs() < 1e-10);
    }

    #[test]
    fn test_stability_check() {
        let nodes = vec![
            NetworkNode::single(0, 1.0), // way too slow
        ];
        let routing = DMatrix::from_row_slice(1, 1, &[0.0]);
        let external = vec![2.0];

        let net = JacksonNetwork::new(nodes, routing, external);
        assert!(!net.is_stable());
    }

    #[test]
    fn test_closed_network_throughputs() {
        let nodes = vec![
            NetworkNode::single(0, 5.0),
            NetworkNode::single(1, 3.0),
        ];
        let routing = DMatrix::from_row_slice(2, 2, &[
            0.0, 1.0,
            1.0, 0.0,
        ]);
        let net = ClosedJacksonNetwork::new(nodes, routing, 5);
        let v = net.relative_throughputs();
        // Symmetric: should be [0.5, 0.5]
        assert!((v[0] - 0.5).abs() < 1e-10);
        assert!((v[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_utilization_open_network() {
        let nodes = vec![NetworkNode::single(0, 10.0)];
        let routing = DMatrix::from_row_slice(1, 1, &[0.0]);
        let external = vec![5.0];

        let net = JacksonNetwork::new(nodes, routing, external);
        let utils = net.utilizations();
        assert!((utils[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_multi_node_with_feedback() {
        let nodes = vec![
            NetworkNode::single(0, 8.0),
            NetworkNode::single(1, 6.0),
        ];
        // Node 0 → 1 (0.7), feedback to 0 (0.1), exit (0.2)
        // Node 1 → 0 (0.5), exit (0.5)
        let routing = DMatrix::from_row_slice(2, 2, &[
            0.1, 0.7,
            0.5, 0.0,
        ]);
        let external = vec![1.0, 0.5];

        let net = JacksonNetwork::new(nodes, routing, external);
        assert!(net.is_stable());

        let lambdas = net.effective_arrival_rates();
        assert!(lambdas[0] > 1.0); // feedback increases it
        assert!(lambdas[1] > 0.5);
    }
}

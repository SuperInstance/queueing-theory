//! Kendall's notation for classifying queueing systems.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Arrival process specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArrivalProcess {
    /// Markovian (Poisson) arrivals — exponential inter-arrival times.
    M,
    /// Deterministic arrivals — constant inter-arrival times.
    D,
    /// General (arbitrary) distribution.
    G,
    /// Erlang-k distribution.
    Ek(usize),
}

/// Service process specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceProcess {
    /// Markovian (exponential) service times.
    M,
    /// Deterministic service times.
    D,
    /// General (arbitrary) distribution.
    G,
    /// Erlang-k distribution.
    Ek(usize),
}

/// Queue discipline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Discipline {
    /// First In, First Out.
    FIFO,
    /// Last In, First Out.
    LIFO,
    /// Service In Random Order.
    SIRO,
    /// Priority-based.
    Priority,
    /// Processor sharing.
    PS,
}

/// Capacity specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Capacity {
    /// Infinite capacity.
    Infinite,
    /// Finite capacity (max customers in system).
    Finite(usize),
}

/// Population specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Population {
    /// Infinite population.
    Infinite,
    /// Finite population size.
    Finite(usize),
}

/// Full Kendall notation descriptor: A/S/c/K/N/D
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KendallNotation {
    pub arrival: ArrivalProcess,
    pub service: ServiceProcess,
    pub servers: usize,
    pub capacity: Capacity,
    pub population: Population,
    pub discipline: Discipline,
}

impl KendallNotation {
    /// Create a standard M/M/1 queue descriptor.
    pub fn mm1() -> Self {
        Self {
            arrival: ArrivalProcess::M,
            service: ServiceProcess::M,
            servers: 1,
            capacity: Capacity::Infinite,
            population: Population::Infinite,
            discipline: Discipline::FIFO,
        }
    }

    /// Create an M/M/c queue descriptor.
    pub fn mmc(c: usize) -> Self {
        Self {
            arrival: ArrivalProcess::M,
            service: ServiceProcess::M,
            servers: c,
            capacity: Capacity::Infinite,
            population: Population::Infinite,
            discipline: Discipline::FIFO,
        }
    }

    /// Create an M/G/1 queue descriptor.
    pub fn mg1() -> Self {
        Self {
            arrival: ArrivalProcess::M,
            service: ServiceProcess::G,
            servers: 1,
            capacity: Capacity::Infinite,
            population: Population::Infinite,
            discipline: Discipline::FIFO,
        }
    }

    /// Create an M/D/1 queue descriptor.
    pub fn md1() -> Self {
        Self {
            arrival: ArrivalProcess::M,
            service: ServiceProcess::D,
            servers: 1,
            capacity: Capacity::Infinite,
            population: Population::Infinite,
            discipline: Discipline::FIFO,
        }
    }

    /// Short Kendall name string (e.g., "M/M/1", "M/M/3").
    pub fn short_name(&self) -> String {
        format!(
            "{}/{}/{}",
            fmt_process(self.arrival),
            fmt_process_service(self.service),
            self.servers
        )
    }
}

impl fmt::Display for KendallNotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}/{}",
            fmt_process(self.arrival),
            fmt_process_service(self.service),
            self.servers
        )?;
        match self.capacity {
            Capacity::Infinite => {}
            Capacity::Finite(k) => write!(f, "/{}", k)?,
        }
        match self.population {
            Population::Infinite => {}
            Population::Finite(n) => write!(f, "/{}", n)?,
        }
        if self.discipline != Discipline::FIFO {
            write!(f, "/{:?}", self.discipline)?;
        }
        Ok(())
    }
}

fn fmt_process(a: ArrivalProcess) -> &'static str {
    match a {
        ArrivalProcess::M => "M",
        ArrivalProcess::D => "D",
        ArrivalProcess::G => "G",
        ArrivalProcess::Ek(_) => "Ek",
    }
}

fn fmt_process_service(s: ServiceProcess) -> &'static str {
    match s {
        ServiceProcess::M => "M",
        ServiceProcess::D => "D",
        ServiceProcess::G => "G",
        ServiceProcess::Ek(_) => "Ek",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mm1_notation() {
        let q = KendallNotation::mm1();
        assert_eq!(q.short_name(), "M/M/1");
        assert_eq!(format!("{}", q), "M/M/1");
    }

    #[test]
    fn test_mmc_notation() {
        let q = KendallNotation::mmc(3);
        assert_eq!(q.short_name(), "M/M/3");
    }

    #[test]
    fn test_mg1_notation() {
        let q = KendallNotation::mg1();
        assert_eq!(q.short_name(), "M/G/1");
    }

    #[test]
    fn test_md1_notation() {
        let q = KendallNotation::md1();
        assert_eq!(q.short_name(), "M/D/1");
    }

    #[test]
    fn test_custom_capacity() {
        let q = KendallNotation {
            capacity: Capacity::Finite(10),
            ..KendallNotation::mm1()
        };
        assert_eq!(format!("{}", q), "M/M/1/10");
    }
}

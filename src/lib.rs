//! # queueing-theory
//!
//! Queueing theory — mathematical analysis of waiting lines and service systems.
//!
//! Covers Kendall's notation, birth-death processes, M/M/1, M/M/c, M/G/1 queues,
//! priority queues, Jackson networks, Erlang formulas, and staffing/capacity planning.

pub mod kendall;
pub mod birth_death;
pub mod mm1;
pub mod mmc;
pub mod mg1;
pub mod priority;
pub mod networks;
pub mod erlang;
pub mod staffing;

pub use kendall::*;
pub use birth_death::*;
pub use mm1::*;
pub use mmc::*;
pub use mg1::*;
pub use priority::*;
pub use networks::*;
pub use erlang::*;
pub use staffing::*;

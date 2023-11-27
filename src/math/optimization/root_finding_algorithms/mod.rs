use num_traits::{Bounded, Num, Signed};

/// Base trait for a root-finding algorithm.
pub trait RootFindingAlgorithm<T>
where
    T: Num + Bounded + Signed + PartialOrd + Copy + Clone,
{
    /// State of the root-finding algorithm.
    type State: Copy + Clone;

    /// Perform 1 iteration of the algorithm.
    fn iterate_once(&mut self, state: Self::State) -> (Self::State, T);

    /// Find the root (within a tolerance, given a maximum number of iterations.)
    fn solve(
        &mut self,
        initial_state: Self::State,
        tolerance: T,
        max_iter: u64,
    ) -> RootFindingResult<Self::State> {
        let mut state = initial_state;
        let mut error;
        let mut iter = max_iter;

        while iter > 0 {
            (state, error) = self.iterate_once(state);
            if error < tolerance {
                return RootFindingResult::Found(state);
            }
            iter -= 1;
        }
        RootFindingResult::MaxIterReached
    }
}

#[derive(Copy, Clone, Debug)]
/// Result of root-finding algorithm.
pub enum RootFindingResult<T> {
    /// Converged on value.
    Found(T),
    /// Could not converge within tolerance.
    MaxIterReached,
}

/// Newton Raphson method.
pub mod newton_raphson;
pub use newton_raphson::*;

/// Secant method.
pub mod secant_method;
pub use secant_method::*;

/// Halley's method.
pub mod halleys_method;
pub use halleys_method::*;

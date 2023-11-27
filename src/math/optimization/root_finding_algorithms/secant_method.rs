use num_traits::{Bounded, Num, Signed};

use crate::math::optimization::root_finding_algorithms::RootFindingAlgorithm;

/// State of secant method.
#[derive(Copy, Clone, Debug)]
pub struct SecantMethodState<T>
where
    T: Num + Copy + Clone,
{
    prev1: T,
    prev2: T,
}

impl<T> SecantMethodState<T>
where
    T: Num + Copy + Clone,
{
    /// Create a new SecantMethodState.
    #[must_use]
    pub fn new(guess: T) -> Self {
        Self {
            prev1: guess,
            prev2: guess + T::one(),
        }
    }
}

/// Secant method solver.
#[derive(Clone, Debug)]
pub struct SecantMethodSolver<T> {
    f: fn(T) -> T,
}

impl<T> RootFindingAlgorithm<T> for SecantMethodSolver<T>
where
    T: Num + Bounded + Signed + PartialOrd + Copy + Clone,
{
    type State = SecantMethodState<T>;

    fn iterate_once(&mut self, state: Self::State) -> (Self::State, T) {
        let prev1 = state.prev1;
        let prev2 = state.prev2;

        let x_new = prev1 - (self.f)(prev1) * (prev1 - prev2) / ((self.f)(prev1) - (self.f)(prev2));
        let error = (self.f)(x_new).abs();
        (
            Self::State {
                prev1: x_new,
                prev2: prev1,
            },
            error,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::math::RootFindingResult;

    use super::*;
    use std::f64::EPSILON as EPS;

    #[test]
    fn test_secant_method() {
        let mut solver = SecantMethodSolver {
            f: |x| x * x - 612.0,
        };
        let guess = SecantMethodState::new(1.0);

        match solver.solve(guess, EPS.sqrt(), 1000) {
            RootFindingResult::Found(ans) => {
                assert!(
                    (solver.f)(ans.prev1).abs() < EPS.sqrt(),
                    "Did not converge to the correct value."
                )
            }
            _ => panic!("Could not converge"),
        }
    }
}

use crate::math::optimization::root_finding_algorithms::RootFindingAlgorithm;

use num_traits::{Bounded, Num, Signed};

/// State of Newton-Raphson solver.
#[derive(Copy, Clone, Debug)]
pub struct NewtonRaphsonState<T>
where
    T: Num + Copy + Clone,
{
    prev_val: T,
}

impl<T> NewtonRaphsonState<T>
where
    T: Num + Copy + Clone,
{
    /// Create a new NewtonRaphsonState.
    #[must_use]
    pub const fn new(prev_val: T) -> Self {
        Self { prev_val }
    }
}

/// Newton-Raphson Solver.
#[derive(Clone, Debug)]
pub struct NewtonRaphsonSolver<T>
where
    T: Num + Bounded + Signed + PartialOrd + Copy + Clone,
{
    f: fn(T) -> T,
    f_prime: fn(T) -> T,
}

impl<T> RootFindingAlgorithm<T> for NewtonRaphsonSolver<T>
where
    T: Num + Bounded + Signed + PartialOrd + Copy + Clone,
{
    type State = NewtonRaphsonState<T>;

    fn iterate_once(&mut self, state: Self::State) -> (Self::State, T) {
        let x_new = state.prev_val - (self.f)(state.prev_val) / (self.f_prime)(state.prev_val);
        let error = (self.f)(x_new).abs();

        (Self::State { prev_val: x_new }, error)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::RootFindingResult;

    use super::*;
    use std::f64::EPSILON as EPS;

    #[test]
    fn test_newton_raphson() {
        let mut solver = NewtonRaphsonSolver {
            f: |x| x * x,
            f_prime: |x| 2.0 * x,
        };
        let guess = NewtonRaphsonState { prev_val: 10.0 };

        match solver.solve(guess, EPS.sqrt(), 100) {
            RootFindingResult::Found(ans) => {
                assert!(
                    (solver.f)(ans.prev_val).abs() < EPS.sqrt(),
                    "Did not converge to the correct value."
                )
            }
            _ => panic!("Could not converge"),
        }
    }
}

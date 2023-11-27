use crate::math::optimization::root_finding_algorithms::RootFindingAlgorithm;

use num_traits::{Bounded, Num, Signed};

/// State of Halley's method.
#[derive(Copy, Clone, Debug)]
pub struct HalleysMethodState<T>
where
    T: Num + Copy + Clone,
{
    prev_val: T,
}

/// Halley's method solver.
#[derive(Clone, Debug)]
pub struct HalleysMethodSolver<T>
where
    T: Num + Bounded + Signed + PartialOrd + Copy + Clone,
{
    f: fn(T) -> T,
    f_prime: fn(T) -> T,
    f_prime_prime: fn(T) -> T,
}

impl<T> RootFindingAlgorithm<T> for HalleysMethodSolver<T>
where
    T: Num + Bounded + Signed + PartialOrd + Copy + Clone,
{
    type State = HalleysMethodState<T>;

    fn iterate_once(&mut self, state: Self::State) -> (Self::State, T) {
        let prev_val = state.prev_val;

        let numerator = {
            // This trick has to be used, as multiplication with an integer is not required.
            let tmp = (self.f)(prev_val) * (self.f_prime)(prev_val);
            tmp + tmp
        };
        let denominator = {
            // This trick has to be used for same reason as above.
            let tmp = (self.f_prime)(prev_val);
            let tmp = tmp * tmp;
            let tmp = tmp + tmp;
            tmp - (self.f)(prev_val) * (self.f_prime_prime)(prev_val)
        };

        let x_new = prev_val - numerator / denominator;
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
    fn test_halleys_method() {
        let mut solver = HalleysMethodSolver {
            f: |x| x * x,
            f_prime: |x| 2.0 * x,
            f_prime_prime: |_x| 2.0,
        };
        let guess = HalleysMethodState { prev_val: 10.0 };

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

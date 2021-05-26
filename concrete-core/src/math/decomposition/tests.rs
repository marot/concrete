use crate::math::decomposition::{DecompositionBaseLog, DecompositionLevelCount, SignedDecomposer};
use crate::math::random::{RandomGenerable, Uniform};
use crate::math::torus::UnsignedTorus;
use crate::numeric::{SignedInteger, UnsignedInteger};
use crate::test_tools::{any_uint, any_usize, random_usize_between};
use std::convert::TryInto;
use std::fmt::Debug;

// Returns a random decomposition valid for the size of the T type.
fn random_decomp<T: UnsignedInteger>() -> SignedDecomposer<T> {
    let mut base_log;
    let mut level_count;
    loop {
        base_log = random_usize_between(2..T::BITS);
        level_count = random_usize_between(2..T::BITS);
        if base_log * level_count < T::BITS {
            break;
        }
    }
    SignedDecomposer::new(
        DecompositionBaseLog(base_log),
        DecompositionLevelCount(level_count),
    )
}

fn test_decompose_recompose<T: UnsignedInteger + Debug + RandomGenerable<Uniform>>()
where
    <T as UnsignedInteger>::Signed: Debug,
{
    // Checks that the decomposing and recomposing a value brings the closest representable
    for _ in 0..100_000 {
        let decomposer = random_decomp::<T>();
        let input = any_uint::<T>();
        let closest = decomposer.closest_representable(input);
        for term in decomposer.decompose(closest) {
            assert!(1 <= term.level().0);
            assert!(term.level().0 <= decomposer.level_count);
            let signed_term = term.value().into_signed();
            let half_basis =
                T::TWO.into_signed().pow(decomposer.base_log as u32) / T::TWO.into_signed();
            assert!(-half_basis <= signed_term);
            assert!(signed_term < half_basis);
        }
        assert_eq!(
            decomposer.closest_representable(input),
            decomposer.decompose(input).recompose()
        );
    }
}

#[test]
fn test_decompose_recompose_u32() {
    test_decompose_recompose::<u32>()
}

#[test]
fn test_decompose_recompose_u64() {
    test_decompose_recompose::<u64>()
}

fn test_round_to_closest_multiple<T: UnsignedTorus>() {
    for _ in 0..1000 {
        let log_b = any_usize();
        let level_max = any_usize();
        let val = any_uint::<T>();
        let delta = any_uint::<T>();
        let bits = T::BITS;
        let log_b = (log_b % ((bits / 4) - 1)) + 1;
        let log_b: usize = log_b.try_into().unwrap();
        let level_max = (level_max % 4) + 1;
        let level_max: usize = level_max.try_into().unwrap();
        let bit: usize = log_b * level_max;

        let val = val << (bits - bit);
        let delta = delta >> (bits - (bits - bit - 1));

        let decomposer = SignedDecomposer::new(
            DecompositionBaseLog(log_b),
            DecompositionLevelCount(level_max),
        );

        assert_eq!(
            val,
            decomposer.closest_representable(val.wrapping_add(delta))
        );
        assert_eq!(
            val,
            decomposer.closest_representable(val.wrapping_sub(delta))
        );
    }
}

#[test]
fn test_round_to_closest_multiple_u32() {
    test_round_to_closest_multiple::<u32>();
}

#[test]
fn test_round_to_closest_multiple_u64() {
    test_round_to_closest_multiple::<u64>();
}

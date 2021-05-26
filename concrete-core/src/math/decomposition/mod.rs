//! Signed decomposition of unsigned integers.
//!
//! Multiple homomorphic operations used in the concrete scheme use a signed decomposition to reduce
//! the amount of noise. This module contains a [`SignedDecomposer`] which offer a clean api for
//! this decomposition.
//!
//! # Description
//!
//! We assume a number $\theta$ lives in $\mathbb{Z}/q\mathbb{Z}$, with $q$ a power of two. Such
//! a number can also be seen as a signed integer in $[ -\frac{q}{2}; \frac{q}{2}-1]$. Assuming a
//! given base $B=2^{b}$ and a number of levels $l$ such that $B^l\leq q$,
//! such a $\theta$ can be approximately decomposed as:
//! $$
//!     \theta \approx \sum_{i=1}^l\tilde{\theta}_i\frac{q}{B^i}
//! $$
//! With the $\tilde{\theta}_i\in[-\frac{B}{2}, \frac{B}{2}-1]$. When $B^l = q$,
//! the decomposition is no longer an approximation, and becomes exact. The rationale behind using
//! an approximate decomposition like that, is that when using this decomposition the
//! approximation error will be located in the least significant bits, which are already erroneous.

use crate::numeric::{Numeric, SignedInteger, UnsignedInteger};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(test)]
mod tests;

/// The logarithm of the base used in a decomposition.
///
/// When decomposing an integer over powers of the $B=2^b$ basis, this type represents the $b$
/// value.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Deserialize, Serialize)]
pub struct DecompositionBaseLog(pub usize);

/// The number of levels used in a decomposition.
///
/// When decomposing an integer over the $l$ levels, this type represents the $l$ value.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Deserialize, Serialize)]
pub struct DecompositionLevelCount(pub usize);

/// The level of a given member of a decomposition.
///
/// When decomposing an integer over the $l$ levels, this type represent the level (in $[0,l)$)
/// currently manipulated.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Deserialize, Serialize)]
pub struct DecompositionLevel(pub usize);

/// A member of the decomposition.
///
/// If we decompose a value $\theta$ as a sum $\sum_{i=1}^l\tilde{\theta}_i\frac{q}{B^i}$, this
/// represents a $\tilde{\theta}_i$.
#[derive(Debug, PartialEq, Eq, Clone, Deserialize, Serialize)]
pub struct DecompositionTerm<T>
where
    T: UnsignedInteger,
{
    level: usize,
    base_log: usize,
    value: T,
}

impl<T> DecompositionTerm<T>
where
    T: UnsignedInteger,
{
    pub(crate) fn new(
        level: DecompositionLevel,
        base_log: DecompositionBaseLog,
        value: T,
    ) -> DecompositionTerm<T> {
        DecompositionTerm {
            level: level.0,
            base_log: base_log.0,
            value,
        }
    }

    /// Turns this term into a summand.
    ///
    /// If our member represents one $\tilde{\theta}_i$ of the decomposition, this method returns
    /// $\tilde{\theta}_i\frac{q}{B^i}$.
    pub fn to_summand(&self) -> T {
        let shift: usize = <T as Numeric>::BITS - self.base_log * self.level;
        self.value << shift
    }

    /// Returns the value of the term.
    ///
    /// If our member represents one $\tilde{\theta}_i$, this returns its actual value.
    pub fn value(&self) -> T {
        self.value
    }

    /// Returns the level of the term.
    ///
    /// If our member represents one $\tilde{\theta}_i$, this returns the value of $i$.
    pub fn level(&self) -> DecompositionLevel {
        DecompositionLevel(self.level)
    }
}

/// A structure which allows to decompose unsigned integers into a set of smaller coefficients.
///
/// See the [module level](super) documentation for a description of the decomposition.
#[derive(Debug)]
pub struct SignedDecomposer<T>
where
    T: UnsignedInteger,
{
    base_log: usize,
    level_count: usize,
    integer_type: PhantomData<T>,
}

impl<T> SignedDecomposer<T>
where
    T: UnsignedInteger,
{
    /// Creates a new decomposer
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::math::decomposition::{
    ///     SignedDecomposer,
    ///     DecompositionBaseLog,
    ///     DecompositionLevelCount
    /// };
    /// let decomposer = SignedDecomposer::<u32>::new(
    ///     DecompositionBaseLog(4),
    ///     DecompositionLevelCount(3)
    /// );
    /// assert_eq!(decomposer.level_count(), DecompositionLevelCount(3));
    /// assert_eq!(decomposer.base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn new(
        base_log: DecompositionBaseLog,
        level_count: DecompositionLevelCount,
    ) -> SignedDecomposer<T> {
        debug_assert!(
            T::BITS > base_log.0 * level_count.0,
            "Decomposed bits exceeds the size of the integer to be decomposed"
        );
        // debug_assert!(
        //     level_count.0 > 1,
        //     "A decomposition should be over two levels at least"
        // );
        SignedDecomposer {
            base_log: base_log.0,
            level_count: level_count.0,
            integer_type: PhantomData,
        }
    }

    /// Returns the logarithm in base two of the base of this decomposer.
    ///
    /// If the decomposer uses a base $B=2^b$, this returns $b$.
    pub fn base_log(&self) -> DecompositionBaseLog {
        DecompositionBaseLog(self.base_log)
    }

    /// Returns the number of levels of this decomposer.
    ///
    /// If the decomposer uses $l$ levels, this returns $l$.
    pub fn level_count(&self) -> DecompositionLevelCount {
        DecompositionLevelCount(self.level_count)
    }

    /// Returns the closet value representable by the decomposition.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::math::decomposition::{
    ///     SignedDecomposer,
    ///     DecompositionBaseLog,
    ///     DecompositionLevelCount
    /// };
    /// let decomposer = SignedDecomposer::<u32>::new(
    ///     DecompositionBaseLog(4),
    ///     DecompositionLevelCount(3)
    /// );
    /// let closest = decomposer.closest_representable(1_340_987_234_u32);
    /// assert_eq!(closest, 1_341_128_704_u32);
    /// ```
    pub fn closest_representable(&self, input: T) -> T {
        // The closest number representable by the decomposition can be computed by performing
        // the rounding at the appropriate bit.

        // We compute the number of least significant bits which can not be represented by the
        // decomposition
        let non_rep_bit_count: usize = <T as Numeric>::BITS - self.level_count * self.base_log;
        // We generate a mask which captures the non representable bits
        let non_rep_mask = T::ONE << (non_rep_bit_count - 1);
        // We retrieve the non representable bits
        let non_rep_bits = input & non_rep_mask;
        // We extract the msb of the  non representable bits to perform the rounding
        let non_rep_msb = non_rep_bits >> (non_rep_bit_count - 1);
        // We remove the non-representable bits and perform the rounding
        let res = input >> non_rep_bit_count;
        let res = res + non_rep_msb;
        res << non_rep_bit_count
    }

    /// Generates an iterator over the terms of the decomposition of the input.
    ///
    /// # Warning
    ///
    /// The returned iterator yields the terms $\tilde{\theta}_i$ in order of decreasing $i$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::math::decomposition::{
    ///     SignedDecomposer,
    /// DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::numeric::UnsignedInteger;
    /// let decomposer = SignedDecomposer::<u32>::new(
    ///     DecompositionBaseLog(4),
    ///     DecompositionLevelCount(3)
    /// );
    /// for term in decomposer.decompose(1_340_987_234_u32){
    ///     assert!(1 <= term.level().0);
    ///     assert!(term.level().0 <= 3);
    ///     let signed_term = term.value().into_signed();
    ///     let half_basis = 2i32.pow(4) / 2i32;
    ///     assert!(- half_basis <= signed_term);
    ///     assert!(signed_term < half_basis);
    /// }
    /// assert_eq!(decomposer.decompose(1).count(), 3);
    /// ```
    pub fn decompose(&self, input: T) -> SignedDecompositionIter<T> {
        SignedDecompositionIter::new(
            self.closest_representable(input),
            DecompositionBaseLog(self.base_log),
            DecompositionLevelCount(self.level_count),
        )
    }
}

/// An iterator that yields the elements of the signed decomposition of a number.
///
/// # Warning
///
/// This iterator yields the decomposition in reverse order. That means that the highest level
/// will be yielded first.
///
///
pub struct SignedDecompositionIter<T>
where
    T: UnsignedInteger,
{
    // The value being decomposed
    input: T,
    // The base log of the decomposition
    base_log: usize,
    // The number of levels of the decomposition
    level_count: usize,
    // The carry from the previous level
    previous_carry: T,
    // The current level
    current_level: usize,
    // A mask which allows to compute the mod B of a value. For B=2^4, this guy is of the form:
    // ...0001111
    mod_b_mask: T,
    // A mask which allows to test whether the value is larger than B/2. For B=2^4, this guy is
    // of the form:
    // ...0001000
    carry_mask: T,
}

impl<T> SignedDecompositionIter<T>
where
    T: UnsignedInteger,
{
    pub(crate) fn new(
        input: T,
        base_log: DecompositionBaseLog,
        level: DecompositionLevelCount,
    ) -> SignedDecompositionIter<T> {
        SignedDecompositionIter {
            input,
            base_log: base_log.0,
            level_count: level.0,
            previous_carry: T::ZERO,
            current_level: level.0,
            mod_b_mask: (T::ONE << base_log.0) - T::ONE,
            carry_mask: T::ONE << (base_log.0 - 1),
        }
    }

    /// Recomposes the decomposed value by summing all the terms.
    ///
    /// If this iterator yields $\tilde{\theta}_i$, this returns
    /// $\sum_{i=1}^l\tilde{\theta}_i\frac{q}{B^i}$.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::math::decomposition::{
    ///     SignedDecomposer,
    /// DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::numeric::UnsignedInteger;
    /// let decomposer = SignedDecomposer::<u32>::new(
    ///     DecompositionBaseLog(4),
    ///     DecompositionLevelCount(3)
    /// );
    /// let val = 1_340_987_234_u32;
    /// assert_eq!(decomposer.closest_representable(val), decomposer.decompose(val).recompose());
    /// ```
    pub fn recompose(self) -> T {
        self.fold(T::ZERO, |acc, term| acc.wrapping_add(term.to_summand()))
    }

    /// Returns the logarithm in base two of the base of this decomposition.
    ///
    /// If the decomposition uses a base $B=2^b$, this returns $b$.
    pub fn base_log(&self) -> DecompositionBaseLog {
        DecompositionBaseLog(self.base_log)
    }

    /// Returns the number of levels of this decomposition.
    ///
    /// If the decomposition uses $l$ levels, this returns $l$.
    pub fn level_count(&self) -> DecompositionLevelCount {
        DecompositionLevelCount(self.level_count)
    }
}

impl<T> Iterator for SignedDecompositionIter<T>
where
    T: UnsignedInteger,
{
    type Item = DecompositionTerm<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // We check if the decomposition is over
        if self.current_level == 0 {
            return None;
        }
        // We perform the division of the input by q/B^i
        let res = self.input >> (T::BITS - self.base_log * self.current_level);
        // We reduce the result modulo B
        let res = res & self.mod_b_mask;
        // The result may already be greater or equal to B/2.
        let carry = res & self.carry_mask;
        // We propagate the carry from the previous level
        let res = res.wrapping_add(self.previous_carry);
        // The previous carry may have made the result larger or equal to B/2.
        let carry = carry | (res & self.carry_mask);
        // If the result is greater or equal to B/2, we subtract B from the result (viewed as a
        // signed integer)
        let res = (res.into_signed() - (carry << 1).into_signed()).into_unsigned();
        // We prepare the output
        let output =
            DecompositionTerm::new(DecompositionLevel(self.current_level), self.base_log(), res);
        // We update the state of the iterator
        self.previous_carry = carry >> (self.base_log - 1);
        self.current_level -= 1;
        // We return the output for this level
        Some(output)
    }
}

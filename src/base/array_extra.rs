use ndarray::prelude::*;
use ndarray::{Data, RemoveAxis, Zip};

use rawpointer::PointerExt;

use std::cmp::Ordering;
use std::ptr::copy_nonoverlapping;

#[derive(Clone, Debug)]
pub struct Permutation {
    pub indices: Vec<usize>,
}

impl Permutation {
    pub fn from_indices(v: Vec<usize>) -> Result<Self, ()> {
        let perm = Permutation { indices: v };
        if perm.correct() {
            Ok(perm)
        } else {
            Err(())
        }
    }

    fn correct(&self) -> bool {
        let axis_len = self.indices.len();
        let mut seen = vec![false; axis_len];
        for &i in &self.indices {
            match seen.get_mut(i) {
                None => return false,
                Some(s) => {
                    if *s {
                        return false;
                    } else {
                        *s = true;
                    }
                }
            }
        }
        true
    }
}

pub trait SortArray {
    fn identity(&self, axis: Axis) -> Permutation;
    fn sort_axis_by<F>(&self, axis: Axis, less_than: F) -> Permutation
    where
        F: FnMut(usize, usize) -> bool;
}

pub trait PermuteArray {
    type Elem;
    type Dim;
    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<Self::Elem, Self::Dim>
    where
        Self::Elem: Clone,
        Self::Dim: RemoveAxis;
}

impl<A, S, D> SortArray for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn identity(&self, axis: Axis) -> Permutation {
        Permutation {
            indices: (0..self.len_of(axis)).collect(),
        }
    }

    fn sort_axis_by<F>(&self, axis: Axis, mut less_than: F) -> Permutation
    where
        F: FnMut(usize, usize) -> bool,
    {
        let mut perm = self.identity(axis);
        perm.indices.sort_by(move |&a, &b| {
            if less_than(a, b) {
                Ordering::Less
            } else if less_than(b, a) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        perm
    }
}

impl<A, D> PermuteArray for Array<A, D>
where
    D: Dimension,
{
    type Elem = A;
    type Dim = D;

    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<A, D>
    where
        D: RemoveAxis,
    {
        let axis_len = self.len_of(axis);
        let axis_stride = self.stride_of(axis);
        assert_eq!(axis_len, perm.indices.len());
        debug_assert!(perm.correct());

        if self.is_empty() {
            return self;
        }

        let mut result = Array::uninit(self.dim());

        unsafe {

            let mut moved_elements = 0;


            let source_0 = self.raw_view().index_axis_move(axis, 0);

            Zip::from(&perm.indices)
                .and(result.axis_iter_mut(axis))
                .for_each(|&perm_i, result_pane| {
                    Zip::from(result_pane)
                        .and(source_0.clone())
                        .for_each(|to, from_0| {
                            let from = from_0.stride_offset(axis_stride, perm_i);
                            copy_nonoverlapping(from, to.as_mut_ptr(), 1);
                            moved_elements += 1;
                        });
                });
            debug_assert_eq!(result.len(), moved_elements);
            let mut old_storage = self.into_raw_vec();
            old_storage.set_len(0);
            result.assume_init()
        }
    }
}


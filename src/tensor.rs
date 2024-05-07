use cudarc::driver::{
    CudaDevice, CudaSlice, CudaView, DeviceRepr, DriverError, LaunchAsync, LaunchConfig,
};

use std::sync::Arc;

//very simplified. Should rather abstract shape to more generic - both for static and dynamic.
pub struct Tensor<T> {
    pub(crate) dev: Arc<CudaDevice>,
    pub(crate) data: CudaSlice<T>,

    pub(crate) shape: Shape,
}

impl<T> Tensor<T>
where
    T: DeviceRepr,
{
    pub fn new(dev: Arc<CudaDevice>, data: CudaSlice<T>, shape: Shape) -> Self {
        Self { dev, data, shape }
    }

    pub fn as_vec(&self) -> Result<Vec<T>, DriverError> {
        self.dev.dtoh_sync_copy(&self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub(crate) shape: Vec<usize>,

    pub(crate) permutation: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

pub fn compute_continuous_strides(shape: &[usize]) -> Vec<usize> {
    let mut res = vec![1; shape.len()];
    for i in 1..shape.len() {
        let idx = shape.len() - i - 1;

        res[idx] = res[idx + 1] * shape[idx + 1];
    }

    res
}

impl Shape {
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    pub fn as_slice(&self) -> &[usize] {
        self.shape.as_slice()
    }

    pub fn elements_count(&self) -> u32 {
        self.shape.iter().product::<usize>() as u32
    }

    pub fn new(shape: Vec<usize>) -> Self {
        let strides = compute_continuous_strides(&shape);
        Self {
            permutation: (0..shape.len()).into_iter().collect(),
            shape,
            strides,
        }
    }

    pub fn revert_permutations(&mut self) {
        unsafe {
            //TODO, test it actually

            permute_unchecked(&mut self.shape, &self.permutation);
            permute_unchecked(&mut self.strides, &self.permutation);
            self.permutation = (0..self.shape.len()).into_iter().collect();
        }
    }

    pub fn permute(&mut self, permutation: impl Into<Vec<usize>>) {
        let n = self.shape.len();

        let permutation: Vec<_> = permutation.into();

        assert!(check_is_permutation(permutation.clone()));

        unsafe {
            permute_unchecked(&mut self.shape, &permutation);
            permute_unchecked(&mut self.strides, &permutation);
        }

        self.permutation = permutation;
    }

    pub fn get_offset(&self, idx: usize) -> Option<usize> {
        //optimize
        if idx as u32 >= self.elements_count() {
            return None;
        }

        //wrong. Double permutation should yield the identity mapping??
        let mut index = compute_shape_index(idx, &self.shape);
        dbg!(&index);
        let offset = compute_strided_offset(&index, &self.strides);

        Some(offset)
    }
}

fn check_is_permutation(permutation: impl Into<Vec<usize>>) -> bool {
    let mut check = permutation.into();
    check.sort();
    check.dedup();
    return check.len() == check.len();
}

unsafe fn permute_unchecked(input: &mut [usize], permutation: &[usize]) {
    //TODO: traverse without copying.

    let input_buf = input.to_vec();
    for i in 0..input.len() {
        input[i] = input_buf[permutation[i]];
    }
}

pub(crate) fn compute_strided_offset(index: &[usize], strides: &[usize]) -> usize {
    assert_eq!(index.len(), strides.len());

    index.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum()
}

pub(crate) fn compute_shape_index(idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut res = vec![0; shape.len()];
    let mut index = idx;
    for i in 0..shape.len() {
        let idx = shape.len() - i - 1;
        res[idx] = index % shape[idx];
        index /= shape[idx];
    }
    res
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(shape: [usize; N]) -> Self {
        Shape::new(shape.into_iter().map(|x| x as usize).collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_double_permute_identity() {
        const REF: [usize; 5] = [0, 1, 2, 3, 4];

        let mut input = REF.clone();
        let mut permutation = [1, 0, 3, 2, 4];

        //"good" engineer puts tests to ensure correctness of theorems lol
        unsafe {
            //double perm should be identity
            permute_unchecked(&mut input, &permutation);
            permute_unchecked(&mut input, &permutation);
        }

        assert_eq!(input, REF);
    }

    #[test]
    fn test_compute_shape_index01() {
        let shape = vec![4, 4, 1];

        assert_eq!(compute_shape_index(0, &shape), vec![0, 0, 0]);
        assert_eq!(compute_shape_index(1, &shape), vec![0, 1, 0]);
        assert_eq!(compute_shape_index(2, &shape), vec![0, 2, 0]);
        assert_eq!(compute_shape_index(3, &shape), vec![0, 3, 0]);

        assert_eq!(compute_shape_index(4, &shape), vec![1, 0, 0]);
        assert_eq!(compute_shape_index(5, &shape), vec![1, 1, 0]);
        assert_eq!(compute_shape_index(6, &shape), vec![1, 2, 0]);
        assert_eq!(compute_shape_index(7, &shape), vec![1, 3, 0]);

        assert_eq!(compute_shape_index(8, &shape), vec![2, 0, 0]);
        assert_eq!(compute_shape_index(9, &shape), vec![2, 1, 0]);
        assert_eq!(compute_shape_index(10, &shape), vec![2, 2, 0]);
        assert_eq!(compute_shape_index(11, &shape), vec![2, 3, 0]);

        assert_eq!(compute_shape_index(12, &shape), vec![3, 0, 0]);
        assert_eq!(compute_shape_index(13, &shape), vec![3, 1, 0]);
        assert_eq!(compute_shape_index(14, &shape), vec![3, 2, 0]);
        assert_eq!(compute_shape_index(15, &shape), vec![3, 3, 0]);
    }

    #[test]
    fn test_compute_shape_index02() {
        let shape = vec![1, 2, 4];

        assert_eq!(compute_shape_index(0, &shape), vec![0, 0, 0]);
        assert_eq!(compute_shape_index(1, &shape), vec![0, 0, 1]);
        assert_eq!(compute_shape_index(2, &shape), vec![0, 0, 2]);
        assert_eq!(compute_shape_index(3, &shape), vec![0, 0, 3]);

        assert_eq!(compute_shape_index(4, &shape), vec![0, 1, 0]);
        assert_eq!(compute_shape_index(5, &shape), vec![0, 1, 1]);
        assert_eq!(compute_shape_index(6, &shape), vec![0, 1, 2]);
        assert_eq!(compute_shape_index(7, &shape), vec![0, 1, 3]);
    }
}

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
    pub(crate) strides_permuted: Vec<usize>,
    pub(crate) strides_init: Vec<usize>,
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
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
        let strides = compute_strides(&shape);
        Self {
            permutation: (0..shape.len()).into_iter().collect(),
            shape,
            strides_init: strides.clone(),
            strides_permuted: strides,
        }
    }

    pub fn permute(&mut self, permutation: &[usize]) {
        let n = self.shape.len();

        //check if valid permutation
        let mut check = permutation.to_vec();
        check.sort();
        check.dedup();
        assert_eq!(check.len(), n);

        unsafe {
            self.permute_unchecked(permutation);
            permute_unchecked(&mut self.permutation, &permutation);
        }
    }

    unsafe fn permute_unchecked(&mut self, permutation: &[usize]) {
        let n = self.shape.len();

        permute_unchecked(&mut self.shape, permutation);
        permute_unchecked(&mut self.strides_permuted, permutation);

        dbg!(&self.shape);
        dbg!(&self.strides_permuted);
    }

    pub fn get_offset(&self, idx: usize) -> Option<usize> {
        //optimize
        if idx as u32 >= self.elements_count() {
            return None;
        }

        //wrong. Double permutation should yield the identity mapping??
        let mut index = compute_strided_index(idx, &self.strides_permuted);
        dbg!(&index);
        unsafe {
            dbg!(&self.permutation);
            permute_unchecked(&mut index, &self.permutation);
        }
        dbg!(&index);

        let old_idx = dot(&index, &self.strides_init);
        dbg!(old_idx);

        Some(old_idx)
    }
}

unsafe fn permute_unchecked(input: &mut [usize], permutation: &[usize]) {
    let input_buf = input.to_vec();

    //can be optimized somehow..
    for i in 0..input.len() {
        input[i] = input_buf[permutation[i]];
    }
}

fn dot(lhs: &Vec<usize>, rhs: &Vec<usize>) -> usize {
    lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l * r).sum()
}

pub(crate) fn compute_strided_index(idx: usize, strides: &[usize]) -> Vec<usize> {
    let mut res = vec![0; strides.len()];
    let mut index = idx;
    for i in 0..strides.len() {
        let idx = strides.len() - i - 1;
        res[idx] = index % strides[idx];
        index /= strides[idx];
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
    fn test_compute_strided_index01() {
        let strides = vec![8, 4, 1];

        assert_eq!(compute_strided_index(0, &strides), vec![0, 0, 0]);
        assert_eq!(compute_strided_index(1, &strides), vec![0, 1, 0]);
        assert_eq!(compute_strided_index(2, &strides), vec![0, 2, 0]);
        assert_eq!(compute_strided_index(3, &strides), vec![0, 3, 0]);

        assert_eq!(compute_strided_index(4, &strides), vec![1, 0, 0]);
        assert_eq!(compute_strided_index(5, &strides), vec![1, 1, 0]);
        assert_eq!(compute_strided_index(6, &strides), vec![1, 2, 0]);
        assert_eq!(compute_strided_index(7, &strides), vec![1, 3, 0]);

        assert_eq!(compute_strided_index(8, &strides), vec![2, 0, 0]);
        //and so on..
    }

    #[test]
    fn test_compute_strided_index02() {
        let strides = vec![1, 2, 4];

        assert_eq!(compute_strided_index(0, &strides), vec![0, 0, 0]);
        assert_eq!(compute_strided_index(1, &strides), vec![0, 0, 1]);
        assert_eq!(compute_strided_index(2, &strides), vec![0, 0, 2]);
        assert_eq!(compute_strided_index(3, &strides), vec![0, 0, 3]);

        assert_eq!(compute_strided_index(4, &strides), vec![0, 1, 0]);
        assert_eq!(compute_strided_index(5, &strides), vec![0, 1, 1]);
        assert_eq!(compute_strided_index(6, &strides), vec![0, 1, 2]);
        assert_eq!(compute_strided_index(7, &strides), vec![0, 1, 3]);
    }
}

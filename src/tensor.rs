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

pub struct Shape {
    pub(crate) shape: Vec<usize>,
}

impl Shape {
    pub fn compute_strides(&self) -> Vec<usize> {
        let mut res = vec![1; self.shape.len()];
        for i in 1..self.shape.len() {
            let idx = self.shape.len() - i - 1;

            res[idx] = res[idx + 1] * self.shape[idx + 1];
        }

        res
    }

    pub fn elements_count(&self) -> u32 {
        self.shape.iter().product::<usize>() as u32
    }

    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

pub(crate) fn compute_strided_index(index: usize, strides: &Vec<usize>) -> Vec<usize> {
    let mut res = vec![0; strides.len()];
    let mut index = index;
    for i in 0..strides.len() {
        res[i] = index / strides[i];
        index %= strides[i];
    }
    res
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(shape: [usize; N]) -> Self {
        Self {
            shape: shape.into_iter().map(|x| x as usize).collect::<Vec<_>>(),
        }
    }
}

use cudarc::driver::{
    CudaDevice, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtrMut, DeviceRepr,
    DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use super::kernel::{load_and_get_kernel, Kernel};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/map_kernel.ptx"));

#[derive(Debug, Copy, Clone)]
pub(crate) struct MapOp;

unsafe impl Kernel<f32> for MapOp {
    const MODULE_NAME: &'static str = "kernel_ops";
    const FN_NAME: &'static str = "map_offsets_f32";

    fn ptx(&self) -> Ptx {
        Ptx::from_src(PTX_SRC)
    }
}

//f32 for now only.
pub(crate) unsafe fn map_offsets_in_place(
    mut input: &mut CudaSlice<f32>,
    input_len: u32,
    dev: Arc<CudaDevice>,
    idx_to_offsets: &CudaSlice<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let function = load_and_get_kernel(&dev, MapOp)?;

    const MAX_BLOCK_LEN: u32 = 1024;

    let block_len: u32 = MAX_BLOCK_LEN.min(input_len);

    let grid_len = (input_len.div_ceil(block_len)) as u32; //should be no remained as

    let type_len = std::mem::size_of::<f32>() as u32;

    let cfg = LaunchConfig {
        block_dim: (block_len, 1, 1),
        grid_dim: (grid_len, 1, 1),
        shared_mem_bytes: type_len * block_len,
    };

    let input: &CudaSlice<_> = input;

    let params = (input, idx_to_offsets, input_len);

    function.launch(cfg, params)?;

    Ok(())
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::tensor::{Shape, Tensor};

    #[test]
    pub fn test_map_offsets_in_place() {
        let cuda_dev = CudaDevice::new(0).expect("could not create cuda device");

        let mut data = (0..10).map(|x| x as f32).collect::<Vec<f32>>();

        let offsets: [usize; 10] = [1, 3, 5, 7, 9, 0, 2, 4, 6, 8];
        let reverse_offsets: [usize; 10] = [5, 0, 6, 1, 7, 2, 8, 3, 9, 4];

        let mut input = cuda_dev.htod_sync_copy(&data).unwrap();
        let offsets = cuda_dev.htod_sync_copy(&offsets).unwrap();

        //CLONE ALLOCATES!
        unsafe {
            map_offsets_in_place(&mut input, 10, cuda_dev.clone(), &offsets).unwrap();
        }

        let output = cuda_dev.dtoh_sync_copy(&input).unwrap();
        let offsets = cuda_dev.dtoh_sync_copy(&offsets).unwrap();

        println!("output: {:?}", output);
        println!("offsets: {:?}", offsets);

        let expected = [1f32, 3f32, 5f32, 7f32, 9f32, 0f32, 2f32, 4f32, 6f32, 8f32];

        assert_eq!(output, expected);

        let offsets = cuda_dev.htod_sync_copy(&reverse_offsets).unwrap();

        unsafe {
            map_offsets_in_place(&mut input, 10, cuda_dev.clone(), &offsets).unwrap();
        }

        let output = cuda_dev.dtoh_sync_copy(&input).unwrap();

        println!("output: {:?}", output);

        //identity operation
        assert_eq!(output, data);
    }
}

use cudarc::driver::{CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

mod exp;
mod map_kernel;
mod reduce;
mod softmax;
mod tensor;
mod kernel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_dev = CudaDevice::new(0)?;

    //    softmax::softmax(cuda_dev)?;

    Ok(())
}

use cudarc::driver::{CudaDevice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

mod softmax;
mod tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_dev = CudaDevice::new(0)?;

//    softmax::softmax(cuda_dev)?;

    Ok(())
}

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::tensor::{Shape, Tensor};

//const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/max.ptx"));

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct MaxOp;

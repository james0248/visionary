export function mul(shape) {
  return shape.reduce((total, value) => total * value, 1);
}

export function tensorByteLength(dtype, shape) {
  const bytesPerElement =
    dtype === 'float32' || dtype === 'int32' || dtype === 'uint32'
      ? 4
      : dtype === 'float16'
        ? 2
        : dtype === 'uint8'
          ? 1
          : 8;
  return mul(shape) * bytesPerElement;
}

export function tensorDataBytes(tensor) {
  const data = tensor.data;
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

export function writeCpuTensorToGpu(device, gpuTensor, cpuTensor) {
  device.queue.writeBuffer(gpuTensor.gpuBuffer, 0, tensorDataBytes(cpuTensor));
}

export function copyGpuTensor(device, source, target, label = 'visionary-gpu-copy') {
  if (source === target) return;
  const byteLength = tensorByteLength(source.type, source.dims);
  const encoder = device.createCommandEncoder({ label });
  encoder.copyBufferToBuffer(source.gpuBuffer, 0, target.gpuBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);
}

export function copyTensorToGpu(device, source, target) {
  if (source?.location === 'gpu-buffer') {
    copyGpuTensor(device, source, target);
  } else {
    writeCpuTensorToGpu(device, target, source);
  }
}

export function createGpuTensorFromCpu(ort, device, tensor) {
  const byteLength = Math.max(16, tensorByteLength(tensor.type, tensor.dims));
  const buffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(tensorDataBytes(tensor));
  buffer.unmap();
  return ort.Tensor.fromGpuBuffer(buffer, {
    dataType: tensor.type,
    dims: tensor.dims,
    dispose: () => buffer.destroy(),
  });
}

export function createEmptyGpuTensor(ort, device, spec) {
  const byteLength = Math.max(16, tensorByteLength(spec.dtype, spec.shape));
  const buffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  return ort.Tensor.fromGpuBuffer(buffer, {
    dataType: spec.dtype,
    dims: spec.shape,
    dispose: () => buffer.destroy(),
  });
}

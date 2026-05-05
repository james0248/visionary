// Deterministic standard-normal noise for browser rollout.
// This matches JAX's normal distribution contract, not JAX's bit-exact Threefry stream.
export class NormalNoiseGenerator {
  state: number;
  spare: number | null;

  constructor(seed = 0) {
    this.state = seed >>> 0;
    this.spare = null;
  }

  uniform() {
    let value = (this.state += 0x6d2b79f5);
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }

  normal() {
    if (this.spare !== null) {
      const value = this.spare;
      this.spare = null;
      return value;
    }

    const u1 = Math.max(this.uniform(), 1e-7);
    const u2 = this.uniform();
    const radius = Math.sqrt(-2.0 * Math.log(u1));
    const angle = 2.0 * Math.PI * u2;
    this.spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  }

  tensorData(size) {
    const values = new Float32Array(size);
    for (let index = 0; index < size; index += 1) {
      values[index] = this.normal();
    }
    return values;
  }
}

import * as bin from '@isopodlabs/binary';
import * as grids from './gguf_grids';

const GGUF_DEFAULT_ALIGNMENT = 32; // defined in ggml.h
const QK_K = 256;	// weights in a K-quant super-block

type QuantReader = (file: bin._stream, offset: number, blocks: number) => Promise<Float32Array>;

interface QuantType {
	name:	string;
	desc:	string;
	block:	number;		// elements per block
	size:	number;		// bytes per block
	read?:	QuantReader;
	glsl?:	string;
}

// A quantised block is described by its layout (ggml's block structs; see gguf-py's quants.py) and decoded from a view of it.
// Each array field access makes a view, so decoders read every field once per block.
function Block<const T extends bin.bitfields.Descriptor>(name: string, desc: string, block: number, layout: T, decode: (x: bin.bitfields.BitOutput<T>, out: Float32Array, o: number) => void, glsl?: string): QuantType {
	const array = bin.typedArray.BitFields(layout);
	return {name, desc, block, size: bin.bitfields.calcBits(layout) / 8,
		async read(file, offset, blocks) {
			const views	= await file.view_at(array, offset, blocks);
			const out	= new Float32Array(blocks * block);
			for (let b = 0; b < blocks; b++)
				decode(views[b], out, b * block);
			return out;
		},
		glsl
	};
}

// unquantised types are one element per block
function Plain(name: string, desc: string, size: number, get: (dv: DataView, offset: number) => number, glsl?: string): QuantType {
	return {name, desc, block: 1, size,
		async read(file, offset, blocks) {
			const src	= await file.view_at(Uint8Array, offset, blocks * size);
			const dv	= new DataView(src.buffer, src.byteOffset, src.byteLength);
			const out	= new Float32Array(blocks);
			for (let i = 0; i < blocks; i++)
				out[i] = get(dv, i * size);
			return out;
		},
		glsl
	};
}

const Unsupported = (name: string, desc: string, block: number, size: number): QuantType => ({name, desc, block, size});

const bytes = <C extends number>(count: C) => bin.bitfields.Array(count, 8);

const half = (h: number) => {
	const e = (h >> 10) & 0x1f, m = h & 0x3ff, sign = h & 0x8000 ? -1 : 1;
	return e === 0 ? sign * m * 2 ** -24 : e === 31 ? (m ? NaN : sign * Infinity) : sign * (1 + m / 1024) * 2 ** (e - 15);
};
const i8 = (b: number) => (b << 24) >> 24;

const bits32	= new Uint32Array(1);
const float32	= new Float32Array(bits32.buffer);
const fromBits	= (bits: number) => (bits32[0] = bits, float32[0]);

// Q3_K stores 16 six-bit scales in 12 bytes: the low 4 bits in the first 8, the high 2 bits in the last 4
const q3kScale = (s: ArrayLike<number>, i: number) => ((i < 8 ? s[i] & 15 : s[i - 8] >> 4) | (((s[8 + (i & 3)] >> (2 * (i >> 2))) & 3) << 4)) - 32;

// Q4_K and Q5_K store 8 six-bit scales and 8 six-bit mins in 12 bytes
const scaleMin = (s: ArrayLike<number>, j: number) => j < 4
	? [s[j] & 63, s[j + 4] & 63]
	: [(s[j + 4] & 15) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)];

export const iq4nl		= [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
export const mxfp4		= [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];	// e2m1, doubled
const pow3		= [1, 3, 9, 27, 81];

// one value per grid element, from the packed hex of gguf_grids.ts
function Grid({entries, width, map, hex}: {entries: number, width: number, map: number[], hex: string}) {
	const bits = Math.ceil(Math.log2(map.length)), per = Math.floor(8 / bits), step = Math.floor(8 / per);
	return Int8Array.from({length: entries * width}, (_, i) => map[(parseInt(hex.substr(Math.floor(i / per) * 2, 2), 16) >> (i % per * step)) & ((1 << bits) - 1)]);
}

const iq2xxs	= Grid(grids.iq2xxs), iq2xs = Grid(grids.iq2xs), iq2s = Grid(grids.iq2s), iq3xxs = Grid(grids.iq3xxs), iq3s = Grid(grids.iq3s), iq1s = Grid(grids.iq1s);

// the sign bits of 8 weights are stored as a 7-bit index, whose parity gives the eighth
const ksigns	= Uint8Array.from({length: 128}, (_, i) => i | ((i.toString(2).match(/1/g)?.length ?? 0) & 1) << 7);

const u16 = (a: ArrayLike<number>, i: number) => a[i] | a[i + 1] << 8;
const u32 = (a: ArrayLike<number>, i: number) => (a[i] | a[i + 1] << 8 | a[i + 2] << 16 | a[i + 3] << 24) >>> 0;

export const QuantizationTypes: Record<number, QuantType> = {
	0: Plain('F32', "32-bit standard IEEE 754 single-precision floating-point number.", 4, (dv, s) => dv.getFloat32(s, true),
		`w = uintBitsToFloat(u32(base + off * 4));`
	),
	1: Plain('F16', '16-bit standard IEEE 754 half-precision floating-point number.', 2, (dv, s) => half(dv.getUint16(s, true)),
		`w = f16At(base + off * 2);`
	),
	2: Block('Q4_0', '4-bit round-to-nearest quantization (q). Each block has 32 weights. Weight formula: w = (q - 8) * block_scale. Legacy quantization method (not used widely as of today).',
		32, {d: bin.float16, qs: bytes(16)}, (x, out, o) => {
			const d = +x.d, qs = x.qs;
			for (let j = 0; j < 16; j++) {
				out[o + j]		= ((qs[j] & 15) - 8) * d;
				out[o + j + 16]	= ((qs[j] >> 4) - 8) * d;
			}
		},
		`w = (float(nibble(base, 2, off)) - 8.0) * f16At(base);`
	),
	3: Block('Q4_1', '4-bit round-to-nearest quantization (q). Each block has 32 weights. Weight formula: w = q * block_scale + block_minimum. Legacy quantization method (not used widely as of today).',
		32, {d: bin.float16, m: bin.float16, qs: bytes(16)}, (x, out, o) => {
			const d = +x.d, m = +x.m, qs = x.qs;
			for (let j = 0; j < 16; j++) {
				out[o + j]		= (qs[j] & 15) * d + m;
				out[o + j + 16]	= (qs[j] >> 4) * d + m;
			}
		},
		`w = float(nibble(base, 4, off)) * f16At(base) + f16At(base + 2);`
	),
	6: Block('Q5_0', '5-bit round-to-nearest quantization (q). Each block has 32 weights. Weight formula: w = (q - 16) * block_scale. Legacy quantization method (not used widely as of today).',
		32, {d: bin.float16, qh: 32, qs: bytes(16)}, (x, out, o) => {
			const d = +x.d, qh = x.qh, qs = x.qs;
			for (let j = 0; j < 16; j++) {
				out[o + j]		= (((qs[j] & 15) | (((qh >>> j) & 1) << 4)) - 16) * d;
				out[o + j + 16]	= (((qs[j] >> 4) | (((qh >>> (j + 16)) & 1) << 4)) - 16) * d;
			}
		},
		`uint q = nibble(base, 6, off) | (((u32(base + 2) >> uint(off)) & 1u) << 4); w = (float(q) - 16.0) * f16At(base);`
	),
	7: Block('Q5_1', '5-bit round-to-nearest quantization (q). Each block has 32 weights. Weight formula: w = q * block_scale + block_minimum. Legacy quantization method (not used widely as of today).',
		32, {d: bin.float16, m: bin.float16, qh: 32, qs: bytes(16)}, (x, out, o) => {
			const d = +x.d, m = +x.m, qh = x.qh, qs = x.qs;
			for (let j = 0; j < 16; j++) {
				out[o + j]		= ((qs[j] & 15) | (((qh >>> j) & 1) << 4)) * d + m;
				out[o + j + 16]	= ((qs[j] >> 4) | (((qh >>> (j + 16)) & 1) << 4)) * d + m;
			}
		},
		`uint q = nibble(base, 8, off) | (((u32(base + 4) >> uint(off)) & 1u) << 4);
		w = float(q) * f16At(base) + f16At(base + 2);`
	),
	8: Block('Q8_0', '8-bit round-to-nearest quantization (q). Each block has 32 weights. Weight formula: w = q * block_scale. Legacy quantization method (not used widely as of today).',
		32, {d: bin.float16, qs: bytes(32)}, (x, out, o) => {
			const d = +x.d, qs = x.qs;
			for (let j = 0; j < 32; j++)
				out[o + j] = i8(qs[j]) * d;
		},
		`w = i8At(base + 2 + off) * f16At(base);`
	),
	9: Unsupported('Q8_1', '8-bit round-to-nearest quantization (q). Each block has 32 weights. Only used for quantizing intermediate results, so not stored in model files.', 32, 40),
	10: Block('Q2_K', '2-bit quantization (q). Super-blocks with 16 blocks, each block has 16 weights. Weight formula: w = q * block_scale(4-bit) - block_min(4-bit), resulting in 2.625 bits-per-weight.',
		QK_K, {scales: bytes(16), qs: bytes(64), d: bin.float16, dmin: bin.float16}, (x, out, o) => {
			const d = +x.d, dmin = +x.dmin, scales = x.scales, qs = x.qs;
			for (let e = 0; e < QK_K; e++) {
				const sc	= scales[e >> 4];
				const j		= e >> 5;
				out[o + e]	= d * (sc & 15) * ((qs[((j >> 2) << 5) + (e & 31)] >> (2 * (j & 3))) & 3) - dmin * (sc >> 4);
			}
		},
		`int sc = int(u8(base + (off >> 4)));
		int j = off >> 5;
		uint q = (u8(base + 16 + ((j >> 2) << 5) + (off & 31)) >> uint(2 * (j & 3))) & 3u;
		w = f16At(base + 80) * float(sc & 15) * float(q) - f16At(base + 82) * float(sc >> 4);`
	),
	11: Block('Q3_K', '3-bit quantization (q). Super-blocks with 16 blocks, each block has 16 weights. Weight formula: w = q * block_scale(6-bit), resulting in 3.4375 bits-per-weight.',
		QK_K, {hmask: bytes(32), qs: bytes(64), scales: bytes(12), d: bin.float16}, (x, out, o) => {
			const d = +x.d, hmask = x.hmask, scales = x.scales, qs = x.qs;
			for (let e = 0; e < QK_K; e++) {
				const j		= e >> 5, l = e & 31;
				out[o + e]	= d * q3kScale(scales, e >> 4) * (((qs[((j >> 2) << 5) + l] >> (2 * (j & 3))) & 3) - (((hmask[l] >> j) & 1) ? 0 : 4));
			}
		},
		`int j = off >> 5, l = off & 31, i = off >> 4;
		int sc = (i < 8 ? int(u8(base + 96 + i) & 15u) : int(u8(base + 88 + i) >> 4))
			| (((int(u8(base + 104 + (i & 3))) >> (2 * (i >> 2))) & 3) << 4);
		uint q = (u8(base + 32 + ((j >> 2) << 5) + l) >> uint(2 * (j & 3))) & 3u;
		w = f16At(base + 108) * float(sc - 32) * (float(q) - (((u8(base + l) >> uint(j)) & 1u) != 0u ? 0.0 : 4.0));`
	),
	12: Block('Q4_K', '4-bit quantization (q). Super-blocks with 8 blocks, each block has 32 weights. Weight formula: w = q * block_scale(6-bit) - block_min(6-bit), resulting in 4.5 bits-per-weight.',
		QK_K, {d: bin.float16, dmin: bin.float16, scales: bytes(12), qs: bytes(128)}, (x, out, o) => {
			const d = +x.d, dmin = +x.dmin, scales = x.scales, qs = x.qs;
			for (let j = 0; j < 8; j++) {
				const [sc, m]	= scaleMin(scales, j);
				const dl = d * sc, ml = dmin * m, shift = 4 * (j & 1), base = (j >> 1) << 5;
				for (let l = 0; l < 32; l++)
					out[o + j * 32 + l] = dl * ((qs[base + l] >> shift) & 15) - ml;
			}
		},
		`int j = off >> 5, l = off & 31, sc, mn;
		if (j < 4) {
			sc = int(u8(base + 4 + j) & 63u);
			mn = int(u8(base + 8 + j) & 63u);
		} else {
			sc = int(u8(base + 8 + j) & 15u) | ((int(u8(base + j)) >> 6) << 4);
			mn = int(u8(base + 8 + j) >> 4) | ((int(u8(base + 4 + j)) >> 6) << 4);
		}
		uint q = (u8(base + 16 + ((j >> 1) << 5) + l) >> uint(4 * (j & 1))) & 15u;
		w = f16At(base) * float(sc) * float(q) - f16At(base + 2) * float(mn);`
	),
	13: Block('Q5_K', '5-bit quantization (q). Super-blocks with 8 blocks, each block has 32 weights. Weight formula: w = q * block_scale(6-bit) - block_min(6-bit), resulting in 5.5 bits-per-weight.',
		QK_K, {d: bin.float16, dmin: bin.float16, scales: bytes(12), qh: bytes(32), qs: bytes(128)}, (x, out, o) => {
			const d = +x.d, dmin = +x.dmin, scales = x.scales, qh = x.qh, qs = x.qs;
			for (let j = 0; j < 8; j++) {
				const [sc, m]	= scaleMin(scales, j);
				const dl = d * sc, ml = dmin * m, shift = 4 * (j & 1), base = (j >> 1) << 5;
				for (let l = 0; l < 32; l++)
					out[o + j * 32 + l] = dl * (((qs[base + l] >> shift) & 15) | (((qh[l] >> j) & 1) << 4)) - ml;
			}
		},
		`int j = off >> 5, l = off & 31, sc, mn;
		if (j < 4) {
			sc = int(u8(base + 4 + j) & 63u);
			mn = int(u8(base + 8 + j) & 63u);
		} else {
			sc = int(u8(base + 8 + j) & 15u) | ((int(u8(base + j)) >> 6) << 4);
			mn = int(u8(base + 8 + j) >> 4) | ((int(u8(base + 4 + j)) >> 6) << 4);
		}
		uint q = (u8(base + 48 + ((j >> 1) << 5) + l) >> uint(4 * (j & 1))) & 15u;
		q |= ((u8(base + 16 + l) >> uint(j)) & 1u) << 4;
		w = f16At(base) * float(sc) * float(q) - f16At(base + 2) * float(mn);`
	),
	14: Block('Q6_K', '6-bit quantization (q). Super-blocks with 16 blocks, each block has 16 weights. Weight formula: w = (q - 32) * block_scale(8-bit), resulting in 6.5625 bits-per-weight.',
		QK_K, {ql: bytes(128), qh: bytes(64), scales: bytes(16), d: bin.float16}, (x, out, o) => {
			const d = +x.d, ql = x.ql, qh = x.qh, scales = x.scales;
			for (let e = 0; e < QK_K; e++) {
				const r		= e >> 5, l = e & 31, h = r >> 2, t = r & 3;
				out[o + e]	= d * i8(scales[e >> 4]) * ((((ql[h * 64 + (t & 1) * 32 + l] >> (4 * (t >> 1))) & 15) | (((qh[h * 32 + l] >> (2 * t)) & 3) << 4)) - 32);
			}
		},
		`int r = off >> 5, l = off & 31, h = r >> 2, t = r & 3;
		uint q = ((u8(base + h * 64 + (t & 1) * 32 + l) >> uint(4 * (t >> 1))) & 15u) | (((u8(base + 128 + h * 32 + l) >> uint(2 * t)) & 3u) << 4);
		w = f16At(base + 208) * i8At(base + 192 + (off >> 4)) * (float(q) - 32.0);`
	),
	15: Block('Q8_K', '8-bit quantization (q). Each block has 256 weights. Only used for quantizing intermediate results. All 2-6 bit dot products are implemented for this quantization type. Weight formula: w = q * block_scale.',
		QK_K, {d: bin.float32, qs: bytes(256), bsums: bin.bitfields.Array(16, 16)}, (x, out, o) => {
			const d = +x.d, qs = x.qs;
			for (let e = 0; e < QK_K; e++)
				out[o + e] = d * i8(qs[e]);
		},
		`w = uintBitsToFloat(u32(base)) * i8At(base + 4 + off);`
	),
	16: Block('IQ2_XXS', '2-bit quantization (q). Super-blocks with 256 weights. Weight w is obtained using super_block_scale & importance matrix, resulting in 2.06 bits-per-weight.',
		QK_K, {d: bin.float16, qs: bytes(64)}, (x, out, o) => {
			const d = +x.d, qs = x.qs;
			for (let ib = 0; ib < 8; ib++) {
				const idx = u32(qs, ib * 8), aux = u32(qs, ib * 8 + 4), db = d * (0.5 + (aux >>> 28)) * 0.25;
				for (let l = 0; l < 4; l++) {
					const grid = ((idx >>> (8 * l)) & 255) * 8, signs = ksigns[(aux >>> (7 * l)) & 127];
					for (let k = 0; k < 8; k++)
						out[o + ib * 32 + l * 8 + k] = db * iq2xxs[grid + k] * ((signs >> k) & 1 ? -1 : 1);
				}
			}
		}
	),
	17: Block('IQ2_XS', '2-bit quantization (q). Super-blocks with 256 weights. Weight w is obtained using super_block_scale & importance matrix, resulting in 2.31 bits-per-weight.',
		QK_K, {d: bin.float16, qs: bytes(64), scales: bytes(8)}, (x, out, o) => {
			const d = +x.d, qs = x.qs, scales = x.scales;
			for (let g = 0; g < 32; g++) {
				const q = u16(qs, g * 2), db = d * (0.5 + ((scales[g >> 2] >> (4 * ((g >> 1) & 1))) & 15)) * 0.25, signs = ksigns[q >> 9];
				for (let k = 0; k < 8; k++)
					out[o + g * 8 + k] = db * iq2xs[(q & 511) * 8 + k] * ((signs >> k) & 1 ? -1 : 1);
			}
		}
	),
	18: Block('IQ3_XXS', '3-bit quantization (q). Super-blocks with 256 weights. Weight w is obtained using super_block_scale & importance matrix, resulting in 3.06 bits-per-weight.',
		QK_K, {d: bin.float16, qs: bytes(64), scales: bytes(32)}, (x, out, o) => {
			const d = +x.d, qs = x.qs;
			for (let ib = 0; ib < 8; ib++) {
				const aux = u32(x.scales, ib * 4), db = d * (0.5 + (aux >>> 28)) * 0.5;
				for (let l = 0; l < 4; l++) {
					const signs = ksigns[(aux >>> (7 * l)) & 127];
					for (let k = 0; k < 8; k++)
						out[o + ib * 32 + l * 8 + k] = db * iq3xxs[qs[ib * 8 + l * 2 + (k >> 2)] * 4 + (k & 3)] * ((signs >> k) & 1 ? -1 : 1);
				}
			}
		}
	),
	19: Block('IQ1_S', '1-bit quantization (q). Super-blocks with 256 weights. Weight w is obtained using super_block_scale & importance matrix, resulting in 1.56 bits-per-weight.',
		QK_K, {d: bin.float16, qs: bytes(32), qh: bytes(16)}, (x, out, o) => {
			const d = +x.d, qs = x.qs, qh = x.qh;
			for (let ib = 0; ib < 8; ib++) {
				const h = u16(qh, ib * 2), dl = d * (2 * ((h >> 12) & 7) + 1), delta = h & 0x8000 ? -0.125 : 0.125;
				for (let l = 0; l < 4; l++) {
					const grid = (qs[ib * 4 + l] | (((h >> (3 * l)) & 7) << 8)) * 8;
					for (let k = 0; k < 8; k++)
						out[o + ib * 32 + l * 8 + k] = dl * (iq1s[grid + k] + delta);
				}
			}
		}
	),
	20: Block('IQ4_NL', '4-bit non-linear quantization. Each block has 32 weights. Weight formula: w = block_scale * kvalues[q].',
		32, {d: bin.float16, qs: bytes(16)}, (x, out, o) => {
			const d = +x.d, qs = x.qs;
			for (let j = 0; j < 16; j++) {
				out[o + j]		= d * iq4nl[qs[j] & 15];
				out[o + j + 16]	= d * iq4nl[qs[j] >> 4];
			}
		},
		`w = f16At(base) * IQ4NL[int(nibble(base, 2, off))];`
	),
	21: Block('IQ3_S', '3-bit quantization (q). Super-blocks with 256 weights. Weight w is obtained using super_block_scale & importance matrix, resulting in 3.44 bits-per-weight.',
		QK_K, {d: bin.float16, qs: bytes(64), qh: bytes(8), signs: bytes(32), scales: bytes(4)}, (x, out, o) => {
			const d = +x.d, qs = x.qs, qh = x.qh, signs = x.signs, scales = x.scales;
			for (let g = 0; g < 32; g++) {
				const db = d * (1 + 2 * ((scales[g >> 3] >> (4 * ((g >> 2) & 1))) & 15));
				for (let k = 0; k < 8; k++) {
					const i = g * 2 + (k >> 2);
					out[o + g * 8 + k] = db * iq3s[(qs[i] | (((qh[i >> 3] >> (i & 7)) & 1) << 8)) * 4 + (k & 3)] * ((signs[g] >> k) & 1 ? -1 : 1);
				}
			}
		}
	),
	22: Block('IQ2_S', '2-bit quantization (q). Super-blocks with 256 weights. Weight w is obtained using super_block_scale & importance matrix, resulting in 2.5 bits-per-weight.',
		QK_K, {d: bin.float16, qs: bytes(32), signs: bytes(32), qh: bytes(8), scales: bytes(8)}, (x, out, o) => {
			const d = +x.d, qs = x.qs, signs = x.signs, qh = x.qh, scales = x.scales;
			for (let g = 0; g < 32; g++) {
				const db = d * (0.5 + ((scales[g >> 2] >> (4 * ((g >> 1) & 1))) & 15)) * 0.25, grid = (qs[g] | (((qh[g >> 2] >> (2 * (g & 3))) & 3) << 8)) * 8;
				for (let k = 0; k < 8; k++)
					out[o + g * 8 + k] = db * iq2s[grid + k] * ((signs[g] >> k) & 1 ? -1 : 1);
			}
		}
	),
	23: Block('IQ4_XS', '4-bit non-linear quantization. Super-blocks with 8 blocks, each block has 32 weights. Weight formula: w = block_scale(6-bit) * kvalues[q], resulting in 4.25 bits-per-weight.',
		QK_K, {d: bin.float16, scalesH: 16, scalesL: bytes(4), qs: bytes(128)}, (x, out, o) => {
			const d = +x.d, scalesH = x.scalesH, scalesL = x.scalesL, qs = x.qs;
			for (let ib = 0; ib < 8; ib++) {
				const dl = d * ((((scalesL[ib >> 1] >> (4 * (ib & 1))) & 15) | (((scalesH >> (2 * ib)) & 3) << 4)) - 32);
				for (let j = 0; j < 16; j++) {
					out[o + ib * 32 + j]		= dl * iq4nl[qs[ib * 16 + j] & 15];
					out[o + ib * 32 + j + 16]	= dl * iq4nl[qs[ib * 16 + j] >> 4];
				}
			}
		}
	),
	24: Plain('I8', '8-bit fixed-width integer number.', 1, (dv, s) => dv.getInt8(s)),
	25: Plain('I16', '16-bit fixed-width integer number.', 2, (dv, s) => dv.getInt16(s, true)),
	26: Plain('I32', '32-bit fixed-width integer number.', 4, (dv, s) => dv.getInt32(s, true)),
	27: Plain('I64', '64-bit fixed-width integer number.', 8, (dv, s) => Number(dv.getBigInt64(s, true))),
	28: Plain('F64', '64-bit standard IEEE 754 double-precision floating-point number.', 8, (dv, s) => dv.getFloat64(s, true)),
	29: Block('IQ1_M', '1-bit quantization (q). Super-blocks with 256 weights. Weight w is obtained using super_block_scale & importance matrix, resulting in 1.75 bits-per-weight.',
		QK_K, {qs: bytes(32), qh: bytes(16), scales: bytes(8)}, (x, out, o) => {
			const qs = x.qs, qh = x.qh, s = [0, 2, 4, 6].map(i => u16(x.scales, i));
			const d = half((s[0] >> 12) | ((s[1] >> 12) << 4) | ((s[2] >> 12) << 8) | ((s[3] >> 12) << 12));	// the fp16 scale is spread over the tops of the four scale words
			for (let g = 0; g < 32; g++) {
				const nibble = (qh[g >> 1] >> (4 * (g & 1))) & 15, j = g >> 1, dl = d * (2 * ((s[j >> 2] >> (3 * (j & 3))) & 7) + 1), delta = nibble & 8 ? -0.125 : 0.125;
				for (let k = 0; k < 8; k++)
					out[o + g * 8 + k] = dl * (iq1s[(qs[g] | ((nibble & 7) << 8)) * 8 + k] + delta);
			}
		}
	),
	30: Plain('BF16', '16-bit shortened version of the 32-bit IEEE 754 single-precision floating-point number.', 2, (dv, s) => fromBits(dv.getUint16(s, true) << 16),
		`w = uintBitsToFloat(u16(base + off * 2) << 16);`
	),
	34: Block('TQ1_0', 'Ternary quantization. Super-blocks with 256 weights, 5 trits per byte. Weight formula: w = (t - 1) * block_scale.',
		QK_K, {qs: bytes(48), qh: bytes(4), d: bin.float16}, (x, out, o) => {
			const d = +x.d, qs = x.qs, qh = x.qh;
			const trit = (byte: number, p: number) => d * (((((byte * pow3[p]) & 255) * 3) >> 8) - 1);
			for (let e = 0; e < 160; e++)
				out[o + e] = trit(qs[e & 31], e >> 5);
			for (let e = 0; e < 80; e++)
				out[o + 160 + e] = trit(qs[32 + (e & 15)], e >> 4);
			for (let e = 0; e < 16; e++)
				out[o + 240 + e] = trit(qh[e & 3], e >> 2);
		}
	),
	35: Block('TQ2_0', 'Ternary quantization. Super-blocks with 256 weights, 2 bits each. Weight formula: w = (q - 1) * block_scale.',
		QK_K, {qs: bytes(64), d: bin.float16}, (x, out, o) => {
			const d = +x.d, qs = x.qs;
			for (let e = 0; e < QK_K; e++)
				out[o + e] = d * (((qs[(e >> 7) * 32 + (e & 31)] >> (2 * ((e >> 5) & 3))) & 3) - 1);
		}
	),
	39: Block('MXFP4', '4-bit Microscaling Block Floating Point. Each block has 32 weights and a shared power-of-two scale.',
		32, {e: 8, qs: bytes(16)}, (x, out, o) => {
			const e = x.e, qs = x.qs;
			const d = fromBits(e < 2 ? 0x00200000 << e : (e - 1) << 23);	// E8M0 scale, halved to match the doubled table
			for (let j = 0; j < 16; j++) {
				out[o + j]		= mxfp4[qs[j] & 15] * d;
				out[o + j + 16]	= mxfp4[qs[j] >> 4] * d;
			}
		},
		`uint e = u8(base);
		float d = uintBitsToFloat(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
		w = d * MXFP4[int(nibble(base, 1, off))];`
	),
	40: Unsupported('NVFP4', '4-bit Microscaling Block Floating Point with global scale.', 64, 36),
	41: Unsupported('Q1_0', '1-bit quantization with fp16 block scale. Each block has 128 weights, where 0 represents -block_scale and 1 represents +block_scale.', 128, 18),
} as const;
type QuantizationTypes = keyof typeof QuantizationTypes

const ValueTypes = {
	UINT8:		0,
	INT8:		1,
	UINT16:		2,
	INT16:		3,
	UINT32:		4,
	INT32:		5,
	FLOAT32:	6,
	BOOL:		7,
	STRING:		8,
	ARRAY:		9,
	UINT64:		10,
	INT64:		11,
	FLOAT64:	12,
} as const;

const Size				= bin.as(bin.UINT64_LE, x => Number(x));	// 32 bits in v1
const QuantizationType	= bin.as(bin.UINT32_LE, x => x as QuantizationTypes);
const ValueType			= bin.as(bin.UINT32_LE, bin.EnumV(ValueTypes));
const String			= bin.String(Size);

export type Block<T> = T | Block<T>[];

async function readBlock<T>(read: (start: number, count: number) => Promise<T>, shape: number[], start: number[], count: number[]) {
	const inner = (dim: number, offset = 0): Promise<Block<T>> => {
		const s = start[dim], c = count[dim];
		if (dim === 0)
			return read(offset + s, c);

		const r = shape.slice(0, dim).reduce((acc: number, val: number) => acc * val, 1);
		return Promise.all(Array.from({length: c}, (_, i) => inner(dim - 1, offset + (s + i) * r)));
	};
	return inner(shape.length - 1);
}


export class Tensor extends bin.Class({
	name:		String,
	shape:		bin.Array(bin.UINT32_LE, bin.UINT64_LE),
	dtype:		QuantizationType,
	offset:		bin.UINT64_LE,
	file:		bin.Func(s => s),
}) {
	parameterCount() {
		return this.shape.reduce((acc: number, val: bigint) => acc * Number(val), 1);
	}

	typeName() {
		return QuantizationTypes[this.dtype]?.name ?? `type ${this.dtype}`;
	}

	private checkRead() {
		const q = QuantizationTypes[this.dtype];
		if (!q)
			throw new Error(`gguf: tensor ${this.name} has unknown type ${this.dtype}`);
		if (!q.read)
			throw new Error(`gguf: dequantising ${q.name} (tensor ${this.name}) is not implemented`);
		return q as QuantType & { read: QuantReader};
	}
	private checkBlock(start: number[], count: number[]) {
		const dims = this.shape.length;
		for (let i = 0; i < dims; i++) {
			const m = Number(this.shape[i]);
			const s = start[i] ?? 0, c = count[i] ?? m;
			if (s < 0 || c < 0 || s + c > m)
				throw new RangeError(`gguf: elements ${start}+${count} are outside tensor ${this.name} (${this.shape})`);
			start[i] = s;
			count[i] = c;
		}
	}

	private async safeReadRaw(q: QuantType, start: number, count: number) {
		const first = Math.floor(start / q.block);
		const last  = Math.ceil((start + count) / q.block);
		return {first, bytes: count === 0 ? new Uint8Array(0) : await this.file.view_at(Uint8Array, Number(this.offset) + first * q.size, (last - first) * q.size)};
	}
	private async safeRead(q: QuantType & { read: QuantReader}, start: number, count: number) {
		if (count === 0)
			return new Float32Array(0);

		const first	= Math.floor(start / q.block);
		const last	= Math.ceil((start + count) / q.block);
		const out	= await q.read(this.file, Number(this.offset) + first * q.size, last - first);
		const skip	= start - first * q.block;
		return skip === 0 && out.length === count ? out : out.slice(skip, skip + count);
	}
	// Decodes count elements starting at flat index start (ggml order: shape[0] varies fastest) to floats.
	// Only the blocks covering the range are read, so this is the way to look at part of a large tensor.
	async read(start = 0, count = this.parameterCount() - start) {
		const q = this.checkRead();
		if (start < 0 || count < 0 || start + count > this.parameterCount())
			throw new RangeError(`gguf: elements ${start}..${start + count} are outside tensor ${this.name} (${this.parameterCount()} elements)`);
		return this.safeRead(q, start, count);
	}
	// The raw bytes of the blocks covering count elements from flat index start, for a reader that expands them itself. The
	// bytes cover whole blocks, so they begin at element floor(start / block) * block: a caller slices to the exact range.
	async readRaw(start = 0, count = this.parameterCount() - start) {
		const q = this.checkRead();
		if (start < 0 || count < 0 || start + count > this.parameterCount())
			throw new RangeError(`gguf: elements ${start}..${start + count} are outside tensor ${this.name} (${this.parameterCount()} elements)`);
		return this.safeReadRaw(q, start, count);
	}

	async readBlock(start: number[], count: number[]) {
		const q = this.checkRead();
		this.checkBlock(start, count);
		return readBlock((s, c) => this.safeRead(q, s, c), this.shape.map(Number), start, count);
	}
	async readBlockRaw(start: number[], count: number[]) {
		const q = this.checkRead();
		this.checkBlock(start, count);
		return readBlock((s, c) => this.safeReadRaw(q, s, c), this.shape.map(Number), start, count);
	}

	async lookup(offset: number) {
		return (await this.read(offset, 1))[0];
	}

	
}
/*
const ValueArray = bin.Switch(ValueType, {
	[ValueTypes.UINT8]:		bin.Array(Size, bin.UINT8),
	[ValueTypes.INT8]:		bin.Array(Size, bin.INT8),
	[ValueTypes.UINT16]:	bin.Array(Size, bin.UINT16_LE),
	[ValueTypes.INT16]:		bin.Array(Size, bin.INT16_LE),
	[ValueTypes.UINT32]:	bin.Array(Size, bin.UINT32_LE),
	[ValueTypes.INT32]:		bin.Array(Size, bin.INT32_LE),
//	[ValueTypes.FLOAT32]:	bin.Array(Size, bin.Float32_LE),
	[ValueTypes.FLOAT32]:	bin.Buffer(Size, bin.typedArray.DataViewTypedArray('Float32', false)),
	[ValueTypes.BOOL]:		bin.Array(Size, bin.UINT8),
	[ValueTypes.STRING]:	bin.Array(Size, String),
	[ValueTypes.ARRAY]:		bin.Array(Size, bin.FuncType((): bin.interop.TypeT<any[]|bin.typedArray.TypedArray<number>> => ValueArray)),
	[ValueTypes.UINT64]:	bin.Array(Size, bin.UINT64_LE),
	[ValueTypes.INT64]:		bin.Array(Size, bin.INT64_LE),
	[ValueTypes.FLOAT64]:	bin.Array(Size, bin.Float64_LE),
});
*/
const ValueBuffer = bin.Switch(ValueType, {
	[ValueTypes.UINT8]:		bin.Buffer(Size, Uint8Array),
	[ValueTypes.INT8]:		bin.Buffer(Size, Int8Array),
	[ValueTypes.UINT16]:	bin.Buffer(Size, bin.typedArray.DataViewTypedArray('Uint16', false)),
	[ValueTypes.INT16]:		bin.Buffer(Size, bin.typedArray.DataViewTypedArray('Int16', false)),
	[ValueTypes.UINT32]:	bin.Buffer(Size, bin.typedArray.DataViewTypedArray('Uint32', false)),
	[ValueTypes.INT32]:		bin.Buffer(Size, bin.typedArray.DataViewTypedArray('Int32', false)),
	[ValueTypes.FLOAT32]:	bin.Buffer(Size, bin.typedArray.DataViewTypedArray('Float32', false)),
	[ValueTypes.BOOL]:		bin.Buffer(Size, Uint8Array),
	[ValueTypes.STRING]:	bin.Array(Size, String),
	[ValueTypes.ARRAY]:		bin.Array(Size, bin.FuncType((): bin.interop.TypeT<any[]|bin.typedArray.TypedArray<any>> => ValueBuffer)),
	[ValueTypes.UINT64]:	bin.Buffer(Size, bin.typedArray.DataViewTypedArray('BigUint64', false)),
	[ValueTypes.INT64]:		bin.Buffer(Size, bin.typedArray.DataViewTypedArray('BigInt64', false)),
	[ValueTypes.FLOAT64]:	bin.Buffer(Size, bin.typedArray.DataViewTypedArray('Float64', false)),
});

const KvPairSpec = {
	key:	String,
	value:	bin.Switch(ValueType, {
		[ValueTypes.UINT8]:		bin.UINT8,
		[ValueTypes.INT8]:		bin.INT8,
		[ValueTypes.UINT16]:	bin.UINT16_LE,
		[ValueTypes.INT16]:		bin.INT16_LE,
		[ValueTypes.UINT32]:	bin.UINT32_LE,
		[ValueTypes.INT32]:		bin.INT32_LE,
		[ValueTypes.FLOAT32]:	bin.Float32_LE,
		[ValueTypes.BOOL]:		bin.UINT8,
		[ValueTypes.STRING]:	String,
		[ValueTypes.ARRAY]:		ValueBuffer,//Array,
		[ValueTypes.UINT64]:	bin.UINT64_LE,
		[ValueTypes.INT64]:		bin.INT64_LE,
		[ValueTypes.FLOAT64]:	bin.Float64_LE,
	})
};

function addValue(obj: any, key: string, value: any) {
	const dot = key.indexOf('.');
	if (dot !== -1)
		addValue(obj[key.substring(0, dot)] ??= {}, key.substring(dot + 1), value);
	else
		obj[key] = value;
	return obj;
}

const GgufSpec = {
	magic:			bin.Expect(bin.String(4), 'GGUF'),
	version:		bin.UINT32_LE,
	tensor_count:	Size,

	metadata:		bin.as(bin.Array(Size, KvPairSpec), array => array.reduce((acc, i) => addValue(acc, i.key, i.value), {}) as any),
	tensors:		bin.Array('tensor_count', Tensor),
};


export async function readGguf(stream: bin.interop._stream) {
	const r = await bin.interop.stream(stream).read(GgufSpec);

	const alignment = r.metadata.general.alignment ?? GGUF_DEFAULT_ALIGNMENT;
	const start 	= BigInt((stream.tell() + alignment - 1) & -alignment);

	for (const t of r.tensors)
		t.offset += start;
	return r;
}

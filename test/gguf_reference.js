"use strict";
// Hand-written quantisation decoders: an independent reference for the descriptive ones in src/gguf.ts, and a drop-in fallback
// if their speed ever matters (bench_gguf_quants.ts measures the difference). Layouts follow ggml, as in gguf-py's quants.py.
Object.defineProperty(exports, "__esModule", { value: true });
exports.reference = void 0;
const QK_K = 256;
const half = (h) => {
    const e = (h >> 10) & 0x1f, m = h & 0x3ff, sign = h & 0x8000 ? -1 : 1;
    return e === 0 ? sign * m * 2 ** -24 : e === 31 ? (m ? NaN : sign * Infinity) : sign * (1 + m / 1024) * 2 ** (e - 15);
};
const i8 = (b) => (b << 24) >> 24;
const bits32 = new Uint32Array(1);
const float32 = new Float32Array(bits32.buffer);
const fromBits = (bits) => (bits32[0] = bits, float32[0]);
const q3kScale = (src, s, i) => ((i < 8 ? src[s + i] & 15 : src[s + i - 8] >> 4) | (((src[s + 8 + (i & 3)] >> (2 * (i >> 2))) & 3) << 4)) - 32;
const scaleMin = (src, s, j) => j < 4
    ? [src[s + j] & 63, src[s + j + 4] & 63]
    : [(src[s + j + 4] & 15) | ((src[s + j - 4] >> 6) << 4), (src[s + j + 4] >> 4) | ((src[s + j] >> 6) << 4)];
const iq4nl = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const mxfp4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
const pow3 = [1, 3, 9, 27, 81];
// by ggml type id: elements per block, bytes per block, decoder
exports.reference = {
    2: { block: 32, size: 18, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s, true));
            for (let j = 0; j < 16; j++) {
                const b = src[s + 2 + j];
                out[o + j] = ((b & 15) - 8) * d;
                out[o + j + 16] = ((b >> 4) - 8) * d;
            }
        } },
    3: { block: 32, size: 20, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s, true)), m = half(dv.getUint16(s + 2, true));
            for (let j = 0; j < 16; j++) {
                const b = src[s + 4 + j];
                out[o + j] = (b & 15) * d + m;
                out[o + j + 16] = (b >> 4) * d + m;
            }
        } },
    6: { block: 32, size: 22, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s, true)), qh = dv.getUint32(s + 2, true);
            for (let j = 0; j < 16; j++) {
                const b = src[s + 6 + j];
                out[o + j] = (((b & 15) | (((qh >>> j) & 1) << 4)) - 16) * d;
                out[o + j + 16] = (((b >> 4) | (((qh >>> (j + 16)) & 1) << 4)) - 16) * d;
            }
        } },
    7: { block: 32, size: 24, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s, true)), m = half(dv.getUint16(s + 2, true)), qh = dv.getUint32(s + 4, true);
            for (let j = 0; j < 16; j++) {
                const b = src[s + 8 + j];
                out[o + j] = ((b & 15) | (((qh >>> j) & 1) << 4)) * d + m;
                out[o + j + 16] = ((b >> 4) | (((qh >>> (j + 16)) & 1) << 4)) * d + m;
            }
        } },
    8: { block: 32, size: 34, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s, true));
            for (let j = 0; j < 32; j++)
                out[o + j] = i8(src[s + 2 + j]) * d;
        } },
    10: { block: QK_K, size: 84, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s + 80, true)), dmin = half(dv.getUint16(s + 82, true));
            for (let e = 0; e < QK_K; e++) {
                const sc = src[s + (e >> 4)];
                const j = e >> 5;
                out[o + e] = d * (sc & 15) * ((src[s + 16 + ((j >> 2) << 5) + (e & 31)] >> (2 * (j & 3))) & 3) - dmin * (sc >> 4);
            }
        } },
    11: { block: QK_K, size: 110, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s + 108, true));
            for (let e = 0; e < QK_K; e++) {
                const j = e >> 5, l = e & 31;
                out[o + e] = d * q3kScale(src, s + 96, e >> 4) * (((src[s + 32 + ((j >> 2) << 5) + l] >> (2 * (j & 3))) & 3) - (((src[s + l] >> j) & 1) ? 0 : 4));
            }
        } },
    12: { block: QK_K, size: 144, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s, true)), dmin = half(dv.getUint16(s + 2, true));
            for (let j = 0; j < 8; j++) {
                const [sc, m] = scaleMin(src, s + 4, j);
                const dl = d * sc, ml = dmin * m, shift = 4 * (j & 1), qs = s + 16 + ((j >> 1) << 5);
                for (let l = 0; l < 32; l++)
                    out[o + j * 32 + l] = dl * ((src[qs + l] >> shift) & 15) - ml;
            }
        } },
    13: { block: QK_K, size: 176, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s, true)), dmin = half(dv.getUint16(s + 2, true));
            for (let j = 0; j < 8; j++) {
                const [sc, m] = scaleMin(src, s + 4, j);
                const dl = d * sc, ml = dmin * m, shift = 4 * (j & 1), qs = s + 48 + ((j >> 1) << 5);
                for (let l = 0; l < 32; l++)
                    out[o + j * 32 + l] = dl * (((src[qs + l] >> shift) & 15) | (((src[s + 16 + l] >> j) & 1) << 4)) - ml;
            }
        } },
    14: { block: QK_K, size: 210, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s + 208, true));
            for (let e = 0; e < QK_K; e++) {
                const r = e >> 5, l = e & 31, h = r >> 2, t = r & 3;
                out[o + e] = d * i8(src[s + 192 + (e >> 4)]) * ((((src[s + h * 64 + (t & 1) * 32 + l] >> (4 * (t >> 1))) & 15) | (((src[s + 128 + h * 32 + l] >> (2 * t)) & 3) << 4)) - 32);
            }
        } },
    15: { block: QK_K, size: 292, decode: (src, dv, s, out, o) => {
            const d = dv.getFloat32(s, true);
            for (let e = 0; e < QK_K; e++)
                out[o + e] = d * i8(src[s + 4 + e]);
        } },
    20: { block: 32, size: 18, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s, true));
            for (let j = 0; j < 16; j++) {
                const b = src[s + 2 + j];
                out[o + j] = d * iq4nl[b & 15];
                out[o + j + 16] = d * iq4nl[b >> 4];
            }
        } },
    23: { block: QK_K, size: 136, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s, true)), scalesH = dv.getUint16(s + 2, true);
            for (let ib = 0; ib < 8; ib++) {
                const dl = d * ((((src[s + 4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((scalesH >> (2 * ib)) & 3) << 4)) - 32);
                for (let j = 0; j < 16; j++) {
                    const b = src[s + 8 + ib * 16 + j];
                    out[o + ib * 32 + j] = dl * iq4nl[b & 15];
                    out[o + ib * 32 + j + 16] = dl * iq4nl[b >> 4];
                }
            }
        } },
    34: { block: QK_K, size: 54, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s + 52, true));
            const trit = (byte, p) => d * (((((byte * pow3[p]) & 255) * 3) >> 8) - 1);
            for (let e = 0; e < 160; e++)
                out[o + e] = trit(src[s + (e & 31)], e >> 5);
            for (let e = 0; e < 80; e++)
                out[o + 160 + e] = trit(src[s + 32 + (e & 15)], e >> 4);
            for (let e = 0; e < 16; e++)
                out[o + 240 + e] = trit(src[s + 48 + (e & 3)], e >> 2);
        } },
    35: { block: QK_K, size: 66, decode: (src, dv, s, out, o) => {
            const d = half(dv.getUint16(s + 64, true));
            for (let e = 0; e < QK_K; e++)
                out[o + e] = d * (((src[s + (e >> 7) * 32 + (e & 31)] >> (2 * ((e >> 5) & 3))) & 3) - 1);
        } },
    39: { block: 32, size: 17, decode: (src, dv, s, out, o) => {
            const x = src[s];
            const d = fromBits(x < 2 ? 0x00200000 << x : (x - 1) << 23);
            for (let j = 0; j < 16; j++) {
                const b = src[s + 1 + j];
                out[o + j] = mxfp4[b & 15] * d;
                out[o + j + 16] = mxfp4[b >> 4] * d;
            }
        } },
};
//# sourceMappingURL=gguf_reference.js.map
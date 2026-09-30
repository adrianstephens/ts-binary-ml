import * as bin from '@isopodlabs/binary';
import {gguf} from '../dist/index';
import { reference } from './gguf_reference';

// Times the descriptive decoders in src/gguf.ts against the hand-written ones in gguf_reference.ts on ~1M random elements each.
// If the ratio ever matters, gguf_reference.ts can replace the decoders (test_gguf_quants.ts checks both against gguf-py).

const u32 = (n: number) => Buffer.from(new Uint32Array([n]).buffer);
const u64 = (n: number) => Buffer.from(new BigUint64Array([BigInt(n)]).buffer);
const str = (s: string) => Buffer.concat([u64(s.length), Buffer.from(s)]);

function makeGguf(type: number, elements: number, bytes: number) {
	const header = Buffer.concat([
		Buffer.from('GGUF'), u32(3), u64(1), u64(1),
		str('general.architecture'), u32(8), str('bench'),
		str('t'), u32(1), u64(elements), u32(type), u64(0),
	]);
	const data = Buffer.alloc(bytes);
	for (let i = 0; i < bytes; i++)
		data[i] = Math.random() * 256;
	return new Uint8Array(Buffer.concat([header, Buffer.alloc(-header.length & 31), data]));
}

const best = async (f: () => unknown) => {
	let ms = Infinity;
	for (let i = 0; i < 7; i++) {
		const t = performance.now();
		await f();
		ms = Math.min(ms, performance.now() - t);
	}
	return ms;
};

(async () => {
	console.log('type      descriptive   hand-written   ratio');
	for (const [type, name] of [[2, 'Q4_0'], [8, 'Q8_0'], [12, 'Q4_K'], [14, 'Q6_K']] as const) {
		const {block, size, decode} = reference[type];
		const blocks	= (1 << 20) / block;
		const data		= makeGguf(type, blocks * block, blocks * size);
		const t			= (await gguf.readGguf(new bin.stream(data))).tensors[0];
		const raw		= data.subarray(data.length - blocks * size);
		const dv		= new DataView(raw.buffer, raw.byteOffset, raw.byteLength);

		const descriptive	= await best(() => t.read());
		const handWritten	= await best(() => {
			const out = new Float32Array(blocks * block);
			for (let b = 0; b < blocks; b++)
				decode(raw, dv, b * size, out, b * block);
		});
		console.log(`${name.padEnd(9)} ${descriptive.toFixed(2).padStart(8)} ms   ${handWritten.toFixed(2).padStart(9)} ms   ${(descriptive / handWritten).toFixed(1).padStart(5)}x`);
	}
})();

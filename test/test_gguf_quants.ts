import * as bin from '@isopodlabs/binary';
import {gguf} from '../dist/index';
import { readFileSync } from 'fs';
import { reference } from './gguf_reference';

// Checks each dequantiser, and the hand-written reference decoders, against gguf-py: quants.gguf holds, for every type,
// a tensor of random blocks and 'ref.<name>' with the float32 values gguf-py decodes them to (regenerate with make_quants_fixture.py)

(async () => {
	const data		= new Uint8Array(readFileSync(__dirname + '/quants.gguf'));
	const gg		= await gguf.readGguf(new bin.stream(data));
	const tensors	= new Map(gg.tensors.map(t => [t.name, t]));
	let failed		= 0;

	for (const t of gg.tensors.filter(t => !t.name.startsWith('ref.'))) {
		const ref	= await tensors.get('ref.' + t.name)!.read();
		const tol	= 2e-6 * Math.max(1e-30, ...ref.map(Math.abs));
		const problems: string[] = [];

		const check = (got: ArrayLike<number>, first: number, what: string) => {
			for (let i = 0; i < got.length; i++) {
				if (!(Math.abs(got[i] - ref[first + i]) <= tol)) {
					problems.push(`${what}: [${first + i}] got ${got[i]}, want ${ref[first + i]}`);
					return;
				}
			}
		};

		try {
			const all = await t.read();
			if (all.length === ref.length)
				check(all, 0, 'read()');
			else
				problems.push(`read() gave ${all.length} elements, want ${ref.length}`);

			const r = reference[t.dtype];
			if (r) {
				const raw	= data.subarray(Number(t.offset), Number(t.offset) + ref.length / r.block * r.size);
				const dv	= new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
				const out	= new Float32Array(ref.length);
				for (let b = 0; b < ref.length / r.block; b++)
					r.decode(raw, dv, b * r.size, out, b * r.block);
				check(out, 0, 'hand-written reference');
			}

			// ranges which start and end inside blocks
			for (const [start, count] of [[1, 5], [31, 3], [30, 40], [ref.length - 7, 7], [ref.length >> 1, 33]]) {
				if (start + count <= ref.length)
					check(await t.read(start, count), start, `read(${start}, ${count})`);
			}
			for (const i of [0, 1, 17, 31, 32, 33, 100, ref.length - 1]) {
				if (i < ref.length)
					check([await t.lookup(i)], i, `lookup(${i})`);
			}
		} catch (e) {
			problems.push('threw: ' + (e as Error).message);
		}

		if (problems.length)
			++failed;
		console.log(`${problems.length ? 'FAIL' : 'ok  '} ${t.typeName().padEnd(7)} ${problems[0] ?? ref.length + ' elements'}`);
	}

	console.log(failed ? `${failed} type(s) differ from gguf-py` : 'all types match gguf-py');
	process.exitCode = failed ? 1 : 0;
})();

# Binary ML
[![npm version](https://img.shields.io/npm/v/@isopodlabs/binary_ml.svg)](https://www.npmjs.com/package/@isopodlabs/binary_ml)
[![GitHub stars](https://img.shields.io/github/stars/adrianstephens/binary_libs.svg?style=social)](https://github.com/adrianstephens/binary_ml)
[![License](https://img.shields.io/npm/l/@isopodlabs/binary_libs.svg)](LICENSE.txt)

This package provides readers for various ML formats, using the @isopodlabs/binary binary file loading library

## ☕ Support My Work  
If you use this package, consider [buying me a cup of tea](https://coff.ee/adrianstephens) to support future updates!  

## Supported File Types

### gguf
GGUF, as used by llama.cpp and ollama. Reading is lazy: `readGguf` parses the header, metadata and tensor descriptions, and tensor data is only read when asked for.
```typescript
import { gguf } from '@isopodlabs/binary_ml';

const {metadata, tensors} = await gguf.readGguf(stream);	// stream: any @isopodlabs/binary stream, e.g. a file read on demand

const t = tensors[0];
t.name;					// 'blk.0.attn_q.weight'
t.shape;				// [4096n, 4096n] - shape[0] varies fastest
t.typeName();			// 'Q4_K'
t.parameterCount();		// 16777216

const rows = await t.read(0, 4096);	// Float32Array of elements 0..4095, dequantised
const one  = await t.lookup(1234);	// a single element (reads its whole block; use read() for more than a few)
```
`read(start, count)` reads and decodes only the blocks covering that range, so it is cheap for any part of a tensor however large.

Dequantisation follows ggml (and gguf-py, which the tests compare against) for F32, F16, BF16, F64, I8, I16, I32, I64,
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0 and MXFP4.
Other types (NVFP4, Q1_0, Q8_1) can be listed but `read()` throws for them.

### onnx
ONNX
```typescript
class ONNX {
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
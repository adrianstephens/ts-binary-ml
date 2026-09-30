# Generates quants.gguf, the fixture for test_gguf_quants.ts.
# For each quantisation type it holds a tensor of random blocks and, as 'ref.<name>', the float32 values gguf-py decodes them to.
# Requires: pip install gguf numpy
import struct
import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as T
from gguf import quants as Q

BLOCKS	= 4		# blocks per tensor
rng		= np.random.default_rng(11)

def string(x):
	b = x.encode()
	return struct.pack('<Q', len(b)) + b

tensors = []	# (name, ggml type, raw bytes, float32 reference values)
def add(name, t, raw, ref):
	tensors.append((name, int(t), raw, np.asarray(ref, dtype=np.float32).reshape(-1)))

for t in [T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0, T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K, T.IQ4_NL, T.IQ4_XS, T.TQ1_0, T.TQ2_0, T.MXFP4,
		T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS, T.IQ3_S, T.IQ1_S, T.IQ1_M]:
	_, size	= GGML_QUANT_SIZES[t]
	cls		= Q._type_traits[t]
	cls.init_grid()
	good	= []
	while len(good) < BLOCKS:	# random scale bytes can decode to NaN/inf; keep only finite blocks
		block	= rng.integers(0, 256, size=(1, size), dtype=np.uint8)
		ref		= cls.dequantize_blocks(block)
		if np.all(np.isfinite(ref)) and np.abs(ref).max() < 1e6:
			good.append((block, ref))
	add(t.name.lower(), t, b''.join(g[0].tobytes() for g in good), np.concatenate([g[1].reshape(-1) for g in good]))

# gguf-py has no Q8_K decoder: float32 d, int8 qs[256], int16 bsums[16]
d	= np.float32(0.0123)
qs	= rng.integers(-128, 128, size=256, dtype=np.int8)
add('q8_k', T.Q8_K, struct.pack('<f', d) + qs.tobytes() + b'\0' * 32, d * qs.astype(np.float32))

f32	= rng.standard_normal(70).astype(np.float32);	add('f32', T.F32, f32.tobytes(), f32)
f16	= rng.standard_normal(70).astype(np.float16);	add('f16', T.F16, f16.tobytes(), f16.astype(np.float32))
bf	= (rng.standard_normal(70).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
add('bf16', T.BF16, bf.tobytes(), (bf.astype(np.uint32) << 16).view(np.float32))
f64	= rng.standard_normal(70);						add('f64', T.F64, f64.tobytes(), f64)
for name, t, dtype in [('i8', T.I8, np.int8), ('i16', T.I16, np.int16), ('i32', T.I32, np.int32), ('i64', T.I64, np.int64)]:
	v = rng.integers(-1000, 1000, size=70).astype(dtype);	add(name, t, v.tobytes(), v)

# every tensor is followed by its reference, stored as F32
entries = []
for name, t, raw, ref in tensors:
	block, size = GGML_QUANT_SIZES[T(t)]
	entries.append((name, t, len(raw) // size * block, raw))
	entries.append(('ref.' + name, int(T.F32), len(ref), ref.tobytes()))

infos, data, offset = b'', b'', 0
for name, t, count, raw in entries:
	pad		= (-offset) % 32
	data	+= b'\0' * pad + raw
	infos	+= string(name) + struct.pack('<I', 1) + struct.pack('<Q', count) + struct.pack('<I', t) + struct.pack('<Q', offset + pad)
	offset	+= pad + len(raw)

header = b'GGUF' + struct.pack('<I', 3) + struct.pack('<Q', len(entries)) + struct.pack('<Q', 2)
header += string('general.architecture') + struct.pack('<I', 8) + string('test')
header += string('general.name') + struct.pack('<I', 8) + string('quantisation test fixture, see make_quants_fixture.py')
header += infos
header += b'\0' * ((-len(header)) % 32)

with open('quants.gguf', 'wb') as f:
	f.write(header + data)
print('wrote quants.gguf:', len(entries) // 2, 'types,', len(header) + len(data), 'bytes')

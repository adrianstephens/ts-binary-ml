import * as bin from '@isopodlabs/binary';
import * as pb from './protobuff';

function makeNameLookup<T extends {name?: string}>(array: T[]) {
	return Object.fromEntries(array.filter(v => v.name).map(v => [v.name!, v] as const));
}

// onnx schemas
const Version = {
	_START_VERSION:			0,
	IR_VERSION_2017_10_10:	0x0000000000000001,
	IR_VERSION_2017_10_30:	0x0000000000000002,
	IR_VERSION_2017_11_3:	0x0000000000000003,
	IR_VERSION_2019_1_22:	0x0000000000000004,
	IR_VERSION_2019_3_18:	0x0000000000000005,
	IR_VERSION_2019_9_19:	0x0000000000000006,
	IR_VERSION_2020_5_8:	0x0000000000000007,
	IR_VERSION_2021_7_30:	0x0000000000000008,
	IR_VERSION_2023_5_5:	0x0000000000000009,
	IR_VERSION_2024_3_25:	0x000000000000000A,
	IR_VERSION_2025_05_12:	0x000000000000000B,
	IR_VERSION_2025_08_26:	0x000000000000000C,
	IR_VERSION:				0x000000000000000D,
};

const TensorDataType = {
	UNDEFINED:		0,
	FLOAT:			1,
	UINT8:			2,
	INT8:			3,
	UINT16:			4,
	INT16:			5,
	INT32:			6,
	INT64:			7,
	STRING:			8,
	BOOL:			9,
	FLOAT16:		10,
	DOUBLE:			11,
	UINT32:			12,
	UINT64:			13,
	COMPLEX64:		14,
	COMPLEX128:		15,
	BFLOAT16:		16,
	FLOAT8E4M3FN:	17,
	FLOAT8E4M3FNUZ:	18,
	FLOAT8E5M2:		19,
	FLOAT8E5M2FNUZ:	20,
	UINT4:			21,
	INT4:			22,
	FLOAT4E2M1:		23,
	FLOAT8E8M0:		24,
	UINT2:			25,
	INT2:			26,
} as const;

const TensorDataTypeBuffer = {
	[TensorDataType.UNDEFINED]:			Uint8Array,
	[TensorDataType.FLOAT]:				Float32Array,
	[TensorDataType.UINT8]:				Uint8Array,
	[TensorDataType.INT8]:				Int8Array,
	[TensorDataType.UINT16]:			Uint16Array,
	[TensorDataType.INT16]:				Int16Array,
	[TensorDataType.INT32]:				Int32Array,
	[TensorDataType.INT64]:				BigInt64Array,
	[TensorDataType.STRING]:			Uint8Array,
	[TensorDataType.BOOL]:				Uint8Array,
	[TensorDataType.FLOAT16]:			bin.typedArray.BitFields(bin.float16),
	[TensorDataType.DOUBLE]:			Float64Array,
	[TensorDataType.UINT32]:			Uint32Array,
	[TensorDataType.UINT64]:			BigUint64Array,
	[TensorDataType.COMPLEX64]:			bin.typedArray.BitFields({r: bin.float32, i: bin.float32}),
	[TensorDataType.COMPLEX128]:		bin.typedArray.BitFields({r: bin.float64, i: bin.float64}),
	[TensorDataType.BFLOAT16]:			bin.typedArray.BitFields(bin.Bfloat16),
	[TensorDataType.FLOAT8E4M3FN]:		bin.typedArray.BitFields(bin.float(3, 4, {noInf: true})),
	[TensorDataType.FLOAT8E4M3FNUZ]:	bin.typedArray.BitFields(bin.float(3, 4, {ebias: 8, noInf: true, noNeg0: true})),
	[TensorDataType.FLOAT8E5M2]:		bin.typedArray.BitFields(bin.float(2, 5, {noInf: true})),
	[TensorDataType.FLOAT8E5M2FNUZ]:	bin.typedArray.BitFields(bin.float(2, 5, {ebias: 16, noInf: true, noNeg0: true})),
	[TensorDataType.UINT4]:				bin.typedArray.Uint(4),
	[TensorDataType.INT4]:				bin.typedArray.Int(4),
	[TensorDataType.FLOAT4E2M1]:		bin.typedArray.BitFields(bin.float4),
	[TensorDataType.FLOAT8E8M0]:		bin.typedArray.BitFields(bin.float(0, 8, {sbit: false})),
	[TensorDataType.UINT2]:				bin.typedArray.Uint(2),
	[TensorDataType.INT2]:				bin.typedArray.Int(2),
} as const;

export class Tensor extends bin.Class(pb.Message({
	1:  pb.Repeat('dims',				pb.int32),
	2:  pb.Optional('data_type',		pb.Enum(TensorDataType)),
	3:  pb.Optional('segment',			pb.Message({
		1: pb.Optional('begin',			pb.int64),
		2: pb.Optional('end',			pb.int64),
	})),
	4:  pb.Repeat('float_data',			pb.float),
	5:  pb.Repeat('int32_data',			pb.int32),
	6:  pb.Repeat('string_data',		pb.bytes),
	7:  pb.Repeat('int64_data',			pb.int64),
	8:  pb.Optional('name',				pb.string),
	9:  pb.Optional('raw_data',			pb.bytes),
	10: pb.Repeat('double_data',		pb.double),
	11: pb.Repeat('uint64_data',		pb.int64),
	12: pb.Optional('doc_string',		pb.string),
	13: pb.Map('external_data',			pb.string, pb.string),
	14: pb.Optional('data_location',	pb.Enum({DEFAULT: 0, EXTERNAL: 1})),
	16: pb.Map('metadata_props',		pb.string, pb.string),
})) {
	get data() {
		let type = this.data_type;
		if (!type)
			type =	this.float_data		? TensorDataType.FLOAT
				:	this.int32_data		? TensorDataType.INT32
				:	this.string_data	? TensorDataType.STRING
				:	this.int64_data		? TensorDataType.INT64
				:	this.double_data	? TensorDataType.DOUBLE
				:	this.uint64_data	? TensorDataType.UINT64
				:	0;

		let raw: bin.typedArray.TypedArray = this.raw_data!;
		if (!raw) {
			switch (type) {
				case TensorDataType.STRING:			return this.string_data;
				case TensorDataType.FLOAT:			return Float32Array.from(this.float_data);
				case TensorDataType.DOUBLE:			return Float64Array.from(this.double_data);
				case TensorDataType.INT64:			return BigInt64Array.from(this.int64_data);
				case TensorDataType.UINT64:			return BigUint64Array.from(this.uint64_data);
				case TensorDataType.COMPLEX64:		raw = Float32Array.from(this.float_data); break;
				case TensorDataType.COMPLEX128:		raw = Float64Array.from(this.double_data); break;

				case TensorDataType.INT32:			return Int32Array.from(this.int32_data);
				case TensorDataType.UINT32:			return Uint32Array.from(this.int32_data);

				case TensorDataType.INT16:			return Int16Array.from(this.int32_data);
				case TensorDataType.UINT16:			return Uint16Array.from(this.int32_data);
				case TensorDataType.FLOAT16:
				case TensorDataType.BFLOAT16:		raw = Uint16Array.from(this.int32_data); break;

				case TensorDataType.INT8:			return Int8Array.from(this.int32_data);
				case TensorDataType.UINT8:
				case TensorDataType.BOOL:			return Uint8Array.from(this.int32_data);
				case TensorDataType.FLOAT8E4M3FN:
				case TensorDataType.FLOAT8E4M3FNUZ:
				case TensorDataType.FLOAT8E5M2:
				case TensorDataType.FLOAT8E5M2FNUZ:
				case TensorDataType.FLOAT8E8M0:		raw = Uint8Array.from(this.int32_data); break;

				case TensorDataType.INT4:			return bin.typedArray.Int(4).from(this.int32_data);
				case TensorDataType.UINT4:			return bin.typedArray.Uint(4).from(this.int32_data);
				case TensorDataType.INT2:			return bin.typedArray.Int(2).from(this.int32_data);
				case TensorDataType.UINT2:			return bin.typedArray.Uint(2).from(this.int32_data);
				case TensorDataType.FLOAT4E2M1:		raw = bin.typedArray.Uint(4).from(this.int32_data); break;
			}
		}
		return new TensorDataTypeBuffer[type](raw.buffer as ArrayBuffer, raw.byteOffset, raw.byteLength);
	}
};

const TensorShape = pb.Message({
	1: pb.Repeat('dim',			pb.Message({
		_: pb.OneOf({
			1: pb.Field('dim_value',	pb.int64),
			2: pb.Field('dim_param',	pb.string),
		}),
		3: pb.Optional('denotation',	pb.string),
	})),
});

const Type = pb.Message({
	_: pb.OneOf({
		1: pb.Field('tensor_type',		pb.Message({
			1: pb.Optional('elem_type',		pb.Enum(TensorDataType)),
			2: pb.Optional('shape',			TensorShape),
		})),
		4: pb.Field('sequence_type',	pb.Message({
			1: pb.Optional('elem_type',		pb.forwardref(():any=>Type)),
		})),
		5: pb.Field('map_type',			pb.Message({
			1: pb.Optional('key_type',		pb.int32),
			2: pb.Optional('value_type',	pb.forwardref(():any=>Type)),
		})),
		9: pb.Field('optional_type',	pb.Message({
			1: pb.Optional('elem_type',		pb.forwardref(():any=>Type)),
		})),
		8: pb.Field('sparse_tensor_type', pb.Message({
			1: pb.Optional('elem_type',		pb.Enum(TensorDataType)),
			2: pb.Optional('shape',			TensorShape),
		})),
		7: pb.Field('opaque_type',		pb.Message({
			1: pb.Optional('domain',		pb.string),
			2: pb.Optional('name',			pb.string),
		})),
	}),
	6: pb.Optional('denotation',		pb.string),
});

const ElemType = {
	UNDEFINED:		0,
	TENSOR:			1,
	SPARSE_TENSOR:	2,
	SEQUENCE:		3,
	MAP:			4,
	OPTIONAL:		5,
} as const;

const Sequence = pb.Message({
	1: pb.Optional('name',				pb.string),
	2: pb.Optional('elem_type',			pb.Enum(ElemType)),
	3: pb.Repeat('tensor_values',		Tensor),
	4: pb.Repeat('sparse_tensor_values',pb.forwardref(():any=>SparseTensor)),
	5: pb.Repeat('sequence_values',		pb.forwardref(():any=>Sequence)),
	6: pb.Repeat('map_values',			pb.forwardref(():any=>MapValues)),
	7: pb.Repeat('optional_values',		pb.forwardref(():any=>OptionalProto)),
});

const MapValues = pb.Message({
	1: pb.Optional('name',				pb.string),
	2: pb.Optional('key_type',			pb.int32),
	3: pb.Repeat('keys',				pb.int64),
	4: pb.Repeat('string_keys',			pb.bytes),
	5: pb.Optional('values',			Sequence),
});

const OptionalProto = pb.Message({
	1: pb.Optional('name',				pb.string),
	2: pb.Optional('elem_type',			pb.Enum(ElemType)),
	3: pb.Optional('tensor_value',		Tensor),
	4: pb.Optional('sparse_tensor_value', pb.forwardref(():any=>SparseTensor)),
	5: pb.Optional('sequence_value',	Sequence),
	6: pb.Optional('map_value',			MapValues),
	7: pb.Optional('optional_value',	pb.forwardref(():any=>OptionalProto)),
});

const SparseTensor = pb.Message({
	1: pb.Optional('values',			Tensor),
	2: pb.Optional('indices',			Tensor),
	3: pb.Repeat('dims',				pb.int64),
});

class ValueInfo extends bin.Class(pb.Message({
	1: pb.Optional('name',				pb.string),
	2: pb.Optional('type',				Type),
	3: pb.Optional('doc_string',		pb.string),
	4: pb.Map('metadata_props',			pb.string, pb.string),
})) {
	getShape() {
		return this.type?.tensor_type?.shape?.dim?.map((d: any) => d.dim_param || d.dim_value);
	}
	getDataType() {
		const elemType = this.type?.tensor_type?.elem_type;
		return elemType !== undefined
			? Object.keys(TensorDataType).find(k => TensorDataType[k as keyof typeof TensorDataType] === elemType)
			: undefined;
	}
}

const AttributeType = {
	UNDEFINED:		0,
	FLOAT:			1,
	INT:			2,
	STRING:			3,
	TENSOR:			4,
	GRAPH:			5,
	SPARSE_TENSOR:	11,
	TYPE_PROTO:		13,
	FLOATS:			6,
	INTS:			7,
	STRINGS:		8,
	TENSORS:		9,
	GRAPHS:			10,
	SPARSE_TENSORS: 12,
	TYPE_PROTOS:	14,
};
const Attribute = pb.Message({
	1:  pb.Optional('name',				pb.string),
	21: pb.Optional('ref_attr_name',	pb.string),
	13: pb.Optional('doc_string',		pb.string),
	20: pb.Optional('type',				pb.Enum(AttributeType)),
	2:  pb.Optional('f',				pb.float),
	3:  pb.Optional('i',				pb.int64),
	4:  pb.Optional('s',				pb.bytes),
	5:  pb.Optional('t',				Tensor),
	6:  pb.Optional('g',				pb.forwardref(():any=>Graph)),
	22: pb.Optional('sparse_tensor',	SparseTensor),
	14: pb.Optional('tp',				Type),
	7:  pb.Optional('floats',			pb.float),
	8:  pb.Optional('ints',				pb.int64),
	9:  pb.Optional('strings',			pb.bytes),
	10: pb.Optional('tensors',			Tensor),
	11: pb.Optional('graphs',			pb.forwardref(():any=>Graph)),
	23: pb.Optional('sparse_tensors',	SparseTensor),
	15: pb.Optional('type_protos',		Type),
});

const SimpleShardedDim = pb.Message({
	_: pb.OneOf({
		1: pb.Field('dim_value',		pb.int64),
		2: pb.Field('dim_param',		pb.string),
	}),
	3: pb.Optional('num_shards',		pb.int64),
});

const ShardedDim = pb.Message({
	1: pb.Optional('axis',				pb.int64),
	2: pb.Repeat('simple_sharding',	SimpleShardedDim),
});

const ShardingSpec = pb.Message({
	1: pb.Optional('tensor_name',			pb.string),
	2: pb.Repeat('device',					pb.int64),
	3: pb.Map('index_to_device_group_map',	pb.int64, pb.int64),
	4: pb.Repeat('sharded_dim',				ShardedDim),
});

const NodeDeviceConfiguration = pb.Message({
	1: pb.Optional('configuration_id',		pb.string),
	2: pb.Repeat('sharding_spec',			ShardingSpec),
	3: pb.Optional('pipeline_stage',		pb.int32),
});

class Node extends bin.Class(pb.Message({
	1: pb.Repeat('input',					pb.string),
	2: pb.Repeat('output',					pb.string),
	3: pb.Optional('name',					pb.string),
	4: pb.Optional('op_type',				pb.string),
	7: pb.Optional('domain',				pb.string),
	8: pb.Optional('overload',				pb.string),
	5: pb.Repeat('attribute',				Attribute),
	6: pb.Optional('doc_string',			pb.string),
	9: pb.Map('metadata_props',				pb.string, pb.string),
	10:pb.Repeat('device_configurations',	NodeDeviceConfiguration),
})) {}

const TensorAnnotation = pb.Message({
	1: pb.Optional('tensor_name',			pb.string),
	2: pb.Map('quant_parameter_tensor_names', pb.string, pb.string),
});

export class Graph extends bin.Class(pb.Message({
	1:  pb.Repeat('node',					Node),
	2:  pb.Optional('name',					pb.string),
	5:  pb.Repeat('initializer',			Tensor),
	15: pb.Repeat('sparse_initializer',		SparseTensor),
	10: pb.Optional('doc_string',			pb.string),
	11: pb.Repeat('input',					ValueInfo),
	12: pb.Repeat('output',					ValueInfo),
	13: pb.Repeat('value_info',				ValueInfo),
	14: pb.Repeat('quantization_annotation',TensorAnnotation),
	16: pb.Map('metadata_props',			pb.string, pb.string),
})) {
	inputs;
	outputs;
	initializers;

	constructor(s: any) {
		super(s);
		this.inputs			= makeNameLookup(this.input);
		this.outputs		= makeNameLookup(this.output);
		this.initializers	= makeNameLookup(this.initializer);
	}

	valueInfoMap(): Map<string, ValueInfo> {
		const map = new Map<string, ValueInfo>();
		for (const v of [...(this.input ?? []), ...(this.output ?? []), ...(this.value_info ?? [])])
			if (v.name)
				map.set(v.name, v);
		return map;
	}

	// navigation
	getNode(name: string) {
		return this.node.find(n => n.name === name);
	}
	getNodes(opType: string) {
		return this.node.filter(n => n.op_type === opType) ?? [];
	}
	getValueInfo(name: string) {
		return this.inputs[name] ?? this.outputs[name];
	}
}

const TrainingInfo = pb.Message({
	1: pb.Optional('initialization',	Graph),
	2: pb.Optional('algorithm',			Graph),
	3: pb.Map('initialization_binding',	pb.string, pb.string),
	4: pb.Map('update_binding',			pb.string, pb.string),
});

const OperatorSetId = pb.Message({
	1: pb.Optional('domain',			pb.string),
	2: pb.Optional('version',			pb.int64),
});

const DeviceConfiguration = pb.Message({
	1: pb.Optional('name',				pb.string),
	2: pb.Optional('num_devices',		pb.int32),
	3: pb.Repeat('device',				pb.string),
});

const Function = pb.Message({
	1:  pb.Optional('name',				pb.string),
	4:  pb.Repeat('input',				pb.string),
	5:  pb.Repeat('output',				pb.string),
	6:  pb.Repeat('attribute',			pb.string),
	11: pb.Repeat('attribute_proto',	Attribute),
	7:  pb.Repeat('node',				Node),
	8:  pb.Optional('doc_string',		pb.string),
	9:  pb.Repeat('opset_import',		OperatorSetId),
	10: pb.Optional('domain',			pb.string),
	13: pb.Optional('overload',			pb.string),
	12: pb.Repeat('value_info',			ValueInfo),
	14: pb.Map('metadata_props',		pb.string, pb.string),
});

class Model extends bin.Class(pb.Message({
	1:  pb.Optional('ir_version',		pb.Enum(Version)),
	8:  pb.Repeat('opset_import',		OperatorSetId),
	2:  pb.Optional('producer_name',	pb.string),
	3:  pb.Optional('producer_version',	pb.string),
	4:  pb.Optional('domain',			pb.string),
	5:  pb.Optional('model_version',	pb.int64),
	6:  pb.Optional('doc_string',		pb.string),
	7:  pb.Optional('graph',			Graph),
	14: pb.Map('metadata_props',		pb.string, pb.string),
	20: pb.Repeat('training_info',		TrainingInfo),
	25: pb.Repeat('functions',			Function),
	26: pb.Repeat('configuration',		DeviceConfiguration),
})) {
	opsets(): Map<string, bigint> {
		const map = new Map<string, bigint>();
		for (const op of this.opset_import ?? [])
			map.set(op.domain ?? '', op.version ?? 0n);
		return map;
	}

	toString() {
		const graph = this.graph!;
		return [
			`ir_version: ${this.ir_version}`,
			`producer: ${this.producer_name} ${this.producer_version}`,
			`opsets: ${[...this.opsets().entries()].map(([d, v]) => `${d || 'ai.onnx'}:${v}`).join(', ')}`,
			`inputs: ${Object.entries(graph.inputs).map(([n, v]) => `${n}${(v.getShape() ?? []).join(',')}]`).join(', ')}`,
			`outputs: ${Object.entries(graph.outputs).map(([n, v]) => `${n}${(v.getShape() ?? []).join(',')}]`).join(', ')}`,
		].join('\n');
	}
}

export function readOnnxModel(data: Uint8Array) {
	const stream = new bin.stream(data);
	return stream.read(Model);
}

import * as bin from '@isopodlabs/binary';

//type Simplify<T> = { [K in keyof T]: T[K] } & {};

const Wire = bin.RemainingArray({
	_: bin.Merge(bin.as(bin.ULEB128, bin.bitfields.BitFields(0, {
		wire:	3,
		field:	29,
	} as const))),
	value: bin.Switch(s => s.obj.wire, {
		0: bin.ULEB128,
		1: bin.UINT64_LE,
		2: bin.Buffer(bin.ULEB128),
		5: bin.UINT32_LE,
	})
});

type Decoder = ((v: any) => any) | {get(s: any): any};
type DecoderReturn<D extends Decoder> = D extends (new (s: any) => infer R) ? R : D extends {get(s: any): infer R} ? bin.NoPromise<R> : D extends ((v: any) => infer R) ? R : never;

function makeDecoder<D extends Decoder>(decode: D): (v: any) => DecoderReturn<D> {
	if ('get' in decode)
		return (data: Uint8Array) => (decode as any).get(new bin.stream(data));
	return decode as any;
}


interface Repeat<K extends string, T> {
	key:		K;
	repeat:		(v: any) => T;
}
interface Optional<K extends string, T> {
	key:		K;
	optional:		(v: any) => T;
}
interface Single<K extends string, T> {
	key:		K;
	single:		(v: any) => T;
}
interface OneOf<T extends Schema2> {
	oneof:		T;
}
type Field2<K extends string, T> = Repeat<K, T> | Optional<K, T> | Single<K, T>;
type SchemaEntry2 = Field2<any, any> | OneOf<Schema2>;
type Schema2 = Record<number, SchemaEntry2>;
type Fields2<T> = T[Extract<keyof T, number>];

type OneOfFields<T extends Schema2> = Fields2<bin.UnionToIntersection<Extract<T[keyof T], OneOf<any>>['oneof']>>

type Decoded2<T extends Schema2> = //Simplify<
	{ [F in Fields2<T> as F extends Field2<infer K, any> ? K : never]:
		  F extends Single<any,   infer V> ? V
		: F extends Repeat<any,   infer V> ? V[]
		: F extends Optional<any, infer V> ? V | undefined
		: never
	} &
	{ [F in OneOfFields<T> as F extends Field2<infer K, any> ? K : never]?:
		  F extends Single<any,   infer V> ? V
		: F extends Repeat<any,   infer V> ? V[]
		: F extends Optional<any, infer V> ? V | undefined
		: never
	}
//>;


/*

interface Field<K extends string, T, R extends boolean> {
	key:		K;
	decode:		(v: any) => T;
	repeated:	R;
}
type Schema = Record<number, Field<any, any, any>>;
type Fields<T extends Schema> = T[Extract<keyof T, number>];
type Decoded<T extends Schema> = Simplify<{
	[F in Fields<T> as F['key']]: F extends Field<any, infer V, infer R> ? R extends true ? V[] : V : never;
}>;


function decode(schema: Schema, fields: bin.ReadType<typeof Wire>) {
	const result: any = {};

	for (const f of fields) {
		const def = schema[f.field];
		if (!def)
			continue;

		const { key, decode, repeated } = def;
		const decoded = decode(f.value);
		if (repeated)
			(result[key] ??= []).push(decoded);
		else
			result[key] = decoded;
	}
	return result;
}

export function Message<T extends Schema>(schema: T) {
	return bin.as(Wire, fields => decode(schema, fields) as Decoded<T>);
}
export function Field<K extends string, D extends Decoder>(key: K, decode: D) : Field<K, DecoderReturn<D>, false> {
	return { key, decode: makeDecoder(decode), repeated: false };
}
export function Repeat<K extends string, D extends Decoder>(key: K, decode: D) : Field<K, DecoderReturn<D>, true> {
	return { key, decode:  makeDecoder(decode), repeated: true };
}

*/

function flatten(schema: Schema2) : Schema2 {
	const oneofs = Object.values(schema).filter(v => 'oneof' in v);
	const normal = Object.fromEntries(Object.entries(schema).filter(([_, v]) => !('oneof' in v)));
	return Object.assign(normal, ...oneofs.map(i => flatten(i.oneof)));
}
function verify(schema: Schema2, result: any) {
	for (const def of Object.values(schema)) {
		if ('single' in def) {
			if (!(def.key in result))
				throw new Error(`missing required field ${def.key}`);

		} else if ('repeat' in def) {
			if (!(def.key in result))
				result[def.key] = [];

		} else if ('oneof' in def) {
			const keys = Object.values(def.oneof).filter(i => 'key' in i).map(i => i.key as string);
			const got = keys.filter(i => i in result);
			if (got.length > 1)
				throw new Error(`oneof has multiple fields set: ${got.join(',')}`);
		}
	}
}

function decode2(schema: Schema2, fields: bin.ReadType<typeof Wire>) {
	const result: any = {};
	const schema2 = flatten(schema);

	for (const f of fields) {
		const def = schema2[f.field];
		if (!def)
			continue;

		if ('single' in def) {
			result[def.key] = def.single(f.value);
		} else if ('optional' in def) {
			result[def.key] = def.optional(f.value);
		} else if ('repeat' in def) {
			(result[def.key] ??= []).push(def.repeat(f.value));
		}
	}

	verify(schema, result);
	return result;
}

export function Message<T extends Schema2>(schema: T) {
	return bin.as(Wire, fields => decode2(schema, fields) as Decoded2<T>);
}
export function Field<K extends string, D extends Decoder>(key: K, decode: D) : Single<K, DecoderReturn<D>> {
	return { key, single: makeDecoder(decode) };
}
export function Optional<K extends string, D extends Decoder>(key: K, decode: D) : Optional<K, DecoderReturn<D>> {
	return { key, optional: makeDecoder(decode) };
}
export function Repeat<K extends string, D extends Decoder>(key: K, decode: D) : Repeat<K, DecoderReturn<D>> {
	return { key, repeat: makeDecoder(decode) };
}
export function OneOf<T extends Schema2>(schema: T) : OneOf<T> {
	return { oneof: schema };
}

// scalar types
//double		
//float			
//int32			Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint32 instead.
//int64			Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint64 instead.
//uint32		Uses variable-length encoding.
//uint64		Uses variable-length encoding.
//sint32		Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int32s.
//sint64		Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int64s.
//fixed32		Always four bytes. More efficient than uint32 if values are often greater than 228.
//fixed64		Always eight bytes. More efficient than uint64 if values are often greater than 256.
//sfixed32		Always four bytes.
//sfixed64		Always eight bytes.
//bool			
//string		A string must always contain UTF-8 encoded or 7-bit ASCII text, and cannot be longer than 232.
//bytes			May contain any arbitrary sequence of bytes no longer than 232.

export const double		= (v: bigint) => { const dv = new DataView(new ArrayBuffer(8)); dv.setBigUint64(0, v, true); return dv.getFloat64(0, true); };
export const float		= (v: number) => { const dv = new DataView(new ArrayBuffer(4)); dv.setUint32(0, v, true); return dv.getFloat32(0, true); };
export const int32		= (v: bigint | number) => Number(v);
export const int64		= (v: bigint | number) => BigInt(v);
export const uint32		= (v: bigint | number) => Number(v);
export const uint64		= (v: bigint | number) => BigInt(v);
export const sint32		= (v: bigint | number) => { const n = Number(v); return (n >>> 1) ^ -(n & 1); };
export const sint64		= (v: bigint | number) => { const n = BigInt(v); return (n >> 1n) ^ -(n & 1n); };
export const fixed32	= (v: number) => v;
export const fixed64	= (v: bigint) => v;
export const sfixed32	= (v: number) => v;
export const sfixed64	= (v: bigint) => v;
export const bool		= (v: bigint | number) => Boolean(Number(v));
export const string		= (v: Uint8Array) => new TextDecoder().decode(v);
export const bytes		= (v: Uint8Array) => v;

//export function ref<T extends Schema>(schema: T) {
//	return (data: Uint8Array) => new bin.stream(data).read(Message(schema));
//}
export function forwardref<T>(func: ()=>bin.TypeT<T>) {
	return (data: Uint8Array) => new bin.stream(data).read(func());
//	return (data: Uint8Array) => new bin.stream(data).read(Message(func()));
}
export function Enum<E extends Record<string, number>>(_e: E) {
	return (data: number|bigint) => data as E[keyof E];
}

export function Map<N extends string, K extends Decoder, V extends Decoder>(name: N, key: K, value: V) {
	return Repeat(name, Message({
		1: Field('key', 	key),
		2: Field('value', 	value),
	}));
}


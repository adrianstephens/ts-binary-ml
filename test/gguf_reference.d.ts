export type Decoder = (src: Uint8Array, dv: DataView, s: number, out: Float32Array, o: number) => void;
export declare const reference: Record<number, {
    block: number;
    size: number;
    decode: Decoder;
}>;

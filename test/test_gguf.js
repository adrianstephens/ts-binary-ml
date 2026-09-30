"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const bin = __importStar(require("@isopodlabs/binary"));
const index_1 = require("../dist/index");
const fs_1 = require("fs");
async function openReadFile(filename) {
    const fd = await fs_1.promises.open(filename, fs_1.promises.constants.O_RDONLY);
    const stat = await fs_1.promises.stat(filename);
    return new bin.async.stream((offset, data) => fd.read(data, 0, data.length, offset).then(r => r.bytesRead), undefined, _s => fd.close(), stat.size);
}
//const fn = '/Volumes/threadripper/Users/adrian/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d';
const fn = '/Volumes/DevSSD/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d';
(async () => {
    const stream = await openReadFile(fn);
    //	const data = await stream.view_at(Uint8Array, 0x59284cd00, 1024);
    //	const stream2 = new bin.stream(data);
    //	const gg = await gguf.readGguf(stream2);
    const gg = await index_1.gguf.readGguf(stream);
    const tensor = gg.tensors.at(-1); //[1];
    console.log(tensor.name, tensor.typeName(), tensor.shape, 'first values:', await tensor.read(0, Math.min(tensor.parameterCount(), 16)));
    const types = {};
    for (const t of gg.tensors)
        types[t.typeName()] = (types[t.typeName()] ?? 0) + 1;
    console.log(types);
    console.log(gg);
})();
//# sourceMappingURL=test_gguf.js.map
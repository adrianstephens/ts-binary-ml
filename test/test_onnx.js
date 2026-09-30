"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const onnx_1 = require("../dist/onnx");
const fs_1 = require("fs");
const fnonnx = '/Users/adrianstephens/Downloads/adv_inception_v3_Opset16.onnx';
//const fn = '/Volumes/threadripper/Users/adrian/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d';
const fn = '/Volumes/DevSSD/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d';
(async () => {
    const data = await fs_1.promises.readFile(fnonnx);
    const x = (0, onnx_1.readOnnxModel)(data);
    console.log(x);
})();
//# sourceMappingURL=test_onnx.js.map
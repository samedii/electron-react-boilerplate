const sharp = require('sharp');
const ort = require('onnxruntime');
const path = require('path');

async function heatmap() {
  console.log('ort', ort);
  const width = 28;
  const height = 28;

  const image = await sharp('./test.jpg')
    .grayscale()
    .resize(width, height)
    .raw()
    .toBuffer();
  console.log(image);

  const uint8Image = new Uint8Array(image.buffer);
  console.log(uint8Image.length);

  // run this in an async method:
  const url = './mnist-8.onnx';
  console.log('loading model', url);
  const session = await ort.InferenceSession.create(url);
  // use the following in an async method
  console.log('session', session);

  const inputs = { Input3: new ort.Tensor('float32', Float32Array.from(uint8Image), [1, height, width])};
  // use this line instead and the program works
  // const inputs = { Input3: new ort.Tensor('float32', Float32Array.from(uint8Image), [1, 1, height, width])};
  console.log(inputs);

  const results = await session.run(inputs);
  console.log('results');
  console.log('results', results);
}

module.exports = heatmap

# BrNet actual forward-pass audit

Reference four-class parameters: **895,812**. Expected: 895,812.

```json
[
  {
    "layer": "features.conv1",
    "type": "Conv2d",
    "input_shape": [
      1,
      1,
      256,
      256
    ],
    "output_shape": [
      1,
      32,
      254,
      254
    ],
    "kernel": "(3, 3)",
    "stride": "(1, 1)",
    "padding": "(0, 0)",
    "trainable_parameters": 320
  },
  {
    "layer": "features.relu1",
    "type": "ReLU",
    "input_shape": [
      1,
      32,
      254,
      254
    ],
    "output_shape": [
      1,
      32,
      254,
      254
    ],
    "kernel": "",
    "stride": "",
    "padding": "",
    "trainable_parameters": 0
  },
  {
    "layer": "features.pool1",
    "type": "MaxPool2d",
    "input_shape": [
      1,
      32,
      254,
      254
    ],
    "output_shape": [
      1,
      32,
      127,
      127
    ],
    "kernel": "2",
    "stride": "2",
    "padding": "0",
    "trainable_parameters": 0
  },
  {
    "layer": "features.conv2",
    "type": "Conv2d",
    "input_shape": [
      1,
      32,
      127,
      127
    ],
    "output_shape": [
      1,
      64,
      125,
      125
    ],
    "kernel": "(3, 3)",
    "stride": "(1, 1)",
    "padding": "(0, 0)",
    "trainable_parameters": 18496
  },
  {
    "layer": "features.relu2",
    "type": "ReLU",
    "input_shape": [
      1,
      64,
      125,
      125
    ],
    "output_shape": [
      1,
      64,
      125,
      125
    ],
    "kernel": "",
    "stride": "",
    "padding": "",
    "trainable_parameters": 0
  },
  {
    "layer": "features.pool2",
    "type": "MaxPool2d",
    "input_shape": [
      1,
      64,
      125,
      125
    ],
    "output_shape": [
      1,
      64,
      62,
      62
    ],
    "kernel": "2",
    "stride": "2",
    "padding": "0",
    "trainable_parameters": 0
  },
  {
    "layer": "features.conv3",
    "type": "Conv2d",
    "input_shape": [
      1,
      64,
      62,
      62
    ],
    "output_shape": [
      1,
      64,
      60,
      60
    ],
    "kernel": "(3, 3)",
    "stride": "(1, 1)",
    "padding": "(0, 0)",
    "trainable_parameters": 36928
  },
  {
    "layer": "features.relu3",
    "type": "ReLU",
    "input_shape": [
      1,
      64,
      60,
      60
    ],
    "output_shape": [
      1,
      64,
      60,
      60
    ],
    "kernel": "",
    "stride": "",
    "padding": "",
    "trainable_parameters": 0
  },
  {
    "layer": "features.pool3",
    "type": "MaxPool2d",
    "input_shape": [
      1,
      64,
      60,
      60
    ],
    "output_shape": [
      1,
      64,
      30,
      30
    ],
    "kernel": "2",
    "stride": "2",
    "padding": "0",
    "trainable_parameters": 0
  },
  {
    "layer": "features.conv4",
    "type": "Conv2d",
    "input_shape": [
      1,
      64,
      30,
      30
    ],
    "output_shape": [
      1,
      64,
      28,
      28
    ],
    "kernel": "(3, 3)",
    "stride": "(1, 1)",
    "padding": "(0, 0)",
    "trainable_parameters": 36928
  },
  {
    "layer": "features.relu4",
    "type": "ReLU",
    "input_shape": [
      1,
      64,
      28,
      28
    ],
    "output_shape": [
      1,
      64,
      28,
      28
    ],
    "kernel": "",
    "stride": "",
    "padding": "",
    "trainable_parameters": 0
  },
  {
    "layer": "features.pool4",
    "type": "MaxPool2d",
    "input_shape": [
      1,
      64,
      28,
      28
    ],
    "output_shape": [
      1,
      64,
      14,
      14
    ],
    "kernel": "2",
    "stride": "2",
    "padding": "0",
    "trainable_parameters": 0
  },
  {
    "layer": "flatten",
    "type": "Flatten",
    "input_shape": [
      1,
      64,
      14,
      14
    ],
    "output_shape": [
      1,
      12544
    ],
    "kernel": "",
    "stride": "",
    "padding": "",
    "trainable_parameters": 0
  },
  {
    "layer": "dense",
    "type": "Linear",
    "input_shape": [
      1,
      12544
    ],
    "output_shape": [
      1,
      64
    ],
    "kernel": "",
    "stride": "",
    "padding": "",
    "trainable_parameters": 802880
  },
  {
    "layer": "relu",
    "type": "ReLU",
    "input_shape": [
      1,
      64
    ],
    "output_shape": [
      1,
      64
    ],
    "kernel": "",
    "stride": "",
    "padding": "",
    "trainable_parameters": 0
  },
  {
    "layer": "classifier",
    "type": "Linear",
    "input_shape": [
      1,
      64
    ],
    "output_shape": [
      1,
      4
    ],
    "kernel": "",
    "stride": "",
    "padding": "",
    "trainable_parameters": 260
  },
  {
    "layer": "softmax",
    "type": "Softmax",
    "input_shape": [
      1,
      4
    ],
    "output_shape": [
      1,
      4
    ],
    "kernel": "",
    "stride": "",
    "padding": "",
    "trainable_parameters": 0
  }
]
```

Grad-CAM candidate: features.conv4, pre-ReLU 64 x 28 x 28. Final pooling is 14 x 14. Figshare requires a separate three-class head: 895,747 parameters. This adaptation cannot produce healthy-class predictions. Training must use logits with cross entropy.

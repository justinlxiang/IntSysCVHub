# IntSysCVHub

## Dependencies

This project requires the following main dependencies:

- Python 3.7+
- PyTorch 1.7+
- torchvision
- detectron2
- numpy
- matplotlib
- Pillow

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

If you have trouble installing detectron2, try installing it from source with these instructions: https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md

## Model Weights

Download the base MaskRCNN model weights from CUAir Box or Google Drive by talking to a intelligent systems member or download your own if developing a new model.

## Usage

To run the project, use the following command:

# Training the model (specify I/O paths in train_net.py)
```bash
python detection/train_net.py --config-file detection/base_config.yaml
```

# Testing the model (specify I/O paths in test_rcnn.py)

```bash
python test_rcnn.py
```
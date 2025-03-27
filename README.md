# Predicting Pathologies in Chest X-ray Images with Rejection Mechanisms

## Overview
This project focuses on Out Of Distribution (OOD) data. IT is predicting pathologies in chest X-ray images while incorporating rejection mechanisms to improve OOD results. The approach is based on inference using the **XRV (Chest X-ray Radiograph) model**, a pre-trained deep learning model for medical imaging analysis. 

This work is inspired by [OoD_Gen-Chest_Xray](https://github.com/etetteh/OoD_Gen-Chest_Xray) but focuses **only on inference** using the **XRV model**.

## Features
- **Inference using XRV Model**: Extracts features and predicts pathologies from chest X-ray images.
- **Out-of-Distribution (OoD) Rejection**: Applies rejection mechanisms to improve model robustness.
- **Evaluation & Metrics**: Analyzes model performance under different rejection thresholds.

## Setup & Installation
### Requierments
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- OpenCV
- Matplotlib
- tqdm

## Usage
### Running Inference
Run inference on chest X-ray images using the XRV model:
```bash
xrv_test.py 
```

### Applying Rejection Mechanisms - ADD
To apply a rejection mechanism,
```bash
Rejection.py 
```

### Applying Entropy-based Rejection Mechanisms
To apply a rejection mechanism,
```bash
Rejection_Entropy_split1.py
Rejection_Entropy_split2.py 
```

## Results & Evaluation
Evaluation is performed by analyzing the model AUC with and without rejection. The key metrics include:
- **AUC (Area Under Curve)**
- **Rejection Rate vs. AUC Tradeoff**

## References
- [Original OoD_Gen-Chest_Xray Repository](https://github.com/etetteh/OoD_Gen-Chest_Xray)
- [TorchXRayVision (XRV Model)](https://github.com/mlmed/torchxrayvision)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or collaborations, feel free to reach out!

---

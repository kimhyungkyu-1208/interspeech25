# Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation

Official PyTorch implementation of the paper:  
**Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation**  
To be presented at **Interspeech 2025**.

---

## 🌟 Core Concept

Traditional speech-driven 3D facial animation models often rely on reconstruction losses like **MSE**, which treat the error of each frame with equal importance. However, this has limitations:

- **Ignores Motion Significance**: Frames with large lip movements (e.g., `/a/`, `/o/`) are weighted the same as frames with minimal movement (e.g., `/s/`).
- **Unnatural Transitions**: Equal weighting can result in *jittery* or abrupt transitions between visemes.

### 🔑 Our Solution: Phonetic Context-Aware Loss
The **Phonetic Context-Aware Loss** addresses this by:

1. **Quantify Motion**  
   Compute vertex displacement between consecutive frames in ground-truth motion data.
2. **Assign Weights**  
   Use a sliding window to measure movement magnitude. Dynamic frames (e.g., mouth opening/closing) get higher weights.
3. **Apply Weighted Loss**  
   Normalize weights with Softmax and apply them to MSE loss.

✅ This encourages the model to focus more on *meaningful motion frames*, leading to smoother, more realistic, and perceptually consistent facial animations.

---

## ⚙️ Installation

The only requirement is **PyTorch**:

```bash
pip install torch
```

---

## 🚀 Usage

Our proposed loss function is a **drop-in replacement** for `torch.nn.functional.mse_loss`.  
Simply replace your reconstruction loss with **PhoneticContextAwareLoss**.

```python
import torch
import torch.nn.functional as F
from loss import PhoneticContextAwareLoss  # Import from loss.py

# --- Your existing variables ---
# gt_motion = ...          # Ground truth 3D mesh sequence
# generated_motion = ...   # Predicted 3D mesh sequence

# --- Original reconstruction loss (BEFORE) ---
# loss_reconstruction = F.mse_loss(generated_motion, gt_motion)

# --- Applying our Phonetic Context-Aware Loss (AFTER) ---
ourloss = PhoneticContextAwareLoss(window_size=5, reduction='mean')
loss_reconstruction = ourloss(generated_motion, gt_motion)

# Continue your pipeline
# loss_reconstruction.backward()
# optimizer.step()
```

---

## 📖 Citation

If you use this repository in your research, please cite our paper:

```bibtex
@inproceedings{kim2025interspeech,
  title={Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation},
  author={Kim, Hyung Kyu and Kim, Hak Gu},
  booktitle={Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2025},
  organization={ISCA}
}
```

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Hyung Kyu Kim, Hak Gu Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👨‍💻 Authors

- **Hyung Kyu Kim**
- **Hak Gu Kim**

**🌐 [Project Page](https://cau-irislab.github.io/interspeech25/) | 📄 [Paper](https://www.isca-archive.org/interspeech_2025/kim25r_interspeech.html)
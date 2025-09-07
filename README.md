<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            line-height: 1.6;
            color: #24292e;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
        }
        h1, h2 {
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
        }
        h1 {
            font-size: 2em;
        }
        h2 {
            font-size: 1.5em;
        }
        p {
            margin-top: 0;
            margin-bottom: 16px;
        }
        strong {
            font-weight: 600;
        }
        code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
            font-size: 85%;
            padding: 0.2em 0.4em;
            margin: 0;
            background-color: rgba(27,31,35,0.05);
            border-radius: 3px;
        }
        pre {
            padding: 16px;
            overflow: auto;
            font-size: 85%;
            line-height: 1.45;
            background-color: #f6f8fa;
            border-radius: 3px;
            margin-bottom: 16px;
            word-wrap: normal;
        }
        pre code {
            padding: 0;
            margin: 0;
            background-color: transparent;
            border: 0;
        }
        .image-placeholder {
            width: 100%;
            height: 150px;
            background-color: #f0f0f0;
            border: 1px dashed #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #888;
            font-style: italic;
            border-radius: 6px;
            margin: 16px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation (Interspeech 2025)</h1>
        
        <p>This is the official repository for the paper: <strong>Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation</strong>, to be presented at Interspeech 2025.</p>
        
        <p>This repository provides the official PyTorch implementation of the <strong>Phonetic Context-Aware Loss</strong>. This loss function is designed to replace the standard frame-wise reconstruction loss (e.g., MSE). It helps 3D facial animation models learn the subtle and natural lip movements that result from coarticulation in speech.</p>
        
        <div class="image-placeholder"></div>
        
        <h2>Core Concept</h2>
        
        <p>Traditional speech-driven 3D facial animation models often rely on reconstruction losses like MSE, which treat the error of each frame with equal importance. This approach has limitations:</p>
        <ul>
            <li><strong>Ignores Motion Significance</strong>: Frames corresponding to pronounced sounds with large lip movements (e.g., '/a/', '/o/') are weighted the same as frames with minimal movement (e.g., '/s/').</li>
            <li><strong>Leads to Unnatural Transitions</strong>: The lack of emphasis on temporal continuity can result in "jittery" or abrupt transitions between visemes, making the animation appear unnatural.</li>
        </ul>
        <p>Our proposed <strong>Phonetic Context-Aware Loss</strong> addresses this problem with the following approach:</p>
        <ol>
            <li><strong>Quantify Motion</strong>: It first calculates the actual vertex displacement between consecutive frames in the ground truth motion data.</li>
            <li><strong>Assign Weights</strong>: Using a sliding window, it measures the average magnitude of movement around each frame. Frames within dynamic segments (e.g., when the mouth is opening or closing) receive a higher weight.</li>
            <li><strong>Apply Weighted Loss</strong>: These weights are normalized using a Softmax function and then multiplied with the standard MSE loss.</li>
        </ol>
        <p>As a result, the model is encouraged to <strong>focus more on reducing errors in frames with significant, meaningful motion</strong>. This leads to the generation of smoother, more realistic, and perceptually consistent facial animations.</p>
        
        <h2>Installation</h2>
        <p>The only requirement is PyTorch.</p>
        <pre><code>pip install torch</code></pre>
        
        <h2>Adaptable Code and Method</h2>
        <p>Our proposed loss function is designed as a direct replacement for the standard reconstruction loss (like <code>torch.nn.functional.mse_loss</code>). You can easily integrate it into any existing speech-driven animation framework.</p>
        <p>The key is to replace the line that calculates the reconstruction loss with our <code>PhoneticContextAwareLoss</code>.</p>
        <p>Here is an example of how to adapt your code:</p>
        <pre><code>import torch
import torch.nn.functional as F
from loss import PhoneticContextAwareLoss # Import our loss from loss.py

# --- Your existing training loop ---
# gt_motion = ...          # Ground truth 3D mesh sequence
# generated_motion = ...   # 3D mesh sequence from your model

# --- Original reconstruction loss (BEFORE) ---
# loss_reconstruction = F.mse_loss(generated_motion, gt_motion)

# --- Applying our Phonetic Context-Aware Loss (AFTER) ---

# 1. Instantiate our loss function
ourloss = PhoneticContextAwareLoss(window_size=5, reduction='mean')

# 2. Replace the original mse_loss with ourloss
# The arguments are in the order (prediction, ground_truth)
loss_reconstruction = ourloss(generated_motion, gt_motion)

# The rest of your training pipeline remains unchanged
# loss_reconstruction.backward()
# optimizer.step()</code></pre>

        <h2>Citation</h2>
        <p>If you use this code in your research, please consider citing our paper.</p>
        <pre><code>@inproceedings{kim2025interspeech,
  title={Learning Phonetic Context-Dependent Viseme for Enhancing Speech-Driven 3D Facial Animation},
  author={Kim, Hyung Kyu and Kim, Hak Gu},
  booktitle={Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2025},
  organization={ISCA}
}</code></pre>

        <h2>License</h2>
        <p>This project is licensed under the MIT License.</p>
        <pre>MIT License

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
</pre>
    </div>
</body>
</html>
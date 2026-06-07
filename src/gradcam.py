import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class: int = 1):
        self.model.zero_grad()

        seg_out, cls_out = self.model(input_tensor)

        # Notebook-aligned class targeting: class index 1 is malignant for 2-class heads.
        if cls_out.ndim == 2 and cls_out.shape[1] > 1:
            score = cls_out[:, target_class].sum()
        else:
            score = cls_out.squeeze()

        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations")

        # GAP (global average pooling)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        # normalize
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-8)

        return cam


def overlay_gradcam(original_img, cam):
    h, w = original_img.shape

    cam = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)

    return overlay
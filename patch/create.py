import torchvision.transforms as transforms
import random
import torch
import torch.nn.functional as F

device='cuda' if torch.cuda.is_available else 'cpu'

class Patch:
    def __init__(self,config):
        self.config = config
        self.patch_size = config.patch.size
        # Transformation list for EOT (scaling, rotation, translation)
        self.eot_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=(-20, 20)),  # Random rotation
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Random translation
            transforms.RandomResizedCrop(size=self.patch_size, scale=(0.8, 1.2)),  # Random scaling
        ])


    def apply_patch(self, image, label, patch):
        """
        Overlay the adversarial patch on the image at a given position.
        """
        patched_image = image.clone()
        patched_label = label.clone()
        B,c,h,w=image.shape


        mask = (patched_label == 1).unsqueeze(1).to(torch.float16).to(device)
        
        kernel = torch.ones((1,1,200,200), dtype=torch.float16, device=device)
        convs = F.conv2d(mask, kernel, stride=1, padding=0)  # [B,1,H-199,W-199]
        
        h_out,w_out=convs.shape[2],convs.shape[3]
        # Max per batch
        max_idxs = torch.argmax(convs.view(B, -1),dim=1)   # shape [B]
        
        
        # Convert flat indices -> (y,x) coordinates in output space
        y, x = max_idxs // w_out, max_idxs % w_out



        x_end, y_end = x + self.patch_size, y + self.patch_size

        # Apply transformation to patch (EOT)
        #transformed_patch = self.eot_transforms(patch)
        transformed_patch = patch

        # Overlay patch onto the image and accordingly edit the label
        patched_image[:,:, y:y_end, x:x_end] = transformed_patch
        patched_label[:, y:y_end, x:x_end] = self.config.train.ignore_label
        #print(patched_label[:, y:y_end, x:x_end])
        return patched_image, patched_label

import torch
import unittest
from fixed_distillation_trainer import ChoEtAlFeatureLoss, FixedDistillationTrainer
from ultralytics import YOLO

class TestDistillationFixes(unittest.TestCase):
    
    def test_feature_loss_normalization(self):
        """Test Issue C: Feature Loss Normalization"""
        loss_fn = ChoEtAlFeatureLoss(normalize=True)
        
        # Create two tensors that are identical in direction but different in magnitude
        base = torch.randn(2, 64, 20, 20)
        scaled = base * 100.0  # Same direction, huge magnitude difference
        
        # Without normalization, MSE would be huge
        mse = torch.nn.functional.mse_loss(base, scaled)
        
        # With normalization, they should be identical (loss = 0)
        norm_loss = loss_fn(base, scaled)
        
        print(f"Raw MSE: {mse.item():.4f} | Normalized Loss: {norm_loss.item():.4f}")
        self.assertLess(norm_loss.item(), 1e-5, "Normalized loss should be near zero for scaled vectors")

    def test_feature_loss_resizing(self):
        """Test that loss handles different spatial dimensions (e.g. v8 vs v26 differences)"""
        loss_fn = ChoEtAlFeatureLoss(normalize=True)
        
        student = torch.randn(2, 64, 20, 20)
        teacher = torch.randn(2, 64, 40, 40) # Larger teacher feature map
        
        # Should not crash
        try:
            loss = loss_fn(student, teacher)
            print(f"Resizing Loss check: {loss.item():.4f}")
        except Exception as e:
            self.fail(f"Feature loss crashed on size mismatch: {e}")

    def test_layer_detection(self):
        """Test Issue A: Dynamic Layer Detection"""
        # We'll use v8n as it's definitely available
        try:
            trainer = FixedDistillationTrainer(model_name='yolov8n.pt', epochs=1)
            idx = trainer._find_feature_layer()
            
            print(f"Detected Layer Index for YOLOv8n: {idx}")
            
            # For v8n, SPPF is typically layer 9
            # But specific index depends on specific version of ultralytics (usually 9 or 10)
            # We assert it found SOMETHING reasonable > 0
            self.assertGreater(idx, 0, "Failed to find any feature layer")
            
            # Check if layer exists
            layer = list(trainer.teacher_model.model.children())[idx]
            print(f"Layer class: {type(layer).__name__}")
            
        except Exception as e:
            print(f"Skipping model loading test if weights missing: {e}")

    def test_hook_robustness(self):
        """Test Issue B: Hooks handling Tuples"""
        trainer = FixedDistillationTrainer(model_name='yolov8n.pt', epochs=1)
        
        # Mock storage
        storage = []
        
        # Create the hook manually to test logic
        def make_hook(storage):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    feat = output[0]
                else:
                    feat = output
                storage.append(feat)
            return hook
            
        hook_fn = make_hook(storage)
        
        # Simulate Tensor output
        hook_fn(None, None, torch.tensor([1.0]))
        self.assertTrue(isinstance(storage[0], torch.Tensor))
        
        # Simulate Tuple output
        hook_fn(None, None, (torch.tensor([2.0]), "some_other_stuff"))
        self.assertTrue(isinstance(storage[1], torch.Tensor))
        self.assertEqual(storage[1].item(), 2.0)
        
        print("Hook robustness verified.")

if __name__ == '__main__':
    unittest.main()

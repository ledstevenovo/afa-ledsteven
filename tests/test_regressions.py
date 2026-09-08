import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.transformer_2d import Transformer2DModel
from transformers import CLIPTextConfig, CLIPTextModel

import evaluate
import train
from models import Model
from models.modules import Aggregator, UNet2DConditionAggregatorModel


def tiny_pipeline(_path):
    return SimpleNamespace(
        vae=AutoencoderKL(
            block_out_channels=(32,), norm_num_groups=32, latent_channels=4,
        ),
        tokenizer=None,
        text_encoder=CLIPTextModel(CLIPTextConfig(
            vocab_size=16, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=8,
        )),
        unet=UNet2DConditionModel(
            sample_size=8, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
            up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'),
            block_out_channels=(32, 64), layers_per_block=2,
            cross_attention_dim=32, attention_head_dim=4,
        ),
    )


def tiny_model(num_layers=1, xformers=False):
    # CPU tests use PyTorch attention; the CUDA training test uses real xformers.
    with patch.object(Model, 'load_sd_model', side_effect=tiny_pipeline):
        if xformers:
            return Model.load_init(['expert_a', 'expert_b'], 'base', 32, num_layers, 4)
        with patch.object(UNet2DConditionAggregatorModel, 'enable_xformers_memory_efficient_attention'), \
                patch.object(Transformer2DModel, 'enable_xformers_memory_efficient_attention'):
            return Model.load_init(['expert_a', 'expert_b'], 'base', 32, num_layers, 4)


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_cli_defaults(self):
        with patch.object(sys, 'argv', ['train.py', '--model_save_path', 'checkpoint']):
            self.assertEqual(train.parse_args().model_save_path, 'checkpoint')
        with patch.object(sys, 'argv', ['evaluate.py']):
            self.assertEqual(evaluate.parse_args().negative_prompt, '')

    def test_requested_aggregator_depth(self):
        for depth in (1, 3):
            with self.subTest(depth=depth):
                model = tiny_model(num_layers=depth)
                for aggregator in model.aggregators:
                    self.assertEqual(aggregator.config['num_layers'], depth)
                    self.assertEqual(len(aggregator.transformer.transformer_blocks), depth)

    def test_train_forward_preserves_gradient_through_frozen_experts(self):
        model = tiny_model()
        model.vae.requires_grad_(False)
        model.text_encoders.requires_grad_(False)
        model.unets.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.aggregators.parameters(), lr=1e-3)
        first_weight = model.aggregators[0].conv_out.weight
        before = first_weight.detach().clone()
        loss = model.train_forward(torch.randn(1, 3, 8, 8), torch.randint(0, 16, (1, 8)))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(first_weight.grad)
        self.assertTrue(torch.isfinite(first_weight.grad).all())
        self.assertGreater(first_weight.grad.abs().sum().item(), 0)
        for module in (model.vae, model.text_encoders, model.unets):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, first_weight))

    def test_aggregator_output_and_checkpoint(self):
        aggregator = Aggregator(32, 32, 2, 1, 4, 32, 128)
        features = torch.randn(1, 2, 32, 4, 4)
        temb, text = torch.randn(1, 128), torch.randn(1, 8, 32)
        output, attention = aggregator(features, temb, text)
        self.assertEqual(output.shape, (1, 32, 4, 4))
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(1, 4, 4))
        with tempfile.TemporaryDirectory() as directory:
            aggregator.save_pretrained(directory)
            restored = Aggregator.load_pretrained(directory)
            torch.testing.assert_close(restored(features, temb, text)[0], output)

    def test_evaluate_saves_distinct_images_and_preserves_single_image_path(self):
        for count in (1, 4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                image_path = Path(directory) / 'sample.png'
                attention_path = Path(directory) / 'attention.pt'
                colors = [(i * 50, 0, 0) for i in range(count)]
                output = SimpleNamespace(
                    images=[Image.new('RGB', (4, 4), color) for color in colors],
                    inference_attn_maps=(torch.ones(1),),
                )
                model = SimpleNamespace(test_forward=lambda *args, **kwargs: output)
                argv = [
                    'evaluate.py', '--model_path', 'unused', '--prompt', 'test',
                    '--num_images_per_prompt', str(count),
                    '--images_saved_path', str(image_path),
                    '--attn_maps_saved_path', str(attention_path),
                ]
                with patch.object(sys, 'argv', argv), \
                        patch.object(evaluate.Model, 'load_pretrained', return_value=model), \
                        patch.object(evaluate.accelerate, 'Accelerator', return_value=SimpleNamespace(
                            device=torch.device('cpu'), process_index=0,
                        )), patch.object(evaluate.torch, 'Generator'):
                    evaluate.main()
                self.assertEqual(len(list(Path(directory).glob('*.png'))), count)
                for i, color in enumerate(colors):
                    saved = image_path if count == 1 else image_path.with_name(f'sample_{i}.png')
                    with Image.open(saved) as image:
                        self.assertEqual(image.getpixel((0, 0)), color)
                self.assertTrue(attention_path.is_file())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for FP16 training')
    def test_cuda_training_entrypoint_updates_fp32_aggregators(self):
        model = tiny_model(xformers=True)
        first_weight = model.aggregators[0].conv_out.weight
        before = first_weight.detach().clone()
        batch = [{'pixel_values': torch.randn(3, 8, 8), 'input_ids': torch.randint(0, 16, (8,))}]
        vae_dtypes, gradients = [], []
        model.vae.encoder.register_forward_hook(
            lambda module, args, output: vae_dtypes.append((args[0].dtype, output.dtype))
        )
        first_weight.register_hook(lambda grad: gradients.append(grad.detach().clone()))
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                'train.py', '--model_base_path', 'unused', '--model_files', 'a,b',
                '--model_save_path', directory, '--batch_size', '1', '--num_workers', '0',
                '--max_train_epochs', '1', '--lr_warmup_steps', '0',
            ]
            with patch.object(sys, 'argv', argv), \
                    patch.object(train.Model, 'load_init', return_value=model), \
                    patch.object(train, 'get_dataset', return_value=batch), \
                    patch.object(model, 'save_pretrained') as save:
                train.main()
                save.assert_called_once_with(directory)
        self.assertEqual(vae_dtypes, [(torch.float32, torch.float32)])
        self.assertEqual(model.vae.dtype, torch.float32)
        self.assertEqual(model.unets[0].dtype, torch.float16)
        self.assertEqual(model.text_encoders[0].dtype, torch.float16)
        self.assertEqual(first_weight.dtype, torch.float32)
        self.assertTrue(gradients and torch.isfinite(gradients[0]).all())
        self.assertGreater(gradients[0].abs().sum().item(), 0)
        self.assertFalse(torch.equal(before, first_weight.detach().cpu()))
        for module in (model.vae, model.text_encoders, model.unets):
            self.assertTrue(all(p.grad is None for p in module.parameters()))


if __name__ == '__main__':
    unittest.main()

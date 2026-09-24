import torch

from fedsechealth.attacks import image_metrics, match_batch, psnr, ssim, total_variation


def test_ssim_identity_and_noise():
    torch.manual_seed(0)
    a = torch.rand(4, 3, 28, 28)
    assert torch.allclose(ssim(a, a), torch.ones(4), atol=1e-5)
    noisy = (a + 0.3 * torch.randn_like(a)).clamp(0, 1)
    assert (ssim(noisy, a) < 0.9).all()


def test_psnr_increases_with_quality():
    a = torch.rand(2, 1, 28, 28)
    assert (psnr(a + 0.01, a) > psnr(a + 0.1, a)).all()


def test_match_batch_recovers_permutation():
    a = torch.rand(5, 1, 28, 28)
    perm = torch.tensor([3, 0, 4, 1, 2])
    cols = match_batch(a[perm], a)
    torch.testing.assert_close(a[perm][cols], a)


def test_image_metrics_is_order_invariant():
    a = torch.rand(3, 1, 28, 28)
    m = image_metrics(a.flip(0), a)
    assert m["success"] and m["ssim"] > 0.999


def test_total_variation_zero_on_constant_image():
    assert total_variation(torch.ones(1, 3, 8, 8)).item() == 0

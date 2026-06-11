"""Example: SFQ disambiguation on synthetic solar magnetogram data.

Generates a synthetic magnetogram with known ground truth, applies 180-degree
ambiguity, then runs SFQ to recover the correct azimuth.
"""

import numpy as np
import matplotlib.pyplot as plt
from sfq import sfq_disambig


def generate_synthetic_magnetogram(nx=128, ny=128):
    """Generate a synthetic solar magnetogram with known transverse field.

    Creates a potential-field-like active region with bipolar structure
    and smooth transverse field, then applies the 180-degree ambiguity.
    """
    x = np.linspace(-100, 100, nx)
    y = np.linspace(-100, 100, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Bipolar magnetic configuration - potential field approximation
    # Bz from two opposite polarity spots
    r1 = np.sqrt((xx + 30) ** 2 + (yy - 20) ** 2)
    r2 = np.sqrt((xx - 30) ** 2 + (yy + 20) ** 2)

    bz = 500 * np.exp(-r1 ** 2 / 800) - 400 * np.exp(-r2 ** 2 / 800)
    bz += 30 * np.random.randn(ny, nx)

    # Transverse field from potential field of this Bz
    # Use Fourier-space relation: Bx_hat = -i*kx*Bz_hat/q, By_hat = -i*ky*Bz_hat/q
    bz_pad = np.zeros((2 * ny, 2 * nx), dtype=float)
    bz_pad[:ny, :nx] = bz

    kx = 2 * np.pi * np.fft.fftfreq(2 * nx)
    ky = 2 * np.pi * np.fft.fftfreq(2 * ny)
    kx, ky = np.meshgrid(kx, ky)
    q = np.sqrt(kx ** 2 + ky ** 2)
    q_safe = np.where(q > 1e-10, q, 1.0)

    bz_hat = np.fft.fft2(bz_pad)
    bx_hat = -1j * kx / q_safe * bz_hat
    by_hat = -1j * ky / q_safe * bz_hat
    bx_hat[0, 0] = 0
    by_hat[0, 0] = 0

    bx_true = np.fft.ifft2(bx_hat).real[:ny, :nx]
    by_true = np.fft.ifft2(by_hat).real[:ny, :nx]

    # Smooth to remove high-frequency noise from Bz
    from scipy.ndimage import gaussian_filter
    bx_true = gaussian_filter(bx_true, sigma=2)
    by_true = gaussian_filter(by_true, sigma=2)

    # Apply azimuth ambiguity to ~50% of pixels randomly
    flip = np.random.rand(ny, nx) > 0.5
    bx_ambig = bx_true.copy()
    by_ambig = by_true.copy()
    bx_ambig[flip] = -bx_ambig[flip]
    by_ambig[flip] = -by_ambig[flip]

    # Add noise
    bx_ambig += 20 * np.random.randn(ny, nx)
    by_ambig += 20 * np.random.randn(ny, nx)

    pos = np.array([-100.0, -100.0, 100.0, 100.0])
    rsun = 960.0

    return bx_ambig, by_ambig, bz, pos, rsun, bx_true, by_true


def main():
    print("SFQ Disambiguation - Synthetic Test")
    print("=" * 50)

    np.random.seed(42)
    bx_ambig, by_ambig, bz, pos, rsun, bx_true, by_true = generate_synthetic_magnetogram()

    print(f"Grid size: {bx_ambig.shape[1]} x {bx_ambig.shape[0]}")
    print(f"Field of view: {pos[2] - pos[0]:.0f} x {pos[3] - pos[1]:.0f} arcsec")
    print(f"Pixels with flipped azimuth: ~50%")
    print()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Ambiguous Bx
    im0 = axes[0, 0].imshow(bx_ambig, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0, 0].set_title('Ambiguous Bx')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Ambiguous By
    im1 = axes[0, 1].imshow(by_ambig, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0, 1].set_title('Ambiguous By')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Bz (LOS)
    im2 = axes[0, 2].imshow(bz, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[0, 2].set_title('Bz (LOS)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Run disambiguation
    print("Running SFQ disambiguation...")
    bx_result, by_result = sfq_disambig(bx_ambig.copy(), by_ambig.copy(), bz, pos, rsun, silent=False)
    print()

    # Disambiguated Bx
    im3 = axes[1, 0].imshow(bx_result, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[1, 0].set_title('Disambiguated Bx')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # Disambiguated By
    im4 = axes[1, 1].imshow(by_result, cmap='RdBu_r', origin='lower', aspect='auto')
    axes[1, 1].set_title('Disambiguated By')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Error map (disambiguated vs true)
    # Account for possible global 180 flip
    err1 = np.sum((bx_result - bx_true) ** 2 + (by_result - by_true) ** 2)
    err2 = np.sum((bx_result + bx_true) ** 2 + (by_result + by_true) ** 2)
    if err2 < err1:
        bx_result_display = -bx_result
        by_result_display = -by_result
    else:
        bx_result_display = bx_result
        by_result_display = by_result

    error = np.sqrt((bx_result_display - bx_true) ** 2 + (by_result_display - by_true) ** 2)
    im5 = axes[1, 2].imshow(error, cmap='hot', origin='lower', aspect='auto')
    axes[1, 2].set_title('Reconstruction Error')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig('sfq_example_result.png', dpi=150, bbox_inches='tight')
    print("Results saved to sfq_example_result.png")

    # Print accuracy metrics
    mask = (bx_true ** 2 + by_true ** 2) > 100  # Only where field is strong
    if np.any(mask):
        angle_error = np.degrees(np.arctan2(
            bx_true[mask] * by_result_display[mask] - by_true[mask] * bx_result_display[mask],
            bx_true[mask] * bx_result_display[mask] + by_true[mask] * by_result_display[mask]
        ))
        print(f"Mean angle error (strong field pixels): {np.mean(np.abs(angle_error)):.2f} degrees")
        print(f"Median angle error: {np.median(np.abs(angle_error)):.2f} degrees")
        correct = np.abs(angle_error) < 45
        print(f"Pixels within 45 degrees: {np.mean(correct) * 100:.1f}%")


if __name__ == '__main__':
    main()

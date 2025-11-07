

## The **Point Spread Function (PSF)** 

It describes how a single point of light spreads in an imaging system.  
In this project:

1. A **motion blur PSF** is generated (based on a defined blur length and angle).
2. The PSF is applied to an input image in the frequency domain.
3. **Gaussian noise** is added.
4. The blurred image is **deconvolved** using the **Wiener filter**.
5. All results are saved and displayed.

---

## Example Outputs

|  | PSF Applied | After Deconvolution |
|-------------|--------------|---------------------|
|  | ![Blurred](https://github.com/0xmnshai/Point-Spread-Function/blob/main/blurred.png) | ![Deconvoled](https://github.com/0xmnshai/Point-Spread-Function/blob/main/deconvoled.png) |

### PSF Visualization on Image:
![PSF Image](https://github.com/0xmnshai/Point-Spread-Function/blob/main/psf_image.png)

---


##  Build & Run

### Prerequisites
- OpenCV 4.x
- g++ with C++17 support

### Compile
```bash
g++ main.cpp -o psf_main `pkg-config --cflags --libs opencv4` -std=c++17

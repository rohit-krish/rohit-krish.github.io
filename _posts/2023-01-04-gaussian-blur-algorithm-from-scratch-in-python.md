---
date: 2023-01-04 16:02:00 +05:30
categories: [Explain, Computer Vision]
tags: [Gaussian Blur, Python, Computer Vision, Mathematics, Coding, OpenCV]
author: rohit
description: >-
    Here We will be discussing about image filters, convolution, etc.. along with the Python implementation.
    As well as, learn to use OpenCV for the same task for real-world applications.
image:
    path: assets/posts/gaussian/banner.jpg
    lqip: data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwC7q3iJ7iAoOtZ2k69NbuRIeKKK3SRy3uf/2Q==
math: true
---

<!-- *Before going into depth, let me make this clear “coding from scratch” means Which is coding the underlying concept without just calling some custom function and be it, rather you are going to be exposed to what is truly happening.* -->

## Gaussian Filter

> **What is a gaussian filter and why do we use it?** <br>
> In digital signal processing, a Gaussian filter is a filter whose impulse response is a Gaussian function.

If that definition is quite a mouthful, let me clear it that for you.

Do you every come across with the term **Normal Distribution, Bell-shaped Curve or Standard Distribution** all of these are the same as **Gaussian distribution**.
So i hope you get the point that, it is that much important in the STEM field.

The equation for the gaussian distribution in 1D is,

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} (\frac{x-\mu}{\sigma})^2}
$$


The equation for the gaussian distribution in 2D (has two variables _x_ and _y_) is,

$$
f(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{( x^2 + y^2 )}{2\sigma^2}}
$$

---

<center>
<div>
$ \sigma $ : standard deviation
</div>
<div>
$ \mu $ : mean of the data
</div>
</center>

---
<br>

You can use the below deptiction to visualize the 1D and 2D Gaussian distribution/kernel.

![depiction of the steps of getting gaussian kernel matrix from it’s 1D plot](assets/posts/gaussian/image1-dark.jpg){: .dark lqip="data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDymwKGUb+lbwjtDAT3oopgf//Z"}
![depiction of the steps of getting gaussian kernel matrix from it’s 1D plot](assets/posts/gaussian/image1-light.jpg){: .light lqip="data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3O4JEZK9awbi4lE4zRRQB/9k="}
_depiction of the steps of getting gaussian kernel matrix from it’s 1D plot_


> There are many filters such as box filter, mean filter, median filter, etc… but in all filters available for this operation gaussian filter stands out, that is because it preserves a bit more details when you compare for example the box filter.


![using box filter v/s gaussian filter](assets/posts/gaussian/image2-dark.jpg){: .dark lqip="data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDl4bm1CCplurXNFFWSf//Z"}
![using box filter v/s gaussian filter](assets/posts/gaussian/image2-light.jpg){: .light lqip="data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDup573zDjpULT3u2iirJP/2Q=="}
_Photo by [Christopher Campbell](https://unsplash.com/@chrisjoelcampbell?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/shallow-focus-photography-of-woman-outdoor-during-day-rDEOVtE7vOs?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) [Edited]_


> There is also another filter called **Bilateral Filter** which performs better than gaussian, but it's computationaly a bit expensive, parhaps we will cover that in another article.

---

So now you know all about gaussian kernel,

before getting into the code, you have to also understand, The Convolution Operation.

## Convolution

Convolution is a matrix operation, which is used to transform a matrix with a given kernel. **It is the heart of the Convolutional Neural Networks (CNNs)**

![animation of how the convolution operation is performed](assets/posts/gaussian/image3-dark.gif){: .dark lqip="data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gAoR0lGIGVkaXRlZCB3aXRoIGh0dHBzOi8vZXpnaWYuY29tL2Nyb3D/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwBfFevW2vqsEZ5Ncr5MPh51eQZ3c0UUoRSiKo7s/9k="}
![animation of how the convolution operation is performed](assets/posts/gaussian/image3-light.gif){: .light lqip="data:image/jpg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gAoR0lGIGVkaXRlZCB3aXRoIGh0dHBzOi8vZXpnaWYuY29tL2Nyb3D/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD2axtlgOQKtlGmaiiphFKIprWx/9k="}
_animation of how the convolution operation is performed_

In the above depiction,

- Left most matrix is the input data
- The matrix in the center is a kernel
- In the rightmost matrix is the output of the convolution operation performed on the input data with the given filter/kernel

Each cell in the output is been calculated by taking the element-wise multiplication (Hadamard product) of the kernel/filter with its corresponding sub-matrix of the input data, and taking the total summation of it.
The sub-matrix will change in each iteration with the given stride.

_Some terms related to this operation,_

### Padding

As the name implies it is the level of empty/zero data point added in the border of the input data, Which is used to give the corner data points more role in the computation. In the above depiction padding was zero.

### Strides

It is the length of the jump in taking the next sub-matrix of the input data. In the above depiction, stride was one.

### Cross-Correlation

**The operation that I explained yet is Cross-Correlation “not Convolution”**, but before putting the full stop, let me make it clear,

if you rotate the given kernel/filter 180° and perform the explained operation, that is Convolution, but in our context, the rotation of the kernel doesn’t matter so both can be considered as the same operation (but not really)

I’m using the term convolution because, it is a bit fancier and more famous than "Cross-Correlation" (i think).

> In our context, the input data is the image that we are giving and kernel/filter is the gaussian kernel (obviously).

**The procedure is to perform convolution operation on an image with the gaussian kernel matrix, which results in a blurred image of the corresponding given image.**

_Ta daa... roll the drums 🥁, so now we can get into the code._

---

{% include adsense/in-article.html %}

## Code

### Gaussian Filter

```python
import numpy as np


def generate_gaussian_filter(sigma: int | float, filter_shape: list | tuple):
    # 'sigma' is the standard deviation of the gaussian distribution

    m, n = filter_shape
    m_half = m // 2
    n_half = n // 2

    # initializing the filter
    gaussian_filter = np.zeros((m, n), np.float32)

    # generating the filter
    for y in range(-m_half, m_half):
        for x in range(-n_half, n_half):
            normal = 1 / (2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))
            gaussian_filter[y+m_half, x+n_half] = normal * exp_term
```
{: file="gaussian_filter.py" }

### Convolution Operation

```python
import numpy as np


def convolution(image: np.ndarray, kernel: list | tuple) -> np.ndarray:
    '''
    It is a "valid" Convolution algorithm implementaion.
    ### Example
    >>> import numpy as np
    >>> from PIL import Image
    >>>
    >>> kernel = np.array(
    >>>     [[-1, 0, 1],
    >>>     [-2, 0, 2],
    >>>     [-1, 0, 1]], np.float32
    >>> )
    >>> img = np.array(Image.open('./lenna.png'))
    >>> res = convolution(img, Kx)
    '''

    if len(image.shape) == 3:
        m_i, n_i, c_i = image.shape

    # if the image is gray then we won't be having an extra channel so handling it
    elif len(image.shape) == 2:
        image = image[..., np.newaxis]
        m_i, n_i, c_i = image.shape
    else:
        raise Exception('Shape of image not supported')

    m_k, n_k = kernel.shape

    y_strides = m_i - m_k + 1  # possible number of strides in y direction
    x_strides = n_i - n_k + 1  # possible number of strides in x direction

    img = image.copy()
    output_shape = (m_i-m_k+1, n_i-n_k+1, c_i)
    output = np.zeros(output_shape, dtype=np.float32)

    count = 0  # taking count of the convolution operation being happening

    output_tmp = output.reshape(
        (output_shape[0]*output_shape[1], output_shape[2])
    )

    for i in range(y_strides):
        for j in range(x_strides):
            for c in range(c_i): # looping over the all channels
                sub_matrix = img[i:i+m_k, j:j+n_k, c]

                output_tmp[count, c] = np.sum(sub_matrix * kernel)

            count += 1

    output = output_tmp.reshape(output_shape)

    return output
```
{: file="convolution_operation_implementation.py" }

The above code is not in anyway optimized or anyting, it is just for teaching purposes, i prefer using [`cv2.filter2D`](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04), [`scipy.signal.correlate2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html) or [`scipy.signal.convolve2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)

> For the complete code refer to my [GitHub Repo](https://github.com/rohit-krish/CVFS)

## Conclusion

I hope you understood how the image blur operation is actually perform under the hood.
Its not a bad practice to code this from scratch,
It actually helps you understand things far better than just an explanation.
But when you work on real-world projects you don’t have to code from scratch, then you can use libraries such as OpenCV for your help.

In Python using OpenCV, you can generate a gaussian blurred image as below,

```python
import cv2
img = cv2.imread('<path_to_image>')
imgBlur = cv2.GaussianBlur(img, (9, 9), 3) # img, kernel_size, sigma
```

## References

- [https://en.wikipedia.org/wiki/Gaussian_blur](https://en.wikipedia.org/wiki/Gaussian_blur)
- [https://youtu.be/C_zFhWdM4ic](https://youtu.be/C_zFhWdM4ic)
- Banner Image by <a href="https://unsplash.com/@dimhou?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Dim Hou</a> on <a href="https://unsplash.com/photos/black-and-white-cat-on-gray-concrete-floor-93AcQpzcASE?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a> [Edited]

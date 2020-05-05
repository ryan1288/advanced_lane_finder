# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

All the steps above are shown in order individually in the Python Jupyter Notebook in "/project_2_advanced_lane_finder.ipynb". 

[//]: # (Image References)

[image1]: report_images/camera_pre.jpg
[image2]: report_images/camera_post.jpg
[image3]: report_images/distort_pre.jpg
[image4]: report_images/distort_post.jpg
[image5]: report_images/image.jpg
[image6]: report_images/image_blur.jpg
[image7]: report_images/image_red_green.jpg
[image8]: report_images/image_combined.jpg
[image9]: report_images/perspective_pre.jpg
[image10]: report_images/perspective_post.jpg
[image11]: report_images/lane_detect.jpg
[image12]: report_images/final_image.jpg
[video1]: report_images/project_vid.mp4

## Camera Calibration using Chessboard Images
Before calibrating, libraries are imported as shown in the first code cell. Including numpy (math), OpenCV (computer vision), glob (grouped images IO), matplotlib (plotting), moviepy (movie IO), and IPython (for in-text video display).

The code for the camera calibration step is in the second code cell of the notebook. I start by preparing object pointers that correspond to the checkerboard pattern used to calibrate the camera (in this case, 9x6 internal vertices). 

Arrays are created for both the object points (x, y, z) and image points (x, y), which are appended to on each successful iteration of images imported using `glob`. 

After iterating through all the checkerboard images, I used the output `objpoints` and `imgpoints` arrays to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()`. 

I used the `cv2.undistort()` function on the test image with the calibration and distortion coefficients and obtained this result:

![][image1]
![][image2]

## Lane Detection Pipeline

### 1. Distortion Correction
Using the same calibration and distortion coefficients and re-applying to each image in the pipeline, the images captured through the same camera were also corrected for distortion as shown below (code block 3 in the notebook):

![][image3]
![][image4]

### 2. Color/Gradient Thresholds with Gaussian Blur
After applying the Gaussian blur using the `cv2.GaussianBlur()` function with a kernel size of 7, both color and gradient thresholding was applied. This was done using the HLS channels as the RGB color channels are insufficient to detect yellow lane markings under shadows and environmental noise. This pipeline step is shown in code block 4.

The saturation was filtered to include only pixels that had above 190/255 and the gradient threshold contained points between (18, 100) inclusive. Afterwards, both thresholding techniques were used concurrently, shown below as red (saturation) and green (gradient).

Original:
![][image5]
Gaussian Blur:
![][image6]
Thresholds:
![][image7]
Combined:
![][image8]

### 3. Perspective Transform (Bird's Eye View)
The perspective transform was done using the `cv2.getPerspectiveTransform()` function. This allows the linear interpolation of points and performing matrix transformations to change it to a flat surface for simpler analysis.

I implemented this using a central offset from the center of the image (90 pixels) shown in code block 5. Using the width and height of the image, the source (`src`) and destination (`dst`) points were hard-coded in like below:

```python
    h = img.shape[0]
    w = img.shape[1]
    img_size = (w,h)
    mid_offset = 90
    # Top left, top right, bottom left, bottom right
    src = np.float32([[w/2-mid_offset, 460], 
                      [w/2+mid_offset, 460], 
                      [0, h-15], 
                      [w, h-15]])
    dst = np.float32([[0, 0], 
                      [w, 0], 
                      [0, h], 
                      [w, h]])
```

This resulted in the following source and destination points:

| Source      | Destination | 
|:-----------:|:-----------:| 
| 570, 460    | 0, 0        | 
| 730, 460    | 1280, 0     |
| 0, 705      | 0, 720      |
| 1280, 705   | 1280, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Before Transform:
![][image9]
After Transform:
![][image10]


### 4. Land Boundary Detection and Vehicle Position
Lane boundary detection is the most complex part of the pipeline and this was broken down into four different functions under this step: top-level function `detect_lane()` (incl. visuals creation), polynomial fitting `fit_poly()`, window histogram `windowHistograms()`, and the detection around the previous polynomial `searchAroundPoly()`. The vehicle position calculation is embedded into the top-level function. This code is displayed in code block 6 to 9.

First, the global left_fit and right_fit parameters are checked to see if it's the first iteration, which chooses whether the historgram windows methods is used or if the points are searched around the previous frame's polynomial.

Using 9 windows, a margin of 80 pixels on each side, and a minimum pixel count of 30 per window, the peaks of the histograms were used to determine the position of the lane. Then, the lane indices that match the requirement and are within the window are concatenated to a list of left and right indices individually and returned for polynomial fitting.

Similarly for searching around the previous polynomial, the fit is determined using new pixels found around the old polynomial with a +_60 pixels margin. 

The polynomial fitting uses `np.polyfit()` with order two and the previously found left and right indices that contain the lane points.

The lane offset is calculated as below, where the center of the image in the x-direction is 640 pixels, and the left/right fits indicate the bottom of the polynomial nearest to the car. The result is multiplied with the meters per pixel estimate:
```python
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    offset = (640 - (left_fitx[-1] + right_fitx[-1])/2)*xm_per_pix
```

Finally, the visuals are created using a fixed pixel margin on each side of the fitted polynomial (30 pixels on each side). In between, the green area dictates the safe area for the vehicle as part of the lane. These are all plotted using `cv2.fillPoly()` and then returned for the inverse image transform.

An example of the final result is shown below:

![][image11]

### 5. Lane Curvature
The lane curvature is calculated using the left/right fit provided in meters shown in code block 10 in the notebook.

Using the curve as `f(y) = Ay^2 + By + C`, the curvature can be estimated as: 
$$r = (1+(2Ay+B)^2)^(3/2)/|2A|$$

The polynomial is a function of y to avoid vertical lines that will have infinite slope as a f(x). The values were checked against US road curvature standards, and the magnitude is on the same order.

### 6. Inverse Perspective Transform and Text/Visuals Overlay
The inverse transform (code block 11) and text overlay (code block 12) is completed before putting the visual overtop the original unprocessed image for each frame in the video.

The destination `dst` and source `src` are flipped in the `cv2.getPerspectiveTransform(0` function before `cv.warpPerspective()` to transform the lane back to the original shape.

The text is overlaid using the `cv2.putText()` function with specified parameters shown in the code. The text is adjusted to have positive positioning reference to either the left or right about the center of the lane (assuming the vehicle is at centered with the iamge).

A sample output is shown below:
![][image12]


## Final Video Output
See _ for a video link to my project video using the pipeline described above.
[Project Video](https://youtu.be/SRAHh0PokSw)


## Discussion

### Problems/Issues in Project Implementation
1. The biggest challenge for me was determining the optimal parameters that allowed the vehicle to succeed under shadow and other drastic color changes on the road. Although the HLS performs better than the RGB spectrum for lane detection, it still contained many false negatives that could skew the final result.

2. Related to the first point, both the binarization and transformation of the lines are very important, as a poor transformation can both skew and blur the lines, leading to an increased likelihood of failure. Robust binarization is essential to maintaining proper lane detection even with shadows and environmental noise.

3. One weak point is the splitting in the middle for detecting the lanes, showing that lane switches will create invalid lane boundary estimates for a short period of time that could be catastrophic is implemented.

4. Another weakness is the reliance on the mask to include all the needed lane points, as different videos can go out of range, leading to failure.

### Potential Areas for Improvement
1. Further testing and optimization of parameters (within transformations, binarization, and lane detection) could be done to minimize fidgeting and improve robustness.

2. Combining more color channel thresholds while being more strict with each individual one could filter out noise. 

3. Increasing use of previous data to improve the next estimate (in addition to the carry-forward polynomial) can make estimates further accurate and weigh more possible lane indices more heavily to filter out noise.

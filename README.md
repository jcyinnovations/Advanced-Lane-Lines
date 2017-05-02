## Advanced Lane Lines Project Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_image.png "Undistorted"
[image2]: ./output_images/test_undistorted.png "Road Transformed"
[image3]: ./output_images/thresholded.png "Binary Example"
[image4]: ./output_images/source_points_and_warped.png "Warp Example"
[image4A]: ./output_images/source_points_and_warped2.png "Warp Example"
[image5]: ./output_images/mapped_lane_lines.png "Fit Visual"
[image6]: ./output_images/final_output.png "Output"
[video1]: https://youtu.be/mk9DIaF3y_A  "Video"

---

### Camera Calibration

#### Computing the camera matrix and distortion coefficients.

The code for this step is contained in the `calibration_camera_from_folder()` function located at lines 75 to 110 of `CameraOperations.py` and it is executed in the first code cell of the IPython notebook located in `./Advanced_Lane_Lines_Project.ipynb`.  

This function iterates over all the calibration chessboard images in the `camera_cal` folder, for each image, computes the distortion parameters and appends them to `imgpoints` if the chessboard was detected correctly. Once all of the images are processed and the `imgpoints` built, `cv2.calibrateCamera()` is used to calibrate the camera with the full `imgpoints` array (from all images).

I then used `cv2.projectPoints()` to estimate the error contribution for each calibration image.  

After calibration, I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Processing Pipeline (single images)

#### Sample corrected images

Images are corrected for distortion using the `correct_distortion()` function at lines 113 to 118 of `CameraOperations.py`. This function uses `cv2.undistort()` to correct the source image. The results are show below:

![alt text][image2]
#### Process description.

Image thresholding is done using the `Road.image_threshold()` method of the `Road` class located at lines 297 to 309 of `LaneLines.py`. This method applies the `sobel_and_gradient()` method to the S (Channel 2 from HLS version of image) and R (channel 0 from RGB version of image) channels of the source image. The `Road.sobel_and_gradient()` method located at lines 265 to 292, combines x, y gradients with gradient magnitude and direction to produce the final binary output for each channel.

The selection of Saturation and Red channels was found to be optimal after investigating several combinations of channels. A future refinement of the method would look at both the Luminance and Saturation channels of the HLS colorspace and select the optimal channel to use based on the level of noise on each side of the image. An initial investigation into this approach found that counting the percentage of dark pixels in the convolutional signal was a good indicator of the level of noise in each half of the image. The Luminance channel seems to perform much better than Saturation in images.

![alt text][image3]

#### Perspective Transformation.

The code for my perspective transform includes a method of the Road object called `perspective_transform()`, which appears in lines 204 through 249 in the file `LaneLines.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `perspective_transform()` function takes as inputs an image (`image`), and a `reverse` boolean flag. The source (`src`) points were derived from analysing an undistorted image in GIMP and saved in the `viewport` Road class property at line 110. The destination (`dst`) points are derived in the method itself. The `reverse` flag allows this method to reverse the warping and lay the mapped lane lines back onto the source image. I chose the setup the source and destination points in the following manner:

```
h = image.shape[0] # Height
w = h = image.shape[1] # Width
offset = 80
viewport = [[540,465],[740,465],[1280,720],[0,720]]

src = np.float32(Road.viewport)

dst = np.float32([[self.offset,0],[w-self.offset,0],[w-self.offset,h],[self.offset,h]])    

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 540, 465      | 80, 0        |
| 740, 465      | 1200, 0      |
| 1280, 720     | 1200, 720    |
| 0, 720        | 0, 720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

![alt text][image4A]

#### Lane line identification and  polynomial fitting

The process of extracting lane lines is encapsulated in the `Road.parse_image()` method of the Road class defined at lines 313 to 336 of `LaneLines.py`. This method uses the `Road.image_threshold()`, `Road.perspective_transform()` and `Road.find_window_centroids()` (lines 168 to 236 of `LaneLines.py`) methods of that same class in sequence to convert the image to binary, warp the image, and detect the lane lines respectively.

If the lane lines are detected in the current image and they pass a basic sanity check (`Road.sanity_check()` at lines 139 to 150 of `LaneLines.py`), they are committed to two `Line` classes; one for each lane using the `Line.update()` method of the `Line` class (lines 45 to 75). If the sanity check fails, the update is skipped and only the recent fit parameters of each line are updated.

If the sanity check passes, the starting point of the found lines are used as the starting point for detection for the next frame (`Road.find_window_centroids()` method, lines 189 to 197).

Finally, the `Line.update()` function updates the running average of detected lines and fits a second order polynomial to the detected points (lines 53 to 65 of `LaneLines.py`). This function also calculates the curvature of the lines based on the best fit (running average of the lines).

Here is an example of the warped thresholded image and the mapped lane lines on that image:

![alt text][image5]

#### Calculating lane radius of curvature and vehicle position with respect to center.

Radius of curvature is calculated in `Line.update()` at lines 67 to 75 of `LaneLines.py`. The vehicle position is calculated in `Road.position_label()` at lines 383 to 394 of `LaneLines.py`.

#### Sample lane plots.

The `Road.draw_overlay()` method at lines 341 to 361 of `LaneLines.py` draws an overlay of the detected lane lines uses `Road.perspective_transform( reverse=True)` to lay the overlay onto the original image. This also generates and overlays the radius of curvature and car position onto the original image.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### Pipeline processed video.

Here's a [link to my video result] https://youtu.be/mk9DIaF3y_A

---

### Discussion

#### Issues and future improvements

The video pipeline is executed in the `Advanced_Lane_Lines_Project.ipynb` ipython notebook's last cell. The approach is very simple: the Road class is instantiated, then each frame of video is processed in `make_frame_lane_markings()`. For each frame, first the correct_distortion() function is called to undistort it, then `Road.parse_image()` is called to do the lane line mapping. Finally `Road.draw_overlay()` is called to map the found lane lines to the original frame. Optionally, in DEBUG mode, the frame making function will emit a series of images at various stages of the pipeline. This helped with debugging issues with the pipeline at various parts of the video.

The pipeline works very well for the `project_video.mp4` but fails on the harder images. Given more time for the project, I would make two changes: reduce the height of the `src` (how far it projects into the image) to better process sharp corners, change the `perspective_transform()` function to crop to the src but then pad back to full image size so that the lines are pushed closer to the center of the viewport. Lastly, the `find_window_centroids()` function needs some tweaking to handle empty image sections better. I implemented a threshold (lower bound) for what constitutes an empty window and code to manage empty windwows  (lines 217 to 220 and 229 to 234 of `LaneLines.py`) but the current threshold is set to 0. Statistical analysis of the project videos would come up with a much better value for this threshold.

In addition to the convolution signal thresholding, I would also implement dark pixel counting to identify whether the current window is too noisy to be a valid lane line. This combined with switching to Luminance channel for shaded areas (found to be better for those areas than saturation) would improve detection on the harder project videos.

Here is some sample code tested for this analysis:

```
l_sum = np.sum(image[int(3*h/4):,:], axis=0)
l_conv_signal = np.convolve(window, l_sum)

#what pixels are non zero in the convolution signal
indicies = np.arange(len(l_conv_signal))
white_count = len( np.extract((l_conv_signal > 0) , indicies) )
pixel_count = len(l_conv_signal)

percentage_dark = 1 - white_count/pixel_count

```

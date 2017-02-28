#################################################################
#################################################################
## Lane Lines: Find lane lines
#################################################################
#################################################################

from CameraOperations import perspective_transform, sobel_LS_threshold
from ImageProcessing import gaussian_blur
import numpy as np
import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


##############################################################################################################
# Tracking detected line characteristics
##############################################################################################################
class Line():
    N = 5                   # Retain the last N measurements
    road_width_pixels = 795
    lane_length_pixels = 115
    ym_per_pix = 3/lane_length_pixels      # Conversion from pixels to meters in y
    xm_per_pix = 3.7/road_width_pixels    # Conversion from pixels to meters in x

    
    def __init__(self, w, h):
        self.detected = False  # was the line detected in the last iteration?
        self.recent_xfitted = [] # x values of the last n fits of the line
        self.bestx = None #average x values of the fitted line over the last n iterations
        self.best_fit = None    #polynomial coefficients averaged over the last n iterations
        self.current_fit = None #polynomial coefficients for the most recent fit
        self.radius_of_curvature = 0.0  #radius of curvature of the line in some units
        self.line_base_pos = 0.0  #distance in meters of vehicle center from the line
        self.diffs = np.array([0,0,0], dtype='float') #difference in fit coefficients between last and new fits
        self.detected_x = None  #x values for detected line pixels
        self.detected_y = None #y values for detected line pixels
        self.h = h
        self.w = w
        self.y = np.array([p for p in range(0,h+1,10)]) # y Coordinates for generated lane lines
        self.y_m = self.y * Line.ym_per_pix             # y Coordinates converted to meters
    
    
    # Update line with new parameters
    def update(self, detected_x, detected_y):
        # Update Current Fit
        self.detected_x = detected_x
        self.detected_y = detected_y
        self.detected = True
        self.current_fit = np.polyfit(detected_y, detected_x, 2)
        x = self.current_fit[0]*self.y**2 + self.current_fit[1]*self.y + self.current_fit[2]
        # Update Best Fit
        if len(self.recent_xfitted) < Line.N:
            self.recent_xfitted.append(x)
        else:
            del self.recent_xfitted[0]
            self.recent_xfitted.append(x)
        
        if len(self.recent_xfitted) > 1:
            self.bestx = np.mean( np.array(self.recent_xfitted) ,axis=0)
        else:
            self.bestx = x
        self.best_fit = np.polyfit(self.y, self.bestx, 2)
        # distance from vehicle center
        self.line_base_pos = math.fabs( self.w/2 - (self.best_fit[0]*self.h**2 + self.best_fit[1]*self.h + self.best_fit[2]) )*Line.xm_per_pix
        
        # Calculate radius
        best_fit_m = np.polyfit(self.y_m, self.bestx*Line.xm_per_pix, 2)
        y_m = self.h*Line.ym_per_pix
        A = best_fit_m[0]
        B = best_fit_m[1]
        if math.fabs(A) > 0:
            self.radius_of_curvature =  ((1 + (2*A*y_m+B)**2)**1.5)/math.fabs(2*A)
        else:
            self.radius_of_curvature = math.inf

            
    # Generate line coordinates for plotting lane
    def fit_line(self):
        if self.best_fit is None:
            coeff = self.current_fit
        else:
            coeff = self.best_fit
        #y = np.array([p for p in range(0,h+1,10)])
        x = coeff[0]*self.y**2 + coeff[1]*self.y + coeff[2]
        pts = np.column_stack((x, self.y)).astype("int32") 
        return pts
    
    # Calculate the starting point for the next detection
    def get_starting_point(self, start_y):
        if self.best_fit is None:
            coeff = self.current_fit
        else:
            coeff = self.best_fit
        start_x = coeff[0]*start_y**2 + coeff[1]*start_y + coeff[2]
        return start_x
    
    
    # Class string for debugging
    def __str__(self):
        return "Radius: {0:.1f}, Distance from center: {1:.2f}m".format(self.radius_of_curvature, self.line_base_pos)
        
    def __repr__(self):
        return self.__str__()
    
##############################################################################################################
# Tracking the road definition as per the line detection algorithms
##############################################################################################################
class Road(object):
    viewport = [[540,465],[740,465],[1280,720],[0,720]]     #Original
    #viewport = [[488,490],[792,490],[1280,720],[0,720]] 
    #viewport = [[570,450],[710,450],[1280,720],[0,720]]    # New 2
    #viewport = [[574,450],[706,450],[1280,720],[0,720]]
    #viewport = [[593,449],[691,449],[1080,720],[204,720]]  # New
    

    def __init__(self, window_height=48, window_width=40, margin=40, w=1280, h=720):
        self.left_lane = Line(w, h)
        self.right_lane = Line(w, h)
        self.radius_of_curvature = None
        self.sobel_thresh=(30, 150)
        self.gradient_thresh=(0.7, 1.3)
        self.sobel_kernel=9
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.offset = 320
        self.shift_tolerance = 0.05             # Tolerance in lane shift
        self.lane_tolerance = 0.10              # Tolerance in lane width
        self.threshold = 0.0                    # minimum weight of found centroid. Any centroids at or under are rejected.
        # Optimzation: Create the yfit once to avoid the overhead of recreating each image parse
        self.y_fit = np.array([s for s in range(int(h-self.window_height/2), 0, -self.window_height)])
        self.y_fit = np.insert(self.y_fit, 0,720) # y positions of detected centroids
        self.w = w
        self.h = h

        
    #Check that the most recent lines are valid
    def sanity_checked(self, l_detected_x, r_detected_x):
        #Checking that they have similar curvature
        # TODO: Can't seem to get consistent curvatures so this check would reject most lines
        #Checking that they are roughly parallel
        separation = r_detected_x - l_detected_x
        avg_separation = np.mean(separation)
        max_separation = np.max(separation)
        min_separation = np.min(separation)
        lines_parallel = (max_separation/avg_separation - 1) < self.lane_tolerance and (1 - min_separation/avg_separation) < self.lane_tolerance
        #Checking that they are separated by approximately the right distance horizontally
        separation_good = math.fabs(avg_separation - Line.road_width_pixels)/Line.road_width_pixels < self.lane_tolerance                          
        return lines_parallel and separation_good
        
        
    # Find the peak location of a signal
    def signal_peak(self, signal):
        if signal is None or len(signal) < 2:
            return 0
            
        idx_start = np.argmax(signal)
        idx_end = len(signal)-np.argmax(signal[::-1])-1
        #print(idx_start, idx_end, signal[idx_start], signal[idx_end])
        if idx_end > idx_start: # Sanity check
            idx = int((idx_start + idx_end)/2)
        else:
            idx = idx_start
        return idx
        
    # Locate the lane lines
    def find_window_centroids(self, image):
        w = image.shape[1]
        h = image.shape[0]  

        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = self.window_width/2
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width) # Create our window template that we will use for convolutions
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        # Sum quarter bottom of image to get slice, could use a different ratio
        if self.left_lane.detected:
            # Last detection was valid so use it as starting point
            l_center = self.left_lane.get_starting_point(h)
        else:
            # Search again
            l_sum = np.sum(image[int(3*h/5):,:int(w/2)], axis=0)
            l_conv_signal = np.convolve(window, l_sum)
            l_idx = self.signal_peak(l_conv_signal)
            l_center = l_idx - offset    
        
        if self.right_lane.detected:
            # Last detection was valid so use it as starting point
            r_center = self.right_lane.get_starting_point(h)
        else:
            # Search again
            r_sum = np.sum(image[int(3*h/5):,int(w/2):], axis=0)
            r_conv_signal = np.convolve(window, r_sum)
            r_idx = self.signal_peak(r_conv_signal)
            r_center = int(w/2) + r_idx - offset 
        
        window_centroids.append((l_center, r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(1+h/self.window_height)):
            ############## INIT
            l_center_old = l_center
            r_center_old = r_center
            image_layer = np.sum(image[int(h-(level)*self.window_height):int(h-(level-1)*self.window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            
            ############## LEFT LINES
            l_min_index = int(max(l_center+offset-self.margin, 0))
            l_max_index = int(min(l_center+offset+self.margin, w))
            #l_conv_signal = conv_signal[:int(w/2)]
            l_conv_signal = conv_signal[l_min_index:l_max_index]
            l_idx = self.signal_peak(l_conv_signal)
            # Reject invalid centroids (noise or empty space) and keep the previous value
            if l_conv_signal[l_idx] > self.threshold:
                l_center = l_idx + l_min_index - offset
            else:
                l_center = l_center_old
                
            ############## RIGHT LINES
            r_min_index = int(max(r_center+offset-self.margin, 0))
            r_max_index = int(min(r_center+offset+self.margin, w))
            #r_conv_signal = conv_signal[int(w/2):]
            r_conv_signal = conv_signal[r_min_index:r_max_index]
            r_idx = self.signal_peak(r_conv_signal)
            # Reject invalid centroids (noise or empty space) and keep the previous value
            if r_conv_signal[r_idx] > self.threshold:
                r_center = r_min_index + r_idx - offset
            else:
                r_center = r_center_old
            # Update detected lines list    
            window_centroids.append((l_center, r_center))
            
        return window_centroids

        
    #Perspective Transform: extract lanes and view from above
    def perspective_transform(self, image, reverse=False):
        w, h = image.shape[1], image.shape[0]
        src = np.float32(Road.viewport)
        dst = np.float32([[self.offset,0],[w-self.offset,0],[w-self.offset,h],[self.offset,h]])

        M = cv2.getPerspectiveTransform(src, dst)
        if reverse:
            M = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(image, M, (w,h), flags=cv2.INTER_LINEAR)
        return warped

    ##########################################################################
    # Sharpen images using a simple filter since a lot of images are blurred
    ##########################################################################
    def image_sharpen(self, img):
        #Create the filter kernel
        kernel = np.zeros( (9,9), np.float32)
        kernel[4,4] = 2.0
        boxFilter = np.ones((9,9), np.float32)/81.0
        kernel = kernel - boxFilter
        #Apply the filter
        return cv2.filter2D(img, -1, kernel)
        
        
    # Combined Sobel plus Gradient Magnitude and direction Thresholding
    def sobel_and_gradient(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel))
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel))
        sobelmag = np.sqrt( np.square(sobelx) + np.square(sobely) )

        gradient = np.arctan2(sobely, sobelx)
        dir_binary = np.zeros_like(gradient).astype("uint8")
        dir_binary[(gradient >= self.gradient_thresh[0]) & (gradient <= self.gradient_thresh[1])] = 1

        scaled_sobelx = np.uint8(255*sobelx/np.max(sobelx))
        gradx = np.zeros_like(scaled_sobelx)
        gradx[(scaled_sobelx >= self.sobel_thresh[0]) & (scaled_sobelx <= self.sobel_thresh[1])] = 1

        scaled_sobely = np.uint8(255*sobely/np.max(sobely))
        grady = np.zeros_like(scaled_sobely)
        grady[(scaled_sobely >= self.sobel_thresh[0]) & (scaled_sobely <= self.sobel_thresh[1])] = 1
        
        scaled_sobelmag = np.uint8(255*sobelmag/np.max(sobelmag))
        mag_binary = np.zeros_like(scaled_sobelmag)
        mag_binary[(scaled_sobelmag >= self.sobel_thresh[0]) & (scaled_sobelmag <= self.sobel_thresh[1])] = 1

        combined = np.zeros_like(dir_binary).astype("uint8")
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined

        
    # Gradient & Threshold on Luminosity and Saturation (HLS) channels
    # Luminosity performs better in shadow whereas Saturation performs better generally. 
    def image_threshold(self, image):
        #image = self.image_sharpen(image)
        r = image[:,:,0]
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l = hls[:,:,1]
        s = hls[:,:,2]
        s = self.sobel_and_gradient(s)
        #l = self.sobel_and_gradient(l)
        r = self.sobel_and_gradient(r)
        
        combined = np.zeros_like(s).astype("uint8")
        combined[((s == 1) | (r == 1))] = 1
        return combined
        
        
    # Parse image for lane lines
    def parse_image(self, image):
        mapped_lanes = None
        lane_markings = None
        w, h = image.shape[1], image.shape[0]
        img_thresholded = self.image_threshold(image)
        warped = self.perspective_transform(img_thresholded)
        window_centroids = self.find_window_centroids(255*warped)
        # Centroids found, fit to a polygon and display
        if len(window_centroids) > 0:
            centroids = np.array(window_centroids)
            l, r = centroids[:,0], centroids[:,1]
            # Update the lane lines fully if sanity checks pass
            if self.sanity_checked(l, r):
                self.left_lane.update(l, self.y_fit)
                self.right_lane.update(r, self.y_fit)
            else:
                #update current fit only
                self.left_lane.detected_x = l
                self.left_lane.detected = False
                self.left_lane.current_fit = np.polyfit(self.y_fit, l, 2)
                
                self.right_lane.detected_x = r
                self.right_lane.detected = False
                self.right_lane.current_fit = np.polyfit(self.y_fit, r, 2)
                
        return warped
         
    # Draw the lane overlay on the
    def draw_overlay(self, image, warped):
        w, h = image.shape[1], image.shape[0]
        l_pts = self.left_lane.fit_line()
        r_pts = self.right_lane.fit_line()
        
        # Paint the polynomial onto a canvas and overlay on image
        found_lines = np.zeros((h,w,3), np.uint8)
        lr_pts = np.append(l_pts,r_pts[::-1], axis=0)
        cv2.fillPoly(found_lines, [lr_pts], (0,255,0))
        cv2.polylines(found_lines,[l_pts], 0, (255,0,0),thickness=5)
        cv2.polylines(found_lines,[r_pts], 0, (255,0,0),thickness=5)
        original = cv2.cvtColor(255*warped, cv2.COLOR_GRAY2RGB)
        mapped_lanes = cv2.addWeighted(original, 0.5, found_lines, 0.5, 0)
        # Now lay the overlay back onto the original image
        overlay = self.perspective_transform(found_lines, reverse=True)
        
        # Annotate the image
        cv2.putText(overlay, self.radius_label(),(20,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),thickness=3)
        cv2.putText(overlay, self.position_label(),(20,120),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),thickness=3)
        lane_markings = cv2.addWeighted(image, 1, overlay, 0.75, 0)
        return lane_markings, mapped_lanes

        
    # Radius Annotation
    def radius_label(self):
        #print(self.left_lane.radius_of_curvature, self.right_lane.radius_of_curvature)
        if self.left_lane.radius_of_curvature is None:
            left = 0.0
        else:
            left = self.left_lane.radius_of_curvature
            
        if self.right_lane.radius_of_curvature is None:
            right = 0.0
        else:
            right = self.right_lane.radius_of_curvature
        radius = (self.left_lane.radius_of_curvature + self.right_lane.radius_of_curvature)/2
        #label = "Radius of Curvature = {0:.0f}, {1:.0f}".format(left, right)
        label = "Radius of Curvature = {0:.0f}".format(radius)
        return label
    
    
    # Vehicle Position Annotation
    def position_label(self):
        l_pos = self.left_lane.get_starting_point(self.h)
        r_pos = self.right_lane.get_starting_point(self.h)
        midpoint = (r_pos + l_pos)/2
        distance = (midpoint -  self.w/2) * Line.xm_per_pix
        #print(l_pos, r_pos, midpoint, distance)
        if distance > 0:
            direction = "right"
        else:
            direction = "left"
        label = "Vehicle is {0:.2f}m {1} of center".format(math.fabs(distance), direction)
        return label
    
    
    # Redraw camera frame with lane overlay and annotations
    def redraw_frame(self, src):
        image = None
        return image
        

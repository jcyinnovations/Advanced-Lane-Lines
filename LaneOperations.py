from CameraOperations import perspective_transform, sobel_LS_threshold
from ImageProcessing import gaussian_blur
import numpy as np
import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Conversions from pixel space to real space in meters
ym_per_pix = 3/138                 # meters per pixel in y dimension
xm_per_pix = 3.7/795               # meters per pixel in x dimension
shift_tolerance = 0.05             # Tolerance in lane shift
lane_tolerance = 0.10              #Tolerance in lane width
threshold = 0.0                    # minimum weight of found centroid. Any centroids at or under are rejected.

##############################################################################################################
# Find the peak location of a signal
##############################################################################################################
def signal_peak(signal):
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


##############################################################################################################
# Locate the center of each lane lines across multiple slices from bottom to top of image
##############################################################################################################
def find_window_centroids(image, window_width, window_height, margin, cache=None,DEBUG=False, DEBUG_ID=""):
    global shift_tolerance
    global threshold
    global ym_per_pix
    global xm_per_pix
    
    w = image.shape[1]
    h = image.shape[0]   

    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    offset = window_width/2
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*h/4):,:int(w/2)], axis=0)
    l_conv_signal = np.convolve(window, l_sum)
    l_idx = signal_peak(l_conv_signal)
    l_center = l_idx - offset    
    
    r_sum = np.sum(image[int(3*h/4):,int(w/2):], axis=0)
    r_conv_signal = np.convolve(window, r_sum)
    r_idx = signal_peak(r_conv_signal)
    r_center = int(w/2) + r_idx - offset 
    
    #Detected lane width
    lane_width = (r_center - l_center) * xm_per_pix
    delta = math.fabs(lane_width-3.7)/3.7
    #print("Lane_width", lane_width, delta)
        
    # Used the cached values (from previous frame) if the detected value differs too much 
    if cache is not None and cache['centers'] is not None:
        #print("Using cache...")
        #Use the cached values from the previous frame
        l_center_cached = cache['centers'][0]
        r_center_cached = cache['centers'][1]
        lane_width_cached = cache['width']
        delta_cached = math.fabs(lane_width_cached-3.7)/3.7
        # Use cached starting point or detected starting point
        # Use whichever is closest to the expected lane width
        if delta > delta_cached:
            l_center = l_center_cached
            r_center = r_center_cached   
    else:
        # No caching so try compensating with expected width
        if delta > lane_tolerance:
            #Fix the starting points
            lgap, rgap = l_center, w-r_center
            if lane_width < 795:
                if lgap > rgap:
                    l_center = r_center - 795
                else:
                    r_center = l_center + 795
            else:
                if lgap > rgap:
                    r_center = l_center + 795
                else:
                    l_center = r_center - 795
            
    # Add what we found for the first layer
    #window_centroids.append((l_center,r_center))
    #f = plt.imshow(image)
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(1+h/window_height)):
        ############## INIT
        l_center_old = l_center
        r_center_old = r_center
        image_layer = np.sum(image[int(h-(level)*window_height):int(h-(level-1)*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        
        ############## LEFT LINES
        l_min_index = int(max(l_center+offset-margin, 0))
        l_max_index = int(min(l_center+offset+margin, w))
        #l_conv_signal = conv_signal[:int(w/2)]
        l_conv_signal = conv_signal[l_min_index:l_max_index]
        l_idx = signal_peak(l_conv_signal)
        # Reject invalid centroids (noise or empty space) and keep the previous value
        if l_conv_signal[l_idx] > threshold:
            l_center = l_idx + l_min_index - offset
        else:
            l_center = l_center_old
            
        ############## RIGHT LINES
        r_min_index = int(max(r_center+offset-margin, 0))
        r_max_index = int(min(r_center+offset+margin, w))
        #r_conv_signal = conv_signal[int(w/2):]
        r_conv_signal = conv_signal[r_min_index:r_max_index]
        r_idx = signal_peak(r_conv_signal)
        # Reject invalid centroids (noise or empty space) and keep the previous value
        if r_conv_signal[r_idx] > threshold:
            r_center = r_min_index + r_idx - offset
        else:
            r_center = r_center_old
        
        #y_pos = int( (h-(level*window_height) + h-(level-1)*window_height)/2)
        #plt.plot(l_center, y_pos, "o", color="green")
        #plt.plot(l_min_index, y_pos, "*", color="yellow")
        #plt.plot(l_max_index, y_pos, "*", color="yellow")       
        #plt.plot(r_center, y_pos, "o", color="green")
        #plt.plot(r_min_index, y_pos, "*", color="yellow")
        #plt.plot(r_max_index, y_pos, "*", color="yellow")
            
        # Update the list of centroids
        window_centroids.append((l_center,r_center))
        # Debugging
        if DEBUG:
            f = plt.figure(figsize=(4,4))
            g = plt.plot(conv_signal)
            plt.plot(l_center, l_conv_signal[l_idx], "*", color="red")
            plt.plot(l_center_old, 100, "o", color="green")
            plt.plot(l_center, 200, "o", color="red")

            plt.plot(r_center, r_conv_signal[r_idx], "*", color="red")
            plt.plot(r_center_old, 100, "o", color="green")
            plt.plot(r_center, 200, "o", color="red")
            plt.title("Slice Convolution {0}".format(level))
            f.savefig("debug_detailed/{0}vslice_convolution{1}_.jpg".format(DEBUG_ID,level))
    return window_centroids

    
##############################################################################################################
# Assumes images were previously corrected for camera distortion
#
# Process an image to map the lane lines as follows:
# 1. Threshold the image on the Luminance, Saturation and Red channels using Sobel in X direction only
# 2. Apply Gaussian Blur to smooth out the lines found and eliminate voids This avoids line segmentation
#    caused by the Sobel filter (some lane lines are split length-wise into two pieces)
# 3. Apply a manually devised viewport to the image to extract the lane portion for mapping the lines
# 4. Apply a perspective transform to create a top-down view
# 5. Slice the image into horizontal sections and find the lane centers in each sections
# 6. Use those lane centers to map a function for each line and calculate road curvature
# 7. Use the functions to highlight the lane on the road.
# 8. Calculate the degree of offset to the center of the lane
##############################################################################################################
def map_lane_lines(image, window_width=50, window_height=80, margin=100, cache=None, DEBUG=False, DEBUG_ID=""):
    global ym_per_pix
    global xm_per_pix
    sobel_thresh=(30, 150)
    gradient_thresh=(0.7, 1.3)
    sobel_kernel=9
    s_thresh=(170,240)
    w, h = image.shape[1], image.shape[0]
    offset = 80
    
    img_thresholded = sobel_LS_threshold(image, sobel_kernel=sobel_kernel, sobel_thresh=sobel_thresh, gradient_thresh=gradient_thresh)

    viewport = [[540,465],[740,465],[1280,720],[0,720]]             # Final viewport corrects distortion at top of transform
    #viewport = [[540,468],[740,468],[1280,720],[0,720]]            # New Viewport (vanishing point intersection)
    warped = perspective_transform(img_thresholded, viewport, offset=offset)

    mapped_lanes = None
    lane_markings = None
    window_centroids = find_window_centroids(warped, window_width, window_height, margin, cache, DEBUG, DEBUG_ID)
    # Centroids found, fit to a polygon and display
    if len(window_centroids) > 0:
        centroids = np.array(window_centroids)
        l, r = centroids[:,0], centroids[:,1]
        #print(r)
        # Fit a polynomial to the centroids found since we're using 80 pixel height windows
        y_fit = np.array([s for s in range(int(h - window_height/2), 0, -window_height)]) # np.array([s for s in range(680,0,-80)])
        y = np.array([p for p in range(0,h+1,10)])
        L_coeff = np.polyfit(y_fit, l, 2)
        L_coeff_m = np.polyfit(y_fit*ym_per_pix, l*xm_per_pix, 2)
        l_x = L_coeff[0]*y**2 + L_coeff[1]*y + L_coeff[2]
        l_pts = np.column_stack((l_x, y)).astype("int32")
        
        y = np.array([p for p in range(h,-2,-10)])
        R_coeff = np.polyfit(y_fit, r, 2)
        R_coeff_m = np.polyfit(y_fit*ym_per_pix, r*xm_per_pix, 2)
        r_x = R_coeff[0]*y**2 + R_coeff[1]*y + R_coeff[2]
        r_pts = np.column_stack((r_x, y)).astype("int32")
        
        # Paint the polynomial onto a canvas and overlay on image
        found_lines = np.zeros((h,w,3), np.uint8)
        
        lr_pts = np.append(l_pts,r_pts, axis=0)
        cv2.fillPoly(found_lines, [lr_pts], (0,255,0))
        cv2.polylines(found_lines,[l_pts], 0, (255,0,0),thickness=4)
        cv2.polylines(found_lines,[r_pts], 0, (255,0,0),thickness=4)
        
        original = cv2.cvtColor(255*warped, cv2.COLOR_GRAY2RGB)
        mapped_lanes = cv2.addWeighted(original, 50, found_lines, 0.3, 0)
        
        r_centroid = np.column_stack((r, y_fit)).astype("int32")
        l_centroid = np.column_stack((l, y_fit)).astype("int32")
        #Radius Calculation
        radius_L, radius_R = radius_of_curvature(L_coeff_m, R_coeff_m, h)
        
        # Now lay the overlay back onto the original image
        overlay = perspective_transform(mapped_lanes, viewport, offset=offset, reverse=True)
        label = "Radius:{0:.0f}, {1:.0f}".format(radius_L, radius_R)
        cv2.putText(overlay, label,(20,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
        lane_markings = cv2.addWeighted(image, 1, overlay, 0.75, 0)
            
        #Update the cache with current parameters
        l_centroid_fit = L_coeff[0]*h**2 + L_coeff[1]*h + L_coeff[2]
        r_centroid_fit = R_coeff[0]*h**2 + R_coeff[1]*h + R_coeff[2]
        lane_width = (r_centroid_fit - l_centroid_fit) * xm_per_pix
        cache = {}
        cache['centers'] = [l_centroid_fit, r_centroid_fit]
        cache['radii'] = (radius_L, radius_R)
        cache['coeffs']= (L_coeff, R_coeff)
        cache['centroids'] = window_centroids
        cache['width'] = lane_width
        
        # Overlay the viewport on the image
        if DEBUG:
            for k in range(len(viewport)):
                x1, y1 = viewport[k][0],viewport[k][1]
                x2, y2 = viewport[k-1][0],viewport[k-1][1]
                cv2.line(lane_markings,(x1,y1),(x2,y2), [255,0,0], 2)
            mpimg.imsave("debug_detailed/{0}.jpg".format(DEBUG_ID), image)
            mpimg.imsave("debug_detailed/{0}_mapped_lanes.jpg".format(DEBUG_ID), mapped_lanes)
            mpimg.imsave("debug_detailed/{0}_lane_markings.jpg".format(DEBUG_ID), lane_markings)
            #Write parameters to a file
            fh = open("debug_detailed/{0}_parameters.json".format(DEBUG_ID),"w")
            print(cache, file=fh)
            fh.close()
    else:
        # Use the previous mapping
        lane_markings = image
    return mapped_lanes, lane_markings, cache


##############################################################################################################
# Measure radius of curvature of the road at the bottom of the image
##############################################################################################################
def radius_of_curvature(L_coeff, R_coeff, height):
    global ym_per_pix
    global xm_per_pix
    y = height*ym_per_pix
    A = L_coeff[0]
    B = L_coeff[1]
    if math.fabs(A) > 0:
        R_Left =  math.pow(1 +  math.pow(2*A*y+B, 2), 1.5)/math.fabs(2*A)
    else:
        R_Left = math.inf
    A = R_coeff[0]
    B = R_coeff[1]
    if math.fabs(A) > 0:
        R_Right = math.pow(1 +  math.pow(2*A*y+B, 2), 3/2)/math.fabs(2*A)
    else:
        R_Right = math.inf
    return R_Left, R_Right

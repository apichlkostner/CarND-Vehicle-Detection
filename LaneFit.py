import numpy as np
import cv2

class LaneFit():
    def __init__(self):
        self.img = None
        self.left_fit = None
        self.right_fit = None
        self.firstCall = True
        self.fircal = True
        self.fit = False
        self.middle_phys = 0.0
        self.middle_pix = 0
        self.cnt = 0
        self.s1 = None
        self.POLYFIT = True
        self.PLOTFITPOINTS = False
        self.POLYUPDATE = False

    def procVideoImg(self, img, numwin=9, margin=100, minpix=50):
        """
        Processing of video images.
        When lane was fitted before an update based on the last fit is done
        """
        if self.firstCall:                
            lane_params = self.fitLanes(img, numwin=numwin, margin=margin, minpix=minpix)
            self.firstCall = False
        else:
            lane_params = self.update(img)

        return lane_params

    
    def update(self, img):
        """
        Update based on fit from last image.

        """

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img[:,:,0].nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        margin = 100
        # not allowed if no call of self.fitLanes was done before
        if not self.firstCall:
            # polynomial or spline fitting
            if self.POLYFIT:
                # update polygon as vehicle position changed till last frame
                if self.POLYUPDATE:
                    # shift of curve while car is driving one frame
                    delta_drive_px = 20
                    self.left_fit[1] += 2. * delta_drive_px * self.left_fit[0]
                    self.left_fit[2] += self.left_fit[0] * delta_drive_px**2 + delta_drive_px + self.left_fit[1]

                    self.right_fit[1] += 2. * delta_drive_px * self.right_fit[0]
                    self.right_fit[2] += self.right_fit[0] * delta_drive_px**2 + delta_drive_px + self.right_fit[1]
                
                # calculate curve
                delta_drive_px = 0    # extrapolate curve for new vehicle position
                left_lane  = self.left_fit[0]*((nonzero_y+delta_drive_px)**2) + self.left_fit[1]*(nonzero_y+delta_drive_px) + self.left_fit[2]
                right_lane = self.right_fit[0] * ((nonzero_y+delta_drive_px)**2) + self.right_fit[1] * (nonzero_y+delta_drive_px) + self.right_fit[2]

            else:
                delta_drive_px = 20

                left_lane = self.s_left(nonzero_y+delta_drive_px)
                right_lane = self.s_right(nonzero_y+delta_drive_px)

            left_lane_inds =  ((nonzero_x > left_lane - margin)) & (nonzero_x < left_lane + margin)
            right_lane_inds = ((nonzero_x > right_lane - margin)) & (nonzero_x < right_lane + margin)
            
            # extract left and right line pixel positions
            leftx = nonzero_x[left_lane_inds]
            lefty = nonzero_y[left_lane_inds] 
            rightx = nonzero_x[right_lane_inds]
            righty = nonzero_y[right_lane_inds]

            # fit curve to found pixels
            self.fitCurves(leftx, lefty, rightx, righty)

            window_img = np.zeros_like(img)

            # plots the found pixel to the image
            if self.PLOTFITPOINTS:
                window_img[lefty, leftx] = [0, 0, 255]
                window_img[righty, rightx] = [0, 0, 255]

            if self.fit:
                # plot lane borders and area
                window_img  = self.plotCurves(window_img)

                y_eval = window_img.shape[0] - 1

                # calculate curve radius
                left_curverad = ((1 + (2*self.left_fit_phys[0]*y_eval + self.left_fit_phys[1])**2)**1.5) \
                                    / np.absolute(2*self.left_fit_phys[0])
                right_curverad = ((1 + (2*self.right_fit_phys[0]*y_eval + self.right_fit_phys[1])**2)**1.5) \
                                    / np.absolute(2*self.right_fit_phys[0])
            else:
                left_curverad = 1
                right_curverad = 1

            lane_params = {'left_fit': self.left_fit, 'right_fit': self.right_fit,
                           'left_curverad': left_curverad, 'right_curverad': right_curverad,
                           'middle_phys': self.middle_phys, 'middle_pix': self.middle_pix,
                           'img': window_img }

            return lane_params
    
    
    def fitCurves(self, leftx, lefty, rightx, righty, kr=6):
        # Fit a second order polynomial to each
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        #print(str(len(lefty))+'   '+str(len(righty)))
        
        # only fit if enough points are found
        if len(lefty) > 1000 and len(righty) > 1000:
            # for every frame with no fit wait one frame
            if self.cnt > 0:
                self.cnt = self.cnt - 1
            else:       
                self.fit = True

                # first call -> no exponential averaging
                if self.fircal:
                    self.left_fit = np.polyfit(lefty, leftx, 2)
                    self.right_fit = np.polyfit(righty, rightx, 2)
                    self.left_fit_phys = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
                    self.right_fit_phys = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
                    self.fircal = False
                else:
                    poly_l = np.polyfit(lefty, leftx, 2)
                    poly_r = np.polyfit(righty, rightx, 2)
                    self.left_fit = (kr * self.left_fit + poly_l) / (kr + 1.)
                    self.right_fit = (kr * self.right_fit + poly_r) / (kr + 1.)
                    
                    poly_l = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
                    poly_r = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
                    self.left_fit_phys = (kr * self.left_fit_phys + poly_l) / (kr + 1.)
                    self.right_fit_phys = (kr * self.right_fit_phys + poly_r) / (kr + 1.)
                
                # middle point, used to calculate vehicle position relative to lane
                bottom = 720
                left_fitx  = self.left_fit[0]*(bottom**2) + self.left_fit[1]*bottom + self.left_fit[2]
                right_fitx = self.right_fit[0] * (bottom**2) + self.right_fit[1] * bottom + self.right_fit[2]
                self.middle = ((self.shape[1] / 2) - ((left_fitx + right_fitx) / 2))
                self.middle_phys = ((self.shape[1] / 2) - ((left_fitx + right_fitx) / 2)) * xm_per_pix
        else:
            self.cnt = self.cnt + 1

    def plotCurves(self, img):
        """
        Plots the fitted curves to the image
        """
        # size of the plotted curve
        plotmargin = 15
            
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        # curve as polygon or spline
        if self.POLYFIT:
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        else:
            left_fitx = self.s_left(ploty)
            right_fitx = self.s_right(ploty)        

        # Plot left and right curve. Fill space inbetween
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-plotmargin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+plotmargin, 
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-plotmargin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+plotmargin, 
                                    ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        inbetween = np.hstack((left_line_window2, right_line_window1))

        # Draw the lane onto the image
        cv2.fillPoly(img, np.int_([left_line_pts]), (100, 0, 255))
        cv2.fillPoly(img, np.int_([inbetween]), (0, 100, 0))
        cv2.fillPoly(img, np.int_([right_line_pts]), (100, 0, 255))

        return img

    def fitLanes(self, img, numwin=9, margin=100, minpix=50):
        """
        First fit to the lane markings.
        Uses a sliding window approach.
        """
        self.shape = img.shape
        out_img = img.copy()
        window_height = np.int(img.shape[0] / numwin)

        # Calculate start position of sliding window for left and right lane marking
        leftx_base, rightx_base = self.calcBase(img)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(numwin):
            # Identify window boundaries in x and y (and right and left)
            win_y_low  = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low   = leftx_current - margin
            win_xleft_high  = leftx_current + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            out_img = cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                                    (win_xleft_high, win_y_high), (0,255,0), 2)
            out_img = cv2.rectangle(out_img, (win_xright_low, win_y_low),
                                    (win_xright_high, win_y_high), (255,0,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
            (nonzero_x >= win_xleft_low) &  (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
            (nonzero_x >= win_xright_low) &  (nonzero_x < win_xright_high)).nonzero()[0]

            # recenter next window on their mean position
            if len(good_left_inds) > minpix:
                window_mean = np.int(np.mean(nonzero_x[good_left_inds]))
                # remove outliers
                good_left_inds = good_left_inds[np.abs(nonzero_x[good_left_inds] - window_mean) < 20]

                if len(good_left_inds) > 1:
                    window_mean = np.int(np.mean(nonzero_x[good_left_inds]))
                    leftx_current = window_mean

            if len(good_right_inds) > minpix:
                window_mean = np.int(np.mean(nonzero_x[good_right_inds]))
                # remove outliers
                good_right_inds = good_right_inds[np.abs(nonzero_x[good_right_inds] - window_mean) < 20]

                if len(good_right_inds) > 1:
                    window_mean = np.int(np.mean(nonzero_x[good_right_inds]))
                    rightx_current = window_mean

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds] 
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds] 

        # fit the curve
        self.fitCurves(leftx, lefty, rightx, righty)

        # output image
        window_img = np.zeros_like(out_img)
        if self.fit:
            # plot lane borders and area
            window_img  = self.plotCurves(window_img)

            # calculate curve radius in m
            y_eval = window_img.shape[0] - 1
            left_curverad = ((1 + (2*self.left_fit_phys[0]*y_eval + self.left_fit_phys[1])**2)**1.5) \
                                / np.absolute(2*self.left_fit_phys[0])
            right_curverad = ((1 + (2*self.right_fit_phys[0]*y_eval + self.right_fit_phys[1])**2)**1.5) \
                                / np.absolute(2*self.right_fit_phys[0])
        else:
            left_curverad = np.nan
            right_curverad = np.nan
        
        #out_img = img.copy()
        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        lane_params = {'left_fit': self.left_fit, 'right_fit': self.right_fit,
                       'left_curverad': left_curverad, 'right_curverad': right_curverad,
                       'middle_phys': self.middle_phys, 'middle_pix': self.middle_pix,
                       'img': window_img }

        return lane_params

    def calcBase(self, img):
        histogram = np.sum(img[img.shape[0]//2:,:-2, 0], axis=0)

        midpoint = np.int(histogram.shape[0]/2)
        
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base
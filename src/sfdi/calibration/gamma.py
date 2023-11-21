import numpy as np
import time

from sfdi.calibration import Calibration

class GammaCalibration(Calibration):
    def __init__(self, width=1280, height=720):
        super().__init__(width, height)

    def run(self):
        print("Gamma calibration routine")
    
        int_val = round(np.linspace(0, 255, 40))

        pre_all_im = np.ones((self.height, self.width, 3))

        int_im = np.zeros((self.height, self.width, 3))

        for k in range(len(int_val)):
            int_im[k] = np.uint8(pre_all_im * int_val(k))
        
        # Set camera exposure
        # cam.exposure = 16666

        imgs = []

        for n in range(len(int_val)):
            # Display image in fullscreen
            self.projector.display(int_im[n])
            time.sleep(.5)

            # Take an image of the pattern
            imgs.append(self.camera.get_image())

            # Clear projector
            self.projector.clear()
        
        # TODO: Get central region of images

        img_width = self.camera.width
        img_height = self.camera.height

        #rows = round(img_width / 2) + (-200:200)
        #cols = round(img_height / 2) + (-200:200)

        for n in range(len(int_val)):
            # Convert image to greyscale
            #%Im_c = rgb2gray(Images{n})
            pass

            # Take averages of images
            # av(n) = mean(mean(Im_c(rows,cols))) ;

        # Characterise gamma curve

        # Calculate max intensity
        #max_int = max(av);

        # Normalise average
        #av = av ./ max_int ;
        #av = av .* 255 ;

        # Determine minimum threshold area
        #vis_int = av >=10 ;

        # Calculate minimum detectable variation and baseline
        #min_int = int_val(find(vis_int,1)) ;
        #baseline = min_int./255 ;
        #baseline = baseline .* 1.05 ;

        # Shift intensity data according to baseline
        #av_vis = av(vis_int) ;
        #int_val_vis = int_val(vis_int) ;

        # Fit polynomial to the curve
        #gamma_curve = fit(av_vis',int_val_vis','poly6') ;

        #plot(gamma_curve,av_vis,int_val_vis) ;
        #xlabel('Captured intensity/Pixels');
        #ylabel('Projected intensity/Pixels');
        #title('Gamma curve')

        # Export data

        #gamma_data.Baseline = baseline ;
        #gamma_data.Gamma_curve = gamma_curve ;
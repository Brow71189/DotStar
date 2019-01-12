import time
import board
import adafruit_dotstar as dotstar
import sys
from math import exp
import threading
import numpy as np

"""
// Compute gamma/dither tables for one color component.  Pass gamma and max
// brightness level (e.g. 2.7, 255) followed by pointers to 'lo', 'hi' and
// 'frac' tables to fill.  Typically will call this 3 times (R, G, B).
void fillGamma(float g, uint8_t m, uint8_t *lo, uint8_t *hi, uint8_t *frac) {
  uint16_t i, j, n;
  for(i=0; i<256; i++) {
    // Calc 16-bit gamma-corrected level
    n = (uint16_t)(pow((double)i / 255.0, g) * (double)m * 256.0 + 0.5);
    lo[i]   = n >> 8;   // Store as 8-bit brightness level
    frac[i] = n & 0xFF; // and 'dither up' probability (error term)
  }
  // Second pass, calc 'hi' level for each (based on 'lo' value)
  for(i=0; i<256; i++) {
    n = lo[i];
    for(j=i; (j<256) && (lo[j] <= n); j++);
    hi[i] = lo[j];
  }
}
"""
def fill_gamma(gamma, max_brightness, lo, hi, frac):
    for i in range(256):
        n = int((i/255)**gamma * max_brightness *256 + 0.5)
        lo[i] = n >> 8
        frac[i] = n & 0xFF
        
    for i in range(256):
        n = lo[i]
        for j in range(i, 256):
            hi[i] = lo[j]
            if lo[j] > n:
                break

class DotStar:
    def __init__(self, num_leds, gamma=2.8, max_brightness=255):
        self._led_colors = np.zeros((num_leds, 3), dtype=np.uint16)
        self._out_buf = np.empty((num_leds, 3), dtype=np.uint8)
        self._err = np.empty((num_leds, 3), dtype=np.uint8)
        self._lo = np.empty((256, 3), dtype=np.uint8)
        self._hi = np.empty((256, 3), dtype=np.uint8)
        self._frac = np.empty((256, 3), dtype=np.uint8)
        self._lock = threading.Lock()
        self._dots = dotstar.DotStar(board.SCK, board.MOSI, num_leds, auto_write=False)
        self._thread = None
        self._stop_event = threading.Event()
        
        self._gamma = gamma
        self._max_brightness = max_brightness
        self.num_leds = num_leds
        
        self.fill_correction_arrays()
        
    def __del__(self):
        self.stop_live()
        self._dots.deinit()
        
    @property
    def led_colors(self):
        with self._lock:
            return self._led_colors
    
    @led_colors.setter
    def led_colors(self, led_colors):
        with self._lock:
            self._led_colors[:] = led_colors
            
    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
        self.fill_correction_arrays()
        
    @property
    def max_brightness(self):
        return self._max_brightness
    
    @max_brightness.setter
    def max_brightness(self, max_brightness):
        self._max_brightness = max_brightness
        self.fill_correction_arrays()
            
    def _magic(self):
        """
        // This function interpolates between two RGB input buffers, gamma- and
        // color-corrects the interpolated result with 16-bit dithering and issues
        // the resulting data to the GPIO port.
        void magic(
         uint8_t *rgbIn1,    // First RGB input buffer being interpolated
         uint8_t *rgbIn2,    // Second RGB input buffer being interpolated
         uint8_t  w2,        // Weighting (0-255) of second buffer in interpolation
         uint8_t *fillBuf,   // SPI data buffer being filled (DotStar-native order)
         uint16_t numLEDs) { // Number of LEDs in buffer
          uint8_t   mix;
          uint16_t  weight1, weight2, pixelNum, e;
          uint8_t  *fillPtr = fillBuf + 5; // Skip 4-byte header + 1 byte pixel marker
        
          weight2 = (uint16_t)w2 + 1; // 1-256
          weight1 = 257 - weight2;    // 1-256
        
          for(pixelNum = 0; pixelNum < numLEDs; pixelNum++, fillPtr += 4) {
            // Interpolate red from rgbIn1 and rgbIn2 based on weightings
            mix = (*rgbIn1++ * weight1 + *rgbIn2++ * weight2) >> 8;
            // fracR is the fractional portion (0-255) of the 16-bit gamma-
            // corrected value for a given red brightness...essentially it's
            // how far 'off' a given 8-bit brightness value is from its ideal.
            // This error is carried forward to the next frame in the errR
            // buffer...added to the fracR value for the current pixel...
            e = fracR[mix] + errR[pixelNum];
            // ...if this accumulated value exceeds 255, the resulting red
            // value is bumped up to the next brightness level and 256 is
            // subtracted from the error term before storing back in errR.
            // Diffusion dithering is the result.
            fillPtr[DOTSTAR_REDBYTE] = (e < 256) ? loR[mix] : hiR[mix];
            // If e exceeds 256, it *should* be reduced by 256 at this point...
            // but rather than subtract, we just rely on truncation in the 8-bit
            // store operation below to do this implicitly. (e & 0xFF)
            errR[pixelNum] = e;
        
            // Repeat same operations for green...
            mix = (*rgbIn1++ * weight1 + *rgbIn2++ * weight2) >> 8;
            e   = fracG[mix] + errG[pixelNum];
            fillPtr[DOTSTAR_GREENBYTE] = (e < 256) ? loG[mix] : hiG[mix];
            errG[pixelNum] = e;
        
            // ...and blue...
            mix = (*rgbIn1++ * weight1 + *rgbIn2++ * weight2) >> 8;
            e   = fracB[mix] + errB[pixelNum];
            fillPtr[DOTSTAR_BLUEBYTE] = (e < 256) ? loB[mix] : hiB[mix];
            errB[pixelNum] = e;
        }
        """
        mix = self.led_colors
        e = self._frac[mix, (0,1,2)].astype(np.uint16) + self._err.astype(np.uint16)
        self._out_buf[:] = np.where(e < 256, self._lo[mix, (0,1,2)], self._hi[mix, (0,1,2)])
        self._err[:] = e
        
    def fill_correction_arrays(self):
        for i in range(3):
            fill_gamma(self.gamma, self.max_brightness, self._lo[:, i], self._hi[:, i], self._frac[:, i])
    
    def run_light(self):
        pass
    
    def fill_gradient(self, center=0, decay=5, color=(255,255,255)):
        indices = np.arange(self.num_leds)
        distances = np.abs(indices - center)
        colors = -decay*np.reshape(np.repeat(distances, 3), (self.num_leds, 3)) + np.array(color)
        colors[colors<0] = 0
        self.led_colors = colors
    
    def show(self):
        self._magic()
        self._dots[:] = self._out_buf
        self._dots.show()

    def start_live(self):
        self._stop_event.clear()
        def update():
            while not self._stop_event.is_set():
                self.show()
        self._thread = threading.Thread(target=update, daemon=True)
        self._thread.start()
    
    def stop_live(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(1)
            self._thread = None
    
def run_light(dots, speed=2, color=(255,255,255)):
    n_dots = len(dots)
    center = 0
    direction = 1
    while True:
        fill_gradient(dots, center=center, decay=10, color=color)
        if direction == 1:
            center += 1
        else:
            center -= 1
        if center == 0 or center == n_dots-1:
            direction *= -1
        time.sleep(1/speed)

def fill_gradient(dots, center=0, decay=5, color=(65536,65536,65536), gamma=(0.6, 0.5, 0.5)):
    n_dots = len(dots)
    #current_color=color
#    step = tuple(n**(1/decay) for n in current_color)
    ts = tuple(color[i]*exp(-gamma[i]*decay)/(1-exp(-gamma[i]*decay)) for i in range(len(color)))
#    Ms = tuple(-color[i]/decay for i in range(len(color)))
    for i in range(n_dots):
        dist = abs(i - center)
#        current_color = tuple(round(color[i]/(step[i]**dist)) for i in range(len(current_color)))
        #brightness = 2**(-dist/decay)
#        brightness = tuple(-color[i]/decay*dist+color[i] for i in range(len(color)))
        current_color = tuple()
        for k in range(len(ts)):
 #           x = Ms[k]*dist+decay
 #           if x < 0:
 #               current_color += (0,)
 #               continue
            val = round((color[k]+ts[k])*exp(-gamma[k]*dist)-ts[k])
            current_color += (val,) if val > 0 else (0,)
        #if brightness < 1/32:
        #    current_color = (0, 0, 0, 0)
        #else:
        #    current_color = color + (brightness,)
        #print(current_color)
        dots[i] = current_color
    dots.show()
if __name__ == '__main__':
    try:
        dots = dotstar.DotStar(board.SCK, board.MOSI, 30, auto_write=False)
        run_light(dots, speed=24, color=(255, 140, 40))
        time.sleep(3600)
        dots.deinit()
    except KeyboardInterrupt:
        dots.deinit()

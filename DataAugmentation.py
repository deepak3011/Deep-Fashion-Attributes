#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:50:17 2019

@author: deepakchoudhary
"""

from PIL import Image, ImageEnhance, ImageFilter
import random
import numpy as np

class DataAugmentation:
    def __call__(self, image):
        image = self.gaussian_noise(image)
        image = self.contrast(image)
        image = self.brightness(image)
        image = self.blur(image)
        #image = self.rotate(image)
        return image
        
    def checkProb(self):
        
        prob = random.random()
        if prob > 0.7:
            return True
        else:
            return False
        
    def gaussian_noise(self, image):
        if self.checkProb() == False:
            return image
        mean = 0
        var = 5
        std_dev = var**0.5
        image = np.asarray(image)
        gauss = np.random.normal(mean, std_dev, image.shape)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255).astype("uint8")
        noisy = Image.fromarray(noisy)
        return noisy
        
    #difference in luminance or colour that makes object distinguishable
    def contrast(self, image):
        if self.checkProb() == False:
            return image
        scale = np.random.rand()*1.5  + 0.5
        return ImageEnhance.Contrast(image).enhance(scale)
    
    def brightness(self, image):
        if self.checkProb() == False:
            return image
        scale = np.random.rand()*1.5  + 0.5
        return ImageEnhance.Brightness(image).enhance(scale)
    
    def blur(self, image):
        if self.checkProb() == False:
            return image
        return image.filter(ImageFilter.BLUR)
    
    def rotate(self, image):
        if self.checkProb() == False:
            return image
        rotateDegrees = [0, 180]
        return image.rotate(np.random.choice(rotateDegrees))
"""   
if __name__ == "__main__":
        obj = DataAugmentation()
        imagePath = "/Users/deepakchoudhary/Desktop/Work/Kaggle/jbm-crack/YE358311_Fender_apron/YE358311_defects/IMG20180905143750.jpg"
        img = Image.open(imagePath)
        a = obj.rotate(img)
        a.show()
"""        
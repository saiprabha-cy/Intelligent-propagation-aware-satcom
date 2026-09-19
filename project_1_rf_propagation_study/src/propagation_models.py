import numpy as np

def fspl(distance_km, frequency_mhz):
    return 32.44 + 20*np.log10(distance_km) + 20*np.log10(frequency_mhz)

def cost231(distance_km, frequency_mhz, hb=30, hm=1.5, C=3):
    a_hm = (1.1*np.log10(frequency_mhz) - 0.7)*hm - \
           (1.56*np.log10(frequency_mhz) - 0.8)
    return (46.3 + 33.9*np.log10(frequency_mhz)
            - 13.82*np.log10(hb) - a_hm
            + (44.9 - 6.55*np.log10(hb))*np.log10(distance_km) + C)

def rain_attenuation(distance_km, rain_rate, k=0.0001, alpha=1.0):
    return k * (rain_rate ** alpha) * distance_km

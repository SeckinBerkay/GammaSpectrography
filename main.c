
#include "gamma_spectrograph.h"

int main(void)
{
    float x_data_all[CHANNELS];

    float amplitude = 500.0, std_dev = 7.0;

    for (uint16_t i = 0; i < CHANNELS; i++) {
        x_data_all[i] = i;
    }
    // Generate synthetic Gaussian peaks
    floatArray peak1 = gaussian_array(x_data_all, CHANNELS, amplitude, 100.0, std_dev);
    floatArray peak2 = gaussian_array(x_data_all, CHANNELS, amplitude, 662.0, std_dev * 1.3);
    floatArray peak3 = gaussian_array(x_data_all, CHANNELS, amplitude, 850.0, std_dev * 0.9);

    // Combine peaks into a single dataset
    intArray y_data_all;
    y_data_all.data = (uint16_t *)malloc(CHANNELS * sizeof(uint16_t));
    y_data_all.length = CHANNELS;

    for (uint16_t i = 0; i < CHANNELS; i++) {
        y_data_all.data[i] = peak1.data[i] + peak2.data[i] + peak3.data[i];
    }

    // Peak Finder Test
    uint8_t confidence = 10;
    uint8_t FWHM = 20;
    float intensity_threshold = 0.001;
    uint8_t z = 6;

    linear_gauss_at_peaks_struct lin_gauss_params = Linear_Gauss_at_Peaks( x_data_all, y_data_all.data,  WINDOW_SIZE,  confidence,  FWHM,  intensity_threshold,  z);

    detection_result_t detections = match_isotopes(&lin_gauss_params);
    
    // Print results
    printf("Number of detections: %d\n", detections.num_detections);
    for (int i = 0; i < detections.num_detections; i++) {
        detected_isotope_t det = detections.detections[i];
        printf("Isotope: %s\n", det.isotope_name);
        printf("Matched energies: ");
        for (int j = 0; j < det.num_matched_energies; j++) {
            printf("%.2f ", det.matched_energies[j]);
        }
        printf("\n\n");
    }
    



    // Free dynamically allocated memory
    freeFloatArray(&peak1);
    freeFloatArray(&peak2);
    freeFloatArray(&peak3);
    freeIntArray(&y_data_all);
}


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>


#define WINDOW_SIZE 20
#define EPSILON 1e-10
#define TOL 1e-6
#define CHANNELS 1024
//#define DATA_LENGTH 6 // For the linear window. It is used in maximum likelihood method.
#define MAX_PEAKS 10  // Maximum number of peaks expected
#define LINEAR_WINDOW 3


#define R2_THRESHOLD 0.8f
#define ENERGY_TOL 20.0f  // Example energy tolerance, if needed


#include "isotope_data.h"  // Contains g_isotopes[] and g_num_isotopes


typedef struct {
    size_t length;
    uint16_t *data;  // Pointer to dynamically allocated data
} intArray;

typedef struct {
    size_t length;
    float *data;  // Pointer to dynamically allocated data
} floatArray;

typedef struct {
    size_t length;               // how many rows are valid
    float (*data)[3];            // Pointer to 2D array with 3 columns
} Jacobian;

typedef struct {
    float data[3][3];  // Fixed-size matrix for J^T * J
} JtJ;

typedef struct {
    float data[3];  // Fixed-size vector for residuals
} Jt_residual;

typedef struct {
    uint8_t status;
    float data[3];
} linear_solver;

typedef struct {
    float data[3];
} levenberg_marquardt;

typedef struct {
    uint16_t num_peaks;
    uint16_t idx_peak[MAX_PEAKS];
} peak_finder_struct;

typedef struct {
    float a;
    float b;
} linear_fit_struct;

typedef struct{

    float x_array[2 * WINDOW_SIZE];
    uint16_t y_array[2 * WINDOW_SIZE];

} near_peak_data_struct;

void freeFloatArray(floatArray *array) {
    if (array->data != NULL) {
        free(array->data);
        array->data = NULL;
    }
}

void freeIntArray(intArray *array) {
    if (array->data != NULL) {
        free(array->data);
        array->data = NULL;
    }
}

void freeJacobian(Jacobian *jacobian) {
    if (jacobian->data != NULL) {
        free(jacobian->data);
        jacobian->data = NULL;
    }
}




/* Function to convert channels to corrected energy */
// high malloc use 4096 size. (4096 * 4 bytes = 16kB memory usage)
floatArray channelToEnergy_d(float x[], uint16_t length) {
    floatArray corrected_x;
    corrected_x.data = (float *)malloc(length * sizeof(float));
    corrected_x.length = length;

    for (uint16_t i = 0; i < length; i++) {
        corrected_x.data[i] = x[i];  // Copy input data
    }

    return corrected_x;
}


/* Function to apply efficiency correction */
// high malloc use 4096 size. (4096 * 4 bytes = 16kB memory usage)
intArray efficiencyCorrection(uint16_t y[],  uint16_t length) {
    intArray corrected_y;
    corrected_y.data = (uint16_t *)malloc(length * sizeof(uint16_t));

    for (uint16_t i = 0; i < length; i++) {
        // Efficiency correction calculation (currently commented out)
        // corrected_y[i] = y[i] / eff[i];

        // Currently, the function returns the input unchanged
        corrected_y.data[i] = y[i];
    }
    return corrected_y;
}

/* Function to compute R² value */
float r2_test(uint16_t y_data[], float y_predicted[], uint16_t length) {
    float sum_y = 0.0;
    float mean_data;
    float ssTot = 0.0;
    float ssRes = 0.0;
    float R2;
    uint16_t i;

    // Compute mean of y_data
    for (i = 0; i < length; i++) {
        sum_y += y_data[i];
    }
    mean_data = sum_y / length;

    // Compute ssTot and ssRes
    for (i = 0; i < length; i++) {
        float diff1 = y_data[i] - mean_data;
        float diff2 = y_data[i] - y_predicted[i];
        ssTot += diff1 * diff1;
        ssRes += diff2 * diff2;
    }

    // Compute R²
    if (ssTot == 0.0) {
        R2 = 0.0;
    } else {
        R2 = 1.0 - (ssRes / ssTot);
    }

    return R2;
}

/* Function to compute the maximum value in an array */
float max_value(uint16_t array[], uint16_t length) {
    float max = array[0];
    uint16_t i;
    for (i = 1; i < length; i++) {
        if (array[i] > max) {
            max = array[i];
        }
    }
    return max;
}

/* Function to compute the mean value of an array */
float mean_value(float array[], uint16_t length) {
    float sum = 0.0;
    for (uint16_t i = 0; i < length; i++) {
        sum += array[i];
    }
    return sum / length;
}



// low malloc use 150 size. (150 * 4 bytes = 0.6kB memory usage)
floatArray multi_gaussian_array(float x_data[], uint16_t length) {

    floatArray result;
    result.data = (float *)malloc(length * sizeof(float));

    for (uint16_t i = 0; i < length; i++) {

    	result.data[i] =
    	    500 * exp(-(3000.0 - 500.0) * (3000.0 - 500.0) / (2.0 * 30.0 * 30.0)) +
    	    500 * exp(-((3000.0 - 1000.0) * (3000.0 - 1000.0)) / (2.0 * 30.0 * 30.0)) +
    	    500 * exp(-((3000.0 - 2000.0) * (3000.0 - 2000.0)) / (2.0 * 30.0 * 30.0));
    }
    result.length = length;
    return result;
}



// low malloc use 150 size. (150 * 4 bytes = 0.6kB memory usage)
floatArray gaussian_array(float x_data[], uint16_t length, float amplitude, float mean, float std) {

    floatArray result;
    result.data = (float *)malloc(length * sizeof(float));
    float denominator = 2.0 * std * std;
    for (uint16_t i = 0; i < length; i++) {
        float diff = x_data[i] - mean;
        result.data[i] = (amplitude * exp(-(diff * diff) / denominator));
    }
    result.length = length;
    return result;
}

// Function that computes and returns a Jacobian matrix as a struct
// low malloc use 450 size. (150 * 3 * 4 bytes = 1.8kB memory usage)
Jacobian computeJacobian(float x_data[], uint16_t data_length, float amplitude, float mean, float std) {
    Jacobian result;

    // Allocate memory for rows dynamically
    result.data = (float (*)[3])malloc(data_length * sizeof(float[3]));
    result.length = data_length;

    float std2 = std * std;
    float std3 = std2 * std;
    float denominator = 2.0 * std2;

    for (uint16_t i = 0; i < data_length; i++) {
        float diff = x_data[i] - mean;
        float exp_term = exp(-(diff * diff) / denominator);

        result.data[i][0] = exp_term;                           // dA
        result.data[i][1] = amplitude * diff / std2 * exp_term; // dMean
        result.data[i][2] = amplitude * (diff * diff) / std3 * exp_term; // dStd
    }

    return result;
}




/* Compute A = J^T * J */
// low malloc use 150 size. (150 * 4 bytes = 0.6kB memory usage)
JtJ compute_JtJ(Jacobian J, uint16_t data_length) {
    uint16_t i, j, k;
    JtJ A;
    // Initialize A to zero
    for (i = 0; i < 3; i++) {
        for (j = 0; j <= i; j++) {
            float sum = 0.0;
            for (k = 0; k < data_length; k++) {
                sum += J.data[k][i] * J.data[k][j];
            }
            A.data[i][j] = sum;
            A.data[j][i] = sum; // Symmetric matrix
        }
    }

    return A;
}


// low malloc use 150 size. (150 * 4 bytes = 0.6kB memory usage)
Jt_residual compute_Jt_residuals(Jacobian J, float residuals[], uint16_t data_length) {
    // Initialize g to zero
    Jt_residual g;
    for (uint16_t i = 0; i < 3; i++) {
        g.data[i] = 0.0;
    }
    // Compute g
    for (uint16_t i = 0; i < 3; i++) {
        for (uint16_t k = 0; k < data_length; k++) {
            g.data[i] += J.data[k][i] * residuals[k];
        }
    }
    return g;
}


/* Compute the maximum absolute value in a vector */
float max_abs(float vec[], uint16_t data_length) {
    float max = 0.0;
    for (uint16_t i = 0; i < data_length; i++) {
        float abs_val = fabs(vec[i]);
        if (abs_val > max) {
            max = abs_val;
        }
    }
    return max;
}

/* Compute the squared norm of a vector */
float norm_squared(float vec[], uint16_t data_length) {
    float sum = 0.0;
    for (uint16_t i = 0; i < data_length; i++) {
        sum += vec[i] * vec[i];
    }
    return sum;
}



/* Solve a 3x3 linear system using Gaussian elimination with partial pivoting */
linear_solver solve_linear_system_3x3(JtJ A, Jt_residual g) {
    float augmented[3][4]; // 3x3 A matrix and 3x1 b vector
    //linear_solver delta;
    linear_solver delta = {0};
    //float lambd = 1;
    uint8_t n_params = 3;
    // In the original code, lambd is added up to A, in the line 394, and substracted in the line 402.
    // This adds 2 * O(t) time complexity. We will add it up here so that it is only O(t) time complexity.


    /*
    This sets all fields to zero:
    delta.status = 0,
    delta.data[0] = 0.0,
    delta.data[1] = 0.0,
    delta.data[2] = 0.0
    */

    int i, j, k;
    float max;
    int maxRow;
    // Copy A and b into augmented matrix
    for (i = 0; i < n_params; i++) {
        for (j = 0; j < 3; j++) {
            augmented[i][j] = A.data[i][j];
        }
        augmented[i][3] = g.data[i];
    }
    // Forward elimination
    for (i = 0; i < 3; i++) {
        // Find the row with the largest pivot
        max = fabs(augmented[i][i]);
        maxRow = i;
        for (k = i + 1; k < 3; k++) {
            if (fabs(augmented[k][i]) > max) {
                max = fabs(augmented[k][i]);
                maxRow = k;
            }
        }
        // Swap maximum row with current row (pivoting)
        if (maxRow != i) {
            for (k = i; k < 4; k++) {
                float tmp = augmented[maxRow][k];
                augmented[maxRow][k] = augmented[i][k];
                augmented[i][k] = tmp;
            }
        }
        // Check for zero pivot element
        if (fabs(augmented[i][i]) < 1e-12) {
            // Singular matrix
            delta.status = -1;
            return delta;
        }
        else {
            delta.status = 0;
        }

        // Eliminate below
        for (k = i + 1; k < 3; k++) {
            float factor = augmented[k][i] / augmented[i][i];
            for (j = i; j < 4; j++) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }
    // Back substitution
    for (i = 2; i >= 0; i--) {
        delta.data[i] = augmented[i][3];
        for (j = i + 1; j < 3; j++) {
            delta.data[i] -= augmented[i][j] * delta.data[j];
        }
        delta.data[i] /= augmented[i][i];
    }

    return delta;
}


/* Levenberg-Marquardt algorithm for Gaussian fitting */
levenberg_marquardt levenberg_marquardt_gauss_fit(
    float        x_data[]                        ,
    float        y_data[]                        ,
    float        initial_params[]                ,
    uint16_t     data_length                     ,
    uint16_t     max_iter                        ,
    float        tol                             )

    {

    // Updated function implementation using 'data_length'
    uint8_t      n_params = 3;
    float        params[3]   ;
    float        lambd = 1.0 ;
    float        v = 2.0     ;
    uint8_t      iter        ;


    // Allocate memory for arrays based on data_length
    float      residuals[data_length]      ;
    float      new_params[3]               ;
    float      new_residuals[data_length]  ;

    levenberg_marquardt optimized_params;

    // Initialize parameters

    memcpy(params, initial_params, n_params * sizeof(float));  // required, as params is switched with new_params
                                                                // at the end of the optimizing params.


    for (iter = 0; iter < max_iter; iter++) {
        // Compute y_pred

        //y_pred = gaussian_array(x_data, data_length, params[0], params[1], params[2], y_pred);
        floatArray y_pred = gaussian_array(x_data, data_length, params[0], params[1], params[2]);

        // Compute residuals = y_data - y_pred
        for (uint16_t i = 0; i < data_length; i++) {
            residuals[i] = y_data[i] - y_pred.data[i];
        }
        freeFloatArray(&y_pred); // Use of y_pred has ended. Free it.

        // Compute Jacobian J
        Jacobian jac = computeJacobian(x_data, data_length, params[0], params[1], params[2]);

        // Compute A = J^T * J
        JtJ A = compute_JtJ(jac, jac.length);
        // Compute g = J^T * residuals
        Jt_residual g = compute_Jt_residuals(jac, residuals, data_length);
        freeJacobian(&jac); // Use of jac is ended. Free it.

        // Check for convergence
        if (max_abs(g.data, n_params) < tol) {
            break;
        }
        // Adjust parameters using LM update rule
        // Add lambd * I to A

        // Solve (A + lambd * I) * delta = g
        //int status = solve_linear_system_3x3(A, g, delta);
        // Add lambd * I to A
        for (uint16_t i = 0; i < n_params; i++) {
            A.data[i][i] += lambd;
        }
        linear_solver delta = solve_linear_system_3x3(A, g);
        for (uint16_t i = 0; i < n_params; i++) {
            A.data[i][i] -= lambd;
        }
        // Subtract lambd from A's diagonal to restore original A

        if (delta.status != 0) {
            continue; // Singular matrix, cannot solve, skip this iteration
        }
        // Compute new_params = params + delta
        for (uint16_t i = 0; i < n_params; i++) {
            new_params[i] = params[i] + delta.data[i];
        }
        // Compute new_y_pred
        floatArray new_y_pred = gaussian_array(x_data, data_length, new_params[0], new_params[1], new_params[2]);

        // Compute new_residuals = y_data - new_y_pred
        for (uint16_t i = 0; i < data_length; i++) {
            new_residuals[i] = y_data[i] - new_y_pred.data[i];
        }
        freeFloatArray(&new_y_pred); // Use of new_y_pred has ended. Free it.

        // Compute rho
        float norm_residuals_sq = norm_squared(residuals, data_length);
        float norm_new_residuals_sq = norm_squared(new_residuals, data_length);
        float numerator = norm_residuals_sq - norm_new_residuals_sq;
        // Compute lambd * delta + g
        float lambd_delta_plus_g[3];
        for (uint16_t i = 0; i < n_params; i++) {
            lambd_delta_plus_g[i] = lambd * delta.data[i] + g.data[i];
        }
        // Compute delta^T * (lambd * delta + g)
        float denominator = 0.0;
        for (uint16_t i = 0; i < n_params; i++) {
            denominator += delta.data[i] * lambd_delta_plus_g[i];
        }
        float rho = numerator / denominator;
        if (rho > 0) {
            // Update the parameters
            memcpy(params, new_params, n_params * sizeof(float));
            // Update lambd
            float tmp = 1.0 - pow(2.0 * rho - 1.0, 3.0);
            if (tmp < 1.0 / 3.0) {
                tmp = 1.0 / 3.0;
            }
            lambd *= tmp;
            if (lambd < 1e-10) {
                lambd = 1e-10; // Prevent lambd from becoming too small
            }
            v = 2.0;
        } else {
            // Increase lambd
            lambd *= v;
            if (lambd > 1e10) {
                lambd = 1e10; // Cap lambd at a large value
            }
            v *= 2.0;
        }
    }
    // Set optimized parameters
    memcpy(optimized_params.data, params, n_params * sizeof(float));

    return optimized_params;
}

/* Function to smooth the data by averaging neighboring values */
// high malloc use 4096 size. (4096 * 4 bytes = 16kB memory usage)
floatArray average(const float A[], int m) {
    uint16_t i, j;
    floatArray B;
    // Initialize B to zeros
    B.data = (float *)calloc(CHANNELS, sizeof(float));  // or malloc + memset

    B.length = CHANNELS;
    // For each element
    for (i = 0; i < CHANNELS; i++) {
        uint16_t count = 0;
        // Sum over neighbors from i - m to i + m - 1
        for (j = i - m; j <= i + m; j++) {
            if (j >= 0 && j < CHANNELS) {
                B.data[i] += A[j];
                count++;
            }
        }
        // Optionally, you can average by dividing by count
        // B[i] /= count;
    }

    return B;
}

static void average_inplace(const float in_array[], float out_array[], uint16_t length, uint16_t m)
{
    // Smooth using neighbors from i - m to i + m
    // out_array[i] = sum of neighbors (optionally / count)
    for (uint16_t i = 0; i < length; i++) {
        float sum   = 0.0f;
        uint16_t count = 0;
        // Sum over i - m to i + m
        int16_t start = (int16_t)i - (int16_t)m;
        int16_t end   = (int16_t)i + (int16_t)m;
        if (start < 0) start = 0;
        if (end >= length) end = length - 1;

        for (int16_t j = start; j <= end; j++) {
            sum += in_array[j];
            count++;
        }
        // If you really want the average, do:
        // out_array[i] = sum / count;
        //
        // If you just want the sum (like your code does),
        // keep out_array[i] = sum. Adjust to your preference.
        out_array[i] = sum;
    }
}

/**
 * @brief Compute the second-difference and store in S[].
 *        Then perform z rounds of smoothing in-place.
 * @param y     Input array of size CHANNELS.
 * @param S     Output array of size CHANNELS (second diff).
 * @param m     Half-window size for smoothing.
 * @param z     Number of smoothing passes.
 */
static void compute_second_diff_and_smooth(
    const uint16_t y[],
    float          S[],
    uint16_t       m,
    uint16_t       z)
{
    // 1) Compute the second difference for each index
    //    S[i] = y[i+1] - 2*y[i] + y[i-1], with boundary checks
    for (uint16_t i = 0; i < CHANNELS; i++) {
        if (i == 0) {
            // i=0 => S[0] = y[1] - 2*y[0]
            S[i] = (i+1 < CHANNELS) ? (float)y[i+1] - 2.0f*(float)y[i] : 0.0f;
        }
        else if (i == CHANNELS - 1) {
            // i=last => S[last] = y[last-1] - 2*y[last]
            S[i] = (i > 0) ? (float)y[i-1] - 2.0f*(float)y[i] : 0.0f;
        }
        else {
            S[i] = (float)y[i+1] - 2.0f*(float)y[i] + (float)y[i-1];
        }
    }

    // 2) Smooth in-place z times using a temporary buffer B[]
    float B[CHANNELS];
    for (uint16_t iter = 0; iter < z; iter++) {
        average_inplace(S, B, CHANNELS, m);
        // copy B back into S
        memcpy(S, B, sizeof(S[0]) * CHANNELS);
    }
}

/**
 * @brief Compute the "standard deviation" array (std_S) for y[]
 *        using the logic in your original standard_dev_S function,
 *        but do it in-place with local arrays, no malloc.
 *
 * @param y      Input array of size CHANNELS.
 * @param std_S  Output array of size CHANNELS (standard dev).
 * @param m      Half-window size for smoothing.
 * @param z      Number of smoothing passes.
 */
static void compute_std_S_and_smooth(
    const uint16_t y[],
    float          std_S[],
    uint16_t       m,
    uint16_t       z)
{
    // "Variance-like measure" array
    float F[CHANNELS];

    // 1) Compute F[i]
    for (uint16_t i = 0; i < CHANNELS; i++) {
        if (i == 0) {
            // F[0] = y[1] + 4*y[0]
            F[i] = (i+1 < CHANNELS) ? (float)y[i+1] + 4.0f*(float)y[i] : 0.0f;
        }
        else if (i == CHANNELS - 1) {
            // F[last] = y[last-1] + 4*y[last]
            F[i] = (float)y[i-1] + 4.0f*(float)y[i];
        }
        else {
            F[i] = (float)y[i+1] + 4.0f*(float)y[i] + (float)y[i-1];
        }
    }

    // 2) Smooth F z times
    float B[CHANNELS];
    for (uint16_t iter = 0; iter < z; iter++) {
        average_inplace(F, B, CHANNELS, m);
        memcpy(F, B, sizeof(F[0]) * CHANNELS);
    }

    // 3) std_S[i] = sqrt(F[i])
    for (uint16_t i = 0; i < CHANNELS; i++) {
        // If F[i] can be negative (shouldn't be, but just in case)
        // clamp to zero to avoid sqrt of negative.
        if (F[i] < 0.0f) {
            F[i] = 0.0f;
        }
        std_S[i] = sqrtf(F[i]);
    }
}

/*
 * Main function to find peaks without big mallocs.
 * We only store second-diff (S[]) and std-dev (std_S[]) in local arrays.
 */
peak_finder_struct Peak_Finder(
    const float   x[],
    const uint16_t y[],
    uint16_t     confidence,
    uint16_t     FWHM,
    float        intensity_threshold,
    int          z)
{
    peak_finder_struct peak_finder_params;
    peak_finder_params.num_peaks = 0;

    // Compute the half-window for smoothing
    float w = 0.6f * (float)FWHM;
    if (((uint16_t)w % 2) == 0) {
        w += 1.0f;
    }
    uint16_t m = (uint16_t)((w - 1.0f) / 2.0f);

    // Arrays for second diff & std dev
    float S[CHANNELS];
    float std_S[CHANNELS];

    // 1) Compute second difference (S) + smoothing
    compute_second_diff_and_smooth(y, S, m, (uint16_t)z);

    // 2) Compute std_S + smoothing
    compute_std_S_and_smooth(y, std_S, m, (uint16_t)z);

    // 3) Find maximum value in y for intensity threshold
    float max_y = (float)y[0];
    for (uint16_t i = 1; i < CHANNELS; i++) {
        if ((float)y[i] > max_y) {
            max_y = (float)y[i];
        }
    }

    // 4) Iterate over channels to detect peaks using S[i] and std_S[i]
    for (uint16_t i = 0; i < CHANNELS; i++) {
        // Original condition: if (fabs(S[i]) > std_S[i] * confidence && S[i] < 0)
        // Converting to float usage:
        float valS   = S[i];
        float valStd = std_S[i];
        if ((fabsf(valS) > (valStd * (float)confidence)) && (valS < 0.0f)) {
            // Check local maximum in y[i - FWHM : i + FWHM]
            int16_t start = (int16_t)i - (int16_t)FWHM;
            int16_t end   = (int16_t)i + (int16_t)FWHM;
            if (start < 0) {
                start = 0;
            }
            if (end >= CHANNELS) {
                end = CHANNELS - 1;
            }

            float local_max = (float)y[start];
            for (int16_t j = start; j <= end; j++) {
                if ((float)y[j] > local_max) {
                    local_max = (float)y[j];
                }
            }
            // Check if y[i] is that local max and meets threshold
            if ((float)y[i] == local_max &&
                (float)y[i] >= (intensity_threshold / 100.0f) * max_y)
            {
                // Record peak index
                peak_finder_params.idx_peak[peak_finder_params.num_peaks] = i;
                peak_finder_params.num_peaks++;
                if (peak_finder_params.num_peaks >= MAX_PEAKS) {
                    printf("Warning: Exceeded maximum number of peaks (%d).\n", MAX_PEAKS);
                    break;
                }
            }
        }
    }

    return peak_finder_params;
}


/* Function implementation */
linear_fit_struct linear_regression_maximum_likelihood(const float x[], const float y[], int n) {
    linear_fit_struct fitted_params;

    float s_x   = 0.0  ;
    float s_y   = 0.0  ;
    float s_x2  = 0.0  ;
    float s_xy  = 0.0  ;

    // Compute the necessary sums
    for (uint16_t i = 0; i < n; i++) {
        s_x += x[i]             ;
        s_y += y[i]             ;
        s_x2 += x[i] * x[i]     ;
        s_xy += x[i] * y[i]     ;
    }

    // Compute the denominator
    float denom = n * s_x2 - s_x * s_x;

    // Check for division by zero
    if (denom == 0.0) {
        printf("Error: Division by zero in linear regression calculation.\n");
        fitted_params.a = 0.0;
        fitted_params.b = 0.0;
        return fitted_params;
    }

    // Compute a and b
    float a_numerator = s_y * s_x2 - s_xy * s_x    ;
    float b_numerator = n * s_xy - s_y * s_x       ;

    fitted_params.a = a_numerator / denom           ;
    fitted_params.b = b_numerator / denom           ;

    return fitted_params;
}


near_peak_data_struct Near_Peak_Data(uint16_t peak_idx, float x[], uint16_t y[]) {
    // RETURNS X[START:END] AROUND THE PEAK INDEX.
    near_peak_data_struct windowed_data = {0};

    int length = 2 * WINDOW_SIZE;

    // Copy slices into the struct's arrays
    memcpy(windowed_data.x_array, &x[peak_idx - WINDOW_SIZE],
           length * sizeof(float));
    memcpy(windowed_data.y_array, &y[peak_idx - WINDOW_SIZE],
           length * sizeof(uint16_t));

    return windowed_data;
}


typedef struct {
    uint16_t peak_indices[MAX_PEAKS];
    float mean_list[MAX_PEAKS];
    float std_list[MAX_PEAKS];
    float R2_list[MAX_PEAKS];
    uint8_t num_peaks;
}  linear_gauss_at_peaks_struct ;


/* Main function */
linear_gauss_at_peaks_struct Linear_Gauss_at_Peaks(float corrected_x[], uint16_t corrected_y[], uint8_t window_size, uint8_t confidence, uint8_t FWHM, float intensity_threshold, uint8_t z) {
    uint16_t i, idx;
    // Define max_iter and tol
    linear_gauss_at_peaks_struct linear_gauss_params;
    memset(&linear_gauss_params, 0, sizeof(linear_gauss_params));

    uint8_t max_iter = 100;
    float tol = 1e-6;

    // Apply corrections to x and y
    //floatArray corrected_x = channelToEnergy_d(x, CHANNELS);
    //floatArray corrected_y = efficiencyCorrection(y, CHANNELS);
    //efficiencyCorrection(y, corrected_y, CHANNELS);

    // Find the peaks
    //Peak_Finder(corrected_x, corrected_y, confidence, FWHM, intensity_threshold, z, peak_indices, num_peaks);
    peak_finder_struct detected_peaks = Peak_Finder(corrected_x, corrected_y, confidence, FWHM, intensity_threshold, z);
    linear_gauss_params.num_peaks = detected_peaks.num_peaks;
    for (idx = 0; idx < detected_peaks.num_peaks; idx++) {

        linear_gauss_params.peak_indices[idx] = detected_peaks.idx_peak[idx];
        int peak_index = detected_peaks.idx_peak[idx];

        // Extract data around the detected peak.
        // Former variables named x_peak and y_peak are peak_data.x_array, and peak_data.y_array from now on.
        near_peak_data_struct peak_data = Near_Peak_Data(peak_index, corrected_x, corrected_y);

        if (sizeof(peak_data.y_array) != 0){
            // Linear regression data preparation
            float x_linear_data[2 * LINEAR_WINDOW];
            float y_linear_data[2 * LINEAR_WINDOW];

            // Extract beginning and ending linear sections for fitting
            for (i = 0; i < LINEAR_WINDOW; i++) {
                x_linear_data[i] = peak_data.x_array[i];
                y_linear_data[i] = peak_data.y_array[i];
                x_linear_data[LINEAR_WINDOW + i] = peak_data.x_array[(2 * WINDOW_SIZE) - LINEAR_WINDOW + i];
                y_linear_data[LINEAR_WINDOW + i] = peak_data.y_array[(2 * WINDOW_SIZE) - LINEAR_WINDOW + i];
            }

            // Perform linear regression on the edges of the window
            linear_fit_struct linear_params = linear_regression_maximum_likelihood(x_linear_data, y_linear_data, 2 * LINEAR_WINDOW);

            // Fit linear trend over the entire x_peak range
            float fitted_linear[2 * WINDOW_SIZE];
            for (i = 0; i < 2 * WINDOW_SIZE; i++) {
                fitted_linear[i] = linear_params.a + linear_params.b * peak_data.x_array[i];
            }

            // Initialize parameters for Gaussian fitting
            float peak_ampl = max_value(peak_data.y_array, 2 * WINDOW_SIZE);
            float mean_x = corrected_x[peak_index]; // Initial guess based on peak index in corrected_x
            float std_x = FWHM / 2.355;  // Convert FWHM to standard deviation
            float initial_params[3] = { peak_ampl, mean_x, std_x };

            // Remove linear trend from y_peak
            float y_peak_minus_fitted_linear[2 * WINDOW_SIZE];
            for (i = 0; i < 2 * WINDOW_SIZE; i++) {
                y_peak_minus_fitted_linear[i] = peak_data.y_array[i] - fitted_linear[i];
            }

            // Perform Gauss fit using Levenberg-Marquardt optimization
            levenberg_marquardt optimized_params = levenberg_marquardt_gauss_fit(peak_data.x_array, y_peak_minus_fitted_linear, initial_params, 2 * WINDOW_SIZE, max_iter, tol);
            // Compute fitted Gaussian values and combined fit
            floatArray fitted_gauss = gaussian_array(peak_data.x_array, 2 * WINDOW_SIZE, optimized_params.data[0], optimized_params.data[1], optimized_params.data[2]);
            float combined_fit[2 * WINDOW_SIZE];
            for (uint8_t i = 0; i < 2 * WINDOW_SIZE; i++){
                combined_fit[i] = fitted_linear[i] + fitted_gauss.data[i];
            }
            freeFloatArray(&fitted_gauss); // Use of fitted_gauss has ended. Free it.


            for (i = 0; i < 2 * WINDOW_SIZE; i++){
                // Calculate R² for the combined fit
                float R2_combined = r2_test(peak_data.y_array, combined_fit, 2 * WINDOW_SIZE);

                // Append results to output lists
                linear_gauss_params.mean_list[idx] = optimized_params.data[1];
                linear_gauss_params.std_list[idx] = fabs(optimized_params.data[2]);
                linear_gauss_params.R2_list[idx] = R2_combined;
            }

        }
    }

    return linear_gauss_params;
}

// The structure to store detection information for one isotope
typedef struct {
    const char *isotope_name;                   // Points to name in g_isotopes
    float       matched_energies[MAX_LINES_PER_ISOTOPE];
    uint8_t     num_matched_energies;
} detected_isotope_t;

// A structure to hold multiple detections plus how many there are
typedef struct {
    detected_isotope_t detections[10];          // Adjust size to your needs
    uint8_t            num_detections;
} detection_result_t;

/**
 * @brief  Detects isotopes by matching known library energies to measured peaks.
 *         Returns a detection_result_t struct that contains the array of matched isotopes
 *         and the total number matched.
 *
 * @param  peak_data  Pointer to a filled linear_gauss_at_peaks_struct instance.
 * @return A detection_result_t struct containing all detection results.
 */
detection_result_t match_isotopes(const linear_gauss_at_peaks_struct *peak_data)
{
    detection_result_t result;
    result.num_detections = 0;  // Initialize count to 0

    // Iterate over every isotope in the library
    for (uint8_t i = 0; i < g_num_isotopes; i++)
    {
        uint8_t matched_count = 0;
        float   matched_energies[MAX_LINES_PER_ISOTOPE];

        // For each known energy in this isotope
        for (uint8_t j = 0; j < g_isotopes[i].num_energies; j++)
        {
            float library_energy = (float)g_isotopes[i].energies[j];

            // Search through measured peaks for a match
            bool matched_this_energy = false;
            for (uint8_t k = 0; k < peak_data->num_peaks; k++)
            {
                float measured_energy = peak_data->mean_list[k];
                float r2_value        = peak_data->R2_list[k];

                // Check if energies match and R² is high enough
                // If you want approximate matching, use something like:
                if (fabsf(measured_energy - library_energy) < ENERGY_TOL && (r2_value >= R2_THRESHOLD))
                //if ((measured_energy == library_energy) && (r2_value >= R2_THRESHOLD))
                {
                    matched_energies[matched_count++] = measured_energy;
                    matched_this_energy = true;
                    break; // Found a match, move on to the next library energy
                }
            }

            // If we didn’t match this library energy, no need to check further
            if (!matched_this_energy)
            {
                matched_count = 0;
                break;
            }
        }

        // If we successfully matched all known lines for this isotope
        if (matched_count == g_isotopes[i].num_energies)
        {
            // Make sure there's still room in the results array
            if (result.num_detections < (sizeof(result.detections) / sizeof(result.detections[0])))
            {
                detected_isotope_t *det = &result.detections[result.num_detections++];
                det->isotope_name = g_isotopes[i].name;
                det->num_matched_energies = matched_count;

                // Copy the matched energies
                for (uint8_t m = 0; m < matched_count; m++)
                {
                    det->matched_energies[m] = matched_energies[m];
                }
            }
        }
    }

    return result;
}



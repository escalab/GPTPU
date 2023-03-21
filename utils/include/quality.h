#ifndef __QUALITY_H__
#define __QUALITY_H__
#include <stdint.h>

class Quality{
    public:
    	// default constructor
        Quality(int row, int col, int ldn, float* target_mat, float* baseline_mat);

	    // main APIs
	    float rmse(int verbose);
        float error_rate(int verbose);
        float error_percentage(int verbose);
        float ssim(int verbose);
        float pnsr(int verbose);
        void print_results(int verbose);

    private:
	    // internal helper functions
        void get_minmax(float* x, float& max, float& min);
        float average(float* mat);
        float sdev(float* mat);
        float covariance();
	    int row;
	    int col;
	    int ldn;
	    float* target_mat;
	    float* baseline_mat;
};
#endif


#ifndef __GPTPU_DIM_H__
#define __GPTPU_DIM_H__

class openctpu_dimension{
public:
    openctpu_dimension(bool transpose);
    void set_dims(int rows, int cols, int ldn);
    void get_dims(int& rows, int& cols, int& ldn);
    void set_transpose(bool flag){ this->transpose = flag; };
    bool get_transpose(){ return this->transpose; };
private:
    int rows;
    int cols;
    int ldn; 
    bool transpose; // To indicate if the 2D data array is transposed or not
};

#endif

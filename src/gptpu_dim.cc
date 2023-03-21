#include "gptpu_dim.h"

openctpu_dimension::openctpu_dimension(bool transpose){
    this->rows = 0;
    this->cols = 0;
    this->ldn  = 0;
    this->set_transpose(transpose);
}

void openctpu_dimension::set_dims(int rows, int cols, int ldn){
    this->rows = rows;
    this->cols = cols;
    this->ldn  = ldn;
}

void openctpu_dimension::get_dims(int& rows, int& cols, int& ldn){
    rows = this->rows;
    cols = this->cols;
    ldn  = this->ldn;
}


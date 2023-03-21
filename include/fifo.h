////////////////////////////////////////////////////////////////////////
//
// Copyright 2014 PMC-Sierra, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License. You may
// obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0 Unless required by
// applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for
// the specific language governing permissions and limitations under the
// License.
//
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
//
//   Author: Logan Gunthorpe
//
//   Date:   Oct 23 2014
//
//   Description:
//     Thread Safe Fifo
//
////////////////////////////////////////////////////////////////////////


#ifndef __LIBDONARD_FIFO_H__
#define __LIBDONARD_FIFO_H__

#include <stdlib.h>

struct fifo;


#ifdef __cplusplus
extern "C" {
#endif

struct fifo *fifo_new(size_t entries);
void fifo_free(struct fifo *f);

void fifo_close(struct fifo *f);
void fifo_push(struct fifo *f, void *x);
void *fifo_pop(struct fifo *f);
int fifo_empty(const struct fifo *f);
int fifo_max_fill(struct fifo *f);

#ifdef __cplusplus
}
#endif


#endif

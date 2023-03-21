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

#include "fifo.h"
#include <pthread.h>

struct fifo {
    void **queue;
    int push_ptr, pop_ptr;
    int mask;
    int closed;

    int max_fill;

    pthread_mutex_t mutex;
    pthread_cond_t push_cond, pop_cond;
};

static int is_power_of_two(size_t x)
{
    return (x & (x-1)) == 0;
}

struct fifo *fifo_new(size_t entries)
{
    if (!is_power_of_two(entries))
        return NULL;

    struct fifo *f = (struct fifo*)malloc(sizeof(*f));
    if (f == NULL)
        return NULL;

    if (pthread_mutex_init(&f->mutex, NULL) ||
        pthread_cond_init(&f->push_cond, NULL) ||
        pthread_cond_init(&f->pop_cond, NULL))
    {
        goto error_free;
    }

    f->queue = (void**)malloc(sizeof(*f->queue) * entries);
    if (f->queue == NULL)
        goto error_free;

    f->push_ptr = f->pop_ptr = 0;
    f->mask = entries - 1;
    f->closed = 0;
    f->max_fill = 0;

    return f;

error_free:
    free(f);
    return NULL;
}

void fifo_free(struct fifo *f)
{
    while (pthread_cond_destroy(&f->pop_cond));
    while (pthread_cond_destroy(&f->push_cond));
    while (pthread_mutex_destroy(&f->mutex));

    free(f->queue);
    free(f);
}

const static inline int isempty(const struct fifo *f)
{
    return f->push_ptr == f->pop_ptr;
}

int fifo_empty(const struct fifo *f)
{
    return f->push_ptr == f->pop_ptr;
}

const static inline int isfull(const struct fifo *f)
{
    return ((f->push_ptr + 1) & f->mask) == f->pop_ptr;
}

const static inline int num_available(const struct fifo *f)
{
    return (f->push_ptr - f->pop_ptr) & f->mask;
}

void fifo_close(struct fifo *f)
{
    pthread_mutex_lock(&f->mutex);

    while (!isempty(f))
        pthread_cond_wait(&f->push_cond, &f->mutex);

    f->closed = 1;
    pthread_cond_broadcast(&f->pop_cond);

    pthread_mutex_unlock(&f->mutex);
}

void fifo_push(struct fifo *f, void *x)
{
    pthread_mutex_lock(&f->mutex);

    while (isfull(f))
        pthread_cond_wait(&f->push_cond, &f->mutex);

    f->queue[f->push_ptr] = x;
    f->push_ptr++;
    f->push_ptr &= f->mask;

    pthread_cond_signal(&f->pop_cond);

    if (num_available(f) > f->max_fill)
        f->max_fill = num_available(f);

    pthread_mutex_unlock(&f->mutex);
}

void *fifo_pop(struct fifo *f)
{
    pthread_mutex_lock(&f->mutex);

    while(isempty(f) && !f->closed)
        pthread_cond_wait(&f->pop_cond, &f->mutex);

    if (isempty(f) && f->closed) {
        pthread_mutex_unlock(&f->mutex);
        return NULL;
    }

    void *ret = f->queue[f->pop_ptr];

    f->pop_ptr++;
    f->pop_ptr &= f->mask;

    pthread_cond_signal(&f->push_cond);

    pthread_mutex_unlock(&f->mutex);

    return ret;
}

int fifo_max_fill(struct fifo *f)
{
    pthread_mutex_lock(&f->mutex);

    int ret = f->max_fill;
    f->max_fill = 0;

    pthread_mutex_unlock(&f->mutex);

    return ret;
}

#define NULL ((void*) 0)

struct point {
    float x, y;
    struct point *prev, *next;
};

void point_init(struct point *p) {
    p->x = 0;
    p->y = 0;
    p->prev = NULL;
    p->next = NULL;
}

/* Type your code here, or load an example. */
void join(struct point *a, struct point *b) {
    a->next = b;
    b->prev = a;
}

extern float sqrtf(float);

// test: pass by value calling convention
int list_len(struct point p) {
    int count = 0;
    struct point *cur = &p;
    for(; cur; cur = cur->next) {
        count++;
    }

    return count;
}



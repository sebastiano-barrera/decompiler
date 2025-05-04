#include <stdio.h>

struct mat {
	const int *numbers;
	int rows, cols;
};

typedef int (*mat_proc_t)(struct mat);

int sum_matrix(struct mat mat);

int main()
{
	const int values[3 * 4] = {
		12,  4,  9,  -1,
		 3,  2, 99, -10,
		88, 42, 51,  -3,
	};

	struct mat mat;
	mat.cols = 4;
	mat.rows = 3;
	mat.numbers = &values[0];

	const int sum = sum_matrix(mat);

	printf("sum: %d\n", sum);
	return 0;
}

int sum_matrix(struct mat mat) {
	int sum = 0;
	for (int i=0; i < mat.rows; i++) {
		for (int j=0; j < mat.cols; j++) {
			sum += mat.numbers[i * mat.cols + j];
		}
	}
	return sum;
}


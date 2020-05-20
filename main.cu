#include <iostream>
#include <time.h>
#include <float.h>
#include <fstream>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"

using namespace std;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at " <<
		file << ":" << line << "'" << func << "/n";
		cudaDeviceReset();
		exit(99);
	}
}


__device__ vec3 color(const ray& r, hitable **world) {
	hit_record rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
		return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
	}
	else {
		vec3 unit_direction(r.direction());
		float t = 0.5f*(unit_direction.y()+1.0f);
		return (1.0f-t)*vec3(1.0, 1.0, 1.0)+t*vec3(0.5,0.7,1.0);
	}
}

__global__ void render(vec3 *fb, int max_x, int max_y, 
					   vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
					   hitable **world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j*max_x+i;
    float u = float(i)/float(max_x);
    float v = float(j)/float(max_y);
    ray r(origin, lower_left_corner+u*horizontal+v*vertical);
    fb[pixel_index] = color(r, world);
}

__global__ void create_world(hitable **d_list, hitable **d_world) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(d_list) = new sphere(vec3(0, 0, -1), (0.5));
		*(d_list+1) = new sphere(vec3(0, -100.5, -1), 100);
		*d_world = new hitable_list(d_list, 2);
	}
}

__global__ void free_world(hitable **d_list, hitable_list **d_world) {
	delete *(d_list);
	delete *(d_list+1);
	delete *(d_world);
}

int main() {
	const auto aspect_ratio = 16.0 / 9.0;
	const int image_width = 384;
	const int image_height = static_cast<int>(image_width / aspect_ratio);

	std::ofstream outfile;
	outfile.open("outfile.ppm");
	outfile << "P3\n" << image_width << " " << image_height << "\n255\n";
	
	auto viewport_height = 2.0f;
	auto viewpor_width = aspect_ratio * viewport_height;
	auto focal_length = 1.0f;

	auto origin = vec3(0, 0, 0);
	auto horizontal = vec3(viewpor_width, 0, 0);
	auto vertical = vec3(0, viewport_height, 0);
	auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

	int num_pixels = image_width*image_height;
	size_t fb_size = num_pixels*sizeof(vec3);

	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	hitable **d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(hitable*)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable_list*)));
	create_world<<<1,1>>>(d_list, d_world);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();

	int tx = 16;
	int ty = 16;

	dim3 blocks(image_width/tx+1, image_height/ty+1);
	dim3 threads(tx, ty);

	render<<<blocks, threads>>>(fb, image_width, image_height, lower_left_corner, 
								horizontal, vertical, origin, d_world);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timer_seconds = ((double)(stop-start) / CLOCKS_PER_SEC);
	cout << "Renderd in " << timer_seconds << " [s]." << endl;

	

	for (int j = image_height-1; j >= 0; j--) {
		for (int i = 0; i < image_width; i++) {
			size_t pixel_index = j*image_width+i;
			int ir = int(255.99*fb[pixel_index].x());
			int ig = int(255.99*fb[pixel_index].y());
			int ib = int(255.99*fb[pixel_index].z());

			outfile << ir << " " << ig << " " << ib << "\n";
		}
	}

	outfile.close();
	checkCudaErrors(cudaFree(fb));
}
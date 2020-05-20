#include <iostream>
#include <time.h>
#include <float.h>
#include <fstream>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "camera.h"
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

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 random_unit_vector(curandState *local_rand_state) {
	return vec3(0,0,0);
}

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
	ray cur_ray = r;
	float cur_attenuation = 1.0f;
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
			cur_attenuation *= 0.5f;
			cur_ray = ray(rec.p, target-rec.p);
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f*(unit_direction.y()+1.0f);
			vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0)+t*vec3(0.5,0.7,1.0);
			return cur_attenuation * c;
		}
	}
	// reached max recursion
	return vec3(0.0, 0.0, 0.0);
	
}

/* Initialize rand function so that each thread will have guaranteed distinct random numbrs*/
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j*max_x+i;
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
	
	int pixel_index = j*max_x+i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for (int s = 0; s < ns; s++) {
    	float u = float(i+curand_uniform(&local_rand_state))/float(max_x);
	    float v = float(j+curand_uniform(&local_rand_state))/float(max_y);
	    ray r = (*cam)->get_ray(u,v);
	    col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
    
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(d_list) = new sphere(vec3(0, 0, -1), (0.5));
		*(d_list+1) = new sphere(vec3(0, -100.5, -1), 100);
		*d_world = new hitable_list(d_list, 2);
		*d_camera = new camera();
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	delete *(d_list);
	delete *(d_list+1);
	delete *d_world;
	delete *d_camera;
}

int main() {
	int image_width = 1200;
	int image_height = 600;
	int ns = 100;
	int tx = 8;
	int ty = 8;

	std::ofstream outfile;
	outfile.open("outfile.ppm");
	outfile << "P3\n" << image_width << " " << image_height << "\n255\n";
	
	int num_pixels = image_width*image_height;
	size_t fb_size = num_pixels*sizeof(vec3);

	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	// allocate random generator
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels*sizeof(curandState)));

	hitable **d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(hitable*)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable_list*)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
	create_world<<<1,1>>>(d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();

	dim3 blocks(image_width/tx+1, image_height/ty+1);
	dim3 threads(tx, ty);

	render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render<<<blocks, threads>>>(fb, image_width, image_height, ns, d_camera, d_world, d_rand_state);
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

	checkCudaErrors(cudaDeviceSynchronize());
	free_world<<<1,1>>>(d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}
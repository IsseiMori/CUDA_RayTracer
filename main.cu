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
#include "material.h"

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


__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0,1.0,1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
			else {
				return vec3(0.0,0.0,0.0);
			}
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

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1004, 0, 0, rand_state);
    }
}

/* Initialize rand function so that each thread will have guaranteed distinct random numbrs*/
__global__ void render_init(int max_x, int max_y, int ns, curandState *rand_state) {
	int i = blockIdx.x;
	int j = blockIdx.y;
	int k = threadIdx.x; 
	if((i >= max_x) || (j >= max_y)  || (k >= ns)) return;

	int pixel_index = j*max_x+i;
	int sample_index = pixel_index * ns + k;
	curand_init(sample_index, 0, 0, &rand_state[sample_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world) {
    int i = blockIdx.x;
	int j = blockIdx.y;
	int k = threadIdx.x; 
	if((i >= max_x) || (j >= max_y)  || (k >= ns)) return;

	extern __shared__ vec3 cols[];
	
	int pixel_index = j*max_x+i;
	int sample_index = pixel_index * ns + k;

	curandState local_rand_state;
	curand_init(sample_index, 0, 0, &local_rand_state);

	float u = float(i+curand_uniform(&local_rand_state))/float(max_x);
    float v = float(j+curand_uniform(&local_rand_state))/float(max_y);
    ray r = (*cam)->get_ray(u,v,&local_rand_state);

    cols[k] = color(r, world, &local_rand_state);

    __syncthreads();

    if (k == 0) {

    	vec3 col = vec3(0,0,0);
    	for (int i = 0; i < ns; i++) {
    		col += cols[i];
    	}
    	col /= float(ns);
	    col[0] = sqrt(col[0]);
	    col[1] = sqrt(col[1]);
	    col[2] = sqrt(col[2]);
	    fb[pixel_index] = col;
    }
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	for (int i = 0; i < 22*22+1+3; i++) {
		delete ((sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

int main() {
	int image_width = 1200;
	int image_height = 800;
	int ns = 200;
	int tx = 16;
	int ty = 16;

	std::ofstream outfile;
	outfile.open("outfile.ppm");
	outfile << "P3\n" << image_width << " " << image_height << "\n255\n";
	
	int num_pixels = image_width*image_height;
	size_t fb_size = num_pixels*sizeof(vec3);

	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	// allocate random generator
	// curandState *d_rand_state;
	// checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels*ns*sizeof(curandState)));
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1*sizeof(curandState)));

	rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

	hitable **d_list;
	int num_hitables = 22*22+1+3;
	checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables*sizeof(hitable*)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable_list*)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
	create_world<<<1,1>>>(d_list, d_world, d_camera, image_width, image_height, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();

	dim3 blocks(image_width, image_height);
	dim3 threads(ns);
	// render_init<<<blocks, threads>>>(image_width, image_height, ns, d_rand_state);
	// checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaDeviceSynchronize());
	render<<<blocks, threads, ns*sizeof(vec3)>>>(fb, image_width, image_height, ns, d_camera, d_world);
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
	checkCudaErrors(cudaFree(d_rand_state2));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}